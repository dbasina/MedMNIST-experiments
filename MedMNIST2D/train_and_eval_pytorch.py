import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from torchvision.models import resnet18, resnet50
from tqdm import trange

def _is_training_ckpt(d):
    return isinstance(d, dict) and 'optimizer' in d and 'epoch' in d and 'scheduler' in d and 'net' in d

def ddp_is_init():
    return dist.is_available() and dist.is_initialized()

def ddp_rank():
    return dist.get_rank() if ddp_is_init() else 0

def is_main_process():
    return ddp_rank() == 0

def setup_ddp():
    """Initialize DDP from torchrun environment."""
    local_rank = int(os.environ["LOCAL_RANK"])     # 0..(nproc_per_node-1)
    torch.cuda.set_device(local_rank)              # <-- set device FIRST
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank

def cleanup_ddp():
    if ddp_is_init():
        dist.destroy_process_group()

def unwrap(m):
    return m.module if isinstance(m, (DDP, torch.nn.DataParallel)) else m

def set_seed(seed, rank = 0):
    seed = int(seed) + int(rank)*1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run, resume, ckpt_every, seed, distributed, per_gpu_batch):
    lr = 0.001
    gamma=0.1
    milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    if distributed:
        local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0
    set_seed(seed, rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # If resuming from a checkpoint, reuse that folder. Otherwise create a fresh timestamped run dir.
    run_dir = os.path.join(output_root, data_flag, run)
    if model_path is not None and resume:
        # trust the checkpoint location for safety
        output_root = os.path.dirname(os.path.abspath(model_path))
    else:
        output_root = run_dir
    
    if is_main_process():
        os.makedirs(output_root, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[device.index])

    if is_main_process():
        print('==> Preparing data...')
        print('==> Building and training model...')

    if as_rgb:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean, std = [0.5], [0.5]

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
     
    do_dl = download and ((not distributed) or is_main_process())
    train_dataset = DataClass(split='train', transform=data_transform, download=do_dl, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=do_dl, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=data_transform, download=do_dl, as_rgb=as_rgb, size=size)
    if distributed:
        dist.barrier(device_ids=[device.index])

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas= world_size, rank = rank, shuffle = True, drop_last = False)
    else:
        train_sampler = None
    
    if distributed and not per_gpu_batch:
        assert batch_size % world_size == 0, "global --batch_size must be divisible by world_size"
        eff_batch = batch_size // world_size
    else:
        eff_batch = batch_size

    # # --- Global batch & LR scaling ---
    # global_bs = eff_batch * world_size

    # # Base LR tuned for global_bs=256 (typical for Adam on MedMNIST baselines)
    # base_lr = 1e-3
    # if global_bs <= 256:
    #     lr = base_lr              # keep 1e-3 at 128–256
    # elif global_bs <= 512:
    #     lr = base_lr * (global_bs / 256)
    # else:
    #     lr = min(base_lr * (global_bs / 256)**0.5, 3e-3)  # sqrt + cap for huge batches

    # if is_main_process():
    #     print(f"[info] per_gpu_batch={eff_batch} world_size={world_size} global_bs={global_bs} lr={lr:.6g}", flush=True)

    # --- Paper LR scaling no batch-size scaling ---
    lr = 1e-3
    if is_main_process():
        print(f"[info] per_gpu_batch={eff_batch} world_size={world_size} global_bs={eff_batch*world_size} lr={lr:.6g}", flush=True)


    try:
        # how many CPUs are *actually* available to this process
        avail_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        avail_cpu = os.cpu_count() or 4

    per_proc_cpu = max(1, avail_cpu // max(1, world_size))
    # Cap to 4 unless you confirm higher is stable on this cluster
    train_workers = min(4, per_proc_cpu)
    eval_workers  = max(1, min(2, train_workers))

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=eff_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last = False,
    )

    eval_kwargs = dict(
        batch_size=eff_batch,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2
    )
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, **eval_kwargs)
    val_loader   = data.DataLoader(dataset=val_dataset, **eval_kwargs)
    test_loader  = data.DataLoader(dataset=test_dataset, **eval_kwargs)
    
    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    if distributed:
        # Convert BN -> SyncBN BEFORE DDP wrap
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()


    # Paper scheduler: MultiStepLR at 50% and 75% of total epochs
    milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


    start_epoch = 0
    best_auc = 0.0
    best_epoch = -1
    best_state = None 

    # --- Resume or evaluate-only logic ---
    if model_path is not None:
        ckpt = torch.load(model_path, map_location=device, weights_only= False)
        print(f"[resume] loading: {model_path}", flush=True)

        if resume:
            if not _is_training_ckpt(ckpt):
                raise ValueError(f"--resume requested but {model_path} is not a training checkpoint.")
            # handle new/old formats
            net_key = 'net_current' if 'net_current' in ckpt else 'net'
            unwrap(model).load_state_dict(ckpt[net_key], strict=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = lr
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_auc   = ckpt.get('best_auc', 0.0)
            best_epoch = ckpt.get('best_epoch', -1)
            # try to restore best_model too
            if 'net_best' in ckpt and ckpt['net_best'] is not None:
                best_state = ckpt['net_best']
            print(f"=> Resumed from ckpt_epoch={ckpt.get('epoch')} -> start_epoch={start_epoch} | best_auc={best_auc:.5f} @ epoch {best_epoch}")
            # one-time sanity eval
            if (not distributed) or is_main_process():
                model.eval()
                with torch.no_grad():
                    sanity_val = test(model, val_evaluator, val_loader, task, criterion, device, run + "_resume_sanity", output_root)
                print(f"[resume] sanity val: auc={sanity_val[1]:.5f} acc={sanity_val[2]:.5f}")
                model.train()

        else:
            # Weight-only file (e.g., best_model.pth): load weights then evaluate-only or continue fresh schedule
            unwrap(model).load_state_dict(ckpt['net'], strict=True)
            print("=> Loaded weights only (no optimizer/scheduler state).")
            # You can still do the three immediate evaluations below if desired:
            train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
            val_metrics   = test(model,  val_evaluator,  val_loader,          task, criterion, device, run, output_root)
            test_metrics  = test(model,  test_evaluator, test_loader,         task, criterion, device, run, output_root)
            print('train  auc: %.5f  acc: %.5f\nval    auc: %.5f  acc: %.5f\ntest   auc: %.5f  acc: %.5f\n'
                % (train_metrics[1], train_metrics[2], val_metrics[1], val_metrics[2], test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results')) if is_main_process() else None

    global iteration
    iteration = 0
    last_epoch = start_epoch - 1

    for epoch in trange(start_epoch, num_epochs):
        if distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        last_epoch = epoch     
        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer)
        
        if (not distributed) or is_main_process():
            # train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
            # val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
            # test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)
            train_metrics = [train_loss, 0.0, 0.0]          # skip full train-set eval
            val_metrics   = test(model, val_evaluator, val_loader, task, criterion, device, run)

            if val_metrics[1] > best_auc:
                # Only now run test (and save ckpt)
                test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)
            else:
                test_metrics = [0.0, 0.0, 0.0]
        else:
            train_metrics = val_metrics = test_metrics = [0.0,0.0,0.0]
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        if writer is not None:
            for key, value in log_dict.items():
                writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_state = deepcopy(unwrap(model).state_dict())
            if is_main_process():
                print('cur_best_val_auc:', best_auc)
                print('cur_val_acc:', val_metrics[2])
                print('cur_test_auc:', test_metrics[1])
                print('cur_test_acc:', test_metrics[2])
                print('cur_best_epoch', best_epoch)

        
        ckpt_last = {
            'epoch': epoch,
            'net': unwrap(model).state_dict(),   # current weights
            'net_best': best_state,              # may be None early on
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_auc': best_auc,
            'best_epoch': best_epoch,
            'data_flag': data_flag,
            'run': run,
        }
        if is_main_process():
            torch.save(ckpt_last, os.path.join(output_root, 'ckpt_last.pth'))
            if ckpt_every and ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                torch.save(ckpt_last, os.path.join(output_root, f'ckpt_epoch{epoch:03d}.pth'))
            if best_epoch == epoch and best_state is not None:
                torch.save(ckpt_last, os.path.join(output_root, 'ckpt_best.pth'))



    if is_main_process() and best_state is not None:
        torch.save({'net': best_state}, os.path.join(output_root, 'best_model.pth'))


    # Final eval on the BEST weights
    if (not distributed) or is_main_process():
        eval_model = deepcopy(model)               # same arch/device/SyncBN/DDP unwrap ok
        if best_state is not None:
            unwrap(eval_model).load_state_dict(best_state, strict=True)

        train_metrics = test(eval_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics   = test(eval_model,  val_evaluator,  val_loader,          task, criterion, device, run, output_root)
        test_metrics  = test(eval_model,  test_evaluator, test_loader,         task, criterion, device, run, output_root)
    else:
        train_metrics = val_metrics = test_metrics = [0.0, 0.0, 0.0]


    if distributed:
        dist.barrier(device_ids=[device.index])

    if (not distributed) or is_main_process():
        train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
        val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
        test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log
        
        print(log)
                
        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log) 
    
    if writer is not None:        
        writer.close()

    if distributed:
        cleanup_ddp()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    use_amp = False

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device, non_blocking=True)
        else:
            targets = torch.squeeze(targets, 1).long().to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        if writer is not None:
            writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    model.eval()
    total_loss = []
    y_chunks = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            outputs = model(inputs)
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device, non_blocking=True)
                loss = criterion(outputs, targets)
                outputs = torch.sigmoid(outputs)
            else:
                targets = torch.squeeze(targets, 1).long().to(device, non_blocking=True)
                loss = criterion(outputs, targets)
                outputs = torch.softmax(outputs, dim=1)
            total_loss.append(loss.item())
            y_chunks.append(outputs.detach().cpu())
    y_score = torch.cat(y_chunks, dim=0).numpy()
    auc, acc = evaluator.evaluate(y_score, save_folder, run)
    test_loss = sum(total_loss) / len(total_loss)
    return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=28,
                        help='the image size of the dataset, 28 or 64 or 128 or 224, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--resume', 
                        action='store_true',
                        help='resume training from a checkpoint (provide --model_path)')
    parser.add_argument('--ckpt_every', 
                        type=int, default=1,
                        help='save a checkpoint every N epochs (default: 1)')
    parser.add_argument('--seed', 
                        type=int, default=0,
                        help='random seed for reproducibility')
    parser.add_argument('--distributed', 
                        action='store_true',
                        help='Use DistributedDataParallel (torchrun).')
    parser.add_argument('--per_gpu_batch', 
                        action='store_true',
                        help='Interpret --batch_size as per-GPU batch size. ''If not set, batch_size is global and will be split across ranks.')



    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    resume = args.resume
    ckpt_every = args.ckpt_every
    seed = args.seed
    distributed = args.distributed
    per_gpu_batch = args.per_gpu_batch

    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run, resume, ckpt_every, seed, distributed, per_gpu_batch)
