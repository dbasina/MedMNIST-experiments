import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy
import torch
import medmnist
import numpy as np
import PIL
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from torchvision.models import resnet18, resnet50
import timm
from tqdm import trange
import math
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run):

    
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
  
    num_visible = torch.cuda.device_count()
    use_cuda = num_visible > 0
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # if using pretrained_weights:    
    as_rgb = True

    data_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    data_transform_eval = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # else:
    #     # your existing behavior for non-Swin models
    #     data_transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=IMAGENET_MEAN if as_rgb else [.5],
    #             std=IMAGENET_STD if as_rgb else [.5]
    #         ),
    #     ])
    #     data_transform_eval = data_transform_train

    train_dataset = DataClass(root=args.data_root, split='train',
                            transform=data_transform_train,
                            download=False, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(root=args.data_root, split='val',
                            transform=data_transform_eval,
                            download=False, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(root=args.data_root, split='test',
                            transform=data_transform_eval,
                            download=False, as_rgb=as_rgb, size=size)

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True)

    print('==> Building and training model...')
    
    
    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  timm.create_model('resnet50', pretrained=False, num_classes = n_classes, in_chans = 3)
    elif model_flag == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes = n_classes, in_chans = 3)
    else:
        raise NotImplementedError

    if use_cuda:
        device_ids = list(range(num_visible))
        model = model.to(device)       
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    else:
        # CPU fallback
        model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size, root = args.data_root)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size, root = args.data_root)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size, root = args.data_root)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    # if model_flag != "swin":
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    base_lr = 2e-4
    weight_decay = 0.02 
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=weight_decay)
    num_epochs_total = num_epochs
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, (num_epochs_total - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))
    
    target_auc = {
    "chestmnist": 0.773,
    "pathmnist": 0.989,
    "dermamnist": 0.912,
    "octmnist": 0.958,
    "pneumoniamnist": 0.962,
    "retinamnist": 0.716,
    "breastmnist": 0.866,
    "bloodmnist": 0.997,
    "tissuemnist": 0.932,
    "organamnist": 0.998,
    "organcmnist": 0.993,
    "organsmnist": 0.975,
    }

    patience = 10
    bad_epochs = 0
    best_auc = -1.0
    best_model = deepcopy(model)
    global iteration
    iteration = 0
    target = target_auc[data_flag]
    print("Target to hit: ", target_auc[data_flag])

    for epoch in trange(num_epochs, disable = True):

        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer)
        # train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run)
        
        scheduler.step()
        
        # for i, key in enumerate(train_logs):
        #     log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        cur_test_auc = test_metrics[1]
        
        if cur_auc > best_auc + 1e-4:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)
            bad_epochs = 0
        else:
            bad_epochs +=1
            if bad_epochs >= patience:
                break

        epoch_log = f"epoch: {epoch}, val auc: {val_metrics[1]},  test auc: {test_metrics[1]}\n"
        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(epoch_log)
        


    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  
            
    writer.close()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
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
                        default=224,
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
    parser.add_argument('--data_root',
                        default='/scratch/dbasina/MedMNIST',
                        type=str,
                        help='where MedMNIST .npz files live')
    

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
    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run)
