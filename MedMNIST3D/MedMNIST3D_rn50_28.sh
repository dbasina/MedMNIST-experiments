#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
OUTROOT=./output
DATA3D=(organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d)
SEEDS=(0 1 2 3 4)

MODEL=resnet50
EPOCHS=100
SIZE=28

BATCH=16            # per-GPU when PER_GPU=1
PER_GPU=1            # 1 => per-GPU batch; 0 => global batch split across GPUs
CKPT_EVERY=0
RUN_PREFIX="r50_3d_${SIZE}"

# 3D specifics
CONV=Conv3d          # {none, ACSConv, Conv2_5d, Conv3d}
PRETRAIN3D=i3d       # only used with Conv3d
USE_SHAPE_TRANSFORM=1
AS_RGB=0             # convert grayscale to RGB (3 channels)

# downloads/args
EXTRA_ARGS="--download"
[[ "${AS_RGB}" -eq 1 ]] && EXTRA_ARGS+=" --as_rgb"
[[ "${USE_SHAPE_TRANSFORM}" -eq 1 ]] && EXTRA_ARGS+=" --shape_transform"

PY=python
TORCHRUN=torchrun
RESUME=1             # set 0 to force fresh runs

# ---- auto-detect GPUs ----
if [[ -n "${SLURM_GPUS_PER_NODE-}" ]]; then
  NGPU="${SLURM_GPUS_PER_NODE%%(*}"
elif [[ -n "${SLURM_NTASKS-}" && -n "${SLURM_GPUS_ON_NODE-}" ]]; then
  NGPU="$SLURM_GPUS_ON_NODE"
else
  NGPU=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
fi

# ---- build launcher & arg separator for torchrun ----
if [[ "${NGPU}" -ge 2 ]]; then
  MP=$(( 10000 + (RANDOM % 50000) ))
  LAUNCH=( ${TORCHRUN} --standalone --nproc_per_node="${NGPU}" --master_port "${MP}" )
  SEP="--"   # everything after this goes to your script (avoids --run collision)
  echo "[info] Using DDP with ${NGPU} GPUs via torchrun (master_port=${MP})"
else
  LAUNCH=( ${PY} )
  SEP=""
  echo "[info] Using single-GPU/CPU launch"
fi

# per-GPU batch flag
if [[ "${PER_GPU}" -eq 1 ]]; then
  PERGPU_FLAG="--per_gpu_batch"
else
  PERGPU_FLAG=""
fi

# ---------- helpers ----------

# Prefer interrupt > last > newest epoch* > best
pick_ckpt () {
  local ds="$1" seed="$2"
  local run_tag="${RUN_PREFIX}_seed${seed}"
  local dir="${OUTROOT}/${ds}/${run_tag}"

  [[ -d "${dir}" ]] || return 1

  if   [[ -f "${dir}/ckpt_interrupt.pth" ]]; then
    echo "${dir}/ckpt_interrupt.pth"
  elif [[ -f "${dir}/ckpt_last.pth" ]]; then
    echo "${dir}/ckpt_last.pth"
  else
    local epoch_ckpt
    epoch_ckpt=$(ls -t "${dir}"/ckpt_epoch*.pth 2>/dev/null | head -n1 || true)
    if [[ -n "${epoch_ckpt}" ]]; then
      echo "${epoch_ckpt}"
    elif [[ -f "${dir}/ckpt_best.pth" ]]; then
      echo "${dir}/ckpt_best.pth"
    else
      return 1
    fi
  fi
}

# - Done if best_model.pth exists
# - Else, done if ckpt_last.pth has epoch >= EPOCHS-1
check_done () {
  local ds="$1" seed="$2"
  local run_tag="${RUN_PREFIX}_seed${seed}"
  local dir="${OUTROOT}/${ds}/${run_tag}"

  [[ -d "${dir}" ]] || return 1
  [[ -f "${dir}/best_model.pth" ]] && return 0

  if [[ -f "${dir}/ckpt_last.pth" ]]; then
    local ep
    ep=$(
      python - <<PY
import torch
p=r"""${dir}/ckpt_last.pth"""
try:
  ckpt=torch.load(p, map_location="cpu", weights_only=False)
  print(ckpt.get("epoch",-1))
except Exception:
  print(-1)
PY
    )
    [[ "${ep:- -1}" -ge $((EPOCHS-1)) ]] && return 0
  fi
  return 1
}

# Ensure a given checkpoint path is readable; prints "OK <epoch>" or "BROKEN"
probe_ckpt () {
  local p="$1"
  python - <<PY
import torch
p = r"""$p"""
try:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    print("OK", ckpt.get("epoch", -1))
except Exception:
    print("BROKEN")
PY
}

# Pre-download to avoid DDP races where non-rank0 tries to use the dataset before it's present
ensure_dataset () {
  local ds="$1"
  local size="$2"
  local as_rgb_flag="$3"   # 0/1

  echo "[info] Ensuring dataset '${ds}' is present (size=${size}, as_rgb=${as_rgb_flag})"
  ${PY} - <<PY
import medmnist, os
from medmnist import INFO
info = INFO["${ds}"]
DataClass = getattr(medmnist, info["python_class"])
# Trigger downloads for all splits (once)
for split in ("train","val","test"):
    DataClass(split=split, download=True, as_rgb=bool(${as_rgb_flag}), size=int(${size}))
print("OK: downloaded/verified ${ds}")
PY
}

run_one () {
  local DS="$1" SEED="$2"
  local RUN_TAG="${RUN_PREFIX}_seed${SEED}"
  local DIR="${OUTROOT}/${DS}/${RUN_TAG}"

  echo "[cfg] DS=${DS} SEED=${SEED} RUN=${RUN_TAG} BATCH(per-GPU)=${BATCH} NGPU=${NGPU} PER_GPU=${PER_GPU}"

  # One-time download (prevents DDP races)
  ensure_dataset "${DS}" "${SIZE}" "${AS_RGB}"

  # Skip finished runs
  if check_done "${DS}" "${SEED}"; then
    echo "[skip] ${RUN_TAG} already complete"
    return 0
  fi

  local RESUME_ARGS=()
  if [[ "${RESUME}" -eq 1 ]]; then
    if CKPT_PATH=$(pick_ckpt "${DS}" "${SEED}"); then
      echo "[resume] ${DS} seed=${SEED} candidate ${CKPT_PATH}"
      status=$(probe_ckpt "${CKPT_PATH}")
      if [[ "${status}" == OK* ]]; then
        echo "[ckpt] $(basename "${CKPT_PATH}") ${status}"
        RESUME_ARGS=( --resume --model_path "${CKPT_PATH}" )
      else
        echo "[warn] primary ckpt unreadable: ${CKPT_PATH}"
        fb=$(ls -t "${DIR}"/ckpt_epoch*.pth 2>/dev/null | head -n1 || true)
        if [[ -n "${fb}" ]]; then
          status=$(probe_ckpt "${fb}")
          if [[ "${status}" == OK* ]]; then
            echo "[resume] falling back to ${fb} ${status}"
            RESUME_ARGS=( --resume --model_path "${fb}" )
          fi
        fi
        if [[ ${#RESUME_ARGS[@]} -eq 0 && -f "${DIR}/ckpt_best.pth" ]]; then
          status=$(probe_ckpt "${DIR}/ckpt_best.pth")
          if [[ "${status}" == OK* ]]; then
            echo "[resume] falling back to ckpt_best.pth ${status}"
            RESUME_ARGS=( --resume --model_path "${DIR}/ckpt_best.pth" )
          else
            echo "[fresh] ${DS} seed=${SEED} (no usable ckpt after fallback)"
          fi
        fi
      fi
    else
      echo "[fresh] ${DS} seed=${SEED} (no checkpoint found)"
    fi
  else
    echo "[fresh] ${DS} seed=${SEED} (RESUME=0)"
  fi

  mkdir -p "${DIR}"
  LOG="${DIR}/train.log"

  "${LAUNCH[@]}" train_and_eval_pytorch.py ${SEP} \
    --data_flag "${DS}" \
    --model_flag "${MODEL}" \
    --num_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --size "${SIZE}" \
    --ckpt_every "${CKPT_EVERY}" \
    --seed "${SEED}" \
    --run "${RUN_TAG}" \
    --output_root "${OUTROOT}" \
    --distributed \
    ${PERGPU_FLAG} ${EXTRA_ARGS} \
    --conv "${CONV}" \
    --pretrained_3d "${PRETRAIN3D}" \
    "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG}"
}

# ---- main loop ----
for DS in "${DATA3D[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "${DS}" "${SEED}"
    sleep 1
  done
done
