#!/usr/bin/env bash
# G1 + Inspire piston LONG-HORIZON — QwenOFT SFT on a single RTX 4080 (16 GB).
#
# Usage:
#   bash examples/realRobots/G1InspirePistonLongHorizon/train_files/run_train.sh          # smoke (20 steps)
#   MAX_STEPS=10000 SAVE_EVERY=1000 RUN_ID=g1-longhorizon-oft-v1 \
#     WANDB_MODE=online bash .../run_train.sh                                             # real SFT
#
# Measured on this machine (Qwen3-VL-2B, micro-batch 1, 224x224, 1 camera,
# grad-ckpt, bf16, action horizon 30):
#   fused AdamW (fp32 states) -> OOM (needs 17.36 GB of optimizer state alone)
#   PagedAdamW8bit            -> 8.95 GB peak / 15.56 GB available.  Use it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
cd "${ROOT_DIR}"

config_yaml=./examples/realRobots/G1InspirePistonLongHorizon/train_files/starvla_qwenoft_g1_longhorizon.yaml
run_root_dir=${RUN_ROOT_DIR:-/home/jren313/research/starvla_rl/checkpoints}
run_id=${RUN_ID:-g1-longhorizon-oft-smoke}

BATCH=${BATCH:-1}
ACCUM=${ACCUM:-8}
MAX_STEPS=${MAX_STEPS:-20}
SAVE_EVERY=${SAVE_EVERY:-100000}
EVAL_EVERY=${EVAL_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-1}

mkdir -p "${run_root_dir}/${run_id}"
cp "${SCRIPT_DIR}/run_train.sh"   "${run_root_dir}/${run_id}/run_train.sh"
cp "${config_yaml#./}"            "${run_root_dir}/${run_id}/$(basename "${config_yaml}")"
cp "${SCRIPT_DIR}/modality.json"  "${run_root_dir}/${run_id}/modality.json"
cp "${SCRIPT_DIR}/data_registry/data_config.py" "${run_root_dir}/${run_id}/data_config.py"
# Reproducibility stamp
{
  echo "starvla_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  echo "starvla_branch=$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)"
  echo "dataset_revision=80fb5a0bbcaf806f5b2cc848fc722f8e005a6830"
  echo "date=$(date -Is)"
  python -c "import torch,transformers;print('torch='+torch.__version__);print('transformers='+transformers.__version__)"
  python -c "import bitsandbytes;print('bitsandbytes='+bitsandbytes.__version__)" 2>/dev/null || true
} > "${run_root_dir}/${run_id}/ENVIRONMENT.txt"

export WANDB_MODE=${WANDB_MODE:-disabled}
# This repo lives outside the editable install target; pin it so the correct
# checkout is imported (the `starvla` env's editable install points elsewhere).
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
# The Accelerator is built at module import, before the cfg loads, so real
# gradient accumulation must be supplied through the environment. Keep this in
# sync with trainer.gradient_accumulation_steps in the YAML.
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${ACCUM}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# train_starvla.py builds a DeepSpeedPlugin at import time by default. This box
# has no CUDA toolkit (no nvcc/CUDA_HOME), so `import deepspeed` raises
# MissingCUDAException. DeepSpeed buys nothing on one GPU -> opt out.
export STARVLA_DISABLE_DEEPSPEED=1
# ...but accelerate ALSO imports deepspeed internally (utils.other.
# extract_model_from_parallel -> unwrap_model), which STARVLA_DISABLE_DEEPSPEED
# cannot prevent. DeepSpeed's installed_cuda_version() only shells out to
# "$CUDA_HOME/bin/nvcc -V" and parses "release X.Y", so a tiny shim satisfies
# the probe. Nothing is compiled: no ZeRO op is ever built or used here.
export CUDA_HOME=${CUDA_HOME:-/home/jren313/research/starvla_rl/pretrained/.cuda_home_shim}

# PLAIN accelerate — no DeepSpeed. ZeRO shards nothing across a single rank, and
# DeepSpeed + gradient_accumulation_steps>1 makes train_starvla.py step the LR
# scheduler per micro-batch, silently compressing the schedule.
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --datasets.vla_data.per_device_batch_size "${BATCH}" \
    --trainer.gradient_accumulation_steps "${ACCUM}" \
    --trainer.max_train_steps "${MAX_STEPS}" \
    --trainer.save_interval "${SAVE_EVERY}" \
    --trainer.logging_frequency "${LOG_EVERY}" \
    --trainer.eval_interval "${EVAL_EVERY}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}"
