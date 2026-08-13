# G1 + Inspire — Piston Long-Horizon (QwenOFT)

StarVLA integration for [`birbirll/g1-inspire-piston-longhorizon`](https://huggingface.co/datasets/birbirll/g1-inspire-piston-longhorizon):
67 episodes / 45,938 frames of a Unitree G1 with Inspire hands performing a
~13.7 s bimanual piston insertion, targeted at a **single 16 GB RTX 4080**.

| Document | Purpose |
|---|---|
| [`ACTION_CONTRACT.md`](ACTION_CONTRACT.md) | **Source of truth.** Dataset facts, action semantics, units, normalization. |
| [`REFERENCE_COMPARISON.md`](REFERENCE_COMPARISON.md) | Field-by-field diff against the reference G1 OFT checkpoint. |
| [`RLINF_INTERFACE.json`](RLINF_INTERFACE.json) | Machine-readable handoff contract for RLinf / IsaacLab. |

## Layout

```
train_files/
├── data_registry/data_config.py           # auto-discovered registry entry
├── modality.json                          # copy into <dataset>/meta/modality.json
├── starvla_qwenoft_g1_longhorizon.yaml    # training config
└── run_train.sh                           # single-GPU launcher
```

## Quick facts

| | |
|---|---|
| Robot / control | `unitree_g1` + Inspire hands, **50 Hz** |
| Camera | 1 × `ego_view`, 240×424 → 224×224 |
| Action | **30-D absolute joint position targets** (`abs_qpos`), horizon **30** (0.6 s) |
| State | **disabled** (`include_state: false`) — see the contract for why |
| Normalization | `q99`, `unnorm_key: new_embodiment` |

## Setup

```bash
# 1. Dataset
hf download birbirll/g1-inspire-piston-longhorizon --repo-type dataset \
  --local-dir ~/research/starvla_rl/data/g1-inspire-piston-longhorizon

# 2. modality.json (REQUIRED — the shipped one has a nested annotation key and
#    omits original_key/absolute/dtype)
cp train_files/modality.json ~/research/starvla_rl/data/g1-inspire-piston-longhorizon/meta/

# 3. Base VLM
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ~/research/starvla_rl/pretrained/Qwen3-VL-2B-Instruct

# 4. Deps not in the base env
pip install bitsandbytes    # PagedAdamW8bit — required to fit 16 GB
```

## Validate, then train

```bash
# dataloader
python -m starVLA.dataloader.lerobot_datasets \
  --config_yaml examples/realRobots/G1InspirePistonLongHorizon/train_files/starvla_qwenoft_g1_longhorizon.yaml

# static invariants
pytest tests/test_g1_longhorizon_integration.py -v

# 20-step smoke
bash examples/realRobots/G1InspirePistonLongHorizon/train_files/run_train.sh

# real SFT
MAX_STEPS=10000 SAVE_EVERY=1000 RUN_ID=g1-longhorizon-oft-v1 \
  bash examples/realRobots/G1InspirePistonLongHorizon/train_files/run_train.sh
```

## Measured on this machine (RTX 4080, 16 GB)

| Configuration | Peak VRAM | Result |
|---|---|---|
| Model load (bf16) | 4.42 GB | — |
| Forward + backward, micro-batch 2 | 10.43 GB | finite loss + grads |
| **Full fine-tune, fused AdamW (fp32 states)** | — | **OOM** (needs 17.36 GB of optimizer state) |
| **Full fine-tune, PagedAdamW8bit, micro-batch 1** | **8.95 GB** | **fits, 6.6 GB headroom** |

Throughput ≈ 1.05 it/s at micro-batch 1.

## Environment notes

Two upstream limitations required minimal, default-preserving patches to
`starVLA/training/train_starvla.py`:

1. `setup_optimizer_and_scheduler` hardcoded `torch.optim.AdamW(fused=True)` and ignored
   `trainer.optimizer.name`. It now honours `paged_adamw_8bit`. Default unchanged.
2. A `DeepSpeedPlugin` was built unconditionally at import. DeepSpeed shards nothing on
   one GPU and `import deepspeed` hard-fails without a CUDA toolkit. Now opt-out via
   `STARVLA_DISABLE_DEEPSPEED=1`. Default unchanged.

`accelerate` also imports `deepspeed` internally from `unwrap_model`, which the flag
cannot prevent; `run_train.sh` points `CUDA_HOME` at a tiny `nvcc -V` shim so DeepSpeed's
version probe succeeds. Nothing is compiled.

> The `starvla` conda env has an editable install of StarVLA pointing at a **different
> checkout** (`/home/jren313/starVLA`). `run_train.sh` pins `PYTHONPATH` to this repo.
> Run other scripts from the repo root or export `PYTHONPATH` yourself.

## Status

Data and SFT plumbing are validated end-to-end (dataloader gates, forward/backward,
20-step training, 91% tiny-overfit reduction). **No policy has been trained and no
closed-loop evaluation has been run** — the only checkpoint produced is a 20-step smoke
artifact.
