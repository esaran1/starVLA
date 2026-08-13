# Reference checkpoint vs. long-horizon dataset

Reference: [`birbirll/g1-inspire-piston-starvla-oft`](https://huggingface.co/birbirll/g1-inspire-piston-starvla-oft)
(rev `fffc5695bda970356901f8fac122abc709731a47`), trained on
`birbirll/g1-inspire-piston-pick-place` — a **different, shorter** dataset.

The long-horizon dataset is authoritative. Fields marked **DIFFERENT** were taken
from the long-horizon data, never copied from the reference.

| Field | Reference checkpoint | Long-horizon | Verdict |
|---|---|---|---|
| Base VLM | Qwen3-VL-2B-Instruct | Qwen3-VL-2B-Instruct | same |
| Framework | QwenOFT (MLP + L1) | QwenOFT (MLP + L1) | same |
| Robot | unitree_g1 + Inspire | unitree_g1 + Inspire | same |
| Camera keys | `ego_view` (1) | `ego_view` (1) | same |
| Native resolution | 240×424 | 240×424 | same |
| Training image size | 224×224 | 224×224 | same |
| Control frequency | 50 Hz | 50 Hz | same |
| `action_dim` | 30 | 30 | same |
| Action slices | arm7/7, hand6/6, base_h1, nav3 | identical | same |
| Action semantics | absolute `abs_qpos` | absolute `abs_qpos` | same |
| `action_horizon` | 30 | 30 | same |
| Normalization | q99 | q99 | same |
| `unnorm_key` | `new_embodiment` | `new_embodiment` | same |
| State enabled | **no** (`include_state: false`) | **no** | same (rationale re-verified) |
| `state_dim` mapped | 29 (tactile excluded) | 29 (tactile excluded) | same |
| Task count | 1 | 1 | same |
| **Episodes** | **102** | **67** | **DIFFERENT** |
| **Frames (`num_transitions`)** | **27,143** | **45,938** | **DIFFERENT** |
| **Episode length** | ~266 frames avg (5.3 s) | **685.6 avg (13.7 s)** | **DIFFERENT** — 2.6× longer |
| **Task string** | "pick up the piston." | "pick up the piston with the right hand, inject it into the tube held by the left hand, then move it over the hole plate." | **DIFFERENT** |
| **Degenerate action dims** | **19**: 0–6, 14–19, 24–29 | **10**: 14–19, 26–29 | **DIFFERENT** |
| **Left arm (dims 0–6)** | **constant (frozen)** | **moves**, std up to 0.10 rad | **DIFFERENT** |
| **`rh_th_p` (dim 24)** | constant | moves (std 0.088) | **DIFFERENT** |
| **`dataset_statistics.json`** | q01/q99 from 27k pick-place frames | **regenerated** from 45,938 frames | **DIFFERENT — do not reuse** |
| Optimizer | PagedAdamW8bit | PagedAdamW8bit | same (needed an upstream patch here) |
| GPU | RTX 4090, 24 GB | **RTX 4080, 16 GB** | **DIFFERENT** |
| `per_device_batch_size` | **8** | **1** | **DIFFERENT** — 16 GB constraint |
| Grad accumulation | 4 (eff. batch 32) | 8 (eff. batch 8) | **DIFFERENT** |
| DeepSpeed | none (plain accelerate) | none (plain accelerate) | same |
| `attn_implementation` | sdpa | sdpa | same |
| Gradient checkpointing | true | true | same |
| Trained steps | 10,000 (~2h15m) | **not yet run** | pending |

## Why the statistics must be regenerated

The reference's q01/q99 mark the entire left arm as degenerate (`q01 == q99`), so
`Normalizer.forward` would pass those dims through **unnormalized**. In the long-horizon
data the left arm genuinely moves. Reusing the reference statistics would feed raw
radians for dims 0–6 into a head trained to emit [-1, 1], and de-normalization at
deployment would return the raw passthrough rather than a scaled joint target.

Fresh statistics are generated automatically by the dataloader into the run directory
(`dataset_statistics.json`) and are the ones the smoke run used.

## Reference material adopted deliberately

1. **`include_state: false`** — their gotcha #1 documents that state on a stationary task
   makes the policy ignore the camera and fail closed-loop. This dataset is *more*
   susceptible (action/state correlate 0.95–0.999, base never moves).
2. **No DeepSpeed on a single GPU** — their gotcha #2 documents LR-schedule compression
   when combining DeepSpeed with gradient accumulation.
3. **Full fine-tune, no frozen VLM** — their gotcha #3 reports a frozen-VLM sibling run
   converging onto the state shortcut. QwenOFT has no projector.
4. **`policy_norm_processor_PATCHED.py`** — upstream's `PolicyNormProcessor` crashes
   (`Video key ego_view not found`) on any data config whose transform includes video
   augmentation, as ours does. **Required before serving this checkpoint.** Not applied
   yet — deployment was out of scope for this pass.
