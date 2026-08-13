# Action Contract — G1 + Inspire Piston Long-Horizon

**Source of truth** for SFT, deployment, RLinf, and IsaacLab. No downstream config may
contradict this file. Every value below was verified against the actual parquet/video
data, not inferred from metadata or copied from the reference checkpoint.

## Provenance

| Field | Value |
|---|---|
| Dataset | `birbirll/g1-inspire-piston-longhorizon` |
| Dataset revision | `80fb5a0bbcaf806f5b2cc848fc722f8e005a6830` |
| Local path | `~/research/starvla_rl/data/g1-inspire-piston-longhorizon` |
| LeRobot version | v2.1 (`codebase_version: "v2.1"`), read via StarVLA's `lerobot_version: v2.0` code path |
| StarVLA commit | `0ed0aad2c83f587714f6167ef60cf7218b786590` (branch `g1-longhorizon-oft`) |
| Origin | Replay of Minji piston CSVs via `replay_piston_csv.py` (simulation replay, IsaacLab) |

## Robot / control

| Field | Value |
|---|---|
| Robot | `unitree_g1` + Inspire hands (6 DoF per hand) |
| Control frequency | **50 Hz** (`fps: 50.0`; parquet `timestamp` dt = 0.02 s exactly) |
| Episodes | 67 (all `success: true`) |
| Frames | 45,938 |
| Episode length | 654–702 frames (mean 685.6) = **13.1–14.0 s** |
| Tasks | 1 |
| Trajectory groups | `trajectory_csv_1` (38), `trajectory_csv_2` (16), `trajectory_csv_3` (13) |

## Cameras

| Field | Value |
|---|---|
| Camera keys | `observation.images.ego_view` (**single camera**) |
| Modality key | `video.ego_view` |
| Ordering | Deterministic — only one camera exists |
| Native resolution | 240 (H) × 424 (W) × 3, h264, yuv420p, 50 fps |
| Training resolution | 224 × 224 (`obs_image_size`), non-aspect-preserving resize by the loader |

## State

| Field | Value |
|---|---|
| Raw key | `observation.state` |
| Raw dimension | **63** |
| Mapped dimension | **29** (`[0:29]`) |
| Slices | `left_arm 0:7` · `right_arm 7:14` · `left_hand 14:20` · `right_hand 20:26` · `waist 26:29` |
| Units | radians (joint positions) |
| Unmapped | `[29:63]` = 34 tactile channels — **identically zero in all 45,938 frames** (min = max = std = 0), and byte-identical to the separate `observation.tactile` column. Excluded. |
| **Enabled for training** | **NO** — `include_state: false`, `state_dim: 0`. See "State decision". |

## Action

| Field | Value |
|---|---|
| Raw key | `action` |
| Dimension | **30** |
| Slices | `left_arm 0:7` · `right_arm 7:14` · `left_hand 14:20` · `right_hand 20:26` · `base_height 26:27` · `navigate_command 27:30` |
| Type | Joint **position targets** (`abs_qpos`) |
| Absolute / delta | **ABSOLUTE** (`action_mode: abs`) |
| Units | radians for the 26 arm/hand joints; **metres** for `base_height`; `navigate_command` = base (vx, vy, vyaw) in m/s, rad/s |
| Reference frame | Joint space; `navigate_command` is base-frame velocity |
| Controller expectation | Downstream PD position controller tracks these targets |

### Evidence for ABSOLUTE semantics

1. Mean `|action|` over the arm dims is ~0.30 rad — far too large for per-step deltas
   (actual per-step `|Δ|` is ~0.002 rad at 50 Hz).
2. `corr(action[:, i], state[:, i])` = 0.95–0.999 across arm and hand dims.
3. Action leads achieved state by a steady-state PD tracking offset (up to ~0.22 rad on
   the loaded right shoulder), constant over time — a position *target*, not the achieved
   state and not a delta.
4. `mean|action[t] − state[t+k]|` is flat (~0.053) for k ∈ {0,1,2,4}: no lag alignment
   makes action equal state, ruling out "action == next state".

### Degenerate dimensions (this dataset)

Globally constant over all 45,938 frames:

| Dims | Meaning | Constant value |
|---|---|---|
| 14–19 | `left_hand` | 1.7 / 1.7 / 1.7 / 1.7 / 0.35 / 0.25 — frozen, holding the tube |
| 26 | `base_height` | 0.76 m |
| 27–29 | `navigate_command` | 0.0 — **stationary task, no locomotion** |

These are **kept** in the 30-D action vector — never shrink the robot action dimension.
`q99` normalization detects `q01 == q99` and passes those dims through unchanged
(`Normalizer.forward`), yielding a constant finite target and no NaN.

> **Differs from the reference checkpoint**, whose dataset had 19 degenerate dims
> including the entire left arm (0–6). Here the left arm **moves** (std up to 0.10 rad)
> and `rh_th_p` (24) moves. Do not reuse reference statistics.

## Task / language

| Field | Value |
|---|---|
| Language key | `annotation.human.task_description` |
| `original_key` | `task_index` (int64 column → `meta/tasks.jsonl`) |
| Task string | "pick up the piston with the right hand, inject it into the tube held by the left hand, then move it over the hole plate." |

> The annotation subkey in `modality.json` must be **flat** (`"human.task_description"`).
> The nested form in the bundled template does not work with this loader.

## Training

| Field | Value |
|---|---|
| Action horizon | **30** (= `len(action_indices)`; `future_action_window_size` 29) |
| Horizon in time | 0.6 s @ 50 Hz |
| Observation indices | `[0]` (current frame only) |
| Execution semantics | Predict 30 absolute joint-position targets; execute open-loop then re-plan |
| Episode boundaries | Chunks clamp to the final frame (`first_last` padding for absolute actions); **verified zero cross-episode leakage** |
| Normalization | `q99` on all state and action keys, clamped to [−2.2, 2.2] |
| Statistics | Generated fresh from THIS dataset — never reuse the reference checkpoint's |
| `unnorm_key` | `new_embodiment` |

## State decision (Stage 8)

**State is OFF** (`include_state: false`, `state_dim: 0`), matching the reference
checkpoint — but adopted here on its documented rationale, not by imitation:

> With state enabled on a stationary task, the model regresses actions from
> proprioception and ignores the camera. It converges in loss and fails closed-loop.
> StarVLA has no state-dropout to mitigate this.
> — `birbirll/g1-inspire-piston-starvla-oft` README, gotcha #1

This dataset is *more* susceptible, not less: the base never moves
(`navigate_command ≡ 0`), and action/state correlate at 0.95–0.999, so proprioception is
a near-perfect shortcut for the next action. The `state_keys` remain defined in
`data_config.py` so a state-enabled ablation is a one-line YAML change.

**UNRESOLVED:** whether the eventual RLinf/IsaacLab controller requires proprioceptive
input. If it does, wire the dormant `StateActionDropout` transform before enabling state.
