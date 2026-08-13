# StarVLA 30-D → IsaacLab 53-D action & camera mapping

Resolves the two interface gaps between the SFT policy and the IsaacLab piston task.
Every row cites direct evidence. Nothing here is inferred from a name alone, and no
missing joint is zero-filled without saying so explicitly.

## Sources of evidence

| Tag | Source |
|---|---|
| `CONTRACT` | `RLinf/docs/contracts/g1_piston_joint_contract.json` — 53 joint names in articulation order, captured from a live sim boot |
| `ADAPTER` | `RLinf/rlinf/envs/isaaclab/tasks/g1_piston.py` — module docstring, upstream contract notes |
| `ACTCFG` | `unitree_sim_isaaclab/tasks/g1_tasks/pick_place_cylinder_g1_29dof_inspire/pickplace_cylinder_g1_29dof_inspire_env_cfg.py:56` (inherited by the piston task) |
| `DEFAULTS` | `unitree_sim_isaaclab/tasks/common_config/robot_configs.py` — default joint pose |
| `INSPIRE` | `unitree_sim_isaaclab/tasks/common_observations/inspire_state.py:32` — Inspire joint names/order |
| `USD` | `unitree_sim_isaaclab/assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd` — the articulation the piston task loads, read with `usd-core` |
| `EGOCAM` | `g1_redball_eval/overlay/.../pickplace_redball_g1_29dof_inspire_joint_env_cfg.py:26-45` — ego camera = D435, dataset resolution |
| `PISTONCAM` | `unitree_sim_isaaclab/tasks/g1_tasks/pick_place_piston_g1_29dof_inspire/...:33` + `tasks/common_config/camera_configs.py` |
| `DATA` | The dataset parquet itself (see `ACTION_CONTRACT.md`) |

## Action space semantics

`ACTCFG`:

```python
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)
```

- `joint_names=[".*"]` → all **53** articulation DOFs are actuated.
- `scale=1.0` → no unit scaling.
- `use_default_offset=True` → the action is an **offset from the articulation default
  pose**: `q_target = q_default + action`.

**Critical**: every entry of the default pose is `0.0` — all 12 leg joints, all 3 waist
joints, all 14 arm joints (`DEFAULTS` lines 27–82) and all 24 Inspire joints
(`DEFAULTS` lines 125–152). Therefore `q_default = 0` and

> `action_53 = q_target_absolute` (radians)

so the dataset's absolute qpos can be written **directly, with no offset conversion**.
This is verified, not assumed — if the default pose is ever changed upstream, this
identity breaks and the conversion column below becomes `q_target − q_default`.

## 30-D → 53-D mapping table

`art` = index into `CONTRACT.joint_names_in_articulation_order`.

| ds dim | dataset name | simulator joint / target | art | conversion | evidence |
|---|---|---|---|---|---|
| 0 | left_arm_shoulder_pitch | `left_shoulder_pitch_joint` | 11 | identity (rad) | CONTRACT+DEFAULTS |
| 1 | left_arm_shoulder_roll | `left_shoulder_roll_joint` | 15 | identity | CONTRACT+DEFAULTS |
| 2 | left_arm_shoulder_yaw | `left_shoulder_yaw_joint` | 19 | identity | CONTRACT+DEFAULTS |
| 3 | left_arm_elbow | `left_elbow_joint` | 21 | identity | CONTRACT+DEFAULTS |
| 4 | left_arm_wrist_roll | `left_wrist_roll_joint` | 23 | identity | CONTRACT+DEFAULTS |
| 5 | left_arm_wrist_pitch | `left_wrist_pitch_joint` | 25 | identity | CONTRACT+DEFAULTS |
| 6 | left_arm_wrist_yaw | `left_wrist_yaw_joint` | 27 | identity | CONTRACT+DEFAULTS |
| 7 | right_arm_shoulder_pitch | `right_shoulder_pitch_joint` | 12 | identity | CONTRACT+DEFAULTS |
| 8 | right_arm_shoulder_roll | `right_shoulder_roll_joint` | 16 | identity | CONTRACT+DEFAULTS |
| 9 | right_arm_shoulder_yaw | `right_shoulder_yaw_joint` | 20 | identity | CONTRACT+DEFAULTS |
| 10 | right_arm_elbow | `right_elbow_joint` | 22 | identity | CONTRACT+DEFAULTS |
| 11 | right_arm_wrist_roll | `right_wrist_roll_joint` | 24 | identity | CONTRACT+DEFAULTS |
| 12 | right_arm_wrist_pitch | `right_wrist_pitch_joint` | 26 | identity | CONTRACT+DEFAULTS |
| 13 | right_arm_wrist_yaw | `right_wrist_yaw_joint` | 28 | identity | CONTRACT+DEFAULTS |
| 14 | left_hand_pinky | `L_pinky_proximal_joint` | 31 | identity | CONTRACT+INSPIRE |
| 15 | left_hand_ring | `L_ring_proximal_joint` | 32 | identity | CONTRACT+INSPIRE |
| 16 | left_hand_middle | `L_middle_proximal_joint` | 30 | identity | CONTRACT+INSPIRE |
| 17 | left_hand_index | `L_index_proximal_joint` | 29 | identity | CONTRACT+INSPIRE |
| 18 | left_hand_thumb_pitch | `L_thumb_proximal_pitch_joint` | 43 | identity | CONTRACT+INSPIRE |
| 19 | left_hand_thumb_yaw | `L_thumb_proximal_yaw_joint` | 33 | identity | CONTRACT+INSPIRE |
| 20 | right_hand_pinky | `R_pinky_proximal_joint` | 36 | identity | CONTRACT+INSPIRE |
| 21 | right_hand_ring | `R_ring_proximal_joint` | 37 | identity | CONTRACT+INSPIRE |
| 22 | right_hand_middle | `R_middle_proximal_joint` | 35 | identity | CONTRACT+INSPIRE |
| 23 | right_hand_index | `R_index_proximal_joint` | 34 | identity | CONTRACT+INSPIRE |
| 24 | right_hand_thumb_pitch | `R_thumb_proximal_pitch_joint` | 48 | identity | CONTRACT+INSPIRE |
| 25 | right_hand_thumb_yaw | `R_thumb_proximal_yaw_joint` | 38 | identity | CONTRACT+INSPIRE |
| 26 | base_height | **no articulation DOF** | — | **see below** | DATA/CONTRACT |
| 27 | nav_vx | **no articulation DOF** | — | **see below** | DATA/CONTRACT |
| 28 | nav_vy | **no articulation DOF** | — | **see below** | DATA/CONTRACT |
| 29 | nav_yaw | **no articulation DOF** | — | **see below** | DATA/CONTRACT |

The 26 joint dims map **1:1 by exact joint name**, with no ambiguity. The Inspire finger
ordering in `INSPIRE` — `(pinky, ring, middle, index, thumb_pitch, thumb_yaw)` — matches
the dataset's hand ordering exactly, which is why dims 14–25 land in a non-monotonic
articulation order (30/31/32 etc.); that is correct, not a bug.

### Dataset dims 26–29 (base_height, nav_vx/vy/yaw)

These have **no articulation DOF** and must **not** be written into the 53-D vector.
They are safe to drop for this task, on direct evidence:

- `DATA`: `nav_vx`, `nav_vy`, `nav_yaw` are **exactly 0.0** in all 45,938 frames, and
  `base_height` is a constant 0.76 m. They carry no signal.
- `PISTONCAM`/scene: the piston task uses `g1_29dof_inspire_base_fix()` — a **fixed-base**
  robot. There is no floating base to command.

The policy will still emit these four numbers (the head is 30-D). The deployment adapter
must slice them off, not forward them.

## The 27 uncovered articulation DOFs

| Group | art indices | Resolution | Why |
|---|---|---|---|
| Legs (12) | 0,1,3,4,6,7,9,10,13,14,17,18 | **Hold at default (0.0)** | Fixed-base task; the dataset never commands them. `q_default = 0`, so an action of 0 holds the default pose. |
| Waist (3) | 2,5,8 | **Hold at default (0.0)** — see caveat | The dataset has waist in **state** (dims 26:29) but **not in action**; it is not commanded during teleop replay. |
| Inspire intermediate/distal (12) | 39–42,44–47,49,50,51,52 | **Hold at default (0.0)** — verified against the USD | Independently driven, no coupling in the asset. See below. |

### Inspire intermediate/distal joints — RESOLVED (they are independently driven)

Read directly from the articulation the piston task loads
(`assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd`,
USD crate 0.8.0, via `usd-core`). Tag: `USD`.

**Finding: there is no coupling in this asset.** All 24 Inspire joints — proximal,
intermediate and distal alike — are `PhysicsRevoluteJoint` prims carrying
`PhysicsDriveAPI:angular`, each independently actuated:

| Joint group | limits (rad) | stiffness | damping | maxForce |
|---|---|---|---|---|
| `*_{index,middle,ring,pinky}_{proximal,intermediate}` | 0 → **1.7000** | 20.0 | 0.01745 | 50.0 |
| `*_thumb_proximal_pitch` | −0.1 → 0.6 | 20.0 | 0.01745 | 150 (L) / 50 (R) |
| `*_thumb_proximal_yaw` | −0.1 → 1.3 | 20.0 | 0.01745 | 150 (L) / 50 (R) |
| `*_thumb_intermediate` | 0 → 0.8 | 20.0 | 0.01745 | 150 (L) / 50 (R) |
| `*_thumb_distal` | 0 → 1.2 | 20.0 | 0.01745 | 150 (L) / 50 (R) |

Exhaustive scan of the stage: **0 mimic attributes**, and the only applied schemas are
`PhysicsArticulationRootAPI`, `PhysicsRigidBodyAPI`, `PhysicsCollisionAPI`,
`PhysicsMeshCollisionAPI`, `PhysicsMassAPI`, `PhysicsDriveAPI:angular`,
`MaterialBindingAPI`, `NodeDefAPI` — **no mimic or tendon schema of any kind**.

Two things this corrects:

- The asset's `config.yaml` records `convert_mimic_joints_to_normal_joints: false`,
  which suggests the source URDF had mimic joints. Whatever the URDF declared, the
  **converted USD that the sim actually loads has no coupling**, so that flag must not
  be read as evidence of coupling at runtime.
- `ADAPTER`'s description of these as "passive/coupled" is accurate about the *physical*
  hand and about what the DDS observation term samples, but **not** about the simulated
  articulation: stiffness 20.0 with maxForce 50–150 is an actively driven joint, not a
  passive one.

IsaacLab's own G1 Inspire task corroborates the joint set, listing exactly these 24 as
`hand_joint_names` with the comment *"All the drive and mimic joints, total 24 joints"*
and `num_hand_joints=24`
(`IsaacLab/source/isaaclab_tasks/.../pickplace_unitree_g1_inspire_hand_env_cfg.py:137-161`),
and its retargeter solves all 12 per hand as independent DOFs rather than deriving the
intermediates from the proximals.

**Consequence for deployment: hold the 12 intermediate/distal joints at their default
target of 0.0.** This is now an evidence-backed choice, not a zero-fill guess:

- `q_default = 0.0` for all 24 Inspire joints (`DEFAULTS`), and `use_default_offset=True`,
  so an action of 0.0 commands exactly the default pose.
- The joints are independently driven, so they will track 0.0 rather than following the
  proximal joint. The fingers will curl at the proximal knuckle and stay straight at the
  middle joint.
- That is a **fidelity limitation, not a correctness error**: the dataset only ever
  commanded 6 DOF per hand, so no information exists in the SFT data to drive the other
  6. The policy cannot be blamed for a pose the demonstrations never contained.

The dataset's commanded hand values were checked against these USD limits and **all 12
hand dims are in range**. The left-hand constant of 1.7000 rad is *exactly* the finger
upper limit — the hand is commanded fully curled to grip the tube — which confirms the
dataset was authored against this same hand model.

### Did the original replay use zero, or a hidden 6→12 expansion? — NOT PROVEN

The USD establishes that 0.0 is a *valid* command that reaches the default pose. It does
**not** establish that 0.0 is what produced the demonstrations. That is a separate
question, and the honest answer is that it could not be settled on this machine.

**A 6→12 coupling expansion demonstrably exists in this ecosystem.** The redball overlay
ships `GR00T-WBC-Bridge/scripts/run_isaac_sim_loop.py`, whose Inspire path does:

```python
# Hands: Inspire 6 -> URDF 12 with coupling
lh_urdf12 = self._mapper.inspire6_to_urdf12(ub[7:13])
rh_urdf12 = self._mapper.inspire6_to_urdf12(ub[20:26])
for j in range(12):
    target[self._lh_art[j]] = float(lh_urdf12[j])
    target[self._rh_art[j]] = float(rh_urdf12[j])
```

with the header documenting `--hand_type inspire  41 DOF, 26 upper body (7+6+7+6),
URDF coupling`. So zero-fill is **not** self-evidently what the data used.

What could not be obtained:

| Artifact | Status |
|---|---|
| `replay_piston_csv.py` (named as the dataset author in `meta/info.json`) | **not present** anywhere on this machine |
| `gr00t_wbc_bridge.hand_mapper.Dex3InspireMapper.inspire6_to_urdf12` | **not installed** — `GR00T-WBC-Bridge` is an optional `install.sh --with-bridge` clone that was never fetched |
| `gr00t_wbc_bridge.g1_inspire_config.URDF_HAND_JOINTS_{LEFT,RIGHT}` | same package, same status |

**The dataset cannot arbitrate this, by construction.** It records exactly 6 hand DOF per
hand in *both* `observation.state` and `action`; the 12 URDF joints appear nowhere. The
upstream observation term
(`get_robot_inspire_joint_states`, `tasks/common_observations/inspire_state.py:108`)
gathers only the 12 **proximal** indices `[36,37,35,34,48,38, 31,32,30,29,43,33]`, so the
intermediates were never observed or logged. And the proximal joints track their own
commands at `corr = 0.993–0.997` regardless of what the intermediates were doing — that
correlation is identical under both hypotheses, so it carries no discriminating
information.

Two further reasons not to assume the bridge path applies to this dataset:

1. The bridge drives a **whole-body 41-DOF** configuration; the piston task the dataset
   targets is the **fixed-base** `g1_29dof_inspire_base_fix` variant.
2. The bridge writes `target = self.robot.data.joint_pos[0].clone()` — it seeds from
   *current positions* so uncontrolled joints hold where they are. The piston task's
   `JointPositionActionCfg(joint_names=[".*"], use_default_offset=True)` instead writes
   an offset from the **default pose** for all 53 DOFs every step. These are different
   write strategies, so the bridge's behaviour does not transfer unexamined.

**Status: documented fidelity caveat, not a correctness blocker** (as directed). Command
0.0 for the 12 joints. If the demonstrations were in fact recorded with coupling applied,
the deployed hand will curl less at the middle phalanges than the demonstrator's did —
a visual/contact-geometry difference confined to the fingers, not a wrong action space,
wrong units, or wrong joint assignment.

To close this properly, obtain `inspire6_to_urdf12` from the `GR00T-WBC-Bridge` repo (or
`replay_piston_csv.py`) and record the ratio here. **Do not reconstruct it by guessing.**
If it turns out the replay used coupling, the fix is a 12-element expansion in the
deployment adapter only — the SFT checkpoint, the 30-D action space and this contract
are unaffected either way.

## Control-rate mismatch (50 Hz vs 100 Hz)

| | Rate | Evidence |
|---|---|---|
| Dataset / policy | **50 Hz** (dt = 0.02 s exactly) | `meta/info.json` `fps: 50.0`; parquet `timestamp` diffs |
| IsaacLab piston task | **100 Hz** | `sim.dt = 0.005`, `decimation = 2` (`pickplace_cylinder_g1_29dof_inspire_env_cfg.py:141,144`, inherited by the piston task) |

The rates differ by exactly 2×, so one policy action must be held for **2 env steps**
(zero-order hold). `hold = ctrl_hz / policy_hz = 100 / 50 = 2`.

### Episode budget — `episode_length_s = 20.0` is sufficient, do NOT raise it

`episode_length_s` is **simulated wall-clock seconds**, not env steps or policy actions.
Under the zero-order hold the two env steps that share one action still consume only
0.02 s of sim time, so sim time and policy time advance together:

```
control dt   = 0.005 * 2                    = 0.01 s      (100 Hz)
env steps    = 20.0 / 0.01                  = 2000
policy acts  = 2000 / 2 (hold)              = 1000
policy time  = 1000 * 0.02                  = 20.0 s      == episode_length_s
demos        = 13.1-14.0 s -> 655-700 acts  = 1310-1400 env steps  (budget 2000)
headroom     = 6.0 s  (600 env steps)
```

The 13.1–14.0 s demonstrations **fit with ~6 s to spare**. An earlier revision of this
document claimed 20 s of sim mapped to ~10 s of policy time and that the episode length
had to be raised; that was wrong — it conflated env steps with policy actions. The
budget is fine as configured.

### Where to implement the 2× hold

The hold belongs at the **simulator/RLinf boundary**, not inside the policy. StarVLA
keeps its native 30-step / 50 Hz action semantics; nothing about the checkpoint, the
action head, or `RLINF_INTERFACE.json` changes.

Do **not** implement it by emitting 60 actions from the policy or by duplicating rows
into the learned representation: that would silently redefine `action_horizon`, break
the contract this directory publishes, and make the SFT checkpoint and the RL rollout
disagree about what one action means.

The natural seam is `IsaaclabBaseEnv.chunk_step` in
`RLinf/rlinf/envs/isaaclab/isaaclab_env.py:153`, which today advances the sim exactly
once per action:

```python
for i in range(chunk_size):
    actions = chunk_actions[:, i]
    ... self.step(actions, auto_reset=False)
```

Applying the hold there — stepping each action `hold` times and accumulating reward
across the repeats — keeps every caller unchanged and confines the rate conversion to
one place. Two consequences the implementer must handle:

1. **Episode-step accounting.** `step()` increments `_elapsed_steps` per env step and
   truncates on `self.cfg.max_episode_steps` (line 130), so `max_episode_steps` must be
   expressed in **env steps** (2000), not policy actions (1000), or episodes truncate at
   half the intended duration.
2. **Chunk-step budget.** Both workers compute
   `n_chunk_steps = max_episode_steps // num_action_chunks`
   (`huggingface_worker.py:103`, `sglang_embodied_worker.py:79`). With
   `max_episode_steps` in env steps and `num_action_chunks = 30`, that yields
   `2000 // 30 = 66` chunk iterations covering 1980 env steps — correct. If instead the
   hold were applied by inflating `num_action_chunks` to 60, the same arithmetic would
   halve the iteration count and the reward/observation bookkeeping would drift.

There is currently **no** action-repeat mechanism anywhere in `rlinf/envs/`, so this has
to be added deliberately. It is an RLinf-side change and is **not** applied here.

## Camera mapping

The dataset's single `ego_view` corresponds to the simulator's **`front_camera`** — same
physical mount — but the *piston* task's optics do **not** match the dataset.

| | Dataset `ego_view` | Piston `front_camera` | Redball overlay ego cam |
|---|---|---|---|
| Mount prim | Intel RealSense D435 on G1 head | `Robot/d435_link/front_cam` | `Robot/d435_link/front_cam` |
| Resolution | 424 × 240 | **640 × 480** | 424 × 240 |
| Aspect | 1.767 | **1.333** | 1.767 |
| Focal length | D435 color | **7.6 mm** | 14.55 mm |
| Horizontal aperture | — | 20.0 mm | 20.0 mm |
| **HFOV** | **~69°** (D435 spec) | **105.5°** | **69.0°** |
| **VFOV** | **~42°** (D435 spec) | **89.2°** | **42.5°** |

`EGOCAM` states it directly: *"dataset observation.images.ego_view is the D435 color
stream at 424x240"*, and derives `focal = 10/tan(34.5°) = 14.55 mm` for a 69° HFOV. My
recomputation reproduces 69.0° / 42.5°, matching the D435 spec (69° / 42°).

The **redball** overlay already applies this fix. The **piston** task does not — it uses
the generic `CameraPresets.g1_front_camera()` default (`PISTONCAM`), which is 640×480 at
7.6 mm → **105.5° HFOV**. That is a far wider view than any training frame: the policy
would receive badly out-of-distribution imagery.

### Required fix before closed-loop evaluation

Give the piston scene the same D435-matched camera the redball overlay uses:

```python
front_camera = CameraBaseCfg.get_camera_config(
    prim_path="/World/envs/env_.*/Robot/d435_link/front_cam",
    height=240, width=424,
    focal_length=14.55, horizontal_aperture=20.0,
    clipping_range=(0.05, 1.0e5))
```

This is a change to `unitree_sim_isaaclab`, **not** to StarVLA, and I have not applied it —
that repo is outside this task's scope.

### Preprocessing to reproduce at inference

Training pipeline, in order (`ACTION_CONTRACT.md`):

1. Decode `ego_view` frame → RGB uint8 HWC at **240 × 424**.
2. Resize to **224 × 224** — a plain, **non-aspect-preserving** squash (no crop, no pad,
   no letterbox). The 1.767 aspect is deliberately distorted; reproduce that squash
   exactly rather than center-cropping.
3. Color jitter (0.3/0.4/0.5/0.08) is **train-only** and must be off at inference
   (`VideoColorJitter.get_transform` is train-gated).

The simulator's 256 × 256 wrist cameras are **not used** — the policy takes exactly one
image. `left_wrist_camera` / `right_wrist_camera` must not be fed.

## Summary

| Item | Status |
|---|---|
| 26 arm/hand dims → articulation | **RESOLVED**, 1:1 by joint name, identity conversion |
| dims 26–29 (base/nav) | **RESOLVED** — drop; no DOF, constant in data, fixed-base task |
| 12 leg + 3 waist DOFs | **RESOLVED** — hold at default 0.0 |
| 12 Inspire intermediate/distal DOFs — *is 0.0 valid?* | **RESOLVED** — independently driven (stiffness 20.0, no mimic/tendon in the USD); 0.0 reaches the default pose |
| 12 Inspire intermediate/distal DOFs — *did the replay use 0.0?* | **NOT PROVEN** — a `inspire6_to_urdf12` coupling exists in `GR00T-WBC-Bridge` but that package and `replay_piston_csv.py` are absent here; the dataset records only 6 DOF/hand so it cannot arbitrate. Documented fidelity caveat. |
| `ego_view` → `front_camera` | **RESOLVED** — same mount, confirmed by EGOCAM |
| Piston camera intrinsics | **MISMATCH** — 105.5° vs 69° HFOV; sim-side fix required |
| Preprocessing | **RESOLVED** — 240×424 → 224×224 squash, no crop, jitter off |
| Control rate | **RESOLVED** — exact 2× ratio; zero-order hold of 2 env steps per action, applied in `chunk_step` at the RLinf boundary |
| `episode_length_s` | **OK as-is (20.0 s)** — 20 s sim = 20 s policy time under the hold; demos need 13.1–14.0 s. Do not raise. |
