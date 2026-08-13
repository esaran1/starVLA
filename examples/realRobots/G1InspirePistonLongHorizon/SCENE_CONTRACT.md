# Training-scene contract — G1 + Inspire piston long-horizon

What the `birbirll/g1-inspire-piston-longhorizon` demonstrations were recorded in, versus
what the currently-installed piston task builds, and what must change to close the gap.

**Status: scene source RECOVERED from tracked git history and RECONSTRUCTED.** Scene-only
validation (no StarVLA policy) has been rendered from the live simulator — see
"Validation" at the end. Closed-loop evaluation has **not** been run.

## Provenance

The authoritative scene is **`g1_redball_eval`, commit `02fddc3`**
*"Add left-palm tube and right-side plate to piston scene"* (zion &lt;zgu78@gatech.edu&gt;,
2026-07-25), on branch **`origin/minji/piston-pot`** — "Minji" matching the dataset's own
provenance string in `meta/info.json`:

> `"author": "bir (replay of Minji piston CSVs via replay_piston_csv.py)"`

`g1_redball_eval/overlay/` is the tracked source of truth: `install.sh` copies it into the
`unitree_sim_isaaclab` working checkout, so the overlay — not the installed copy — is what
must be edited.

**The current branch (`pipette-stand-collision`, HEAD `d34971c`) is a REGRESSION of that
scene.** `02fddc3` is not an ancestor of HEAD; the branches diverged. Two later commits
(`c910509`, `2d35f58`) added the barrel lip / StandTest work and, in doing so, reverted or
commented out the tube, the pot, the de-cluttered tables, and the tube-rack position.

### Sources searched

| Order | Source | Result |
|---|---|---|
| 1 | Dataset metadata (`info.json`, `episodes.jsonl`, README) | **HIT** — names the author and `replay_piston_csv.py`; `final_xyz_sim` gives a hard target coordinate |
| 2 | StarVLA G1 examples / dataset docs | miss — no scene definitions |
| 3 | `g1_redball_eval` branches / history / stashes | **PRIMARY HIT** — `02fddc3` on `origin/minji/piston-pot`; no stashes |
| 4 | `unitree_sim_isaaclab` branches / history | miss — upstream vendor repo, contains no piston work |
| 5 | `replay_piston_csv.py` | **NOT PRESENT** on this machine |
| 6 | Asset names (stand, rack, hole plate) | **HIT** — `holeplate_object.py`, `pipette_stand_object.py`, `pot_object.py`, `piston_object.py`, `external_mesh_object.py` all present |
| 7 | Old launch/config scripts | partial — `run_isaac_sim_loop.py` (whole-body bridge, different config) |
| 8 | Reference checkpoint / deployment files | miss — only "Closed-loop stable in IsaacLab" |
| 9 | Original producing machine/repo | **NOT AVAILABLE** — `parallel/batched_autonomous_piston.py`, cited in `02fddc3`'s comments as the source of the measured grip offsets, is not on this machine |

## Numeric evidence from the dataset

Over all 67 episodes, the piston's final position (`final_xyz_sim`):

| axis | mean | std | min | max |
|---|---|---|---|---|
| x | **0.1505** | 0.0022 | 0.1452 | 0.1541 |
| y | **0.3270** | 0.0034 | 0.3156 | 0.3373 |
| z | **1.0091** | 0.0026 | 1.0035 | 1.0202 |

Two conclusions:

1. **No reset randomization of the target.** A 2–3 mm spread across 67 episodes is
   settling noise, not sampling.
2. It **confirms `02fddc3`**: that commit's `PLATE_OFFSET` puts the pot at
   xy = (0.15, 0.30), and the observed final xy is (0.150, 0.327). The z of 1.009 sits
   ~0.21 m above the table top (0.794), consistent with the piston being held above the
   plate rather than resting on it.

Also recorded per episode: `tube_disp_sim_m` mean 0.094 (the tube moves ~9 cm — it is
genuinely grasped and carried), and `final_pos_err_m` mean 0.0037.

## Scene contract

Legend — confidence: **High** = recovered from tracked config/code or dataset numerics;
**Medium** = corroborated by demo video + partial config; **Low** = inferred.

| Element | Training / demo configuration | Current simulator configuration | Authoritative evidence | Required change | Confidence |
|---|---|---|---|---|---|
| **Piston pose** | Spawn xy = `PISTON_XY` (−0.15, 0.40), z = `SPAWN_Z` 0.92 | identical | `02fddc3` + HEAD agree | none | **High** |
| **Piston orientation** | **UPRIGHT**, identity quat `(1,0,0,0)`, rod axis along +Z | identical | `_UPRIGHT_ROT`, both revisions; demo frames show the piston vertical | none — **note: the task brief's "horizontal on the stand" does not match the data; the piston is upright** | **High** |
| **Piston stand** | A **light-blue open-topped box** with the piston barrel seated inside it. This **is** the socket-wall assembly: 4 kinematic cuboids (`_SOCKET_HALF` 0.032, `_WALL_T` 0.02, `_WALL_H`) with `diffuse_color=(0.3, 0.5, 0.9)` — blue — forming a square well around the piston base | identical (same walls, same blue material) | `_socket_wall()` in both revisions; demo zoom `ZOOM_piston_ep000.png` | none — **no missing pedestal prop** | **High** |
| **Graspable tube (left palm)** | `tube` = dynamic `CylinderCfg`, r = 0.010, h = 0.10, mass 0.02, static/dynamic friction 1.5, spawned at `_TUBE_POS` = **(−0.287, 0.364, 0.878)** in the closed left-palm grip channel | **ABSENT** — never defined | `02fddc3`; offsets measured live from `parallel/batched_autonomous_piston.py` DBG dump | **ADD** | **High** |
| **Tube rack** | `_TUBE_RACK_XY` = **(−0.45, 0.35)** — out to the robot's left, clear of the arm and centrifuge | `(0.10, 0.40)` — back-left of tray | `02fddc3` vs HEAD; demo frames show the white multi-well rack on the **left** | **MOVE** to (−0.45, 0.35) | **High** |
| **Hole plate** | `pot` = `RigidObjectCfg(spawn=PotObjectCfg())` at `_POT_POS`, with `02fddc3` noting *"Swap in HolePlateObjectCfg() for the hole plate"*. Demo frames show a tan **5×3 grid** plate on the right — matching `HolePlateObjectCfg` defaults (`holes_x=5, holes_y=3`, colour (0.6,0.4,0.2)) | `pot` is **aliased to the centrifuge-tube mesh** (`pot = AssetBaseCfg(... CentrifugeTube ...)`); the real pot is commented out | `02fddc3` vs HEAD; `holeplate_object.py` defaults; demo frames | **REPLACE** — restore `pot` as a real object and spawn `HolePlateObjectCfg()` | **High** (that a plate belongs) / **Medium** (that `HolePlateObjectCfg` defaults are the exact authored parameters) |
| **Hole-plate position** | `_POT_POS` = `PISTON_XY + PLATE_OFFSET` = **(0.15, 0.30, 0.80)** | same formula, same constants | `02fddc3`; corroborated by `final_xyz_sim` xy = (0.150, 0.327) | none | **High** |
| **Mini centrifuge** | `_MINI_CENTRIFUGE_XY` = (−0.32, 0.575), static trimesh collider | identical | both revisions; demo frames show it back-centre | none | **High** |
| **Pipette stand** | `STAND_XY` = (−0.05, 0.60), z = 0.80, yaw −120° `_STAND_ROT` = (0.5, 0, 0, −0.8660254) | identical **except** the StandTest variant overrides the yaw | both revisions; demo frames show it back-left with coloured tips | none for the base task; **do not use `TablePistonStandTestCfg`** | **High** |
| **Table / workspace** | 6 × `PackingTable` at `_TABLE_POSES`, spawned through the generated **`PackingTable_NoProps.usda`** de-clutter layer (deactivates `container_h20` + 7 lower-shelf crates). Top at z = 0.794, x ∈ [−1.219, 1.219], y ∈ [0.169, 0.931] | de-clutter layer **ABSENT** — stock cluttered tables | `02fddc3`; demo frames show a clean white table with no basket/crates | **ADD** the `_packing_table` wrapper + 6 table entries | **High** |
| **Robot initial pose** | `G1RobotPresets.g1_29dof_inspire_base_fix()`, **fixed base**, all 53 default joint positions 0.0 | identical | both revisions; `robot_configs.py` | none | **High** |
| **Camera mount** | `/World/envs/env_.*/Robot/d435_link/front_cam` | identical mount | `EGOCAM`; both revisions | none | **High** |
| **Camera intrinsics** | D435 colour: 424×240, focal 14.55 mm, aperture 20 mm, clip (0.05, 1e5) → **69.0° H / 42.5° V** | **FIXED** in `g1_redball_eval@6c1190b` (was 640×480 / 7.6 mm → 105.5°/89.2°) | `EGOCAM`; datasheet; FOV arithmetic | done — now in the tracked overlay | **High** |
| **Collisions** | Socket walls kinematic; centrifuge + rack **static trimesh**; hole plate kinematic; tube dynamic with friction 1.5; tables kinematic | same, minus the missing tube/pot; plus barrel-lip colliders from `c910509`/`2d35f58` that post-date the dataset | `02fddc3` vs HEAD | **verify** the added barrel lip is disabled for the base task (`lip_enabled` is StandTest-only) | **Medium** |
| **Reset / randomization** | **None.** `final_xyz_sim` std ≤ 3.4 mm across 67 episodes | already zeroed for this task: `__post_init__` sets `pose_range` to `{x:[0,0], y:[0,0]}` on both `reset_object` and the DDS `reset_object_self` event, because the piston is pinned by fixed socket walls | dataset numerics; `pickplace_piston_..._env_cfg.py::__post_init__` | none — **do not add randomization** before parity is confirmed | **High** |
| **Left-hand reset pose** | Fingers **CLOSED around the tube**: commanded (1.7, 1.7, 1.7, 1.7, 0.35, 0.25) rad; achieved frame-0 state (1.698, 1.680, 1.579, 1.571, 0.329, 0.239) | articulation resets with joint **targets** at 0 → hand springs OPEN, tube not held | dataset: commanded value constant across all 45,938 frames (std ≤ 3.6e-7); frame-0 state confirms settled | **OPEN — see "Left-hand grip" below.** Must be a reset-time *target* write, NOT a `default_joint_pos` change | **High** (what) / **unresolved** (where) |

## What still has to be inferred

| Item | Why | Risk |
|---|---|---|
| `HolePlateObjectCfg` exact parameters | `02fddc3` ships `PotObjectCfg()` and only *suggests* the swap. Demo frames match the 5×3 defaults, but the authored values are not recorded anywhere | Medium — hole count/size affects insertion geometry |
| ~~Inspire left-hand closing at reset~~ | **RESOLVED from the dataset, not inferred.** The controller in the missing `parallel/batched_autonomous_piston.py` is unavailable, but the grip pose it produced is recoverable: the dataset commands the left-hand DOFs at a constant **(1.7, 1.7, 1.7, 1.7, 0.35, 0.25) rad** in all 45,938 frames (std ≤ 3.6e-7), and frame 0 of every episode already reports the fingers settled there (state 1.698/1.680/1.579/1.571/0.329/0.239). Applied as `scene.robot.init_state.joint_pos` overrides on the 6 left proximal joints. Verified by render. | resolved |

## Rules for the reconstruction

- Restore from `02fddc3` by cherry-pick/port, **not** by re-deriving coordinates from the
  video. The video is validation evidence only.
- Do not touch the learned model, normalization, the 30→53 mapper, the 2× hold, or the
  task instruction.
- Run scene-only visual validation (no StarVLA policy) before any closed-loop test.

## Validation (scene-only, no StarVLA policy)

Rendered headless with `smoke_render.py` (path-corrected copy): `env.reset()`, 60 sim
steps to settle, then the `front_camera` RGB dumped to PNG. The policy is never loaded.

Command:

```bash
conda activate env_isaaclab
cd /home/jren313/unitree_sim_isaaclab
PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab python <scratchpad>/scene_render.py \
  --task Isaac-PickPlace-Piston-G129-Inspire-Joint --out front_AFTER.png --steps 60
```

All renders come back `shape=(240, 424, 3)` with `robot DOFs: 53`, confirming the camera
correction and the articulation contract at the same time.

Saved frames:

| File | What |
|---|---|
| `demo_frames/ep{000,001,033,066}_frame000.png` | demonstration frame 0, four episodes |
| `front_BEFORE.png` | reset render, scene as installed before this work |
| `front_AFTER.png` | reset render, reconstructed scene |
| `compare_demo_before_after.png` | 3-up strip: demo / before / after |
| `ZOOM_piston_ep000.png`, `ZOOM_lefthand_{DEMO,AFTER}.png` | crops used for the piston-orientation and grip checks |

| Check | Result |
|---|---|
| Frame size = dataset 240×424 | **PASS** |
| Articulation = 53 DOF | **PASS** |
| Piston upright in the blue socket box, centre | **PASS** (unchanged before/after) |
| Tube rack on the robot's LEFT | **PASS** (was on the right) |
| Hole plate present on the RIGHT | **PASS** (was absent) |
| Table de-cluttered (no basket/crates) | **PASS** (clutter box gone from top-right) |
| Mini centrifuge + pipette tips at back | **PASS** (unchanged) |
| Robot initial pose | **PASS** (unchanged) |
| Tube spawns at the grip channel | **PASS** (world pos (−0.264, 0.365, 0.845)) |
| Tube actually **held** by closed fingers | **FAIL** — hand open at reset; see "Left-hand grip" below |

## Left-hand grip at reset — diagnosed, fix deliberately NOT applied

The tube spawns in the grip channel but is **not held**: the hand is open at reset.

**Mechanism (proven by live probe, not inferred):**

1. `reset_scene_to_default()` writes `default_joint_pos` into the articulation *state*,
   but its signature is `reset_joint_targets: bool = False` — the PD **targets stay at
   zero**. The controller therefore drives the fingers straight back open within a few
   steps. Probe: joints read `-0.0005 … -0.0011` right after `reset()`.
2. Writing the targets fixes it. With
   `robot.set_joint_position_target(robot.data.default_joint_pos)` the fingers settle to
   **1.6979 / 1.6987 / 1.6989 / 1.6974 / 0.3505 / 0.2500** after 60 steps — matching the
   demonstrations' achieved frame-0 state.

**Why the obvious fix is wrong.** Raising `scene.robot.init_state.joint_pos` for the left
fingers *does* propagate (probe confirmed `default_joint_pos = 1.6990`), but the action
term is
`JointPositionActionCfg(joint_names=[".*"], scale=1.0, use_default_offset=True)`, and
`joint_actions.py:195` sets `self._offset = default_joint_pos`. The effective command is
therefore `target = action + default_joint_pos`. With the default at 1.699 the policy's
commanded 1.700 becomes **3.399 rad — past the 1.700 joint limit**.

That would silently destroy the identity 30→53 mapping the SFT contract depends on: the
whole mapping is valid *because* every default is 0.0. Changing any default shifts the
action origin for that DOF.

The override was applied, proven harmful by this analysis, and **reverted**. Two safe
options, neither applied here (both touch deployment, which is out of scope for this
task):

- **A — reset-time target write (preferred).** In the env's reset path, after
  `reset_scene_to_default`, call `set_joint_position_target` for the 6 left proximal
  joints only. Leaves all 53 defaults at 0.0, so the mapping identity holds.
- **B — rely on the policy.** The policy commands left-hand 1.7 from its very first
  chunk, so the hand closes ~1 control step in. The tube may fall in that step; whether
  it does is an empirical question for the first closed-loop run.

**This is the one gap between the reconstructed reset observation and the demonstrations.**
