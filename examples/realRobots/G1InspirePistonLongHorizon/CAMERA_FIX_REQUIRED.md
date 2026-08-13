# Required simulator camera correction (apply OUTSIDE this repo)

**This is the one hard blocker for closed-loop evaluation of the G1 long-horizon SFT
policy.** It is a change to `unitree_sim_isaaclab`, deliberately **not** applied from the
StarVLA repo. Everything needed to make the fix is recorded here.

## The problem

The piston task renders its ego camera with a much wider field of view than the camera
that produced the training data. A policy trained on 69° imagery would receive 105.5°
imagery at evaluation — badly out of distribution, through no fault of the checkpoint.

| | Dataset `ego_view` | Piston task `front_camera` | Δ |
|---|---|---|---|
| Mount prim | `Robot/d435_link/front_cam` | `Robot/d435_link/front_cam` | same mount |
| Resolution | 424 × 240 | 640 × 480 | aspect 1.767 → 1.333 |
| Focal length | 14.55 mm (D435 color) | **7.6 mm** | −6.95 mm |
| Horizontal aperture | 20.0 mm | 20.0 mm | same |
| **HFOV** | **69.0°** | **105.5°** | **+36.5°** |
| **VFOV** | **42.5°** | **89.2°** | **+46.7°** |
| Clipping range | (0.05, 1e5) | (0.1, 1e5) | near plane |

`ego_view` **is** `front_camera` — same physical mount, established below. Only the
optics differ.

## Evidence

1. **The dataset camera is a RealSense D435 colour stream at 424×240**, stated
   explicitly in
   `g1_redball_eval/overlay/unitree_sim_isaaclab/tasks/g1_tasks/pick_place_redball_g1_29dof_inspire/pickplace_redball_g1_29dof_inspire_joint_env_cfg.py:26-27`:

   > `# --- ego camera = Intel RealSense D435 color module, as on the Unitree G1 head ---`
   > `# dataset observation.images.ego_view is the D435 *color* stream at 424x240 (h264).`

   The same block derives the focal length from the D435's 69° HFOV:
   `focal = 10 / tan(34.5°) = 14.55 mm`, and defines
   `EGO_H, EGO_W = 240, 424`, `D435_FOCAL_MM = 14.55`, `D435_H_APERTURE_MM = 20.0`,
   `D435_CLIP = (0.05, 1.0e5)`.

2. **The redball task already applies the fix**; the piston task never inherited it. The
   piston scene uses the generic preset
   (`unitree_sim_isaaclab/tasks/g1_tasks/pick_place_piston_g1_29dof_inspire/pickplace_piston_g1_29dof_inspire_joint_env_cfg.py:33`):

   ```python
   front_camera = CameraPresets.g1_front_camera()
   ```

   which is `CameraBaseCfg.get_camera_config()` with **all defaults** —
   `height=480, width=640, focal_length=7.6, horizontal_aperture=20.0,
   clipping_range=(0.1, 1e5)` (`tasks/common_config/camera_configs.py:22-33, 87-89`).

3. **The FOV arithmetic** — `HFOV = 2·atan(aperture / (2·focal))`,
   `VFOV = 2·atan((aperture·H/W) / (2·focal))`:

   | focal | W×H | HFOV | VFOV |
   |---|---|---|---|
   | 7.6 mm | 640×480 | 105.5° | 89.2° |
   | 14.55 mm | 424×240 | **69.0°** | **42.5°** |
   | *D435 datasheet* | — | *69°* | *42°* |

   The 14.55 mm / 424×240 configuration reproduces the real D435 to within 0.5°.

## The fix

In the piston scene cfg
(`unitree_sim_isaaclab/tasks/g1_tasks/pick_place_piston_g1_29dof_inspire/pickplace_piston_g1_29dof_inspire_joint_env_cfg.py`),
replace `front_camera = CameraPresets.g1_front_camera()` with the D435-matched config —
identical to what the redball overlay already uses:

```python
from tasks.common_config import CameraBaseCfg  # add to the existing common_config import

# --- ego camera = Intel RealSense D435 color module, matching the training data ---
# dataset observation.images.ego_view is the D435 color stream at 424x240 (h264).
# HFOV = 2*atan(h_aperture / (2*focal)); 20.0 mm aperture and 69 deg HFOV
# -> focal = 10/tan(34.5 deg) = 14.55 mm.
EGO_H, EGO_W = 240, 424
D435_FOCAL_MM = 14.55
D435_H_APERTURE_MM = 20.0
D435_CLIP = (0.05, 1.0e5)

front_camera = CameraBaseCfg.get_camera_config(
    prim_path="/World/envs/env_.*/Robot/d435_link/front_cam",
    height=EGO_H, width=EGO_W,
    focal_length=D435_FOCAL_MM,
    horizontal_aperture=D435_H_APERTURE_MM,
    clipping_range=D435_CLIP,
)
```

Leave `left_wrist_camera` / `right_wrist_camera` untouched — the policy consumes exactly
one image and never sees them.

> If the physical device's `calibration.json` is available, its `fx/fy/ppx/ppy` give a
> per-unit exact match (`Camera.set_intrinsic_matrices` also works at runtime). The
> nominal 14.55 mm is accurate to the datasheet FOV and is what the redball task uses.

## Verification after the fix

```python
import math
f, ap, W, H = 14.55, 20.0, 424, 240
assert round(2*math.degrees(math.atan(ap/(2*f))), 1) == 69.0          # HFOV
assert round(2*math.degrees(math.atan((ap*H/W)/(2*f))), 1) == 42.5    # VFOV
```

Then confirm the rendered frame is 240×424×3 before it reaches the preprocessing below.

## Preprocessing the policy expects (unchanged by the fix)

1. RGB uint8 HWC at **240 × 424**.
2. Resize to **224 × 224** — a plain **non-aspect-preserving squash**. No crop, no pad,
   no letterbox: the 1.767 aspect is deliberately distorted, and training did exactly
   this. Reproduce the squash, do not "improve" it with a centre crop.
3. **No colour jitter at inference.** The training jitter (0.3/0.4/0.5/0.08) is
   train-gated in `VideoColorJitter.get_transform` and must stay off.

Feeding a 640×480 frame through step 2 does **not** rescue the mismatch: the FOV is baked
into the render, so the resize would squash a 105.5° view into the square instead of a
69° one.
