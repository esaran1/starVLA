"""Unitree G1 + Inspire hands — piston LONG-HORIZON data registry for QwenOFT.

Dataset : birbirll/g1-inspire-piston-longhorizon (LeRobot v2.1)
          revision 80fb5a0bbcaf806f5b2cc848fc722f8e005a6830
Task    : "pick up the piston with the right hand, inject it into the tube held
           by the left hand, then move it over the hole plate."  (1 task, 67 eps,
           45938 frames @ 50 fps, ~13.7 s/episode)

Raw LeRobot columns (verified against the parquet, not inferred from metadata):
  observation.state           : (T, 63) float32
        [0:29]  arm/hand/waist proprio  -> mapped below
        [29:63] tactile                 -> IDENTICALLY ZERO in every frame of
                                           this dataset; NOT mapped.
  observation.tactile         : (T, 34) float32, all-zero, byte-identical to
                                state[:, 29:63]. Not mapped.
  action                      : (T, 30) float32, ABSOLUTE joint position targets
                                in radians (base_height in metres).
  observation.images.ego_view : mp4 240x424 @ 50 fps, single camera.
  task_index                  : int64 -> meta/tasks.jsonl

Action semantics (established numerically, see docs/ACTION_CONTRACT.md):
  Actions are absolute position *targets*, not deltas and not achieved state.
  corr(action[:, i], state[:, i]) = 0.95..0.999 with a steady-state PD tracking
  offset (up to ~0.22 rad on the loaded right shoulder), and mean |action| over
  the arm dims is ~0.30 rad — far too large for deltas (per-step |delta| is
  ~0.002 rad at 50 Hz).

Degenerate dims in THIS dataset (globally constant over all 45938 frames):
  action[14:20] left_hand  (frozen holding the tube), action[26] base_height
  (0.76 m), action[27:30] navigate_command (all exactly 0.0 -> stationary task).
  They are kept in the action vector: never shrink the robot action dimension.
  `q99` normalization detects q01 == q99 and passes those dims through
  unchanged (Normalizer.forward), so they contribute a constant, finite target.

Auto-discovered by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``
because this file lives at ``examples/**/train_files/data_registry/data_config.py``
and exports ROBOT_TYPE_CONFIG_MAP / ROBOT_TYPE_TO_EMBODIMENT_TAG /
DATASET_NAMED_MIXTURES.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoToNumpy,
    VideoToTensor,
)

# Action horizon. 30 steps @ 50 Hz = 0.6 s of lookahead, matching the reference
# G1 piston checkpoint. Must equal framework.action_model.action_horizon in the
# YAML (share_tools.apply_config_compat enforces
# future_action_window_size == action_horizon - 1).
ACTION_HORIZON = 30


class UnitreeG1InspirePistonLongHorizonDataConfig:
    # NEW_EMBODIMENT -> action-expert projector index 31
    # (embodiment_tags.EMBODIMENT_TAG_MAPPING).
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = ["video.ego_view"]

    # Proprioception groups. Defined so state can be switched on for ablations,
    # but the shipped YAML sets `include_state: false` — see the state-shortcut
    # note in docs/ACTION_CONTRACT.md. Tactile [29:63] is excluded (all-zero).
    state_keys = [
        "state.left_arm",    # 7
        "state.right_arm",   # 7
        "state.left_hand",   # 6  (Inspire, 6 DoF per hand)
        "state.right_hand",  # 6
        "state.waist",       # 3
    ]  # -> state_dim = 29

    action_keys = [
        "action.left_arm",          # 7
        "action.right_arm",         # 7
        "action.left_hand",         # 6
        "action.right_hand",        # 6
        "action.base_height",       # 1
        "action.navigate_command",  # 3  (base vx / vy / vyaw)
    ]  # -> action_dim = 30

    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]                        # current frame only
    action_indices = list(range(ACTION_HORIZON))     # predict t .. t+29

    # For auditing / YAML cross-checks (asserted in tests, not read by loader).
    state_key_dims = {
        "state.left_arm": 7,
        "state.right_arm": 7,
        "state.left_hand": 6,
        "state.right_hand": 6,
        "state.waist": 3,
    }
    action_key_dims = {
        "action.left_arm": 7,
        "action.right_arm": 7,
        "action.left_hand": 6,
        "action.right_hand": 6,
        "action.base_height": 1,
        "action.navigate_command": 3,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.video_keys
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.state_keys
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices, modality_keys=self.action_keys
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.language_keys
            ),
        }

    def transform(self):
        # q99 for every continuous joint/base channel (StarVLA's documented
        # default for continuous state/action). No "binary" mode: the Inspire
        # hands are continuous multi-DoF, there is no binary gripper.
        return ComposedModalityTransform(
            transforms=[
                # Train-gated video augmentation (no-op in eval; see
                # VideoColorJitter.get_transform). VideoToTensor/VideoToNumpy
                # bracket the jitter so the sample re-emerges as numpy for the
                # Qwen image processor.
                VideoToTensor(apply_to=self.video_keys),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: "q99" for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: "q99" for k in self.action_keys},
                ),
            ]
        )


# ---------------------------------------------------------------------------
# REQUIRED top-level exports — names matter, do not rename.
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_piston_longhorizon": UnitreeG1InspirePistonLongHorizonDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "unitree_g1_piston_longhorizon": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    # mixture_name : [(dataset_subdir_relative_to_data_root_dir, weight, robot_type)]
    "unitree_g1_piston_longhorizon": [
        ("g1-inspire-piston-longhorizon", 1.0, "unitree_g1_piston_longhorizon"),
    ],
}
