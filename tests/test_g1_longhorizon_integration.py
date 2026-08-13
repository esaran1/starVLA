"""Static integration tests for the G1 + Inspire piston long-horizon dataset.

CPU-only and dataset-free: these check the registration/config invariants that
silently corrupt training when they drift. Loading actual frames or the VLM is
covered by the Stage 9/10 validation scripts, not here.

Run:  pytest tests/test_g1_longhorizon_integration.py -v
"""
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples/realRobots/G1InspirePistonLongHorizon"
YAML = EX / "train_files/starvla_qwenoft_g1_longhorizon.yaml"
MODALITY = EX / "train_files/modality.json"

ROBOT_TYPE = "unitree_g1_piston_longhorizon"
MIXTURE = "unitree_g1_piston_longhorizon"
ACTION_DIM = 30
STATE_DIM = 29
HORIZON = 30


@pytest.fixture(scope="module")
def data_cfg():
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP

    assert ROBOT_TYPE in ROBOT_TYPE_CONFIG_MAP, (
        f"{ROBOT_TYPE} not auto-discovered; expected "
        "examples/**/train_files/data_registry/data_config.py to export ROBOT_TYPE_CONFIG_MAP"
    )
    return ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE]


@pytest.fixture(scope="module")
def modality():
    return json.loads(MODALITY.read_text())


@pytest.fixture(scope="module")
def yaml_cfg():
    return OmegaConf.load(YAML)


def test_dataset_registration():
    from starVLA.dataloader.gr00t_lerobot.registry import (
        DATASET_NAMED_MIXTURES,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
    )
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

    assert MIXTURE in DATASET_NAMED_MIXTURES
    spec = DATASET_NAMED_MIXTURES[MIXTURE]
    assert spec == [("g1-inspire-piston-longhorizon", 1.0, ROBOT_TYPE)]
    assert ROBOT_TYPE_TO_EMBODIMENT_TAG[ROBOT_TYPE] == EmbodimentTag.NEW_EMBODIMENT


def test_slice_widths_sum_to_dims(data_cfg):
    assert sum(data_cfg.action_key_dims.values()) == ACTION_DIM
    assert sum(data_cfg.state_key_dims.values()) == STATE_DIM
    assert list(data_cfg.action_key_dims) == data_cfg.action_keys
    assert list(data_cfg.state_key_dims) == data_cfg.state_keys


def test_modality_slices_are_contiguous_and_in_range(modality):
    for section, total in (("action", ACTION_DIM), ("state", STATE_DIM)):
        spans = [(v["start"], v["end"]) for v in modality[section].values()]
        spans.sort()
        assert spans[0][0] == 0, f"{section} must start at 0"
        assert spans[-1][1] == total, f"{section} must end at {total}"
        for (_, prev_end), (start, _) in zip(spans, spans[1:]):
            assert start == prev_end, f"{section} slices must be contiguous, gap at {prev_end}"
        for start, end in spans:
            assert 0 <= start < end <= total


def test_modality_matches_data_config(modality, data_cfg):
    """modality.json group widths must equal data_config's declared dims."""
    for key, dim in data_cfg.action_key_dims.items():
        g = modality["action"][key.split(".", 1)[1]]
        assert g["end"] - g["start"] == dim, f"{key} width mismatch"
    for key, dim in data_cfg.state_key_dims.items():
        g = modality["state"][key.split(".", 1)[1]]
        assert g["end"] - g["start"] == dim, f"{key} width mismatch"


def test_horizon_consistency(data_cfg, yaml_cfg):
    """len(action_indices) == action_horizon, and indices are 0..H-1."""
    assert len(data_cfg.action_indices) == HORIZON
    assert data_cfg.action_indices == list(range(HORIZON))
    assert yaml_cfg.framework.action_model.action_horizon == HORIZON
    assert data_cfg.observation_indices == [0]


def test_yaml_dims_match_data_config(yaml_cfg, data_cfg):
    am = yaml_cfg.framework.action_model
    assert am.action_dim == sum(data_cfg.action_key_dims.values())
    # state is intentionally disabled; see ACTION_CONTRACT.md
    assert yaml_cfg.datasets.vla_data.include_state is False
    assert am.state_dim == 0


def test_yaml_action_semantics(yaml_cfg):
    """Actions are ABSOLUTE joint position targets — never silently delta."""
    vla = yaml_cfg.datasets.vla_data
    assert vla.action_type == "abs_qpos"
    assert vla.action_mode == "abs"
    assert vla.data_mix == MIXTURE


def test_language_annotation_key_is_flat_task_index(modality):
    """The loader looks up the subkey after 'annotation.' verbatim, and
    original_key must be the int64 task index column."""
    ann = modality["annotation"]
    assert "human.task_description" in ann, "annotation subkey must be FLAT"
    assert ann["human.task_description"]["original_key"] == "task_index"


def test_single_camera_mapping(modality, data_cfg):
    assert data_cfg.video_keys == ["video.ego_view"]
    assert list(modality["video"]) == ["ego_view"]
    assert modality["video"]["ego_view"]["original_key"] == "observation.images.ego_view"


def test_tactile_is_not_mapped(modality):
    """state[29:63] is all-zero tactile and must stay unmapped."""
    ends = [v["end"] for v in modality["state"].values()]
    assert max(ends) == STATE_DIM == 29


def test_normalization_modes_are_q99(data_cfg):
    tf = data_cfg.transform()
    modes = {}
    for t in tf.transforms:
        modes.update(getattr(t, "normalization_modes", {}) or {})
    for key in data_cfg.action_keys + data_cfg.state_keys:
        assert modes.get(key) == "q99", f"{key} should use q99 normalization"


def test_q99_normalization_roundtrip_including_degenerate_dims():
    """q99 must round-trip, and must not produce NaN on constant dims
    (this dataset has 10 globally-constant action dims)."""
    import torch
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer

    q01 = torch.tensor([-1.0, 0.0, 0.5, -0.25])
    q99 = torch.tensor([1.0, 0.0, 2.5, 0.75])  # dim 1 is degenerate
    n = Normalizer(mode="q99", statistics={"q01": q01.clone(), "q99": q99.clone()})
    x = torch.tensor([[0.3, 0.0, 1.5, 0.25], [-0.9, 0.0, 2.4, -0.2]])
    z = n.forward(x)
    assert torch.isfinite(z).all(), "degenerate dim produced non-finite value"
    back = n.inverse(z) if hasattr(n, "inverse") else None
    if back is not None:
        assert torch.allclose(back, x, atol=1e-5)
