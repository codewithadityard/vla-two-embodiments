"""M1 tests: split rules and stats on toy data; integrity on the real dataset if it is downloaded."""
from pathlib import Path

import numpy as np
import pytest

from minivla import data
from minivla.config import load_config

# Toy table: 3 tasks with 10, 20 and 3 episodes.
TOY_EP_TASK = {ep: (0 if ep < 10 else 1 if ep < 30 else 2) for ep in range(33)}


def test_split_disjoint_and_complete():
    train, val = data.split_episodes(TOY_EP_TASK, seed=0, val_fraction=0.1)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(TOY_EP_TASK)


def test_split_every_task_in_val_with_expected_count():
    _, val = data.split_episodes(TOY_EP_TASK, seed=0, val_fraction=0.1)
    per_task = np.bincount([TOY_EP_TASK[e] for e in val], minlength=3)
    # max(1, round(0.1 * n)) for n = 10, 20, 3
    assert per_task.tolist() == [1, 2, 1]


def test_split_seed_determinism():
    assert data.split_episodes(TOY_EP_TASK, 0, 0.1) == data.split_episodes(TOY_EP_TASK, 0, 0.1)
    assert data.split_episodes(TOY_EP_TASK, 0, 0.1) != data.split_episodes(TOY_EP_TASK, 1, 0.1)


def test_norm_stats_known_values():
    x = np.array([[0.0, 1.0], [2.0, 1.0]])
    s = data.norm_stats(x)
    assert s["mean"] == [1.0, 1.0]
    assert s["std"] == [1.0, 0.0]
    assert s["min"] == [0.0, 1.0] and s["max"] == [2.0, 1.0]


def test_train_stats_ignore_val_frames():
    frames = {
        "episode_index": np.array([0, 0, 1, 1]),
        "state": np.array([[1.0], [3.0], [100.0], [200.0]]),
        "action": np.array([[0.0], [2.0], [-50.0], [50.0]]),
    }
    stats = data.train_stats(frames, train_eps=[0])
    assert stats["state"]["mean"] == [2.0]
    assert stats["action"]["mean"] == [1.0]


def test_inherit_merges_nested(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: 3\n")
    (tmp_path / "run.yaml").write_text("inherit: base.yaml\na: {y: 5}\n")
    assert load_config(tmp_path / "run.yaml") == {"a": {"x": 1, "y": 5}, "b": 3}


# --- real data (skipped until `python -m scripts.data_report` has fetched the laptop copies) ---
CFG = load_config("configs/base.yaml")["data"]
needs_data = pytest.mark.skipif(not Path(CFG["lowdim_path"]).exists(), reason="lowdim data not fetched")
needs_sample = pytest.mark.skipif(not Path(CFG["sample_path"]).exists(), reason="image sample not fetched")


@pytest.fixture(scope="module")
def frames():
    return data.load_frames(CFG["lowdim_path"])


@needs_data
def test_real_totals(frames):
    info = data.load_info(CFG["meta_dir"])
    assert len(frames["index"]) == info["total_frames"] == 52042
    assert len(np.unique(frames["episode_index"])) == info["total_episodes"] == 428
    assert len(data.load_tasks(CFG["meta_dir"])) == info["total_tasks"] == 10


@needs_data
def test_real_shapes_and_finite(frames):
    assert frames["state"].shape == (52042, 8)
    assert frames["action"].shape == (52042, 7)
    assert np.isfinite(frames["state"]).all() and np.isfinite(frames["action"]).all()


@needs_data
def test_real_split_every_task_has_val(frames):
    ep_task = data.episode_tasks(frames)
    split = load_config("configs/base.yaml")["split"]
    _, val = data.split_episodes(ep_task, split["split_seed"], split["val_fraction"])
    assert {ep_task[e] for e in val} == set(range(10))


@needs_sample
def test_sample_images_decode():
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(CFG["sample_path"])
    assert sorted(set(table["episode_index"].to_pylist())) == sorted(CFG["sample_episodes"])
    for key in ["observation.images.image", "observation.images.wrist_image"]:
        cell = table[key][0].as_py()
        assert np.array(Image.open(io.BytesIO(cell["bytes"]))).shape == (256, 256, 3)
