"""LIBERO-Goal data: fetch from the Hub without images, split by episode, normalization stats."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCALAR_COLS = ["index", "episode_index", "frame_index", "task_index", "timestamp"]
VECTOR_COLS = {"observation.state": "state", "action": "action"}


# ---------- fetching from the Hub (pinned revision) ----------

def _hub_data_files(repo_id, revision):
    """(filesystem, sorted remote parquet paths) for the dataset at this revision."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    return fs, sorted(fs.glob(f"datasets/{repo_id}@{revision}/data/*/*.parquet"))


def _read_remote(fs, path, columns):
    # cache_type="none": only the byte ranges of the requested columns are fetched.
    with fs.open(path, cache_type="none") as f:
        return pq.ParquetFile(f).read(columns=columns)


def fetch_meta(repo_id, revision, meta_dir):
    """Download only the meta/ folder (info.json, stats, tasks, episodes) into meta_dir/meta/."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=["meta/*"], local_dir=meta_dir)


def fetch_lowdim(repo_id, revision, out_path):
    """Read the non-image columns of every data file from the Hub; save them as one local parquet."""
    fs, files = _hub_data_files(repo_id, revision)
    table = pa.concat_tables(_read_remote(fs, p, SCALAR_COLS + list(VECTOR_COLS)) for p in files)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def fetch_sample(repo_id, revision, meta_dir, episodes, out_path):
    """Save all columns (images included) for a few episodes only, as one local parquet."""
    eps = pq.read_table(Path(meta_dir) / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    eps = eps[eps["episode_index"].isin(episodes)]
    fs, _ = _hub_data_files(repo_id, revision)
    parts = []
    for (chunk, file), _ in eps.groupby(["data/chunk_index", "data/file_index"]):
        path = f"datasets/{repo_id}@{revision}/data/chunk-{chunk:03d}/file-{file:03d}.parquet"
        table = _read_remote(fs, path, None)
        parts.append(table.filter(pc.is_in(table["episode_index"], pa.array(episodes))))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(parts), out_path)


# ---------- reading local copies ----------

def _vector(column) -> np.ndarray:
    """Parquet list column (one vector per row) -> float32 array [rows, dim]."""
    column = column.combine_chunks()
    flat = pc.list_flatten(column).to_numpy()
    return flat.reshape(len(column), -1).astype(np.float32)


def table_to_frames(table) -> dict[str, np.ndarray]:
    """Arrow table -> dict of numpy arrays, sorted by global frame index."""
    frames = {c: table.column(c).to_numpy() for c in SCALAR_COLS}
    for col, name in VECTOR_COLS.items():
        frames[name] = _vector(table.column(col))
    order = np.argsort(frames["index"], kind="stable")
    return {k: v[order] for k, v in frames.items()}


def load_frames(lowdim_path) -> dict[str, np.ndarray]:
    """Every frame's index/state/action columns (no images), from the local lowdim parquet."""
    return table_to_frames(pq.read_table(lowdim_path))


def load_tasks(meta_dir) -> dict[int, str]:
    """task_index -> instruction string, from meta/tasks.parquet."""
    df = pq.read_table(Path(meta_dir) / "meta" / "tasks.parquet").to_pandas()
    return {int(i): str(task) for task, i in zip(df.index, df["task_index"])}


def load_info(meta_dir) -> dict:
    return json.loads((Path(meta_dir) / "meta" / "info.json").read_text())


# ---------- split and stats ----------

def episode_tasks(frames) -> dict[int, int]:
    """episode_index -> task_index (each episode has exactly one task)."""
    out = {}
    for ep, task in zip(frames["episode_index"], frames["task_index"]):
        if out.setdefault(int(ep), int(task)) != int(task):
            raise ValueError(f"episode {ep} has more than one task")
    return out


def split_episodes(ep_task: dict[int, int], seed: int, val_fraction: float):
    """Split episodes into train/val, stratified by task.

    For each task (in sorted order): shuffle its sorted episode ids with one seeded RNG,
    put the first max(1, round(val_fraction * n)) in val. Returns sorted (train, val) lists.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for task in sorted(set(ep_task.values())):
        eps = sorted(ep for ep, t in ep_task.items() if t == task)
        shuffled = rng.permutation(eps).tolist()
        n_val = max(1, round(val_fraction * len(eps)))
        val += shuffled[:n_val]
        train += shuffled[n_val:]
    return sorted(train), sorted(val)


def split_hash(val_eps) -> str:
    """Short hash of the val episode list, to compare splits at a glance."""
    text = ",".join(str(e) for e in sorted(val_eps))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def norm_stats(x: np.ndarray) -> dict[str, list[float]]:
    """Per-dimension stats of x [rows, dim]."""
    x = x.astype(np.float64)
    return {
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "p01": np.percentile(x, 1, axis=0).tolist(),
        "p99": np.percentile(x, 99, axis=0).tolist(),
    }


def train_stats(frames, train_eps) -> dict[str, dict]:
    """Normalization stats for state and action, from train-split frames only."""
    mask = np.isin(frames["episode_index"], list(train_eps))
    return {name: norm_stats(frames[name][mask]) for name in VECTOR_COLS.values()}
