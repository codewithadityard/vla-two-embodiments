"""P3-0a (Kaggle, CPU): write docs/data_report_metaworld.md for lerobot/metaworld_mt50. Every number is computed here.

Run from the repo root:  python -m scripts.data_report_mw
Exits with code 1 if any check fails (the report is still written, with the failures).

Reads only the dataset's files (meta/ and the data parquet files), with numpy, pyarrow, pandas, PIL and
huggingface_hub, which Kaggle preinstalls; it is run without the runner's install step. No metaworld, LeRobot,
MuJoCo or renderer is imported (checked at the end).
Everything that needs the simulator or the Meta-World / LeRobot code moves to P3-0b (first GPU run): the MT10 task
list, instruction -> env mapping by text, image orientation, env state vs. dataset state, the MT10 split and its
normalization stats, the reference policy's n_action_steps re-check and the MuJoCo version (docs/p3_metaworld_plan.md).
"""
import importlib.metadata as md
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from minivla import data
from minivla.config import load_config
from scripts.data_report import fmt, git_commit, table

CONFIG = Path("configs/metaworld.yaml")
REPORT = Path("docs/data_report_metaworld.md")
IMG_DIR = Path("outputs/p3_0a")  # sample frame (not committed)
EXTRA_COLS = ["index", "next.success", "task_id"]  # dataset-only facts beyond data.fetch_lowdim's columns
SIM_MODULES = ["metaworld", "lerobot", "mujoco", "OpenGL", "gymnasium"]


def fetch_extra(repo, rev, out_path):
    """Read EXTRA_COLS of every data file from the Hub (column selection only); save as one local parquet."""
    fs, files = data._hub_data_files(repo, rev)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(data._read_remote(fs, p, EXTRA_COLS) for p in files), out_path)


def flat(column):
    """Column of 1-element lists (or scalars) -> 1-D numpy array."""
    col = column.combine_chunks()
    return (pc.list_flatten(col) if pa.types.is_list(col.type) else col).to_numpy(zero_copy_only=False)


def main():
    cfg = load_config(CONFIG)
    dcfg = cfg["data"]
    repo, rev, meta = dcfg["repo_id"], dcfg["revision"], dcfg["meta_dir"]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- metadata and all non-image columns ---
    if not Path(meta, "meta/info.json").exists():
        print("fetching meta/ ...", flush=True)
        data.fetch_meta(repo, rev, meta)
    if not Path(dcfg["lowdim_path"]).exists():
        print("fetching non-image columns (all files) ...", flush=True)
        data.fetch_lowdim(repo, rev, dcfg["lowdim_path"])
    if not Path(dcfg["extra_path"]).exists():
        print(f"fetching {EXTRA_COLS} (all files) ...", flush=True)
        fetch_extra(repo, rev, dcfg["extra_path"])
    info, tasks = data.load_info(meta), data.load_tasks(meta)
    frames = data.load_frames(dcfg["lowdim_path"])
    ep_task = data.episode_tasks(frames)
    n = len(frames["index"])
    lengths = np.bincount(frames["episode_index"])
    print("metadata and columns read", flush=True)

    check("episodes == info.json total_episodes", len(ep_task) == info["total_episodes"],
          f"{len(ep_task)} vs {info['total_episodes']}")
    check("frames == info.json total_frames", n == info["total_frames"], f"{n} vs {info['total_frames']}")
    check("tasks == info.json total_tasks", len(tasks) == info["total_tasks"] == len(set(ep_task.values())),
          f"{len(tasks)} in tasks.parquet, {len(set(ep_task.values()))} used, {info['total_tasks']} in info.json")
    check("global index is 0..N-1", np.array_equal(frames["index"], np.arange(n)))
    check("frame_index is 0..len-1 in every episode", all(
        np.array_equal(np.sort(frames["frame_index"][frames["episode_index"] == e]), np.arange(lengths[e]))
        for e in ep_task))
    check("no NaN/Inf in state, action, timestamp", all(np.isfinite(frames[k]).all() for k in ["state", "action", "timestamp"]))
    ts_err = float(np.abs(frames["timestamp"] - frames["frame_index"] / info["fps"]).max())
    check("timestamp == frame_index / fps", ts_err < 1e-4, f"max abs diff {ts_err:.2e} s")
    meta_eps = pa.concat_tables([pq.read_table(p) for p in sorted(Path(meta, "meta/episodes").glob("*/*.parquet"))]).to_pandas()
    meta_len = dict(zip(meta_eps["episode_index"], meta_eps["length"]))
    check("episode lengths match meta/episodes", all(meta_len[e] == lengths[e] for e in ep_task))
    for name, feat in [("state", "observation.state"), ("action", "action")]:
        want = info["features"][feat]["shape"][0]
        check(f"{name} dim == info.json {feat} shape", frames[name].shape[1] == want, f"{frames[name].shape[1]} vs {want}")

    # --- success flag and task_id (extra columns), sorted by global index like `frames` ---
    extra = pq.read_table(dcfg["extra_path"])
    x_idx, x_succ, x_tid = (flat(extra.column(c)) for c in EXTRA_COLS)
    order = np.argsort(x_idx, kind="stable")
    x_idx, x_succ, x_tid = x_idx[order], x_succ[order].astype(bool), x_tid[order]
    check("extra columns cover the same frames (global index 0..N-1)", np.array_equal(x_idx, np.arange(n)))
    ep_col, last = frames["episode_index"], frames["frame_index"] == lengths[frames["episode_index"]] - 1
    succ_any = {e: bool(x_succ[ep_col == e].any()) for e in ep_task}
    succ_last = {int(e): bool(s) for e, s in zip(ep_col[last], x_succ[last])}
    tid_of_task = {}
    for t, tid in zip(frames["task_index"], x_tid):
        tid_of_task.setdefault(int(t), set()).add(int(tid))
    check("each task_index has exactly one task_id", all(len(v) == 1 for v in tid_of_task.values()),
          f"{sum(len(v) == 1 for v in tid_of_task.values())}/{len(tid_of_task)}")

    # --- images: every frame of the dataset's first episode ---
    sample_ep = min(ep_task)
    if not Path(dcfg["sample_path"]).exists():
        print(f"fetching sample episode {sample_ep} ...", flush=True)
        data.fetch_sample(repo, rev, meta, [sample_ep], dcfg["sample_path"])
    sample = data.table_to_frames(pq.read_table(dcfg["sample_path"]))
    check("sample holds exactly the first episode", np.unique(sample["episode_index"]).tolist() == [sample_ep])
    check("sample state/action == column-only read for the same frames",
          np.array_equal(sample["state"], frames["state"][sample["index"]]) and
          np.array_equal(sample["action"], frames["action"][sample["index"]]))
    cam_keys = [k for k, v in info["features"].items() if v["dtype"] in ("image", "video")]
    check("exactly one camera", len(cam_keys) == 1, f"{cam_keys}")
    cam_rows = []
    for k in cam_keys:
        cells = pq.read_table(dcfg["sample_path"], columns=[k]).column(k).to_pylist()
        imgs = [Image.open(io.BytesIO(c["bytes"])) for c in cells]
        shapes = {np.asarray(im).shape for im in imgs}
        want = info["features"][k]["shape"]  # (C, H, W) in info.json
        check(f"every decoded {k} frame of episode {sample_ep} is HxWxC == info.json CxHxW", shapes == {(want[1], want[2], want[0])},
              f"decoded {sorted(shapes)}, info {want}")
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        imgs[0].save(IMG_DIR / f"{k}.png")
        cam_rows.append([f"`{k}`", info["features"][k]["dtype"], ", ".join(sorted({im.format for im in imgs})),
                         " / ".join(sorted({im.mode for im in imgs})), "x".join(map(str, sorted(shapes)[0])),
                         len(imgs), f"`{IMG_DIR / k}.png`"])

    loaded = [m for m in SIM_MODULES if m in sys.modules]
    check("no metaworld / LeRobot / MuJoCo / OpenGL / gymnasium module imported", not loaded, f"{loaded}")

    # --- report ---
    versions = {p: md.version(p) for p in ["numpy", "pyarrow", "pandas", "pillow", "huggingface_hub"]}
    L = ["# Data report: Meta-World MT50 dataset (P3-0a)", "",
         "Generated on Kaggle (CPU, no install step) by `python -m scripts.data_report_mw`. Every number below is "
         "computed from the dataset's files; nothing is typed by hand. No simulator, Meta-World or LeRobot code was "
         "used: the MT10 task list, instruction -> env mapping, image orientation, env vs. dataset state, the MT10 "
         "split and its stats come in P3-0b (first GPU run).", "", "## Source", ""]
    L += table(["Item", "Value"], [
        ["Dataset", f"`{repo}`"], ["Hub revision", f"`{rev}`"], ["LeRobot format", info["codebase_version"]],
        ["Robot", info["robot_type"]],
        ["How it was read", "non-image columns of every file via `HfFileSystem` + pyarrow column selection; "
         f"images only for episode {sample_ep}"],
        ["Versions", ", ".join(f"{k} {v}" for k, v in versions.items())],
        ["Code commit", f"`{git_commit()}`"], ["Config", f"`{CONFIG}`"]])
    L += ["", "## Checks", ""]
    L += table(["Check", "Result", "Detail"], [[c, "PASS" if ok else "**FAIL**", d] for c, ok, d in checks])

    all_lens = lengths[sorted(ep_task)]
    L += ["", "## Counts (whole dataset)", ""]
    L += table(["Item", "Value"], [
        ["Episodes", len(ep_task)], ["Frames", n], ["Tasks", len(tasks)],
        ["Episode length min / mean / median / max (frames)",
         f"{all_lens.min()} / {fmt(all_lens.mean(), 1)} / {fmt(float(np.median(all_lens)), 1)} / {all_lens.max()}"],
        ["fps (metadata)", info["fps"]],
        ["Mean episode duration at that fps (s)", fmt(all_lens.mean() / info["fps"], 2)],
        ["Episodes with `next.success` true at the last frame", f"{sum(succ_last.values())}/{len(ep_task)}"],
        ["Episodes with `next.success` true at any frame", f"{sum(succ_any.values())}/{len(ep_task)}"]])

    L += ["", f"## Tasks (all {len(tasks)}; MT10 is selected in P3-0b)", ""]
    rows = []
    for t in sorted(tasks):
        eps = [e for e in ep_task if ep_task[e] == t]
        lens = lengths[eps]
        rows.append([t, ", ".join(map(str, sorted(tid_of_task.get(t, [])))), tasks[t], len(eps), int(lens.sum()),
                     f"{lens.min()} / {fmt(lens.mean(), 1)} / {lens.max()}", sum(succ_last[e] for e in eps)])
    L += table(["Task index", "task_id", "Instruction (tasks.parquet)", "Episodes", "Frames",
                "Length min / mean / max", "Success at last frame"], rows)

    L += ["", "## Cameras", ""]
    L += table(["Key", "Stored as", "Encoding", "Mode", "Decoded shape (H x W x C)", f"Frames decoded (ep {sample_ep})",
                "Sample frame (not committed)"], cam_rows)

    L += ["", "## Other features (info.json)", ""]
    L += table(["Key", "dtype", "shape", "names"], [[f"`{k}`", v["dtype"], v["shape"], v.get("names")]
                                                     for k, v in info["features"].items() if k not in cam_keys])

    for name, feat in [("state", "observation.state"), ("action", "action")]:
        names = info["features"][feat]["names"]
        names = (names.get("axes") or names.get("motors") or []) if isinstance(names, dict) else (names or [])
        s = data.norm_stats(frames[name])
        L += ["", f"## {name.capitalize()} (`{feat}`, {len(s['mean'])}-D): all frames, for information", "",
              "Normalization stats for training come from the MT10 train split (P3-0b), not from this table.", ""]
        L += table(["Dim", "Name (metadata)", "Mean", "Std", "Min", "Max", "P1", "P99"],
                   [[i, names[i] if i < len(names) else "", *(fmt(s[k][i]) for k in ["mean", "std", "min", "max", "p01", "p99"])]
                    for i in range(len(s["mean"]))])
    vals = np.unique(frames["action"][:, -1])
    L += ["", "## Gripper action (last action dim, all frames)", "",
          f"{len(vals)} distinct values, range {fmt(vals.min())} to {fmt(vals.max())}."]

    REPORT.write_text("\n".join(L) + "\n")
    failed = [c for c, ok, _ in checks if not ok]
    print("\n".join(L))
    print(f"Wrote {REPORT}. Checks: {len(checks) - len(failed)}/{len(checks)} PASS.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
