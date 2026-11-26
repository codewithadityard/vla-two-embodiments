"""M1: write docs/data_report.md for LIBERO-Goal. Every number is computed here from the data.

Run from the repo root:  python -m scripts.data_report
Exits with code 1 if any check fails (the report is still written, with the failures).
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from minivla import data
from minivla.config import load_config

CONFIG = Path("configs/base.yaml")
REPORT = Path("docs/data_report.md")
IMG_DIR = Path("outputs/m1")  # gitignored


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def git_commit():
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    # Only code and config count as "dirty"; the report itself changes every time it is written.
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "minivla", "scripts", "configs"],
                           capture_output=True, text=True).stdout.strip()
    return head + (" (uncommitted changes)" if dirty else "")


def decode_first_image(sample_path, key):
    """Decode the first frame of an image column in the sample; save it as PNG; return its shape."""
    cell = pq.read_table(sample_path, columns=[key]).column(key)[0].as_py()
    img = np.array(Image.open(io.BytesIO(cell["bytes"])))
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(IMG_DIR / f"{key}.png")
    return img.shape


def state_layout(state):
    """Is the 8-D state [pos3, quat4, grip1] or [pos3, axis-angle3, finger qpos2] (the env's layout)?"""
    quat_norm = np.linalg.norm(state[:, 3:7], axis=1)
    frac_unit = float(np.mean(np.abs(quat_norm - 1) < 1e-3))
    frac_opposite = float(np.mean(state[:, 6] * state[:, 7] <= 0))
    finger_sum = float(np.mean(np.abs(state[:, 6] + state[:, 7])))
    finger_corr = float(np.corrcoef(state[:, 6], state[:, 7])[0, 1])
    aa_norm = np.linalg.norm(state[:, 3:6], axis=1)
    # Rule (approved 2026-11-26, after the first run's -0.99 correlation threshold proved too strict).
    if frac_unit > 0.99:
        verdict = "quaternion in dims 3-6 (DIFFERENT from the env's layout)"
    elif frac_unit < 0.01 and frac_opposite >= 0.99 and finger_sum < 0.005 and finger_corr < -0.8:
        verdict = "axis-angle in dims 3-5 + two mirrored gripper fingers in dims 6-7 (SAME as the env's layout)"
    else:
        verdict = "UNCLEAR"
    rows = [
        ["fraction of frames with norm(dims 3-6) within 1e-3 of 1 (quaternion if > 0.99; need < 0.01)", fmt(frac_unit)],
        ["fraction of frames where dims 6 and 7 have opposite signs (need >= 0.99)", fmt(frac_opposite)],
        ["mean abs(dim6 + dim7) (need < 0.005)", fmt(finger_sum, 6)],
        ["correlation of dims 6 and 7 (need < -0.8)", fmt(finger_corr)],
        ["norm(dims 3-5) min / max (robosuite's quat2axisangle gives angles in [0, 2 pi])",
         f"{fmt(aa_norm.min())} / {fmt(aa_norm.max())}"],
    ]
    return verdict, rows


def compare_reference(ref_dir, goal_tasks, frames, lengths):
    """Compare with the goal tasks of the reference dataset, using its meta/ files only."""
    ref_tasks = data.load_tasks(ref_dir)
    ref_eps = pq.read_table(Path(ref_dir) / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    ref_eps["task"] = ref_eps["tasks"].map(lambda t: t[0])
    has_stats = "stats/observation.state/mean" in ref_eps.columns
    rows, all_ok = [], True
    for t, text in sorted(goal_tasks.items()):
        ours = frames["episode_index"][frames["task_index"] == t]
        our_eps = np.unique(ours)
        our_lens = sorted(int(lengths[e]) for e in our_eps)
        ref = ref_eps[ref_eps["task"] == text]
        ref_lens = sorted(int(x) for x in ref["length"])
        same = text in ref_tasks.values() and our_lens == ref_lens
        diff = "n/a"
        if has_stats and len(ref) > 0:
            # length-weighted mean state over the task's episodes, ours vs reference
            ref_mean = (np.stack(ref["stats/observation.state/mean"]) * ref["length"].to_numpy()[:, None]).sum(0)
            ref_mean /= ref["length"].sum()
            our_mean = frames["state"][frames["task_index"] == t].mean(0)
            diff = fmt(float(np.abs(ref_mean - our_mean).max()), 6)
        all_ok &= same
        rows.append([t, len(our_eps), len(ref), "yes" if same else "NO", diff])
    return all_ok, rows


def fetch_missing(dcfg):
    """Fetch the small laptop copies from the Hub (pinned revisions) if they are not there yet."""
    repo, rev, meta = dcfg["repo_id"], dcfg["revision"], dcfg["meta_dir"]
    if not Path(meta, "meta/info.json").exists():
        print("fetching meta/ ...", flush=True)
        data.fetch_meta(repo, rev, meta)
    if not Path(dcfg["reference_meta_dir"], "meta/info.json").exists():
        print("fetching reference meta/ ...", flush=True)
        data.fetch_meta(dcfg["reference_repo_id"], dcfg["reference_revision"], dcfg["reference_meta_dir"])
    if not Path(dcfg["lowdim_path"]).exists():
        print("fetching non-image columns (all files) ...", flush=True)
        data.fetch_lowdim(repo, rev, dcfg["lowdim_path"])
    if not Path(dcfg["sample_path"]).exists():
        print(f"fetching sample episodes {dcfg['sample_episodes']} ...", flush=True)
        data.fetch_sample(repo, rev, meta, dcfg["sample_episodes"], dcfg["sample_path"])


def main():
    cfg = load_config(CONFIG)
    dcfg, scfg = cfg["data"], cfg["split"]
    fetch_missing(dcfg)
    meta = dcfg["meta_dir"]
    info = data.load_info(meta)
    tasks = data.load_tasks(meta)
    frames = data.load_frames(dcfg["lowdim_path"])
    ep_task = data.episode_tasks(frames)
    n = len(frames["index"])
    lengths = np.bincount(frames["episode_index"])
    checks = []  # (name, ok, detail)

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- totals and integrity ---
    check("episodes == info.json total_episodes", len(ep_task) == info["total_episodes"],
          f"{len(ep_task)} vs {info['total_episodes']}")
    check("frames == info.json total_frames", n == info["total_frames"], f"{n} vs {info['total_frames']}")
    check("tasks == info.json total_tasks", len(tasks) == info["total_tasks"] == len(set(ep_task.values())),
          f"{len(tasks)} in tasks.parquet, {len(set(ep_task.values()))} used, {info['total_tasks']} in info.json")
    check("global index is 0..N-1", np.array_equal(frames["index"], np.arange(n)))
    contiguous = all(
        np.array_equal(np.sort(frames["frame_index"][frames["episode_index"] == e]), np.arange(lengths[e]))
        for e in ep_task)
    check("frame_index is 0..len-1 in every episode", contiguous)
    finite = all(np.isfinite(frames[k]).all() for k in ["state", "action", "timestamp"])
    check("no NaN/Inf in state, action, timestamp", finite)
    ts_err = float(np.abs(frames["timestamp"] - frames["frame_index"] / info["fps"]).max())
    check("timestamp == frame_index / fps", ts_err < 1e-4, f"max abs diff {ts_err:.2e} s")
    meta_eps = pq.read_table(Path(meta) / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    meta_len = dict(zip(meta_eps["episode_index"], meta_eps["length"]))
    check("episode lengths match meta/episodes", all(meta_len[e] == lengths[e] for e in ep_task))

    # --- split ---
    train, val = data.split_episodes(ep_task, scfg["split_seed"], scfg["val_fraction"])
    train2, val2 = data.split_episodes(ep_task, scfg["split_seed"], scfg["val_fraction"])
    check("split: train and val are disjoint", not set(train) & set(val))
    check("split: train + val cover every episode", set(train) | set(val) == set(ep_task))
    check("split: every task has val episodes", {ep_task[e] for e in val} == set(ep_task.values()))
    check("split: same seed gives the same split", (train, val) == (train2, val2))
    stats = data.train_stats(frames, train)

    # --- image sample (all columns, sample episodes only) ---
    sample = data.table_to_frames(pq.read_table(dcfg["sample_path"]))
    sample_eps = sorted(np.unique(sample["episode_index"]).tolist())
    check("sample holds exactly the configured episodes", sample_eps == sorted(dcfg["sample_episodes"]), f"{sample_eps}")
    same = np.array_equal(sample["state"], frames["state"][sample["index"]]) and \
        np.array_equal(sample["action"], frames["action"][sample["index"]])
    check("sample state/action == column-only read for the same frames", same)

    # --- cameras ---
    image_keys = [k for k, v in info["features"].items() if v["dtype"] in ("image", "video")]
    cam_rows = []
    for k in image_keys:
        shape = decode_first_image(dcfg["sample_path"], k)
        ok = list(shape) == info["features"][k]["shape"]
        check(f"decoded {k} shape matches info.json", ok, f"{shape}")
        cam_rows.append([f"`{k}`", info["features"][k]["dtype"], "x".join(map(str, shape)), f"`{IMG_DIR / k}.png`"])

    # --- state layout and reference comparison ---
    verdict, layout_rows = state_layout(frames["state"])
    check("state layout is decided (not UNCLEAR)", verdict != "UNCLEAR", verdict)
    ref_ok, ref_rows = compare_reference(dcfg["reference_meta_dir"], tasks, frames, lengths)
    check(f"episode counts and lengths match {dcfg['reference_repo_id']} for all 10 goal tasks", ref_ok)

    # --- write the report ---
    L = ["# Data report: LIBERO-Goal (M1)", "",
         "Generated by `python -m scripts.data_report`. Every number below is computed from the data; nothing is typed by hand.", "",
         "## Source", ""]
    L += table(["Item", "Value"], [
        ["Dataset", f"`{dcfg['repo_id']}`"], ["Hub revision", f"`{dcfg['revision']}`"],
        ["LeRobot format", info["codebase_version"]], ["Robot", info["robot_type"]],
        ["How it was read", "non-image columns of every file via `HfFileSystem` + pyarrow column selection; "
         f"images only for sample episodes {dcfg['sample_episodes']}"],
        ["Code commit", f"`{git_commit()}`"], ["Config", f"`{CONFIG}`"]])
    L += ["", "## Checks", ""]
    L += table(["Check", "Result", "Detail"], [[c, "PASS" if ok else "**FAIL**", d] for c, ok, d in checks])

    L += ["", "## Counts", ""]
    ep_lens = lengths[sorted(ep_task)]
    L += table(["Item", "Value"], [
        ["Episodes", len(ep_task)], ["Frames", n], ["Tasks", len(tasks)],
        ["Episode length min / mean / max", f"{ep_lens.min()} / {fmt(ep_lens.mean(), 1)} / {ep_lens.max()}"],
        ["fps (metadata)", info["fps"]]])
    L += ["", "The LeRobot LIBERO env steps at 20 Hz (`control_freq`), while the metadata says "
          f"{info['fps']} fps. We index chunks by frame, so one dataset action is one env step; the fps value only sets timestamps."]

    L += ["", "## Tasks", ""]
    rows = []
    val_set = set(val)
    for t, text in sorted(tasks.items()):
        eps = [e for e in ep_task if ep_task[e] == t]
        lens = lengths[eps]
        rows.append([t, text, len(eps), int(lens.sum()), f"{lens.min()} / {fmt(lens.mean(), 1)} / {lens.max()}",
                     sum(e not in val_set for e in eps), sum(e in val_set for e in eps)])
    L += table(["Task", "Instruction", "Episodes", "Frames", "Length min / mean / max", "Train eps", "Val eps"], rows)

    L += ["", "## Cameras", ""]
    L += table(["Key", "Stored as", "Decoded shape (H x W x C)", "Sample frame (not committed)"], cam_rows)

    for name, feat in [("state", "observation.state"), ("action", "action")]:
        names = info["features"][feat]["names"]
        names = names.get("motors", names) if isinstance(names, dict) else names
        s = stats[name]
        L += ["", f"## {name.capitalize()} (`{feat}`, {len(s['mean'])}-D): stats on the train split", ""]
        L += table(["Dim", "Name (metadata)", "Mean", "Std", "Min", "Max", "P1", "P99"],
                   [[i, names[i] if i < len(names) else "", *(fmt(s[k][i]) for k in ["mean", "std", "min", "max", "p01", "p99"])]
                    for i in range(len(s["mean"]))])

    grip = frames["action"][:, -1]
    vals, counts = np.unique(grip, return_counts=True)
    L += ["", "## Gripper action (last action dim)", ""]
    if len(vals) <= 10:
        L += table(["Value", "Frames"], [[fmt(v), c] for v, c in zip(vals, counts)])
    else:
        L += [f"{len(vals)} distinct values, range {fmt(vals.min())} to {fmt(vals.max())}."]

    L += ["", "## State layout", "", f"**Verdict: {verdict}.**", "",
          "The env (`lerobot/sim/integrations/libero.py` at the pinned LeRobot commit) builds the state as "
          "eef position (3) + axis-angle (3) + gripper finger qpos (2). The dataset's metadata names say "
          "`x, y, z, rx, ry, rz, rw, gripper`, so we test the values instead of trusting the names. "
          "M2 checks this directly: an env reset on Kaggle is compared with the first frame of a dataset episode.", ""]
    L += table(["Test", "Value"], layout_rows)

    L += ["", "## Normalization", "",
          "Mean/std per dimension, from train-split frames only (the Mean and Std columns above). "
          "Code recomputes them with `minivla.data.train_stats`; this report is the record."]

    L += ["", "## Train/val split", ""]
    L += table(["Item", "Value"], [
        ["Rule", f"by episode, stratified by task; per task, val = first max(1, round({scfg['val_fraction']} x n)) of a seeded shuffle"],
        ["split_seed", scfg["split_seed"]], ["Train episodes / frames", f"{len(train)} / {int(lengths[train].sum())}"],
        ["Val episodes / frames", f"{len(val)} / {int(lengths[val].sum())}"], ["Val list hash", f"`{data.split_hash(val)}`"],
        ["Val episodes", ", ".join(map(str, val))]])

    L += ["", f"## Comparison with `{dcfg['reference_repo_id']}` (the dataset `smolvla_libero` trained on)", "",
          f"Reference revision `{dcfg['reference_revision']}`, `meta/` files only. Tasks are matched by exact instruction string. "
          "The last column is the largest per-dimension difference between the length-weighted mean state of the task's "
          "episodes (reference, from its per-episode stats) and ours.", ""]
    L += table(["Task", "Our episodes", "Reference episodes", "Same lengths", "Max abs diff, mean state"], ref_rows)

    REPORT.write_text("\n".join(L) + "\n")
    failed = [c for c, ok, _ in checks if not ok]
    print(f"Wrote {REPORT}. Checks: {len(checks) - len(failed)}/{len(checks)} PASS.")
    for c in failed:
        print("FAIL:", c)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
