"""P3-0 (Kaggle, CPU): write docs/data_report_metaworld.md for Meta-World MT10. Every number is computed here.

Run from the repo root:  python -m scripts.data_report_mw
Exits with code 1 if any check fails (the report is still written, with the failures).
Same structure as the M1 report (scripts/data_report.py): non-image columns of every file, images for sample
episodes only (the first episode of each MT10 task).
"""
import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from minivla import data
from minivla.config import load_config
from scripts.data_report import decode_first_image, fmt, git_commit, table
import scripts.data_report as m1

CONFIG = Path("configs/metaworld.yaml")
REPORT = Path("docs/data_report_metaworld.md")
m1.IMG_DIR = Path("outputs/p3_0")  # sample frames (not committed)


def lerobot_metaworld_config():
    """TASK_DESCRIPTIONS etc. from the installed LeRobot (the pinned commit), not typed here."""
    import lerobot
    path = Path(lerobot.__file__).parent / "sim" / "integrations" / "metaworld_config.json"
    return json.loads(path.read_text()), path


def mt10_names():
    """MT10 env names from the installed metaworld's env_dict.py (`MT10_V3 = _get_env_dict([...])`), read with ast.
    Importing metaworld imports mujoco's renderer, which needs OpenGL; the Kaggle CPU session has none (P3-0 v2)."""
    import ast
    import importlib.util
    src = Path(importlib.util.find_spec("metaworld").submodule_search_locations[0]) / "env_dict.py"
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MT10_V3" for t in node.targets):
            return sorted(ast.literal_eval(node.value.args[0])), src
    raise ValueError(f"MT10_V3 not found in {src}")


def reference_policy_config(rcfg):
    from huggingface_hub import hf_hub_download
    out = {}
    for name in ["config.json", "train_config.json"]:
        p = hf_hub_download(rcfg["repo_id"], name, revision=rcfg["revision"])
        out[name] = json.loads(Path(p).read_text())
    return out


def main():
    cfg = load_config(CONFIG)
    dcfg, scfg, rcfg = cfg["data"], cfg["split"], cfg["reference_policy"]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- cheap parts first, so a failure here costs minutes, not the 30-min column download ---
    mt10, mt10_src = mt10_names()
    lr_cfg, lr_path = lerobot_metaworld_config()
    ref = reference_policy_config(rcfg)
    versions = {p: md.version(p) for p in ["lerobot", "metaworld", "mujoco", "gymnasium"]}

    # --- metadata, all non-image columns ---
    repo, rev, meta = dcfg["repo_id"], dcfg["revision"], dcfg["meta_dir"]
    if not Path(meta, "meta/info.json").exists():
        data.fetch_meta(repo, rev, meta)
    if not Path(dcfg["lowdim_path"]).exists():
        print("fetching non-image columns (all files) ...", flush=True)
        data.fetch_lowdim(repo, rev, dcfg["lowdim_path"])
    info, tasks = data.load_info(meta), data.load_tasks(meta)
    frames = data.load_frames(dcfg["lowdim_path"])
    ep_task_all = data.episode_tasks(frames)
    n = len(frames["index"])
    lengths = np.bincount(frames["episode_index"])

    check("episodes == info.json total_episodes", len(ep_task_all) == info["total_episodes"],
          f"{len(ep_task_all)} vs {info['total_episodes']}")
    check("frames == info.json total_frames", n == info["total_frames"], f"{n} vs {info['total_frames']}")
    check("tasks == info.json total_tasks", len(tasks) == info["total_tasks"] == len(set(ep_task_all.values())),
          f"{len(tasks)} in tasks.parquet, {len(set(ep_task_all.values()))} used, {info['total_tasks']} in info.json")
    check("global index is 0..N-1", np.array_equal(frames["index"], np.arange(n)))
    check("frame_index is 0..len-1 in every episode", all(
        np.array_equal(np.sort(frames["frame_index"][frames["episode_index"] == e]), np.arange(lengths[e]))
        for e in ep_task_all))
    check("no NaN/Inf in state, action, timestamp", all(np.isfinite(frames[k]).all() for k in ["state", "action", "timestamp"]))
    ts_err = float(np.abs(frames["timestamp"] - frames["frame_index"] / info["fps"]).max())
    check("timestamp == frame_index / fps", ts_err < 1e-4, f"max abs diff {ts_err:.2e} s")
    meta_eps = pa.concat_tables([pq.read_table(p) for p in sorted(Path(meta, "meta/episodes").glob("*/*.parquet"))]).to_pandas()
    meta_len = dict(zip(meta_eps["episode_index"], meta_eps["length"]))
    check("episode lengths match meta/episodes", all(meta_len[e] == lengths[e] for e in ep_task_all))

    # --- task names: dataset instruction -> Meta-World env name (LeRobot's TASK_DESCRIPTIONS), MT10 from metaworld ---
    desc_to_env = {}
    for env_name, text in lr_cfg["TASK_DESCRIPTIONS"].items():
        desc_to_env.setdefault(text, []).append(env_name)
    env_of = {t: desc_to_env.get(text, []) for t, text in tasks.items()}
    check("every dataset instruction matches exactly one env in LeRobot's TASK_DESCRIPTIONS",
          all(len(v) == 1 for v in env_of.values()),
          f"{sum(len(v) == 1 for v in env_of.values())}/{len(tasks)}; unmatched: "
          f"{[tasks[t] for t, v in env_of.items() if len(v) != 1][:5]}")
    task_of_env = {v[0]: t for t, v in env_of.items() if len(v) == 1}
    mt10_tasks = {name: task_of_env.get(name) for name in mt10}
    check("metaworld MT10_V3 has 10 tasks, all in the dataset", len(mt10) == 10 and None not in mt10_tasks.values(),
          f"{mt10}; missing: {[k for k, v in mt10_tasks.items() if v is None]}")

    # --- MT10 subset, split, stats ---
    keep = {t for t in mt10_tasks.values() if t is not None}
    ep_task = {e: t for e, t in ep_task_all.items() if t in keep}
    sel = np.isin(frames["task_index"], list(keep))
    mt10_frames = {k: v[sel] for k, v in frames.items()}
    train, val = data.split_episodes(ep_task, scfg["split_seed"], scfg["val_fraction"])
    train2, val2 = data.split_episodes(ep_task, scfg["split_seed"], scfg["val_fraction"])
    check("split: train and val are disjoint", not set(train) & set(val))
    check("split: train + val cover every MT10 episode", set(train) | set(val) == set(ep_task))
    check("split: every MT10 task has val episodes", {ep_task[e] for e in val} == keep)
    check("split: same seed gives the same split", (train, val) == (train2, val2))
    stats = data.train_stats(mt10_frames, train)

    # --- images: the first episode of each MT10 task ---
    sample_eps = sorted(min(e for e in ep_task if ep_task[e] == t) for t in keep)
    if not Path(dcfg["sample_path"]).exists():
        print(f"fetching sample episodes {sample_eps} ...", flush=True)
        data.fetch_sample(repo, rev, meta, sample_eps, dcfg["sample_path"])
    sample = data.table_to_frames(pq.read_table(dcfg["sample_path"]))
    check("sample holds exactly the first episode of each MT10 task",
          sorted(np.unique(sample["episode_index"]).tolist()) == sample_eps, f"{sample_eps}")
    check("sample state/action == column-only read for the same frames",
          np.array_equal(sample["state"], frames["state"][sample["index"]]) and
          np.array_equal(sample["action"], frames["action"][sample["index"]]))
    cam_rows = []
    for k in [k for k, v in info["features"].items() if v["dtype"] in ("image", "video")]:
        shape = decode_first_image(dcfg["sample_path"], k)
        want = info["features"][k]["shape"]  # [3, 480, 480] (C, H, W) in this dataset's info.json
        check(f"decoded {k} shape matches info.json", sorted(shape) == sorted(want), f"decoded {shape}, info {want}")
        cam_rows.append([f"`{k}`", info["features"][k]["dtype"], "x".join(map(str, shape)), f"`{m1.IMG_DIR / k}.png`"])
    check("exactly one camera", len(cam_rows) == 1, f"{len(cam_rows)}")

    # --- reference policy (eval settings we copy) and env versions ---
    nas = ref["config.json"]["n_action_steps"]
    check("smolvla_metaworld n_action_steps at the pinned revision == configs/metaworld.yaml",
          nas == rcfg["n_action_steps"], f"{nas} vs {rcfg['n_action_steps']}")
    env_cfg = ref["train_config.json"].get("env", {})
    check("MuJoCo version unchanged by the metaworld extra (3.8.1, as in all Project 1 runs)",
          versions["mujoco"] == "3.8.1", versions["mujoco"])

    # --- report ---
    L = ["# Data report: Meta-World MT10 (P3-0)", "",
         "Generated on Kaggle (CPU) by `python -m scripts.data_report_mw`. Every number below is computed from the data "
         "and the pinned configs; nothing is typed by hand.", "", "## Source", ""]
    L += table(["Item", "Value"], [
        ["Dataset", f"`{repo}`"], ["Hub revision", f"`{rev}`"], ["LeRobot format", info["codebase_version"]],
        ["Robot", info["robot_type"]], ["MT10 task list", f"`MT10_V3` in the installed metaworld's `{mt10_src.name}`, read without importing metaworld (its import needs OpenGL, absent on the CPU session)"],
        ["Instruction -> env name", f"exact match in LeRobot's `TASK_DESCRIPTIONS` (`{lr_path.name}` at the pinned LeRobot commit)"],
        ["How it was read", "non-image columns of every file via `HfFileSystem` + pyarrow column selection; "
         "images only for the first episode of each MT10 task"],
        ["Versions", ", ".join(f"{k} {v}" for k, v in versions.items())],
        ["Code commit", f"`{git_commit()}`"], ["Config", f"`{CONFIG}`"]])
    L += ["", "## Checks", ""]
    L += table(["Check", "Result", "Detail"], [[c, "PASS" if ok else "**FAIL**", d] for c, ok, d in checks])

    mt10_lens = lengths[sorted(ep_task)]
    all_lens = lengths[sorted(ep_task_all)]
    L += ["", "## Counts", ""]
    L += table(["Item", "MT50 (whole dataset)", "MT10 (used)"], [
        ["Episodes", len(ep_task_all), len(ep_task)], ["Frames", n, int(mt10_lens.sum())],
        ["Tasks", len(tasks), len(keep)],
        ["Episode length min / mean / max (frames)", f"{all_lens.min()} / {fmt(all_lens.mean(), 1)} / {all_lens.max()}",
         f"{mt10_lens.min()} / {fmt(mt10_lens.mean(), 1)} / {mt10_lens.max()}"],
        ["fps (metadata)", info["fps"], info["fps"]]])

    L += ["", "## MT10 tasks", ""]
    rows, val_set = [], set(val)
    for name in mt10:
        t = mt10_tasks[name]
        if t is None:
            rows.append([name, "–", "not in dataset", "", "", "", "", ""])
            continue
        eps = [e for e in ep_task if ep_task[e] == t]
        lens = lengths[eps]
        rows.append([name, t, tasks[t], len(eps), int(lens.sum()), f"{lens.min()} / {fmt(lens.mean(), 1)} / {lens.max()}",
                     sum(e not in val_set for e in eps), sum(e in val_set for e in eps)])
    L += table(["Env name", "Task index", "Instruction", "Episodes", "Frames", "Length min / mean / max",
                "Train eps", "Val eps"], rows)

    L += ["", "## Cameras", ""]
    L += table(["Key", "Stored as", "Decoded shape", "Sample frame (not committed)"], cam_rows)

    for name, feat in [("state", "observation.state"), ("action", "action")]:
        names = info["features"][feat]["names"]
        names = (names.get("axes") or names.get("motors") or []) if isinstance(names, dict) else (names or [])
        s = stats[name]
        L += ["", f"## {name.capitalize()} (`{feat}`, {len(s['mean'])}-D): stats on the MT10 train split", ""]
        L += table(["Dim", "Name (metadata)", "Mean", "Std", "Min", "Max", "P1", "P99"],
                   [[i, names[i] if i < len(names) else "", *(fmt(s[k][i]) for k in ["mean", "std", "min", "max", "p01", "p99"])]
                    for i in range(len(s["mean"]))])
    grip = mt10_frames["action"][:, -1]
    vals = np.unique(grip)
    L += ["", "## Gripper action (last action dim, MT10)", "",
          f"{len(vals)} distinct values, range {fmt(vals.min())} to {fmt(vals.max())}."]

    L += ["", "## Reference policy settings (`lerobot/smolvla_metaworld`)", "",
          f"Read from `config.json` and `train_config.json` at revision `{rcfg['revision']}`. "
          "n_action_steps is fixed from here before any eval and never changed based on eval results.", ""]
    pc = ref["config.json"]
    L += table(["Setting", "Value"], [
        ["chunk_size", pc["chunk_size"]], ["n_action_steps", nas], ["n_obs_steps", pc["n_obs_steps"]],
        ["input features", ", ".join(f"{k} {v['shape']}" for k, v in pc["input_features"].items())],
        ["env.episode_length (train_config)", env_cfg.get("episode_length")], ["env.fps", env_cfg.get("fps")],
        ["env.features_map", json.dumps(env_cfg.get("features_map"))],
        ["eval.n_episodes / batch_size", f"{ref['train_config.json']['eval']['n_episodes']} / "
                                         f"{ref['train_config.json']['eval']['batch_size']}"],
        ["dataset", f"{ref['train_config.json']['dataset']['repo_id']} (revision "
                    f"{ref['train_config.json']['dataset']['revision']})"]])

    L += ["", "## Train/val split (MT10)", ""]
    L += table(["Item", "Value"], [
        ["Rule", f"by episode, stratified by task; per task, val = first max(1, round({scfg['val_fraction']} x n)) of a seeded shuffle"],
        ["split_seed", scfg["split_seed"]], ["Train episodes / frames", f"{len(train)} / {int(lengths[train].sum())}"],
        ["Val episodes / frames", f"{len(val)} / {int(lengths[val].sum())}"], ["Val list hash", f"`{data.split_hash(val)}`"],
        ["Val episodes", ", ".join(map(str, val))]])

    REPORT.write_text("\n".join(L) + "\n")
    failed = [c for c, ok, _ in checks if not ok]
    print("\n".join(L))
    print(f"Wrote {REPORT}. Checks: {len(checks) - len(failed)}/{len(checks)} PASS.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
