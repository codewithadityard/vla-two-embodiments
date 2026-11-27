"""Project 2 GPU runs (Kaggle): DINO-WM on Wall with the official code, in the P2-R0 venv recipe.

  python -m scripts.p2_run r1     official checkpoint + official planner (50 evals, seed 99); two 8-minute
                                  training probes (decoder on / off) to measure s/step; data value check
  python -m scripts.p2_run r2 official   the same without the reader test: official reader, option a (P2-R2 v1's
                                  test failed: conversion timed out at 20 min, then /tmp became read-only)
  python -m scripts.p2_run r2     frame-level reader test (<= 30 min; used only if exactly equal and >= 2x faster
                                  than R1's decoder-off probe), then our reduced-budget training (E epochs, GPU 0)
                                  while the official checkpoint is planned without its decoder (GPU 1), then
                                  planning with our final checkpoint

Needs the P2-R0 run output attached (wall_single.zip + the Wall checkpoint). Writes results/p2_r1.md.
Training settings: the official checkpoint's saved hydra.yaml where it has the key, else the repo's train.yaml;
every override is listed in TRAIN_OVERRIDES with its source (docs/deviations.md has the same list).
"""
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.p2_env import (COMMIT, DATA_DIR, P2_OUT, PY, REPO, clone, env_pins, extract_data, install_mujoco_py,
                            make_venv, disk, sh, uv_pip, venv_env, write_constraints)

REQUIREMENTS = Path("results/p2_requirements.txt")  # exact package set from P2-R0
DINOV2_COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"  # dinov2 main as of 2024-09-12 (commit of 2024-02-22)
PROBE_MIN = 8
# (override, source). Source "ckpt" = the official Wall checkpoint's hydra.yaml; "hw" = our hardware.
TRAIN_OVERRIDES = [
    ("env=wall", "ckpt: env.name wall"),
    ("frameskip=5", "ckpt (= repo)"),
    ("num_hist=1", "ckpt (repo default 3)"),
    ("training.batch_size=128", "ckpt (repo default 32)"),
    ("predictor.emb_dropout=0.1", "ckpt (repo default 0)"),
    ("~decoder.channel", "ckpt has no channel key -> VQVAE default 128 (repo yaml sets 384)"),
    ("env.num_workers=4", "hw: Kaggle has 4 CPUs (ckpt: 0, with 32 processes of batch 4)"),
]


def find_input(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no /kaggle/input/**/{pattern}: is the P2-R0 output attached?")
    return Path(hits[0])


def setup():
    """Venv with exactly P2-R0's packages, MuJoCo, data extracted to /tmp. Returns (ckpt_base_path, minutes)."""
    t0 = time.time()
    clone()
    make_venv()
    write_constraints()
    install_mujoco_py(env_pins())
    keep = [l for l in REQUIREMENTS.read_text().splitlines()
            if l and not re.match(r"(torch|torchvision|triton|nvidia-)", l)]  # torch family comes from the cu121 index
    Path("/tmp/p2_req.txt").write_text("\n".join(keep) + "\n")
    uv_pip("-r", "/tmp/p2_req.txt")
    # train.yaml / plan_wall.yaml name the submitit launcher in their defaults, so its plugin must be installed
    # for the configs to compose (P2-R0 installed it after writing p2_requirements.txt; P2-R1 v1 failed on this)
    uv_pip(env_pins()["hydra-submitit-launcher"])
    # models/dino.py calls torch.hub.load("facebookresearch/dinov2", ...) unpinned, i.e. today's main, which now
    # needs Python >= 3.10 (P2-R1 v2). torch.hub reuses its cache folder if present, so we place dinov2 at the main
    # commit of the official checkpoint's training date (2024-09-12) there; the official code is unchanged.
    hub = Path("/root/.cache/torch/hub/facebookresearch_dinov2_main")
    if not hub.exists():
        sh(["git", "clone", "-q", "https://github.com/facebookresearch/dinov2.git", str(hub)])
        sh(["git", "-C", str(hub), "checkout", "-q", DINOV2_COMMIT])
    t1 = time.time()
    extract_data(find_input("p2/wall_single.zip"))
    ckpt_base = find_input("p2/checkpoints/outputs/wall_single/hydra.yaml").parents[2]
    return ckpt_base, (t1 - t0) / 60, (time.time() - t1) / 60


def start_logged(cmd, log_path, gpu=None):
    """Start in the repo with the venv env (on one GPU if given); output to a file (tqdm included)."""
    env = venv_env()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # no EGL here (MuJoCo is CPU-only), so this is safe
    f = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT), f, time.time()


def wait_logged(job, timeout_min):
    """Returns (exit code or 'timeout', minutes)."""
    p, f, t = job
    try:
        rc = p.wait(timeout=max(1, timeout_min * 60 - (time.time() - t)))
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        rc = "timeout"
    f.close()
    return rc, (time.time() - t) / 60


def run_logged(cmd, log_path, timeout_min, gpu=None):
    return wait_logged(start_logged(cmd, log_path, gpu), timeout_min)


TQDM = re.compile(r"Epoch \d+ Train:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+):(\d+)<[^,]*,\s*([\d.]+)(s/it|it/s)")


def train_cmd(extra, ckpt_base):
    return [str(PY), "train.py", "--config-name", "train.yaml", *[o for o, _ in TRAIN_OVERRIDES], *extra,
            f"ckpt_base_path={ckpt_base}"]


def read_log(path):
    return path.read_text(errors="replace").replace("\r", "\n")


def probe(name, extra, log_dir, minutes_max=PROBE_MIN):
    """Train for minutes_max minutes, then stop. Returns dict with steps done, steps/epoch, s/step."""
    cmd = train_cmd(extra, f"/tmp/p2_probe_{name}")
    print("$", " ".join(cmd), flush=True)
    rc, minutes = run_logged(cmd, log_dir / f"probe_{name}.log", minutes_max)
    text = read_log(log_dir / f"probe_{name}.log")
    hits = TQDM.findall(text)
    r = {"name": name, "exit": rc, "minutes": round(minutes, 1), "tail": text.strip()[-1500:]}
    if hits:
        n, total, mm, ss, rate, unit = hits[-1]
        s_it = float(rate) if unit == "s/it" else 1 / float(rate)
        r.update(steps=int(n), steps_per_epoch=int(total), s_per_step_tqdm=s_it,
                 s_per_step_avg=(int(mm) * 60 + int(ss)) / max(1, int(n)),
                 min_per_epoch=int(total) * s_it / 60)
    print("probe", {k: v for k, v in r.items() if k != "tail"}, flush=True)
    return r


def data_check():
    """Are the stored float32 pixels exact integers in 0..255 (so uint8 storage would be lossless)? 20 episodes."""
    code = f"""
import torch, random
from pathlib import Path
files = sorted(Path({str(DATA_DIR)!r}).rglob('obses/episode_*.pth'))
random.seed(0)
ok, lo, hi, dt = True, 1e9, -1e9, None
for p in random.sample(files, 20):
    x = torch.load(p)
    dt = x.dtype
    ok &= bool((x == x.round()).all())
    lo, hi = min(lo, x.min().item()), max(hi, x.max().item())
print('dtype', dt, 'integer-valued', ok, 'min', lo, 'max', hi)
"""
    p = subprocess.run([str(PY), "-c", code], cwd=REPO, env=venv_env(), capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()[-500:]


def plan_start(model_name, ckpt_base, log_dir, tag, gpu=None):
    """Official planner (plan_wall.yaml: MPC-CEM, 50 evals, seed 99), started in the background."""
    cmd = [str(PY), "plan.py", "--config-name", "plan_wall.yaml", f"model_name={model_name}",
           f"ckpt_base_path={ckpt_base}"]
    print("$", " ".join(cmd), flush=True)
    return start_logged(cmd, log_dir / f"plan_{tag}.log", gpu), log_dir / f"plan_{tag}.log", model_name


def plan_wait(started, timeout_min=150):
    """Returns the final logs.json entry of that planning run."""
    job, log, _ = started
    rc, minutes = wait_logged(job, timeout_min)
    # newest logs.json written since this run started (callers never run two plannings at once)
    logs = sorted((p for p in REPO.rglob("logs.json") if p.stat().st_mtime > job[2]), key=lambda p: p.stat().st_mtime)
    final = json.loads(logs[-1].read_text().strip().splitlines()[-1]) if logs else {}
    return {"exit": rc, "minutes": round(minutes, 1), "final": final, "logs_json": str(logs[-1]) if logs else None,
            "tail": read_log(log).strip()[-2000:]}


def plan(model_name, ckpt_base, log_dir, tag, gpu=None):
    return plan_wait(plan_start(model_name, ckpt_base, log_dir, tag, gpu))


def r1():
    checks = []

    def check(n, ok, detail=""):
        checks.append((n, bool(ok), detail))
        print("PASS" if ok else "FAIL", n, str(detail)[:400], flush=True)

    log_dir = Path("/kaggle/working/outputs/p2/r1_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base, install_min, extract_min = setup()
    dc = data_check()
    check("data value check ran", "integer-valued" in dc, dc)

    # training probes (before planning, so nothing else competes for the GPU, CPUs or disk)
    on = probe("decoder_on", [], log_dir)
    off = probe("decoder_off", ["has_decoder=False"], log_dir)
    for r in (on, off):
        check(f"probe {r['name']}: training ran and tqdm timing was read", "s_per_step_tqdm" in r,
              f"{r.get('steps')} steps in {r['minutes']} min")
    saving = 1 - off["s_per_step_tqdm"] / on["s_per_step_tqdm"] if "s_per_step_tqdm" in on and "s_per_step_tqdm" in off else None

    # official checkpoint, official planner
    res = plan("wall_single", ckpt_base, log_dir, "official")
    sr = next((v for k, v in res["final"].items() if "success_rate" in k), None)
    check("official checkpoint: planning finished and logs.json has a success rate", res["exit"] == 0 and sr is not None,
          f"exit {res['exit']}, {res['minutes']} min")

    L = ["# P2-R1: official Wall checkpoint + planner; training speed probes (Kaggle T4)", "",
         "Generated by `python -m scripts.p2_run r1`. Every number is measured in this run.", "",
         f"- Repo gaoyuezhou/dino_wm @ `{COMMIT}`; venv = P2-R0's package set; setup {install_min:.1f} min, "
         f"data extraction {extract_min:.1f} min",
         f"- Data value check (20 episodes): `{dc}`", "", "## Checks", "",
         "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {str(d).replace('|', '/')[:300]} |" for n, ok, d in checks]
    L += ["", "## Official checkpoint (epoch 65), plan_wall.yaml unchanged (MPC-CEM, 50 evals, seed 99)", "",
          f"- Success rate: **{sr}** (paper Table 1: 0.96 with MPC; Table 8: CEM alone 0.74)",
          f"- Planning time: {res['minutes']} min; logs: `{res['logs_json']}`", "", "```json",
          json.dumps(res["final"], indent=1)[:3000], "```"]
    L += ["", f"## Training speed probes ({PROBE_MIN} min each, then stopped)", "",
          "Overrides (source): " + "; ".join(f"`{o}` ({s})" for o, s in TRAIN_OVERRIDES), "",
          "| Probe | Steps done | Steps/epoch | s/step (tqdm) | s/step (average incl. start) | min/epoch |",
          "|---|---|---|---|---|---|"]
    for r in (on, off):
        L.append(f"| {r['name']} | {r.get('steps', '–')} | {r.get('steps_per_epoch', '–')} | "
                 f"{r.get('s_per_step_tqdm', float('nan')):.2f} | {r.get('s_per_step_avg', float('nan')):.2f} | "
                 f"{r.get('min_per_epoch', float('nan')):.1f} |")
    if saving is not None:
        L += ["", f"Decoder off saves **{100 * saving:.1f} %** of the time per step "
              f"(approved rule: train without the decoder only if the saving is >= 15 %)."]
    for r in (on, off):
        L += ["", f"### Probe {r['name']}: last output", "", "```", r["tail"], "```"]
    L += ["", "### Planning: last output", "", "```", res["tail"], "```"]
    Path("results").mkdir(exist_ok=True)
    Path("results/p2_r1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


# ---------------------------------------------------------------- R2
R1_OFF_S_PER_STEP, R1_OFF_MIN_PER_EPOCH = 8.39, 77.5  # P2-R1 v3 decoder-off probe, official reader (results/p2_r1.md)
BUDGET_MIN, MAX_EPOCHS = 180, 100  # approved: E = 3 h / measured min per epoch, capped at 100
FAST_TEST_MAX_MIN, CONVERT_MAX_MIN = 30, 20  # approved: at most 30 min on the frame-level reader test
WALL = DATA_DIR / "wall_single"
TRAIN_BASE = Path("/tmp/p2_train")  # our run's checkpoints (the final one is copied to the output)


def fastdata(cmd, log_dir, timeout_min):
    """Run scripts/p2_fastdata.py in the venv. Returns (exit code, minutes, output tail)."""
    env = venv_env()
    env["PYTHONPATH"] = str(REPO)  # the official datasets/ package
    t = time.time()
    try:
        p = subprocess.run([str(PY), str(Path("scripts/p2_fastdata.py").resolve()), cmd, str(WALL)], cwd=REPO,
                           env=env, capture_output=True, text=True, timeout=timeout_min * 60)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", str(e.stdout or "") + str(e.stderr or "")
    (log_dir / f"fastdata_{cmd}.log").write_text(out)
    return rc, (time.time() - t) / 60, out.strip()[-600:]


def official_ckpt_without_decoder(ckpt_base):
    """Copy of the official checkpoint folder whose hydra.yaml says has_decoder: false (approved 2026-11-27): the
    planner then loads no decoder (plan.py load_model), which it uses only to plot predicted frames
    (planning/evaluator.py); success comes from env.eval_state on simulator states. Checkpoint file unchanged."""
    src, dst = Path(ckpt_base) / "outputs" / "wall_single", Path("/tmp/p2_official_nodec/outputs/wall_single")
    dst.mkdir(parents=True, exist_ok=True)
    text = (src / "hydra.yaml").read_text()
    assert text.count("has_decoder: true") == 1, "expected exactly one 'has_decoder: true'"
    (dst / "hydra.yaml").write_text(text.replace("has_decoder: true", "has_decoder: false"))
    if not (dst / "checkpoints").exists():
        (dst / "checkpoints").symlink_to(src / "checkpoints")
    return dst.parents[1]


def r2(mode="test"):
    checks, R = [], {"mode": mode}

    def check(n, ok, detail=""):
        checks.append((n, bool(ok), detail))
        print("PASS" if ok else "FAIL", n, str(detail)[:400], flush=True)
        return ok

    log_dir = P2_OUT / "r2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base, R["install_min"], R["extract_min"] = setup()
    sh(["git", "-C", str(REPO), "apply", str(Path("scripts/p2_fastdata.patch").resolve())])
    try:
        # 1) frame-level reader test, <= 30 min: convert, exact-equality check, training probe (decoder off)
        t = time.time()
        if mode == "official":  # option a, decided by P2-R2 v1's failed test (approved fallback)
            rc, R["convert_min"], R["convert_out"] = "skipped", float("nan"), ""
            converted = False
        else:
            rc, R["convert_min"], R["convert_out"] = fastdata("convert", log_dir, CONVERT_MAX_MIN)
            converted = check("frame-level reader: conversion finished", rc == 0,
                              f"exit {rc}, {R['convert_min']:.1f} min")
        equal = False
        if converted:
            rc, _, R["equal_out"] = fastdata("check", log_dir, 10)
            equal = check("frame-level reader: 64/64 training samples exactly equal to the official reader",
                          rc == 0 and "exactly equal samples: 64/64" in R["equal_out"], R["equal_out"][-200:])
        fast = {}
        left = FAST_TEST_MAX_MIN - (time.time() - t) / 60
        if equal and left >= 4:
            fast = probe("fast_reader", ["has_decoder=False"], log_dir, min(PROBE_MIN, left))
        R["fast_test_min"] = (time.time() - t) / 60
        R["fast"] = fast
        R["speedup"] = R1_OFF_S_PER_STEP / fast["s_per_step_tqdm"] if "s_per_step_tqdm" in fast else None
        use_fast = bool(R["speedup"] and R["speedup"] >= 2 and R["fast_test_min"] <= FAST_TEST_MAX_MIN + 1)
        R["use_fast"] = use_fast
        if not use_fast and (WALL / "obses.npy").exists():
            (WALL / "obses.npy").rename(WALL / "obses.npy.unused")  # official reader (patch falls back to it)
        R["min_per_epoch"] = fast["min_per_epoch"] if use_fast else R1_OFF_MIN_PER_EPOCH
        R["epochs"] = min(MAX_EPOCHS, max(1, math.floor(BUDGET_MIN / R["min_per_epoch"])))
        print("R2 decision:", {k: R[k] for k in ("speedup", "use_fast", "min_per_epoch", "epochs")}, flush=True)
        disk("before training")

        # 2) our training on GPU 0 while the official checkpoint is planned (no decoder) on GPU 1
        official = plan_start("wall_single", official_ckpt_without_decoder(ckpt_base), log_dir, "official_nodec", gpu=1)
        cmd = train_cmd(["has_decoder=False", f"training.epochs={R['epochs']}"], TRAIN_BASE)
        print("$", " ".join(cmd), flush=True)
        rc, R["train_min"] = run_logged(cmd, log_dir / "train.log", 2 * BUDGET_MIN + 60, gpu=0)
        R["train_exit"] = rc
        R["train_tail"] = read_log(log_dir / "train.log").strip()[-2500:]
        R["official"] = plan_wait(official)
        sr = next((v for k, v in R["official"]["final"].items() if "success_rate" in k), None)
        check("official checkpoint (no decoder): planning finished and logs.json has a success rate",
              R["official"]["exit"] == 0 and sr is not None, f"exit {R['official']['exit']}, {R['official']['minutes']} min")
        R["official_sr"] = sr
        latest = sorted(TRAIN_BASE.rglob("checkpoints/model_latest.pth"))
        trained = check(f"our training: {R['epochs']} epochs finished, model_latest.pth saved",
                        rc == 0 and len(latest) == 1, f"exit {rc}, {R['train_min']:.1f} min, {len(latest)} found")

        # 3) planning with our final checkpoint (never chosen by eval results: the last epoch)
        if trained:
            model_dir = latest[0].parents[1]
            out = P2_OUT / "ours"
            (out / "checkpoints").mkdir(parents=True, exist_ok=True)
            shutil.copy(model_dir / "hydra.yaml", out / "hydra.yaml")
            shutil.copy(latest[0], out / "checkpoints" / "model_latest.pth")
            R["ours_dir"] = str(model_dir)
            R["ours"] = plan(str(model_dir.relative_to(TRAIN_BASE / "outputs")), TRAIN_BASE, log_dir, "ours", gpu=0)
            sr = next((v for k, v in R["ours"]["final"].items() if "success_rate" in k), None)
            check("our checkpoint: planning finished and logs.json has a success rate",
                  R["ours"]["exit"] == 0 and sr is not None, f"exit {R['ours']['exit']}, {R['ours']['minutes']} min")
            R["ours_sr"] = sr
    finally:
        write_r2(checks, R)
    sys.exit(0 if checks and all(ok for _, ok, _ in checks) else 1)


def write_r2(checks, R):
    f = R.get("fast", {})
    sp = R.get("speedup")
    L = ["# P2-R2: frame-level reader test, reduced-budget training, planning (Kaggle T4 x2)", "",
         "Generated by `python -m scripts.p2_run r2`. Every number is measured in this run, except the R1 baseline "
         f"({R1_OFF_S_PER_STEP} s/step, {R1_OFF_MIN_PER_EPOCH} min/epoch: P2-R1 v3, official reader, decoder off).", "",
         ("- Mode `official`: reader test skipped. P2-R2 v1 ran it: the one-time conversion did not finish within "
          "its 20-minute limit (so no 2x speed-up was shown within the approved 30 minutes) and /tmp then became "
          "read-only; per the approved rule this run uses the official reader (option a)." if R.get("mode") == "official"
          else "- Mode `test`: frame-level reader test first"),
         f"- Setup {R.get('install_min', 0):.1f} min, data extraction {R.get('extract_min', 0):.1f} min", "",
         "## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {str(d).replace('|', '/')[:300]} |" for n, ok, d in checks]
    L += ["", "## 1. Frame-level reader test (approved: <= 30 min; use it only if >= 2x faster on the same data)", "",
          f"- Conversion: {R.get('convert_min', float('nan')):.1f} min; equality check: `{R.get('equal_out', '–')[-120:]}`",
          f"- Probe (decoder off): {f.get('steps', '–')} steps, {f.get('s_per_step_tqdm', float('nan')):.2f} s/step, "
          f"{f.get('min_per_epoch', float('nan')):.1f} min/epoch",
          f"- Speed-up vs R1: **{sp:.2f}x**" if sp else "- Speed-up vs R1: not measured",
          f"- Test time: {R.get('fast_test_min', float('nan')):.1f} min. Decision: "
          f"**{'frame-level reader' if R.get('use_fast') else 'official reader (option a)'}**", "",
          "## 2. Training (reduced budget)", "",
          f"- **Reduced budget: {R.get('epochs', '–')} epochs** (E = floor(180 min / {R.get('min_per_epoch', float('nan')):.1f} "
          "min per epoch), vs 100 in the paper; the official checkpoint is from epoch 65)",
          f"- Exit {R.get('train_exit', '–')}, {R.get('train_min', float('nan')):.1f} min; model folder `{R.get('ours_dir', '–')}`",
          "", "```", R.get("train_tail", ""), "```", "",
          "## 3. Planning (plan_wall.yaml unchanged: MPC-CEM, 50 evals, seed 99; final checkpoints only)", "",
          "| Model | Success rate | Minutes |", "|---|---|---|"]
    for tag, name in (("official", "Official checkpoint (epoch 65), no decoder"),
                      ("ours", f"Ours, reduced budget ({R.get('epochs', '–')} epochs)")):
        r = R.get(tag, {})
        L.append(f"| {name} | {R.get(tag + '_sr', '–')} | {r.get('minutes', '–')} |")
    L += ["", "Paper (Table 1): 0.96 with MPC; Table 8: CEM alone 0.74."]
    for tag in ("official", "ours"):
        r = R.get(tag)
        if r:
            L += ["", f"### Planning {tag}: final logs.json entry, last output", "", "```json",
                  json.dumps(r["final"], indent=1)[:3000], "```", "", "```", r["tail"], "```"]
    Path("results").mkdir(exist_ok=True)
    Path("results/p2_r2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:45]), flush=True)


if __name__ == "__main__":
    {"r1": r1, "r2": r2}[sys.argv[1]](*sys.argv[2:])
