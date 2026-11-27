"""Project 2 GPU runs (Kaggle): DINO-WM on Wall with the official code, in the P2-R0 venv recipe.

  python -m scripts.p2_run r1     official checkpoint + official planner (50 evals, seed 99); two 8-minute
                                  training probes (decoder on / off) to measure s/step; data value check

Needs the P2-R0 run output attached (wall_single.zip + the Wall checkpoint). Writes results/p2_r1.md.
Training settings: the official checkpoint's saved hydra.yaml where it has the key, else the repo's train.yaml;
every override is listed in TRAIN_OVERRIDES with its source (docs/deviations.md has the same list).
"""
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.p2_env import (COMMIT, DATA_DIR, PY, REPO, clone, extract_data, install_mujoco_py, make_venv, uv_pip,
                            venv_env, write_constraints, env_pins)

REQUIREMENTS = Path("results/p2_requirements.txt")  # exact package set from P2-R0
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
    t1 = time.time()
    extract_data(find_input("p2/wall_single.zip"))
    ckpt_base = find_input("p2/checkpoints/outputs/wall_single/hydra.yaml").parents[2]
    return ckpt_base, (t1 - t0) / 60, (time.time() - t1) / 60


def run_logged(cmd, log_path, timeout_min):
    """Run in the repo with the venv env; output to a file (tqdm included). Returns (exit code or 'timeout', minutes)."""
    t = time.time()
    with open(log_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=REPO, env=venv_env(), stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = "timeout"
    return rc, (time.time() - t) / 60


TQDM = re.compile(r"Epoch \d+ Train:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+):(\d+)<[^,]*,\s*([\d.]+)(s/it|it/s)")


def probe(name, extra, log_dir):
    """Train for PROBE_MIN minutes, then stop. Returns dict with steps done, steps/epoch, s/step."""
    cmd = [str(PY), "train.py", "--config-name", "train.yaml", *[o for o, _ in TRAIN_OVERRIDES], *extra,
           f"ckpt_base_path=/tmp/p2_probe_{name}"]
    print("$", " ".join(cmd), flush=True)
    rc, minutes = run_logged(cmd, log_dir / f"probe_{name}.log", PROBE_MIN)
    text = (log_dir / f"probe_{name}.log").read_text(errors="replace").replace("\r", "\n")
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


def plan(model_name, ckpt_base, log_dir, tag):
    """Official planner (plan_wall.yaml: MPC-CEM, 50 evals, seed 99). Returns the final logs.json entry."""
    t = time.time()
    cmd = [str(PY), "plan.py", "--config-name", "plan_wall.yaml", f"model_name={model_name}",
           f"ckpt_base_path={ckpt_base}"]
    print("$", " ".join(cmd), flush=True)
    rc, minutes = run_logged(cmd, log_dir / f"plan_{tag}.log", 150)
    logs = [p for p in REPO.rglob("logs.json") if p.stat().st_mtime > t]
    final = json.loads(logs[-1].read_text().strip().splitlines()[-1]) if logs else {}
    tail = (log_dir / f"plan_{tag}.log").read_text(errors="replace").replace("\r", "\n").strip()[-2000:]
    return {"exit": rc, "minutes": round(minutes, 1), "final": final, "logs_json": str(logs[-1]) if logs else None,
            "tail": tail}


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


if __name__ == "__main__":
    {"r1": r1}[sys.argv[1]]()
