"""P2-R0 (Kaggle, CPU, no GPU quota): set up DINO-WM for Wall and find out what it needs.

1. Clone the official repo at the pinned commit; make a Python 3.9 venv with torch 2.3.0 (environment.yaml pins).
2. Find the packages the Wall code path needs by importing train.py, plan.py, the Wall env, the Wall dataset
   loader and the planner; each missing module is installed at its environment.yaml pin (or unpinned, logged).
3. Download wall_single.zip and the Wall checkpoint files from OSF into the run output (/kaggle/working/outputs/p2),
   so later runs attach them instead of downloading again.
4. Inspect: dataset shapes, checkpoint files, the checkpoint's saved Hydra config vs the repo's composed train
   config (env=wall), the success criterion in the code, and how train.py uses accelerate/AMP/wandb.

Writes results/p2_r0.md and results/p2_requirements.txt. Exits 1 if any check fails.
"""
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

from scripts.p2_env import COMMIT, P2_OUT, PY, REPO, clone, download, env_pins, make_venv, norm, osf_files, uv_pip

MODULE_TO_PACKAGE = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn",
                     "skimage": "scikit-image", "hydra": "hydra-core", "mujoco_py": "mujoco-py",
                     "dm_control": "dm-control", "hydra_plugins": "hydra-submitit-launcher"}
REPORT = Path("results/p2_r0.md")


def venv_run(code, timeout=600):
    """Run Python code in the venv, inside the repo (so its packages import). Returns (exit code, output)."""
    p = subprocess.run([str(PY), "-c", code], cwd=REPO, capture_output=True, text=True, timeout=timeout,
                       env={"WANDB_MODE": "disabled", "DATASET_DIR": str(P2_OUT / "data"), "PATH": "/usr/bin:/bin",
                            "HOME": "/root"})
    return p.returncode, p.stdout + p.stderr


def discover_packages(modules, pins, max_rounds=40):
    """Import `modules`; install each missing top-level module's package until all import. Returns (ok, installed, log)."""
    code = ("import importlib, sys, traceback\n"
            f"for m in {modules!r}:\n"
            "    try:\n        importlib.import_module(m); print('OK', m)\n"
            "    except ModuleNotFoundError as e:\n        print('MISSING', e.name); sys.exit(3)\n"
            "    except Exception:\n        traceback.print_exc(); sys.exit(4)\n")
    installed, log = [], []
    for _ in range(max_rounds):
        rc, out = venv_run(code)
        if rc == 0:
            return True, installed, log
        if rc != 3:
            log.append(out[-3000:])
            return False, installed, log
        top = re.search(r"^MISSING (\S+)", out, re.M).group(1).split(".")[0]  # stdout line, not stderr warnings
        if (REPO / top).exists() or (REPO / f"{top}.py").exists():
            log.append(f"repo-internal module missing: {top}\n{out[-2000:]}")
            return False, installed, log
        pkg = MODULE_TO_PACKAGE.get(top, top)
        spec = pins.get(norm(pkg), pkg)
        installed.append((top, spec, "pinned" if "==" in spec else "UNPINNED (not in environment.yaml)"))
        print(f"missing module {top} -> installing {spec}", flush=True)
        uv_pip(spec)
    return False, installed, ["too many install rounds"]


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def grep(pattern, paths):
    p = subprocess.run(["grep", "-rnE", "-A6", pattern, *paths], cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip()


def main():
    t0 = time.time()
    checks, L = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print("PASS" if ok else "FAIL", name, detail, flush=True)

    # 1. repo + venv
    clone()
    make_venv()
    pins = env_pins()
    uv_pip(pins["numpy"], pins.get("pyyaml", "pyyaml"))
    rc, out = venv_run("import sys, torch; print(sys.version.split()[0], torch.__version__)")
    check("venv: Python 3.9 + torch 2.3.0", rc == 0 and out.startswith("3.9") and "2.3.0" in out, out.strip())

    # 2. packages for the Wall code path
    wall_env = sorted(str(p.relative_to(REPO).with_suffix("")).replace("/", ".")
                      for p in REPO.glob("env/**/*.py") if "wall" in str(p).lower() and p.name != "__init__.py")
    modules = ["train", "plan", "datasets.wall_dset", "planning.mpc", "planning.cem", "planning.objectives", *wall_env]
    ok, installed, log = discover_packages(modules, pins)
    check("train.py, plan.py, Wall env, Wall dataset loader and planner import in the venv", ok,
          f"{len(installed)} packages added" + ("" if ok else "; see log below"))
    freeze = subprocess.run([sys.executable, "-m", "uv", "pip", "freeze", "--python", str(PY)],
                            capture_output=True, text=True).stdout
    Path("results").mkdir(exist_ok=True)
    Path("results/p2_requirements.txt").write_text(freeze)
    install_min = (time.time() - t0) / 60

    # 3. data + Wall checkpoint from OSF
    files = osf_files()
    data_dir, ckpt_dir = P2_OUT / "data", P2_OUT / "checkpoints"
    zip_entry = [f for f in files if f[0].endswith("/wall_single.zip")]
    ckpt_zip = [f for f in files if f[0] == "/checkpoints/outputs.zip"]  # all pretrained models in one zip
    check("OSF lists wall_single.zip and /checkpoints/outputs.zip", zip_entry and ckpt_zip,
          f"{(ckpt_zip[0][1] or 0) / 1e9:.2f} GB checkpoint zip" if ckpt_zip else "")
    t1 = time.time()
    if zip_entry:
        z = data_dir / "wall_single.zip"
        download(zip_entry[0][2], z)
        zipfile.ZipFile(z).extractall(data_dir)
        z.unlink()
    wall_members = []
    if ckpt_zip:  # keep only the Wall model (outputs/<model_name>/...), so plan.py finds it under ckpt_base_path
        z = Path("/tmp/ckpt_outputs.zip")
        download(ckpt_zip[0][2], z)
        with zipfile.ZipFile(z) as zf:
            print("checkpoint zip top-level entries:", sorted({n.split("/")[1] for n in zf.namelist() if "/" in n}))
            wall_members = [n for n in zf.namelist() if "wall" in n.lower()]
            zf.extractall(ckpt_dir, members=wall_members)
        z.unlink()
    check("checkpoint zip contains Wall model files", wall_members, f"{len(wall_members)} entries")
    download_min = (time.time() - t1) / 60

    # 4a. dataset
    rc, out = venv_run(f"""
import torch, numpy as np, os
from pathlib import Path
root = Path({str(data_dir / 'wall_single')!r})
n_files = 0
for p in sorted(root.rglob('*')):
    if p.is_dir(): continue
    n_files += 1
    if n_files > 40: continue
    desc = ''
    try:
        if p.suffix in ('.pth', '.pt'):
            x = torch.load(p, map_location='cpu')
            desc = tuple(x.shape) if hasattr(x, 'shape') else type(x).__name__ + (f' len {{len(x)}}' if hasattr(x, '__len__') else '')
        elif p.suffix in ('.npy',):
            desc = np.load(p, mmap_mode='r').shape
    except Exception as e:
        desc = 'error: ' + repr(e)[:100]
    print(f'{{p.relative_to(root)}} | {{p.stat().st_size / 1e6:.1f}} MB | {{desc}}')
print('total files', n_files)
""", timeout=1800)
    data_listing = out.strip()
    sizes_ok = "(1920, 50" in data_listing or "(1920, 51" in data_listing or "len 1920" in data_listing
    check("dataset loads; a tensor/list with 1920 trajectories x ~50 steps is present (paper: 1920 x 50)",
          rc == 0 and sizes_ok, "see listing")

    # 4b. checkpoint files + saved Hydra config
    ckpt_listing = "\n".join(f"{p.relative_to(ckpt_dir)} | {p.stat().st_size / 1e6:.1f} MB"
                             for p in sorted(ckpt_dir.rglob("*")) if p.is_file())
    model_files = [p for p in ckpt_dir.rglob("*") if p.suffix in (".pth", ".pt", ".ckpt")]
    rc, out = venv_run("import torch\n" + "".join(
        f"x = torch.load({str(p)!r}, map_location='cpu'); print({p.name!r}, type(x).__name__, "
        f"sorted(x)[:20] if isinstance(x, dict) else '')\n" for p in model_files), timeout=1800)
    check("Wall checkpoint file(s) load with torch.load (repo on the path)", rc == 0 and model_files,
          out.strip()[-500:])
    ckpt_load = out.strip()
    cfgs = [p for p in ckpt_dir.rglob("*.yaml") if "frameskip" in p.read_text()]
    check("the checkpoint folder has its saved training config (a yaml with frameskip)", cfgs,
          ", ".join(str(p.relative_to(ckpt_dir)) for p in cfgs))

    # 4c. composed repo config for env=wall vs the checkpoint's saved config
    def compose():
        return subprocess.run([str(PY), "train.py", "--config-name", "train.yaml", "env=wall", "hydra/launcher=basic",
                               "--cfg", "job"], cwd=REPO, capture_output=True, text=True,
                              env={"WANDB_MODE": "disabled", "DATASET_DIR": str(data_dir), "PATH": "/usr/bin:/bin",
                                   "HOME": "/root"})
    composed = compose()
    if composed.returncode != 0 and "launcher" in composed.stderr:  # the default launcher plugin may be needed
        spec = pins.get("hydra-submitit-launcher", "hydra-submitit-launcher")
        installed.append(("hydra_plugins (launcher)", spec, "pinned" if "==" in spec else "UNPINNED"))
        uv_pip(spec)
        composed = compose()
    check("repo train config composes with env=wall (hydra --cfg job, nothing trained)", composed.returncode == 0,
          composed.stderr.strip()[-300:])
    diff_rows = []
    if cfgs and composed.returncode == 0:
        saved, repo = flatten(yaml.safe_load(cfgs[0].read_text())), flatten(yaml.safe_load(composed.stdout))
        for k in sorted(set(saved) | set(repo)):
            if saved.get(k, "—") != repo.get(k, "—"):
                diff_rows.append(f"| `{k}` | `{repo.get(k, '—')}` | `{saved.get(k, '—')}` |")

    # 4d. success criterion and training mechanics, quoted from code
    success_code = grep(r"def eval_state|success", [p for p in ["env", "plan.py", "planning"] if (REPO / p).exists()])
    train_code = grep(r"accelerat|autocast|mixed_precision|GradScaler|bf16|fp16|wandb\.(init|log)|WANDB", ["train.py"])
    wall_success = any("wall" in line.lower() for line in success_code.splitlines())
    check("success criterion found in the Wall env code", wall_success, "quoted below")

    # report
    L = ["# P2-R0: DINO-WM setup for Wall (Kaggle CPU)", "",
         "Generated by `python -m scripts.p2_setup`. Every number is measured in this run.", "",
         f"- Repo: gaoyuezhou/dino_wm @ `{COMMIT}`; venv Python 3.9 + torch 2.3.0 (cu121)",
         f"- Minutes: venv + packages {install_min:.1f}, downloads {download_min:.1f}, total {(time.time() - t0) / 60:.1f}",
         f"- Output saved for later runs: `{P2_OUT}` (data + checkpoints)", "", "## Checks", "",
         "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {str(d).replace('|', '/')[:300]} |" for n, ok, d in checks]
    L += ["", "## Packages added for the Wall code path", "", "| Module | Installed | Pin |", "|---|---|---|"]
    L += [f"| {m} | `{s}` | {p} |" for m, s, p in installed]
    if log:
        L += ["", "## Import failure log", "", "```", *log, "```"]
    L += ["", "## OSF files", "", "| Path | MB |", "|---|---|"]
    L += [f"| {p} | {(s or 0) / 1e6:.1f} |" for p, s, _ in files]
    L += ["", "## Dataset listing (first 40 files)", "", "```", data_listing, "```",
          "", "## Checkpoint files", "", "```", ckpt_listing, "", ckpt_load, "```"]
    if cfgs:
        L += ["", f"## Saved training config of the official Wall checkpoint (`{cfgs[0].relative_to(ckpt_dir)}`)",
              "", "```yaml", cfgs[0].read_text().strip(), "```"]
    L += ["", "## Repo train config (env=wall) vs the checkpoint's saved config (differing keys only)", "",
          "| Key | Repo (train.yaml, env=wall) | Checkpoint |", "|---|---|---|", *diff_rows]
    L += ["", "## Success criterion and evaluation code (grep: `def eval_state|success`)", "", "```",
          success_code[:6000], "```", "", "## train.py: accelerate / AMP / wandb (grep)", "", "```",
          train_code[:3000], "```"]
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    failed = [n for n, ok, _ in checks if not ok]
    print(f"Checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
