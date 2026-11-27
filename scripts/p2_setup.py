"""P2-R0 (Kaggle, CPU, no GPU quota): set up DINO-WM for Wall and find out what it needs.

1. Clone the official repo at the pinned commit; Python 3.9 venv with torch 2.3.0; every install is constrained
   by environment.yaml's pip pins. MuJoCo 2.1 + mujoco-py are installed because plan.py imports env/__init__,
   which imports PointMaze -> gym's MuJoCo env -> mujoco_py (found in P2-R0 v3).
2. Import train.py, plan.py, the Wall env, the Wall dataset loader and the planner; each missing module is
   installed at its environment.yaml pin (or unpinned, logged).
3. Download wall_single.zip (kept as output; extracted to /tmp because it expands to > 20 GB) and the Wall model
   from the pretrained-checkpoint zip (kept as output).
4. Inspect: dataset shapes, checkpoint files, the checkpoint's saved Hydra config vs the repo's composed train
   config (env=wall), the success criterion in the code, and how train.py uses accelerate/AMP/wandb.

Every finding is printed as it happens, and results/p2_r0.md is written even if a step crashes.
Also writes results/p2_requirements.txt. Exits 1 if any check fails.
"""
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import yaml

from scripts.p2_env import (COMMIT, DATA_DIR, P2_OUT, PY, REPO, clone, disk, download, env_pins, extract_data,
                            install_mujoco_py, make_venv, norm, osf_files, uv_pip, venv_env, write_constraints)

MODULE_TO_PACKAGE = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn",
                     "skimage": "scikit-image", "hydra": "hydra-core", "mujoco_py": "mujoco-py",
                     "dm_control": "dm-control", "hydra_plugins": "hydra-submitit-launcher"}
REPORT = Path("results/p2_r0.md")
JUNK = ("/._", "__MACOSX", ".DS_Store")  # macOS files inside the OSF zips


def venv_run(code, timeout=1800):
    """Run Python code in the venv, inside the repo. Returns (exit code, output)."""
    p = subprocess.run([str(PY), "-c", code], cwd=REPO, capture_output=True, text=True, timeout=timeout,
                       env=venv_env())
    return p.returncode, p.stdout + p.stderr


def discover_packages(modules, pins, max_rounds=40):
    """Import `modules`; install each missing top-level module's package until all import. Returns (ok, installed, log)."""
    code = ("import importlib, sys, traceback\n"
            f"for m in {modules!r}:\n"
            "    try:\n        importlib.import_module(m); print('OK', m)\n"
            "    except ModuleNotFoundError as e:\n        print('MISSING', e.name); sys.exit(3)\n"
            "    except Exception:\n        traceback.print_exc(); sys.exit(4)\n")
    installed = []
    for _ in range(max_rounds):
        rc, out = venv_run(code)
        if rc == 0:
            return True, installed, out
        if rc != 3:
            return False, installed, out[-4000:]
        top = re.search(r"^MISSING (\S+)", out, re.M).group(1).split(".")[0]  # stdout line, not stderr warnings
        if (REPO / top).exists() or (REPO / f"{top}.py").exists():
            return False, installed, f"repo-internal module missing: {top}\n{out[-3000:]}"
        pkg = MODULE_TO_PACKAGE.get(top, top)
        spec = pins.get(norm(pkg), pkg)
        installed.append((top, spec, "pinned" if "==" in spec else "UNPINNED (not in environment.yaml)"))
        print(f"missing module {top} -> installing {spec}", flush=True)
        uv_pip(spec)
    return False, installed, "too many install rounds"


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def grep(pattern, paths):
    p = subprocess.run(["grep", "-rnE", "-A6", pattern, *paths], cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip()


LISTING = """
import torch, numpy as np
from pathlib import Path
root = Path({root!r})
files = [p for p in sorted(root.rglob('*')) if p.is_file() and '.huggingface' not in p.parts]  # skip HF lock files
print('total files', len(files), '| total GB', round(sum(p.stat().st_size for p in files) / 1e9, 2))
obses = [p for p in files if p.parent.name == 'obses']
print('obses/ episode files:', len(obses))
for p in [p for p in files if p.parent.name != 'obses'][:30] + obses[:3]:
    desc = ''
    try:
        if p.suffix in ('.pth', '.pt'):
            try:
                x = torch.load(p, map_location='cpu', mmap=True)  # big files: no full read into RAM
            except Exception:
                if p.stat().st_size > 4e9: raise
                x = torch.load(p, map_location='cpu')
            desc = tuple(x.shape) if hasattr(x, 'shape') else type(x).__name__ + (f' len {{len(x)}}' if hasattr(x, '__len__') else '')
            if isinstance(x, (list, tuple)) and len(x) and hasattr(x[0], 'shape'):
                desc += f' first item {{tuple(x[0].shape)}}'
        elif p.suffix == '.npy':
            desc = np.load(p, mmap_mode='r').shape
    except Exception as e:
        desc = 'error: ' + repr(e)[:150]
    print(f'{{p.relative_to(root)}} | {{p.stat().st_size / 1e6:.1f}} MB | {{desc}}')
"""


def main():
    t0 = time.time()
    checks, S = [], {}  # S: report sections, filled as we go

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print("PASS" if ok else "FAIL", name, str(detail)[:500], flush=True)

    def section(title, text):
        S[title] = text
        print(f"\n===== {title} =====\n{text}\n", flush=True)

    try:
        # 1. repo, venv, MuJoCo
        disk("start")
        clone()
        make_venv()
        pins = env_pins()
        write_constraints()
        uv_pip(*[pins[k] for k in ["numpy", "setuptools", "pyyaml"] if k in pins])
        install_mujoco_py(pins)
        rc, out = venv_run("import sys, torch; print(sys.version.split()[0], torch.__version__)")
        check("venv: Python 3.9 + torch 2.3.0", rc == 0 and out.startswith("3.9") and "2.3.0" in out, out.strip())
        rc, out = venv_run("import mujoco_py; print('mujoco_py', mujoco_py.__version__)")
        check("mujoco_py imports (builds its CPU extension on first import)", rc == 0, out.strip()[-1500:])

        # 2. packages for the Wall code path
        wall_env = sorted(str(p.relative_to(REPO).with_suffix("")).replace("/", ".")
                          for p in REPO.glob("env/**/*.py") if "wall" in str(p).lower() and p.name != "__init__.py")
        modules = ["train", "plan", "datasets.wall_dset", "planning.mpc", "planning.cem", "planning.objectives",
                   *wall_env]
        ok, installed, log = discover_packages(modules, pins)
        check("train.py, plan.py, Wall env, Wall dataset loader and planner import in the venv", ok,
              f"{len(installed)} packages added")
        section("Packages added for the Wall code path",
                "\n".join(f"- {m}: `{s}` ({p})" for m, s, p in installed) or "(none)")
        section("Import log", "```\n" + log.strip() + "\n```")
        freeze = subprocess.run([sys.executable, "-m", "uv", "pip", "freeze", "--python", str(PY)],
                                capture_output=True, text=True).stdout
        Path("results").mkdir(exist_ok=True)
        Path("results/p2_requirements.txt").write_text(freeze)
        S["_install_min"] = (time.time() - t0) / 60

        # 3. data + Wall checkpoint from OSF
        disk("before downloads")
        files = osf_files()
        section("OSF files", "\n".join(f"- {p}: {(s or 0) / 1e6:.1f} MB" for p, s, _ in files))
        zip_entry = [f for f in files if f[0].endswith("/wall_single.zip")]
        ckpt_zip = [f for f in files if f[0] == "/checkpoints/outputs.zip"]  # all pretrained models in one zip
        check("OSF lists wall_single.zip and /checkpoints/outputs.zip", zip_entry and ckpt_zip)
        t1 = time.time()
        z = P2_OUT / "wall_single.zip"  # kept as output (1.67 GB); every run extracts it to /tmp
        download(zip_entry[0][2], z)
        S["_unzipped_gb"] = extract_data(z) / 1e9
        ckpt_dir = P2_OUT / "checkpoints"
        cz = Path("/tmp/ckpt_outputs.zip")
        download(ckpt_zip[0][2], cz)
        with zipfile.ZipFile(cz) as zf:
            names = [n for n in zf.namelist() if not any(j in n for j in JUNK)]
            section("Checkpoint zip entries (macOS junk removed)", "\n".join(f"- {n}" for n in names[:200]))
            wall = [n for n in names if "wall" in n.lower()]
            zf.extractall(ckpt_dir, members=wall)
        cz.unlink()
        check("checkpoint zip contains Wall model files", wall, f"{len(wall)} entries")
        S["_download_min"] = (time.time() - t1) / 60
        disk("after downloads")

        # 4a. dataset
        rc, out = venv_run(LISTING.format(root=str(DATA_DIR)))
        section("Dataset listing (DATASET_DIR; HF lock files skipped; first 3 episode files)", "```\n" + out.strip() + "\n```")
        sizes_ok = bool(re.search(r"\(1920, 5[01]\b|len 1920\b|obses/ episode files: 1920\b", out))
        check("dataset: something with 1920 trajectories x ~50 steps is present (paper: 1920 x 50)",
              rc == 0 and sizes_ok, "see listing")

        # 4b. checkpoint files + saved Hydra config
        ckpt_files = [p for p in sorted(ckpt_dir.rglob("*")) if p.is_file()]
        section("Wall checkpoint files", "\n".join(f"- {p.relative_to(ckpt_dir)}: {p.stat().st_size / 1e6:.1f} MB"
                                                    for p in ckpt_files))
        model_files = [p for p in ckpt_files if p.suffix in (".pth", ".pt", ".ckpt")]
        rc, out = venv_run("import torch\n" + "".join(
            f"x = torch.load({str(p)!r}, map_location='cpu'); print({p.name!r}, type(x).__name__, "
            f"sorted(map(str, x))[:30] if isinstance(x, dict) else '', 'epoch:', "
            f"x.get('epoch') if isinstance(x, dict) else '')\n" for p in model_files))
        section("Checkpoint torch.load", "```\n" + out.strip()[-3000:] + "\n```")
        check("Wall checkpoint file(s) load with torch.load (repo on the path)", rc == 0 and model_files)
        cfgs = [p for p in ckpt_files if p.suffix == ".yaml" and "frameskip" in p.read_text(errors="replace")]
        check("the checkpoint folder has its saved training config (a yaml with frameskip)", cfgs,
              ", ".join(str(p.relative_to(ckpt_dir)) for p in cfgs))
        if cfgs:
            section(f"Saved training config of the official Wall checkpoint ({cfgs[0].relative_to(ckpt_dir)})",
                    "```yaml\n" + cfgs[0].read_text().strip() + "\n```")

        # 4c. composed repo config for env=wall vs the checkpoint's saved config
        # train.yaml's defaults name the submitit launcher (and set its keys), so its plugin must be installed for
        # the config to compose; a plain (non --multirun) run never uses a launcher.
        uv_pip(pins.get("hydra-submitit-launcher", "hydra-submitit-launcher"))
        composed = subprocess.run([str(PY), "train.py", "--config-name", "train.yaml", "env=wall", "--cfg", "job"],
                                  cwd=REPO, capture_output=True, text=True, env=venv_env())
        check("repo train config composes with env=wall (hydra --cfg job, default launcher, nothing trained)", composed.returncode == 0,
              composed.stderr.strip()[-500:])
        if cfgs and composed.returncode == 0:
            saved, repo = flatten(yaml.safe_load(cfgs[0].read_text())), flatten(yaml.safe_load(composed.stdout))
            rows = [f"| `{k}` | `{repo.get(k, '—')}` | `{saved.get(k, '—')}` |"
                    for k in sorted(set(saved) | set(repo)) if saved.get(k, "—") != repo.get(k, "—")]
            section("Repo train config (env=wall) vs the checkpoint's saved config (differing keys only)",
                    "| Key | Repo (train.yaml, env=wall) | Checkpoint |\n|---|---|---|\n" + "\n".join(rows))

        # 4d. success criterion and training mechanics, quoted from code
        success = grep(r"def eval_state|success", [p for p in ["env", "plan.py", "planning"] if (REPO / p).exists()])
        section("Success criterion and evaluation code (grep: def eval_state|success)",
                "```\n" + success[:8000] + "\n```")
        check("success criterion found in the Wall env code", any("wall" in l.lower() for l in success.splitlines()))
        section("train.py: accelerate / AMP / wandb (grep)", "```\n" + grep(
            r"accelerat|autocast|mixed_precision|GradScaler|bf16|fp16|wandb\.(init|log)|WANDB", ["train.py"])[:4000]
            + "\n```")
    except Exception:
        check("script ran to the end", False, traceback.format_exc()[-1500:])

    # report (always written)
    L = ["# P2-R0: DINO-WM setup for Wall (Kaggle CPU)", "",
         "Generated by `python -m scripts.p2_setup`. Every number is measured in this run.", "",
         f"- Repo: gaoyuezhou/dino_wm @ `{COMMIT}`; venv Python 3.9 + torch 2.3.0 (cu121); installs constrained by "
         "environment.yaml's pip pins",
         f"- Minutes: venv + packages {S.get('_install_min', float('nan')):.1f}, downloads "
         f"{S.get('_download_min', float('nan')):.1f}, total {(time.time() - t0) / 60:.1f}",
         f"- Output kept for later runs: `{P2_OUT}` (wall_single.zip + Wall checkpoint). The data expands to "
         f"{S.get('_unzipped_gb', float('nan')):.1f} GB and is extracted to `{DATA_DIR}` (DATASET_DIR) in each run.",
         "", "## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {str(d).replace('|', '/').replace(chr(10), ' ')[:300]} |"
          for n, ok, d in checks]
    for title, text in S.items():
        if not title.startswith("_"):
            L += ["", f"## {title}", "", text]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    failed = [n for n, ok, _ in checks if not ok]
    print(f"Checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
