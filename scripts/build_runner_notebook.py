"""Write notebooks/runner.ipynb from the cell sources below (keeps the notebook readable in diffs).

Run from the repo root:  python -m scripts.build_runner_notebook
"""
import json
from pathlib import Path

INTRO = """# Runner: one Kaggle notebook for every milestone

Import once into Kaggle, then for each run: edit the `RUN` cell, pick the accelerator
(**CPU** for tests that only need torch, **GPU T4** for model work), Internet **on**,
attach any input datasets, then **Save Version → Save & Run All (Commit)**.

What it does: clone `main` of the public repo at `RUN["commit"]` into `/tmp/repo` (not saved),
install `requirements-kaggle.txt`, run `python -m <module> <args>` inside the repo,
and save `outputs/` plus changed `results/`/`docs/` files to `/kaggle/working`.
The last cell prints a **summary** to paste back into the chat (also saved as `summary.md`)."""

CONFIG = '''# ---- edit this cell for each run ----
RUN = {
    "commit": "main",               # branch or full commit hash; the exact hash is logged
    "module": "pytest",             # run as: python -m <module> <args>
    "args": ["-q", "tests"],
    "install": True,                # install requirements-kaggle.txt (about 1 min)
    "purpose": "describe the run",  # copied into the summary and the GPU ledger
}
REPO_URL = "https://github.com/codewithadityard/vla-two-embodiments.git"'''

SETUP = '''import os, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
OUT = WORK / "outputs"           # saved by Kaggle
REPO = Path("/tmp/repo")         # not saved: downloads under repo/data/ don't fill the output
HAS_GPU = shutil.which("nvidia-smi") is not None


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def out_of(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


shutil.rmtree(REPO, ignore_errors=True)
sh(f"git clone -q {REPO_URL} {REPO}")
sh(f"git -C {REPO} checkout -q {RUN['commit']}")
COMMIT = out_of(f"git -C {REPO} rev-parse HEAD")
OUT.mkdir(exist_ok=True)
(REPO / "outputs").symlink_to(OUT)  # scripts write outputs/... -> lands in /kaggle/working
print("commit", COMMIT, "| GPU" if HAS_GPU else "| CPU only")'''

INSTALL = '''if RUN["install"]:
    if HAS_GPU:
        # Default torch wheels are CUDA 13 and need driver >= 580.65 (LeRobot install docs).
        driver = out_of("nvidia-smi --query-gpu=driver_version --format=csv,noheader").splitlines()[0]
        if tuple(int(x) for x in driver.split(".")[:2]) < (580, 65):
            sh(f"{sys.executable} -m pip install -q --index-url https://download.pytorch.org/whl/cu126 torch torchvision")
            sh(f"{sys.executable} -m pip install -q --index-url https://download.pytorch.org/whl/cpu torchcodec")
    sh(f"{sys.executable} -m pip install -q -r {REPO}/requirements-kaggle.txt")
    # Kaggle preinstalls torchao 0.10; peft rejects torchao < 0.16 (feasibility run 1). We don't use it.
    sh(f"{sys.executable} -m pip uninstall -y -q torchao")
INSTALL_MIN = round((time.time() - T0) / 60, 1)'''

RUN_CELL = '''cmd = [sys.executable, "-m", RUN["module"], *RUN["args"]]
env = {**os.environ, "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl" if HAS_GPU else "osmesa"}
print("$", " ".join(cmd), flush=True)
t = time.time()
lines = []
with subprocess.Popen(cmd, cwd=REPO, env=env, text=True, stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
    p.stdin.write("N\\n")  # answers LIBERO's first-import question (keep default paths)
    p.stdin.close()
    for line in p.stdout:
        print(line, end="")
        lines.append(line)
EXIT = p.returncode
RUN_MIN = round((time.time() - t) / 60, 1)
(OUT / "log.txt").write_text("".join(lines))'''

SUMMARY = '''# Copy results/ and docs/ files the script created or changed, so they can go back to the repo.
changed = [l[3:] for l in out_of(f"git -C {REPO} status --porcelain --untracked-files=all -- results docs").splitlines()]
for rel in changed:
    dst = OUT / "repo_files" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / rel, dst)

py = f"{sys.executable} -c"
versions = out_of(f"""{py} 'import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'""")
lerobot = out_of(f"{py} 'import importlib.metadata as m; print(m.version(\\"lerobot\\"))'")

S = ["===== RUN SUMMARY (paste this back) =====",
     f"date_utc: {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}",
     f"purpose: {RUN['purpose']}",
     f"repo commit: {COMMIT}",
     f"command: python -m {RUN['module']} {' '.join(RUN['args'])}",
     f"exit code: {EXIT}  ({'OK' if EXIT == 0 else 'FAILED'})",
     f"accelerator: {'GPU' if HAS_GPU else 'CPU'} | {versions} | lerobot {lerobot or 'not installed'}",
     f"minutes: install {INSTALL_MIN}, script {RUN_MIN}, total {round((time.time() - T0) / 60, 1)}",
     "", "--- last 40 lines of output ---", *[l.rstrip() for l in lines[-40:]],
     "", "--- results/docs files created or changed ---"]
for rel in changed:
    text = (REPO / rel).read_text(errors="replace") if (REPO / rel).stat().st_size < 20_000 else "(larger than 20 KB: download it)"
    S += [f"### {rel}", "```", text.rstrip(), "```"]
S += ["", "--- outputs/ ---"]
S += [f"{p.relative_to(OUT)}  {p.stat().st_size / 1e6:.2f} MB" for p in sorted(OUT.rglob("*")) if p.is_file()]
S += ["===== END SUMMARY ====="]
(OUT / "summary.md").write_text("\\n".join(S) + "\\n")
print("\\n".join(S))'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {
    "cells": [cell("markdown", INTRO), cell("code", CONFIG), cell("code", SETUP), cell("code", INSTALL),
              cell("code", RUN_CELL), cell("code", SUMMARY)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
Path("notebooks/runner.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
print("wrote notebooks/runner.ipynb")
