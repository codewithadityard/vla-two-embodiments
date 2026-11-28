"""Launch and read Kaggle runs with the kaggle CLI (token: ~/.kaggle/access_token; never printed or copied).

Each run tag gets its own private Kaggle notebook "<owner>/vla-<tag>", so two runs can go at once.

  python -m scripts.kaggle_run push TAG --module scripts.x [--args "a b c"] [--commit HASH] [--gpu] [--no-install] [--dataset owner/slug] [--kernel-source owner/notebook] --purpose "..."
  python -m scripts.kaggle_run status TAG
  python -m scripts.kaggle_run summary TAG      # downloads ONLY outputs/summary.md (small), prints it
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.build_runner_notebook import config_cell, make_notebook

OWNER = "adityard"
KAGGLE = str(Path(sys.executable).parent / "kaggle")
WORK = Path("/tmp/claude-1000/kaggle_runs")  # kernel folders and downloaded summaries (outside the repo)


def ref(tag):
    return f"{OWNER}/vla-{tag}"


def head_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def push(a):
    commit = a.commit or head_commit()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout
    if "origin/main" not in pushed:
        sys.exit(f"commit {commit[:12]} is not on origin/main yet: push first")
    folder = WORK / a.tag
    folder.mkdir(parents=True, exist_ok=True)
    run = {"commit": commit, "module": a.module, "args": shlex.split(a.args), "install": not a.no_install,
           "purpose": a.purpose}
    (folder / "runner.ipynb").write_text(json.dumps(make_notebook(config_cell(run)), indent=1))
    meta = {"id": ref(a.tag), "title": f"vla-{a.tag}", "code_file": "runner.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": a.gpu, "enable_internet": True,
            "dataset_sources": a.dataset, "competition_sources": [], "kernel_sources": a.kernel_source}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    cmd = [KAGGLE, "kernels", "push", "-p", str(folder)]
    if a.gpu:
        cmd += ["--accelerator", a.accelerator]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def status(a):
    subprocess.run([KAGGLE, "kernels", "status", ref(a.tag)], check=True)


def summary(a):
    out = WORK / a.tag / "out"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "kernels", "output", ref(a.tag), "-p", str(out), "-o", "-q",
                    "--file-pattern", r"(^|/)summary\.md$"], check=True)
    files = list(out.rglob("summary.md"))
    if not files:
        sys.exit("summary.md not found in the run's output (run failed early or still running?)")
    print(files[0].read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("push")
    pp.add_argument("tag")
    pp.add_argument("--module", required=True)
    pp.add_argument("--args", default="")  # one string, split like a shell
    pp.add_argument("--commit")
    pp.add_argument("--purpose", required=True)
    pp.add_argument("--gpu", action="store_true")
    pp.add_argument("--no-install", action="store_true")  # skip requirements-kaggle.txt (Kaggle preinstalls only)
    pp.add_argument("--accelerator", default="NvidiaTeslaT4")
    pp.add_argument("--dataset", action="append", default=[])
    pp.add_argument("--kernel-source", action="append", default=[])  # attach another notebook's output
    for name in ["status", "summary"]:
        sub.add_parser(name).add_argument("tag")
    a = p.parse_args()
    {"push": push, "status": status, "summary": summary}[a.cmd](a)
