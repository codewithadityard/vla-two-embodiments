"""Project 2 helpers (Kaggle): the official DINO-WM repo in its own Python 3.9 venv, plus OSF downloads.

Kaggle's Python is 3.13 with torch 2.11; DINO-WM pins Python 3.9.19 and torch 2.3.0, so it runs in a uv venv
(in /tmp, not saved as output). Packages come from the repo's environment.yaml pins.
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/gaoyuezhou/dino_wm.git"
COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"  # main on 2025-03-24 ("add checkpoints")
REPO = Path("/tmp/dino_wm")
VENV = Path("/tmp/p2venv")
PY = VENV / "bin" / "python"
UV_VERSION = "0.13.7"  # same as the feasibility notebook
PYTHON = "3.9.19"  # environment.yaml
TORCH = ["torch==2.3.0", "torchvision==0.18.0"]  # environment.yaml; cu121 wheels support the T4
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
OSF_NODE, OSF_VIEW_ONLY = "bmw48", "a56a296ce3b24cceaf408383a175ce28"  # DINO-WM README
P2_OUT = Path("/kaggle/working/outputs/p2")  # saved as the run's output, attached by later runs
DATA_DIR = Path("/tmp/p2_data")  # DATASET_DIR: wall_single.zip expands to > 20 GB (the output disk), so it is
                                 # extracted to /tmp in every run; only the zip is kept as output
CONSTRAINTS = Path("/tmp/p2_constraints.txt")  # environment.yaml pip pins, applied to every install
MUJOCO_URL = "https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
MUJOCO_DIR = Path("/root/.mujoco/mujoco210")  # where mujoco-py looks (README: ~/.mujoco)


def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def uv_pip(*args):
    # --no-cache: no second copy of the ~4.5 GB torch/CUDA wheels (P2-R0 v1 ran out of disk)
    extra = ["-c", str(CONSTRAINTS)] if CONSTRAINTS.exists() else []  # dependencies get the official versions too
    sh([sys.executable, "-m", "uv", "pip", "install", "--no-cache", "--python", str(PY), *extra, *args])


def clone():
    if not REPO.exists():
        sh(["git", "clone", "-q", REPO_URL, str(REPO)])
    sh(["git", "-C", str(REPO), "checkout", "-q", COMMIT])


def make_venv():
    sh([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"])
    sh([sys.executable, "-m", "uv", "venv", "--python", PYTHON, str(VENV)])
    uv_pip("--index-url", TORCH_INDEX, *TORCH)


def write_constraints():
    """Constraints file from environment.yaml's pip section (+ numpy, setuptools from conda). A constraint only
    applies if the package gets installed, so this pins e.g. sentry-sdk for wandb 0.13.1 without installing it."""
    import yaml
    deps = yaml.safe_load((REPO / "environment.yaml").read_text())["dependencies"]
    lines = [p for d in deps if isinstance(d, dict) for p in d.get("pip", []) if "==" in p]
    lines += [f"{d.split('=')[0]}=={d.split('=')[1]}" for d in deps
              if isinstance(d, str) and d.split("=")[0] in ("numpy", "setuptools")]
    CONSTRAINTS.write_text("\n".join(lines) + "\n")
    return lines


def extract_data(zip_path):
    import zipfile
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        zf.extractall(DATA_DIR)
    return total


def env_pins():
    """{normalized package name: 'name==version'} from the repo's environment.yaml: pip pins, plus conda pins
    (name=version=build) for packages that pip also provides (e.g. numpy, setuptools)."""
    import yaml
    deps = yaml.safe_load((REPO / "environment.yaml").read_text())["dependencies"]
    pins = {}
    for d in deps:
        if isinstance(d, dict):
            for p in d.get("pip", []):
                if "==" in p:
                    pins[norm(p.split("==")[0])] = p
        elif isinstance(d, str) and d.count("=") >= 1 and not d.startswith(("python=", "_", "lib")):
            name, version = d.split("=")[:2]
            pins.setdefault(norm(name), f"{name}=={version}")
    return pins


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def install_requirements(path):
    uv_pip("-r", str(path))


def osf_files():
    """Every file in the OSF project: [(path, size, download_url)] (folders walked, pages followed)."""
    def walk(url):
        while url:
            page = json.load(urllib.request.urlopen(url, timeout=60))
            for f in page["data"]:
                if f["attributes"]["kind"] == "folder":
                    href = f["relationships"]["files"]["links"]["related"]["href"]
                    if "view_only" not in href:
                        href += ("&" if "?" in href else "?") + f"view_only={OSF_VIEW_ONLY}"
                    yield from walk(href)
                else:
                    yield f["attributes"]["materialized_path"], f["attributes"]["size"], f["links"]["download"]
            url = page["links"].get("next")
    return list(walk(f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/?view_only={OSF_VIEW_ONLY}"))


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if "view_only" not in url:
        url += ("&" if "?" in url else "?") + f"view_only={OSF_VIEW_ONLY}"
    sh(["curl", "-sSL", "--fail", "-o", str(dest), url])


def disk(label):
    """Print free space where the venv (/tmp) and outputs (/kaggle/working) live."""
    print(f"[disk] {label}:", flush=True)
    subprocess.run(["df", "-h", "/tmp", "/kaggle/working"])


def venv_env():
    """Environment for every command run in the venv. MUJOCO_PY_FORCE_CPU: mujoco-py builds its CPU extension
    (Wall never renders with MuJoCo; mujoco-py is only imported because env/__init__ imports PointMaze)."""
    # WANDB_DISABLE_SERVICE: plan.py hard-codes wandb_logging=True; with the service off, wandb.init (disabled mode)
    # starts no background process (P2-R2 v2: that service failed to start and planning crashed)
    return {"PATH": "/usr/bin:/bin", "HOME": "/root", "WANDB_MODE": "disabled", "WANDB_DISABLE_SERVICE": "true",
            "DATASET_DIR": str(DATA_DIR),
            "LD_LIBRARY_PATH": f"{MUJOCO_DIR}/bin", "MUJOCO_PY_FORCE_CPU": "1"}


def install_mujoco_py(pins):
    """README: MuJoCo 2.1 binaries in ~/.mujoco + mujoco-py (pinned). mujoco-py compiles itself on first import,
    so it needs gcc, the GL/OSMesa headers, patchelf and Cython < 3."""
    sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
       "libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3 patchelf > /dev/null")
    if not MUJOCO_DIR.exists():
        MUJOCO_DIR.parent.mkdir(parents=True, exist_ok=True)
        sh(f"curl -sSL --fail {MUJOCO_URL} | tar -xz -C {MUJOCO_DIR.parent}")
    uv_pip(pins.get("mujoco-py", "mujoco-py==2.1.2.14"), pins.get("cython", "cython<3"))
