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


def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def uv_pip(*args):
    sh([sys.executable, "-m", "uv", "pip", "install", "--python", str(PY), *args])


def clone():
    if not REPO.exists():
        sh(["git", "clone", "-q", REPO_URL, str(REPO)])
    sh(["git", "-C", str(REPO), "checkout", "-q", COMMIT])


def make_venv():
    sh([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"])
    sh([sys.executable, "-m", "uv", "venv", "--python", PYTHON, str(VENV)])
    uv_pip("--index-url", TORCH_INDEX, *TORCH)


def env_pins():
    """{normalized package name: 'name==version'} from the repo's environment.yaml (pip section + numpy)."""
    import yaml
    deps = yaml.safe_load((REPO / "environment.yaml").read_text())["dependencies"]
    pins = {}
    for d in deps:
        if isinstance(d, dict):
            for p in d.get("pip", []):
                if "==" in p:
                    pins[norm(p.split("==")[0])] = p
        elif isinstance(d, str) and d.startswith("numpy="):
            pins["numpy"] = "numpy==" + d.split("=")[1]
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
    sep = "&" if "?" in url else "?"
    sh(["curl", "-sSL", "--fail", "-o", str(dest), f"{url}{sep}view_only={OSF_VIEW_ONLY}"])
