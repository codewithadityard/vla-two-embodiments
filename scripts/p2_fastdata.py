"""P2 frame-level reader for the Wall data (runs in the DINO-WM venv, cwd = the dino_wm repo).

  python p2_fastdata.py convert <wall_single dir>   stack every obses/episode_XXX.pth into obses.npy (float32,
                                                    the same values), so a sample reads only the frames it uses
  python p2_fastdata.py check <wall_single dir>     official reader vs frame-level reader on 64 random training
                                                    samples: every tensor must be exactly equal

scripts/p2_fastdata.patch makes the official loader use obses.npy when it exists (and the official path otherwise).
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

SHAPE = (51, 3, 224, 224)  # frames per episode file (P2-R0)


def write_chunk(args):
    tmp, data, ks = args
    mm = np.load(tmp, mmap_mode="r+")
    for k in ks:
        x = torch.load(data / "obses" / f"episode_{k:03d}.pth").numpy()
        assert x.shape == SHAPE and x.dtype == np.float32, (k, x.shape, x.dtype)
        mm[k] = x
    mm.flush()
    return len(ks)


def convert(data):
    t = time.time()
    n = len(torch.load(data / "states.pth"))
    tmp = data / "obses.npy.tmp"
    np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, *SHAPE)).flush()
    with Pool(4) as pool:  # Kaggle has 4 CPUs
        done = sum(pool.map(write_chunk, [(tmp, data, ks) for ks in np.array_split(np.arange(n), 16)]))
    tmp.rename(data / "obses.npy")  # only a complete file is ever named obses.npy
    print(f"converted {done}/{n} episodes in {(time.time() - t) / 60:.1f} min", flush=True)


def check(data):
    from datasets.img_transforms import default_transform
    from datasets.wall_dset import load_wall_slice_train_val
    # same dataset arguments as training (conf/env/wall.yaml + the checkpoint's num_hist 1, num_pred 1, frameskip 5)
    slices, traj = load_wall_slice_train_val(default_transform(224), n_rollout=None, data_path=str(data),
                                             normalize_action=True, split_ratio=0.9, num_hist=1, num_pred=1,
                                             frameskip=5)
    base, mm = traj["train"].dataset, traj["train"].dataset.obs_mm
    assert mm is not None, "obses.npy not found"
    rng = np.random.default_rng(0)
    equal, n = 0, 64
    for idx in rng.choice(len(slices["train"]), n, replace=False):
        base.obs_mm = mm
        fast = slices["train"][idx]
        base.obs_mm = None  # official path
        official = slices["train"][idx]
        pairs = [(fast[0]["visual"], official[0]["visual"]), (fast[0]["proprio"], official[0]["proprio"]),
                 (fast[1], official[1]), (fast[2], official[2])]
        equal += all(a.dtype == b.dtype and torch.equal(a, b) for a, b in pairs)
    print(f"exactly equal samples: {equal}/{n}", flush=True)


if __name__ == "__main__":
    {"convert": convert, "check": check}[sys.argv[1]](Path(sys.argv[2]))
