"""Two jobs at once on Kaggle's T4 x2: each gets its own GPU (MINIVLA_GPU -> torch.cuda.set_device),
so its numbers are the same as if it ran alone.

  python -m scripts.run_pair configs/runs/B_s0.yaml configs/runs/A_s1.yaml                 # train + eval
  python -m scripts.run_pair --module scripts.eval_topup configs/runs/A_s0.yaml configs/runs/C_s0.yaml

For training, the feature cache is read once here, before the jobs start, so both share one copy in the OS
page cache (two jobs reading it cold at the same time took 16.7 min in vla-pair-s0; alone it takes ~1 min).
Exits 1 if either job fails.
"""
import argparse
import sys
import time
from pathlib import Path

from minivla.parallel import run_parallel
from minivla.train import find_input, warm_page_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--module", default="scripts.train_eval")
    a = p.parse_args()
    if a.module == "scripts.train_eval":
        cache_dir = Path(find_input("prefix.npy")).parent
        t = time.time()
        warm_page_cache([cache_dir / "prefix.npy", cache_dir / "text.npy"])
        print(f"cache warmed once for both jobs in {(time.time() - t) / 60:.1f} min", flush=True)
    codes = run_parallel([(path.split("/")[-1].removesuffix(".yaml"), gpu, [a.module, path])
                          for gpu, path in enumerate(a.configs)])
    print("exit codes:", codes, flush=True)
    sys.exit(1 if any(codes.values()) else 0)


if __name__ == "__main__":
    main()
