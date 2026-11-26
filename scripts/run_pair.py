"""Two models at once on Kaggle's T4 x2: each scripts.train_eval run gets its own GPU
(MINIVLA_GPU -> torch.cuda.set_device), so its numbers are the same as if it ran alone.

  python -m scripts.run_pair configs/runs/A_s0.yaml configs/runs/C_s0.yaml

Both share one copy of the feature cache in the OS page cache. Exits 1 if either run fails.
"""
import sys

from minivla.parallel import run_parallel


def main(paths):
    codes = run_parallel([(p.split("/")[-1].removesuffix(".yaml"), gpu, ["scripts.train_eval", p])
                          for gpu, p in enumerate(paths)])
    print("exit codes:", codes, flush=True)
    sys.exit(1 if any(codes.values()) else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
