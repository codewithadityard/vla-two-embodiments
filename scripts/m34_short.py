"""M4 worker, started by scripts/m34_checks.py (one per GPU, both at once):
a short training run, then a smoke eval of the closed-loop pipeline (2 episodes) with that model.

  python -m scripts.m34_short configs/C_l1_lang.yaml C

Writes outputs/m34/short_<tag>.csv (training log), outputs/m34/smoke_<tag>.csv (episodes) and
outputs/m34/short_<tag>.json (numbers for the report). The smoke eval only tests that evaluation runs
end to end and measures seconds per episode; its successes are not used for anything.
"""
import json
import sys
import time
from pathlib import Path

import torch

from minivla.backbone import Backbone
from minivla.config import load_config, seed_everything
from minivla.eval_libero import Policy, evaluate
from minivla.train import prepare, setup_determinism, train, zero_baseline_l1

SHORT_STEPS = 2000
SMOKE_TASKS, SMOKE_EPISODES = [0, 1], 1


def main(cfg_path, tag):
    setup_determinism()
    cfg = load_config(cfg_path)
    cfg["train"]["val_every"] = 250
    seed_everything(0)
    out = Path("outputs/m34")
    d, train_rows, val_rows, stats = prepare(cfg)
    baseline = zero_baseline_l1(d, val_rows)
    expert, head, hist = train(cfg, d, train_rows, val_rows, SHORT_STEPS, out / f"short_{tag}.csv", seed=0)
    n_prefix, n_text, cache_info = d.n_prefix, d.n_text, d.describe()
    del d
    bb = Backbone(cfg["backbone"]["name"], cfg["backbone"]["revision"], cfg["backbone"]["num_layers"])
    t = time.time()
    rows = evaluate(cfg, Policy(cfg, expert, head, bb, n_text, stats), f"smoke_{tag}", out / f"smoke_{tag}.csv",
                    out / f"videos_{tag}", tasks=SMOKE_TASKS, episodes=SMOKE_EPISODES)
    res = {"tag": tag, "gpu": torch.cuda.get_device_name(), "history": hist, "baseline": baseline,
           "cache": cache_info, "prefix_len_online": bb.prefix_len, "prefix_len_cache": n_prefix,
           "smoke_rows": rows, "smoke_min": (time.time() - t) / 60}
    (out / f"short_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"done: val L1 {hist[-1][2]:.4f} (baseline {baseline:.4f}), {hist[-1][4]:.4f} s/step, "
          f"smoke {len(rows)} episodes in {res['smoke_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
