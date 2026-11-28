"""P2-R4 (runs in the DINO-WM venv, cwd = the dino_wm repo): open-loop prediction error vs. horizon of the official
Wall checkpoint on the official held-out trajectories. No planning, no decoder.

  python p2_pred_error.py <ckpt_base_path> <out.json>

For each validation trajectory (the official split: datasets/traj_dset.py split_traj_datasets, seed 42, 10 %),
frame 0 is encoded and the model is rolled forward with the real actions (wm.rollout, as the planner uses it);
prediction h is compared with the encoded real frame h (frame index 5h). Error = mean squared error over the visual
patch features (as z_visual_loss in models/visual_world_model.py) and over the proprio embedding (z_proprio_loss).
Baseline "nothing moves": the encoded frame 0 compared with the encoded frame h.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())  # the official repo (cwd), so its modules import as plan.py's do

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from plan import load_model  # noqa: E402  (official)

H_MAX = 9  # states/actions have 50 steps (P2-R0); at frameskip 5 the official data gives frames 0, 5, ..., 45
BATCH = 16


def main(ckpt_base, out):
    t0 = time.time()
    model_path = Path(ckpt_base) / "outputs" / "wall_single"
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    fs = cfg.frameskip
    # the same call as plan.py planning_main, so the same split and normalisation as training
    _, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=fs)
    val = dset["valid"]
    wm = load_model(model_path / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, torch.device("cuda"))
    wm.eval()
    assert cfg.num_hist == 1 and wm.decoder is None, (cfg.num_hist, type(wm.decoder))
    errs = {k: [] for k in ("visual", "proprio", "visual_copy", "proprio_copy")}
    with torch.no_grad():
        for b0 in range(0, len(val), BATCH):
            items = [val[i] for i in range(b0, min(b0 + BATCH, len(val)))]
            act = torch.stack([it[1] for it in items])  # (b, 50, 2), normalised as in training
            assert act.shape[1] == fs * (H_MAX + 1), act.shape
            frames = list(range(0, act.shape[1], fs))  # 0, 5, ..., 45
            obs = {k: torch.stack([it[0][k][frames] for it in items]).cuda() for k in ("visual", "proprio")}
            act = rearrange(act, "b (n f) d -> b n (f d)", f=fs).cuda()  # concatenated as in TrajSlicerDataset
            z_pred, _ = wm.rollout({k: v[:, :1] for k, v in obs.items()}, act[:, :H_MAX])  # frames 0..H_MAX
            z_true = wm.encode_obs(obs)
            for k in ("visual", "proprio"):
                pred, true = z_pred[k][:, 1:H_MAX + 1], z_true[k][:, 1:H_MAX + 1]
                copy = z_true[k][:, :1].expand_as(true)
                dims = tuple(range(2, true.dim()))
                errs[k].append(((pred - true) ** 2).mean(dim=dims).cpu())
                errs[k + "_copy"].append(((copy - true) ** 2).mean(dim=dims).cpu())
            print(f"{b0 + len(items)}/{len(val)} trajectories, {(time.time() - t0) / 60:.1f} min", flush=True)
    res = {"n_traj": len(val), "frameskip": fs, "minutes": (time.time() - t0) / 60, "val_indices_first5": [int(i) for i in list(getattr(val, "indices", []))[:5]]}
    for k, v in errs.items():
        e = torch.cat(v).double().numpy()  # (n_traj, H_MAX)
        res[k] = {"mean": e.mean(0).tolist(), "ci95": (1.96 * e.std(0, ddof=1) / np.sqrt(len(e))).tolist(),
                  "finite": bool(np.isfinite(e).all())}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
