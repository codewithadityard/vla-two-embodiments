"""Closed-loop evaluation on LIBERO-Goal with LeRobot's env (pinned commit). Kaggle only.

Same semantics as lerobot-eval: LeRobot's LiberoEnv (300-step limit for libero_goal, 10 no-op steps
after reset, success = env check_success), observations through LeRobot's canonical_observation
(180-degree image flip, 8-D state). Episode e of every task uses init state e and seed eval_seed + e,
so every model sees exactly the same starts. The policy is our cached-feature pipeline run online:
same processor + 16-layer backbone, then the expert; it executes the whole 50-step chunk, then replans.
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from minivla import data
from minivla.stats import wilson


def to_hwc_uint8(x):
    x = np.asarray(x).squeeze()
    if x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return x if x.dtype == np.uint8 else np.clip(x * (255 if x.max() <= 1 else 1), 0, 255).astype(np.uint8)


class Policy:
    """Observation -> 50-step action chunk (unnormalized), using the online backbone + trained expert."""

    def __init__(self, cfg, expert, head, backbone, n_text, stats):
        self.cfg, self.expert, self.head, self.bb = cfg, expert.eval(), head, backbone
        self.n_text, self.stats = n_text, stats

    @torch.no_grad()
    def chunk(self, main, wrist, state, instruction, generator):
        pre, txt, mask = self.bb.encode([main], [wrist], [instruction])
        b, t, dim = txt.shape
        text = torch.zeros(b, self.n_text, dim, dtype=txt.dtype, device=txt.device)
        text_mask = torch.zeros(b, self.n_text, dtype=torch.bool, device=txt.device)
        if self.cfg["use_language"]:  # config B: text masked out, as in training
            text[:, :t], text_mask[:, :t] = txt, mask
        feats = torch.cat([pre, text], dim=1)
        fmask = torch.cat([torch.ones(b, pre.shape[1], dtype=torch.bool, device=pre.device), text_mask], dim=1)
        s = torch.from_numpy(data.normalize(np.asarray(state, np.float32)[None], self.stats["state"])).to(pre.device)
        with torch.autocast("cuda", dtype=torch.float16):
            a = self.head.predict(self.expert, feats, fmask, s, generator=generator)
        return data.unnormalize(a[0].float().cpu().numpy(), self.stats["action"])


def save_gif(frames, path, every=2, size=128):
    imgs = [Image.fromarray(f).resize((size, size)) for f in frames[::every]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def evaluate(cfg, policy, run_name, csv_path, video_dir, tasks=None, episodes=None):
    """Run every task x episodes 0..episodes_per_task-1, or only the given task / episode lists (smoke tests,
    top-ups). Episode e always uses init state e and seed eval_seed + e, so any subset gives the same episodes.
    Writes one CSV row per episode (flushed as it goes). Returns rows."""
    from libero.libero import benchmark
    from lerobot.sim.integrations.libero import LiberoEnv, canonical_observation

    ecfg = cfg["eval"]
    suite_name = cfg["env"]["suite"]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    video_dir.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "env_task", "instruction", "episode", "seed", "success", "steps", "seconds"])
        for task in (range(suite.n_tasks) if tasks is None else tasks):
            for ep in (range(ecfg["episodes_per_task"]) if episodes is None else episodes):
                seed = ecfg["eval_seed"] + ep
                t0 = time.time()
                env = LiberoEnv(task_suite=suite, task_id=task, task_suite_name=suite_name,
                                obs_type="pixels_agent_pos", init_states=True, episode_index=ep)
                obs = canonical_observation(env.reset(seed=seed)[0])
                instruction = env.task_description
                g = torch.Generator(device="cuda").manual_seed(seed * 100 + task)
                queue, frames, success, steps = [], [], False, 0
                for steps in range(1, env._max_episode_steps + 1):
                    main = to_hwc_uint8(obs["observation.images.image"])
                    frames.append(main)
                    if not queue:
                        chunk = policy.chunk(main, to_hwc_uint8(obs["observation.images.image2"]),
                                             obs["observation.state"], instruction, g)
                        queue = list(chunk[:ecfg["n_action_steps"]])
                    raw, _, terminated, truncated, info = env.step(queue.pop(0))
                    obs = canonical_observation(raw)
                    if info.get("is_success"):
                        success = True
                        break
                    if terminated or truncated:
                        break
                env.close()
                sec = time.time() - t0
                save_gif(frames, video_dir / f"task{task}_ep{ep}_{'success' if success else 'fail'}.gif")
                row = [run_name, task, instruction, ep, seed, int(success), steps, round(sec, 1)]
                w.writerow(row)
                f.flush()
                rows.append(row)
                print(f"task {task} ep {ep}: {'SUCCESS' if success else 'fail'} in {steps} steps, {sec:.0f} s", flush=True)
    return rows


def summary_lines(rows):
    """Markdown: per-task successes, overall rate with Wilson 95 % CI, seconds per episode."""
    k, n = sum(r[5] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    sec = np.array([r[7] for r in rows])
    succ = np.array([r[5] for r in rows], bool)
    L = ["| Env task | Instruction | Successes |", "|---|---|---|"]
    for task in sorted({r[1] for r in rows}):
        rr = [r for r in rows if r[1] == task]
        L.append(f"| {task} | {rr[0][2]} | {sum(r[5] for r in rr)}/{len(rr)} |")
    L += ["", f"**Overall: {k}/{n} = {100 * k / n:.1f} % (Wilson 95 % CI {100 * lo:.1f}–{100 * hi:.1f} %).**", "",
          f"Seconds per episode: mean {sec.mean():.1f} (successes {sec[succ].mean() if succ.any() else float('nan'):.1f}, "
          f"failures {sec[~succ].mean() if (~succ).any() else float('nan'):.1f}); total {sec.sum() / 60:.1f} min."]
    return L
