"""Train the action expert on cached backbone features (Kaggle, GPU). Used by scripts/train_eval.py and m34_checks.py."""
import csv
import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from minivla import cache, data
from minivla.expert import ActionExpert
from minivla.heads import FlowMatchingHead, RegressionHead, masked_mean


def setup_determinism():
    """Same seed -> same result on the same GPU. Call before any CUDA work."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (in the log) instead of crashing a long run
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)  # the math attention kernel is deterministic
    torch.backends.cuda.enable_mem_efficient_sdp(False)


def find_input(pattern):
    """First file under /kaggle/input matching a glob pattern (attached Kaggle datasets)."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no /kaggle/input/**/{pattern}: is the dataset attached?")
    return hits[0]


def check_cache_meta(meta, cfg):
    """The cache must come from the pinned dataset, backbone and layer count."""
    want = {"dataset_revision": cfg["data"]["revision"], "backbone_revision": cfg["backbone"]["revision"],
            "num_layers": cfg["backbone"]["num_layers"]}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"cache does not match config: {bad}")


class CachedData:
    """Cached features + normalized state/action chunks, batched on the GPU."""

    def __init__(self, cache_dir, frames, stats, use_language, chunk, device):
        self.meta, c = cache.open_cache(cache_dir)
        self.use_language, self.device = use_language, device
        need = c["prefix"].nbytes + (c["text"].nbytes if use_language else 0)
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        self.in_ram = ram > 1.5 * need
        load = np.ascontiguousarray if self.in_ram else (lambda a: a)  # read into RAM if it fits
        t = time.time()
        self.prefix = load(c["prefix"])
        self.text = load(c["text"]) if use_language else None
        self.text_mask = np.asarray(c["text_mask"])
        self.load_min = (time.time() - t) / 60
        self.ram_gb, self.need_gb = ram / 1e9, need / 1e9
        self.n_prefix, self.n_text, self.dim = self.meta["n_prefix"], self.meta["n_text"], self.meta["dim"]
        self.state = data.normalize(frames["state"], stats["state"]).astype(np.float32)
        self.actions = data.normalize(frames["action"], stats["action"]).astype(np.float32)
        self.chunk_idx, self.pad = data.chunk_indices(frames, chunk)

    def batch(self, rows):
        rows = np.sort(np.asarray(rows))
        b = len(rows)
        prefix = self.prefix[rows]
        if self.use_language:
            text, text_mask = self.text[rows], self.text_mask[rows]
        else:  # config B: instruction tokens masked out (equals an empty instruction, checked in M2)
            text = np.zeros((b, self.n_text, self.dim), np.float16)
            text_mask = np.zeros((b, self.n_text), bool)
        to = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device, non_blocking=True)
        feats = to(np.concatenate([prefix, text], axis=1))
        mask = to(np.concatenate([np.ones((b, self.n_prefix), bool), text_mask], axis=1))
        return feats, mask, to(self.state[rows]), to(self.actions[self.chunk_idx[rows]]), to(self.pad[rows])


def make_expert_and_head(cfg, n_feat_tokens, feat_dim):
    m, flow = cfg["model"], cfg["head"] == "flow"
    expert = ActionExpert(feat_dim, n_feat_tokens, chunk=m["chunk"], d=m["d_model"], layers=m["layers"],
                          heads=m["heads"], ffn=m["ffn"], dropout=m["dropout"], flow=flow)
    head = FlowMatchingHead(cfg["flow"]["steps"], cfg["flow"]["beta_a"], cfg["flow"]["beta_b"]) if flow else RegressionHead()
    return expert, head


def lr_factor(step, warmup, total, peak, final):
    """Linear warm-up, then cosine decay from peak to final (as LeRobot's CosineDecayWithWarmup)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return (final + (peak - final) * 0.5 * (1 + math.cos(math.pi * progress))) / peak


@torch.no_grad()
def val_l1(expert, head, d, rows, noise_seed, batch=256):
    """Masked L1 (normalized units) of predicted chunks on fixed rows; flow noise fixed by noise_seed."""
    expert.eval()
    g = torch.Generator(device=d.device).manual_seed(noise_seed)
    total, n = 0.0, 0
    for s in range(0, len(rows), batch):
        feats, mask, state, actions, pad = d.batch(rows[s:s + batch])
        with torch.autocast("cuda", dtype=torch.float16):
            pred = head.predict(expert, feats, mask, state, generator=g)
        keep = (~pad).sum().item() * actions.shape[-1]
        total += masked_mean((pred - actions).abs(), pad).item() * keep
        n += keep
    expert.train()
    return total / n


@torch.no_grad()
def zero_baseline_l1(d, rows, batch=256):
    """L1 of always predicting the mean chunk (0 in normalized units)."""
    total, n = 0.0, 0
    for s in range(0, len(rows), batch):
        _, _, _, actions, pad = d.batch(rows[s:s + batch])
        keep = (~pad).sum().item() * actions.shape[-1]
        total += masked_mean(actions.abs(), pad).item() * keep
        n += keep
    return total / n


def train(cfg, d, train_rows, val_rows, steps, log_path, seed, log_every=250):
    """Train for exactly `steps` steps. Returns (expert, head, history rows)."""
    tr = cfg["train"]
    torch.manual_seed(seed)
    expert, head = make_expert_and_head(cfg, d.n_prefix + d.n_text, d.dim)
    expert = expert.to(d.device).train()
    opt = torch.optim.AdamW(expert.parameters(), lr=tr["lr"], betas=tuple(tr["betas"]), eps=1e-8,
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, tr["warmup"], steps, tr["lr"], tr["final_lr"]))
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    history, running, t0 = [], [], time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_l1", "lr", "sec_per_step"])
        for step in range(1, steps + 1):
            feats, mask, state, actions, pad = d.batch(rng.choice(train_rows, tr["batch"]))
            with torch.autocast("cuda", dtype=torch.float16):
                loss = head.loss(expert, feats, mask, state, actions, pad)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(expert.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.item())
            if step % log_every == 0 or step == steps:
                v = val_l1(expert, head, d, val_rows, cfg["split"]["split_seed"]) \
                    if step % tr["val_every"] == 0 or step == steps else ""
                row = [step, float(np.mean(running)), v, sched.get_last_lr()[0], (time.time() - t0) / step]
                w.writerow(row)
                f.flush()
                history.append(row)
                print(f"step {step}/{steps} loss {row[1]:.4f} val_l1 {v} lr {row[3]:.2e} {row[4]:.3f}s/step", flush=True)
                running = []
    return expert, head, history


def prepare(cfg, device="cuda"):
    """Load inputs on Kaggle: cache + lowdim table; split; train-only stats. Returns (data, train_rows, val_rows, stats)."""
    cache_dir = Path(find_input("prefix.npy")).parent
    frames = data.load_frames(find_input("libero_goal_lowdim.parquet"))
    ep_task = data.episode_tasks(frames)
    train_eps, val_eps = data.split_episodes(ep_task, cfg["split"]["split_seed"], cfg["split"]["val_fraction"])
    stats = data.train_stats(frames, train_eps)
    d = CachedData(cache_dir, frames, stats, cfg["use_language"], cfg["model"]["chunk"], device)
    check_cache_meta(d.meta, cfg)
    train_rows = np.flatnonzero(np.isin(frames["episode_index"], train_eps))
    val_all = np.flatnonzero(np.isin(frames["episode_index"], val_eps))
    # same fixed val subset for every run (depends only on split_seed)
    val_rows = np.sort(np.random.default_rng(cfg["split"]["split_seed"]).choice(
        val_all, min(cfg["train"]["val_samples"], len(val_all)), replace=False))
    return d, train_rows, val_rows, stats
