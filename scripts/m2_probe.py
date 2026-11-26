"""M2 probe (Kaggle, GPU T4): every check before building the feature cache -> results/m2_probe.md.

Run on Kaggle through notebooks/runner.ipynb with module "scripts.m2_probe".
Exits with code 1 if any check fails (the report is still written).
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image

from minivla import cache, data
from minivla.backbone import Backbone, load_vlm
from minivla.config import load_config, seed_everything

CONFIG = Path("configs/base.yaml")
REPORT = Path("results/m2_probe.md")
OUT = Path("outputs/m2_probe")
PROBE_SAMPLE = "data/m2_probe_sample.parquet"
PLANNED_SAMPLES_PER_MODEL = 20_000 * 64  # plan's step cap x batch (docs/IMPLEMENTATION_PLAN.md, M3)


# ---------- small helpers ----------

def fmt(x, d=4):
    return f"{x:.{d}g}" if isinstance(x, float) else str(x)


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]


def decode(cell):
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def as_hwc_uint8(x):
    """Env image -> HxWx3 uint8 (drops batch dims, moves channels last, rescales floats)."""
    x = np.asarray(x).squeeze()
    if x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    if x.dtype != np.uint8:
        x = np.clip(x * (255 if x.max() <= 1 else 1), 0, 255).astype(np.uint8)
    return x


def gray_corr(a, b):
    return float(np.corrcoef(a.astype(np.float64).mean(-1).ravel(), b.astype(np.float64).mean(-1).ravel())[0, 1])


def compare(a, b, mask=None):
    """Max abs difference and min cosine similarity over tokens (mask: [B, T] True = compare)."""
    a, b = a.float(), b.float()
    if mask is not None:
        a, b = a[mask], b[mask]
    max_abs = (a - b).abs().max().item()
    min_cos = nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).min().item()
    return max_abs, min_cos


def timed(fn, steps=50, warmup=5):
    """Mean seconds per call over `steps` calls after `warmup` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / steps


class StandInExpert(nn.Module):
    """Timing stand-in with the planned size (6 layers, width 512, chunk 50 x 7). The real expert is M3."""

    def __init__(self, dim_in, d=512, layers=6, heads=8, chunk=50, act_dim=7):
        super().__init__()
        self.mem_in = nn.Linear(dim_in, d)
        self.queries = nn.Parameter(torch.zeros(1, chunk, d))
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d, act_dim)

    def forward(self, memory):
        q = self.queries.expand(len(memory), -1, -1)
        return self.out(self.decoder(q, self.mem_in(memory)))


# ---------- steps ----------

def fetch_data(dcfg):
    """Meta, non-image columns, and frame-0 images of one episode per task (first episode of each task)."""
    data.fetch_meta(dcfg["repo_id"], dcfg["revision"], dcfg["meta_dir"])
    data.fetch_lowdim(dcfg["repo_id"], dcfg["revision"], dcfg["lowdim_path"])
    frames = data.load_frames(dcfg["lowdim_path"])
    ep_task = data.episode_tasks(frames)
    episodes = sorted(min(e for e, t in ep_task.items() if t == task) for task in set(ep_task.values()))
    data.fetch_sample(dcfg["repo_id"], dcfg["revision"], dcfg["meta_dir"], episodes, PROBE_SAMPLE)
    df = pq.read_table(PROBE_SAMPLE).to_pandas()
    df = df[df["frame_index"] == 0]
    images = {int(r["task_index"]): (decode(r["observation.images.image"]), decode(r["observation.images.wrist_image"]))
              for _, r in df.iterrows()}
    return frames, images


def env_observations(suite_name, task_ids, seed):
    """Reset LeRobot's LIBERO env (init state 0) for each task; return its instruction and canonical obs."""
    from libero.libero import benchmark
    from lerobot.sim.integrations.libero import LiberoEnv, canonical_observation

    suite = benchmark.get_benchmark_dict()[suite_name]()
    out = {}
    for t in task_ids:
        env = LiberoEnv(task_suite=suite, task_id=t, task_suite_name=suite_name,
                        obs_type="pixels_agent_pos", init_states=True, episode_index=0)
        obs, _ = env.reset(seed=seed)
        out[t] = (env.task_description, canonical_observation(obs))
        env.close()
    return out


def find_key(obs, test):
    keys = [k for k in obs if test(k)]
    if len(keys) != 1:
        raise KeyError(f"expected one matching key, got {keys} from {list(obs)}")
    return keys[0]


def main():
    cfg = load_config(CONFIG)
    seed_everything(cfg["seed"])
    if not torch.cuda.is_available():
        sys.exit("m2_probe needs a GPU session (T4).")
    dcfg, bcfg = cfg["data"], cfg["backbone"]
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1. data
    frames, data_imgs = fetch_data(dcfg)
    tasks = data.load_tasks(dcfg["meta_dir"])
    task_ids = sorted(tasks)
    n_frames = len(frames["index"])

    # 2. env vs data
    env = env_observations(cfg["env"]["suite"], task_ids, cfg["seed"])
    obs_keys = list(env[task_ids[0]][1])
    env_rows = []
    all_text, all_orient, all_state = True, True, True
    first = frames["frame_index"] == 0
    for t in task_ids:
        text, obs = env[t]
        e_main = as_hwc_uint8(obs[find_key(obs, lambda k: k.endswith("image"))])
        e_wrist = as_hwc_uint8(obs[find_key(obs, lambda k: k.endswith("image2"))])
        e_state = np.asarray(obs[find_key(obs, lambda k: "state" in k or "agent_pos" in k)]).squeeze()
        d_main, d_wrist = data_imgs[t]
        Image.fromarray(np.concatenate([e_main, d_main, e_wrist, d_wrist], axis=1)).save(OUT / f"task{t}_env_vs_data.png")
        c = [gray_corr(e_main, d_main), gray_corr(e_main[::-1, ::-1], d_main),
             gray_corr(e_wrist, d_wrist), gray_corr(e_wrist[::-1, ::-1], d_wrist)]
        s0 = frames["state"][first & (frames["task_index"] == t)]
        ratio = np.abs(e_state - s0.mean(0)) / np.maximum(3 * s0.std(0), 0.01)
        text_ok, orient_ok, state_ok = text == tasks[t], c[0] > c[1] and c[2] > c[3], ratio.max() <= 1
        all_text &= text_ok
        all_orient &= orient_ok
        all_state &= state_ok
        env_rows.append([t, "yes" if text_ok else f"NO: env says '{text}'", c[0], c[1], c[2], c[3], float(ratio.max())])
    check("env instruction == dataset instruction (all tasks)", all_text)
    check("env images (after LeRobot's flip) match dataset orientation (all tasks, both cameras)", all_orient)
    check("env reset state within dataset first-frame range (all tasks, all dims)", all_state,
          "per dim: |env - mean| <= max(3 std, 0.01) over the task's first frames")

    # 3. token layout
    bb = Backbone(bcfg["name"], bcfg["revision"], bcfg["num_layers"])
    mains = [data_imgs[t][0] for t in task_ids]
    wrists = [data_imgs[t][1] for t in task_ids]
    instr = [tasks[t] for t in task_ids]
    batch = bb.tokenize(mains, wrists, instr)
    empty = bb.tokenize(mains, wrists, [""] * len(task_ids))
    P = bb.prefix_len
    ids = batch["input_ids"]
    n_image_tok = (ids == bb.image_token_id).sum(1)
    text_len = batch["attention_mask"][:, P:].sum(1)
    n_text = int(text_len.max())
    T = P + n_text
    check("128 image tokens per frame (2 x 64)", (n_image_tok == 128).all().item(), f"{n_image_tok.tolist()}")
    check("prompt tokens before the instruction are identical to the empty-instruction prompt",
          empty["input_ids"].shape[1] == P and torch.equal(ids[:, :P], empty["input_ids"]), f"prefix length {P}")

    # 4. layer cut: our 16-layer model == final_norm(full model hidden_states[16])
    h_cut = bb.forward(batch)
    full = load_vlm(bcfg["name"], bcfg["revision"], "cuda")
    n_full = len(full.text_model.layers)
    with torch.no_grad():
        b = {k: v.to("cuda") for k, v in batch.items()}
        b["pixel_values"] = b["pixel_values"].half()
        hs = full(**b, output_hidden_states=True).hidden_states
        h_ref = full.text_model.norm(hs[bcfg["num_layers"]])
    valid = batch["attention_mask"].bool().to("cuda")
    cut_abs, cut_cos = compare(h_cut, h_ref, valid)
    check("full model has 32 LM layers; we keep the first 16", n_full == 32 and len(bb.vlm.text_model.layers) == 16,
          f"full {n_full}, kept {len(bb.vlm.text_model.layers)}")
    check("16-layer output == final_norm(full hidden_states[16]) (max abs <= 1e-3)", cut_abs <= 1e-3,
          f"max abs {cut_abs:.2e}, min cos {cut_cos:.6f}")
    del full, hs
    torch.cuda.empty_cache()

    # 5. one cache for A, B and C: prefix tokens don't depend on the instruction
    pre_real, _, _ = bb.encode(mains, wrists, instr)
    pre_empty, _, _ = bb.encode(mains, wrists, [""] * len(task_ids))
    eq_abs, eq_cos = compare(pre_real, pre_empty)
    check("prefix features with real == with empty instruction (max abs <= 1e-2, cos >= 0.9999)",
          eq_abs <= 1e-2 and eq_cos >= 0.9999, f"max abs {eq_abs:.2e}, min cos {eq_cos:.6f}")

    # 6. storage
    dim = h_cut.shape[-1]
    gb_all = cache.size_gb(n_frames, T, dim)
    gb_half = cache.size_gb((n_frames + 1) // 2, T, dim)

    # 7. timing (T4)
    def tile(x, n):
        return [x[i % len(x)] for i in range(n)]

    b16 = bb.tokenize(tile(mains, 16), tile(wrists, 16), tile(instr, 16))
    b32 = bb.tokenize(tile(mains, 32), tile(wrists, 32), tile(instr, 32))
    t_tok32 = timed(lambda: bb.tokenize(tile(mains, 32), tile(wrists, 32), tile(instr, 32)), steps=10, warmup=2)
    t_fwd32 = timed(lambda: bb.forward(b32), steps=20)
    expert = StandInExpert(dim).cuda()
    opt = torch.optim.AdamW(expert.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def expert_step(memory):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = expert(memory).float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    t_online16 = timed(lambda: expert_step(bb.forward(b16)))
    mem16 = torch.randn(16, T, dim, device="cuda", dtype=torch.float16)
    mem64 = torch.randn(64, T, dim, device="cuda", dtype=torch.float16)
    t_cached16 = timed(lambda: expert_step(mem16))
    t_cached64 = timed(lambda: expert_step(mem64))
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    # 8. decision (planning arithmetic from the measured timings above)
    build_h = lambda n: n / 32 * (t_fwd32 + t_tok32) / 3600  # tokenizing in one process: upper bound
    online_h = 6 * PLANNED_SAMPLES_PER_MODEL * t_online16 / 16 / 3600
    cached_train_h = 6 * PLANNED_SAMPLES_PER_MODEL * t_cached64 / 64 / 3600
    max_gb = cfg["cache"]["max_gb"]
    if gb_all <= max_gb and build_h(n_frames) + cached_train_h < online_h:
        decision = f"CACHE all {n_frames} frames ({gb_all:.1f} GB)"
    elif gb_half <= max_gb and build_h(n_frames / 2) + cached_train_h < online_h:
        decision = f"CACHE every 2nd frame ({gb_half:.1f} GB): pre-agreed fallback, log in docs/deviations.md"
    else:
        decision = "ONLINE features (cache too large or not faster)"

    # ---------- report ----------
    gpu = torch.cuda.get_device_name(0)
    L = ["# M2 probe: backbone features, env vs data, cache decision", "",
         "Generated on Kaggle by `python -m scripts.m2_probe` (via `notebooks/runner.ipynb`). "
         "Every number is measured in this run.", "",
         f"- GPU: {gpu}; torch {torch.__version__}; backbone `{bcfg['name']}` @ `{bcfg['revision'][:12]}`, "
         f"first {bcfg['num_layers']} LM layers; seed {cfg['seed']}",
         f"- Env observation keys (after `canonical_observation`): {obs_keys}", "", "## Checks", ""]
    L += table(["Check", "Result", "Detail"], [[c, "PASS" if ok else "**FAIL**", d] for c, ok, d in checks])
    L += ["", "## Env vs data (init state 0, frame 0 of the first episode of each task)", "",
          "Correlations are between grayscale images; orientation passes if the as-is correlation beats the "
          "180°-flipped one. State ratio = max over dims of |env - mean| / max(3 std, 0.01); must be <= 1. "
          f"Side-by-side images: `{OUT}/task*_env_vs_data.png` (env main | data main | env wrist | data wrist).", ""]
    L += table(["Task", "Instruction match", "Main corr", "Main corr (flipped)", "Wrist corr", "Wrist corr (flipped)",
                "State ratio"], env_rows)
    L += ["", "## Token layout", ""]
    L += table(["Item", "Value"], [
        ["Prefix tokens P (BOS, image wrappers, 2 x 64 image tokens)", P],
        ["Instruction tokens, min / max over the 10 tasks", f"{int(text_len.min())} / {n_text}"],
        ["Tokens per frame T = P + max instruction", T], ["Feature dim", dim]])
    L += ["", "## Storage (fp16, all T tokens)", ""]
    L += table(["Frames cached", "GB", f"Within {max_gb} GB?"], [
        [f"all {n_frames}", gb_all, "yes" if gb_all <= max_gb else "no"],
        [f"every 2nd ({(n_frames + 1) // 2})", gb_half, "yes" if gb_half <= max_gb else "no"]])
    L += ["", f"## Timing on {gpu} (mean of 50 steps after 5 warm-up unless noted)", ""]
    L += table(["Measurement", "Seconds"], [
        ["Tokenize/preprocess batch 32 (CPU, one process; 10 calls)", t_tok32],
        ["Backbone forward batch 32 (16 layers, no grad; 20 calls)", t_fwd32],
        ["Online training step, batch 16 (backbone + stand-in expert fwd/bwd)", t_online16],
        ["Cached training step, batch 16 (stand-in expert only, memory on GPU)", t_cached16],
        ["Cached training step, batch 64", t_cached64]])
    L += [f"", f"Peak GPU memory allocated: {peak_gb:.2f} GB. The stand-in expert has the planned size "
          "(6 layers, width 512) but is not the M3 expert. Cached timings exclude reading features from disk.", ""]
    L += ["## Decision", "",
          f"Planning arithmetic from the timings above, for 6 models x {PLANNED_SAMPLES_PER_MODEL:,} training samples "
          "(plan's cap: 20,000 steps x batch 64):", ""]
    L += table(["Option", "GPU hours"], [
        ["Online: 6 models", online_h],
        ["Cache all frames: build + 6 models", build_h(n_frames) + cached_train_h],
        ["Cache every 2nd frame: build + 6 models", build_h(n_frames / 2) + cached_train_h]])
    L += ["", f"**Rule result: {decision}.** Needs your approval before R2.", ""]

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    failed = [c for c, ok, _ in checks if not ok]
    print("\n".join(L))
    print(f"Checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
