"""M2 R2 (Kaggle, GPU T4): build the per-token feature cache for every frame.

Writes the cache to outputs/cache/ (saved as the Kaggle version's output) and results/m2_cache.md.
Preprocessing is HF's own processor, run in parallel DataLoader workers (approved option (a)).
Run through notebooks/runner.ipynb with module "scripts.build_cache". Exits 1 if any check fails.
"""
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from minivla import cache, data
from minivla.backbone import Backbone, make_processor, tokenize
from minivla.config import load_config, seed_everything

CONFIG = Path("configs/base.yaml")
REPORT = Path("results/m2_cache.md")
LOCAL = "data/libero_goal_image"  # full dataset, inside the /tmp clone on Kaggle (not saved)
IMG_COLS = ["observation.images.image", "observation.images.wrist_image"]
BATCH = 32
N_CHECK = 10  # frames for the cached == online test


def decode(cell):
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


class FrameBatches(IterableDataset):
    """Yields (global frame indices, processor batch). Each worker reads its own share of the parquet files."""

    def __init__(self, files, tasks, name, revision):
        self.files, self.tasks, self.name, self.revision = files, tasks, name, revision

    def __iter__(self):
        w = get_worker_info()
        files = self.files if w is None else self.files[w.id::w.num_workers]
        processor = make_processor(self.name, self.revision)
        for f in files:
            t = pq.read_table(f, columns=["index", "task_index", *IMG_COLS])
            idx, task = t["index"].to_numpy(), t["task_index"].to_numpy()
            mains, wrists = t[IMG_COLS[0]].to_pylist(), t[IMG_COLS[1]].to_pylist()
            for s in range(0, len(idx), BATCH):
                rows = slice(s, s + BATCH)
                batch = tokenize(processor, [decode(c) for c in mains[rows]], [decode(c) for c in wrists[rows]],
                                 [self.tasks[int(k)] for k in task[rows]])
                yield idx[rows], dict(batch)


def compare(a, b, mask=None):
    """Max abs difference and min cosine similarity over tokens (mask: [B, T] True = compare)."""
    a, b = a.float(), b.float()
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    return (a - b).abs().max().item(), nn.functional.cosine_similarity(a, b, dim=-1).min().item()


def pad_text(text, mask, n_text):
    """[B, t, D] + [B, t] -> zero/False padded to n_text tokens."""
    b, t, d = text.shape
    out = torch.zeros(b, n_text, d, dtype=text.dtype, device=text.device)
    out_mask = torch.zeros(b, n_text, dtype=torch.bool, device=text.device)
    out[:, :t], out_mask[:, :t] = text, mask
    return out, out_mask


def read_frames(files, wanted):
    """Images and task of the wanted global indices, read from the local parquet files."""
    found = {}
    for f in files:
        idx = pq.read_table(f, columns=["index"])["index"].to_numpy()
        if not np.isin(wanted, idx).any():
            continue
        t = pq.read_table(f, columns=["index", "task_index", *IMG_COLS]).to_pandas()
        for _, r in t[t["index"].isin(wanted)].iterrows():
            found[int(r["index"])] = (decode(r[IMG_COLS[0]]), decode(r[IMG_COLS[1]]), int(r["task_index"]))
    return [found[int(i)] for i in wanted]


def main():
    cfg = load_config(CONFIG)
    seed_everything(cfg["seed"])
    if not torch.cuda.is_available():
        sys.exit("build_cache needs a GPU session (T4).")
    dcfg, bcfg, ccfg = cfg["data"], cfg["backbone"], cfg["cache"]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1. full dataset to local disk (pinned revision)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(dcfg["repo_id"], repo_type="dataset", revision=dcfg["revision"], local_dir=LOCAL,
                      allow_patterns=["data/*", "meta/*"])
    download_min = (time.time() - t0) / 60
    files = sorted(Path(LOCAL).glob("data/chunk-*/file-*.parquet"))
    tasks = data.load_tasks(LOCAL)
    n_frames = data.load_info(LOCAL)["total_frames"]

    # 2. layout
    bb = Backbone(bcfg["name"], bcfg["revision"], bcfg["num_layers"])
    P = bb.prefix_len
    dim = bb.vlm.config.text_config.hidden_size
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    task_ids = sorted(tasks)
    lay = bb.tokenize([blank] * len(task_ids), [blank] * len(task_ids), [tasks[t] for t in task_ids])
    text_len = dict(zip(task_ids, lay["attention_mask"][:, P:].sum(1).tolist()))
    n_text = max(text_len.values())
    need_gb = cache.size_gb(n_frames, P + n_text, dim)
    out_dir = Path(ccfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_dir.resolve()).free / 1e9
    if need_gb > ccfg["max_gb"] or free_gb < need_gb + 1:
        sys.exit(f"Not enough room: cache needs {need_gb:.2f} GB (limit {ccfg['max_gb']}), free {free_gb:.1f} GB.")

    # 3. build
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"dataset": dcfg["repo_id"], "dataset_revision": dcfg["revision"], "backbone": bcfg["name"],
            "backbone_revision": bcfg["revision"], "num_layers": bcfg["num_layers"], "prompt": "<image><image>{instruction}",
            "dtype": "float16", "code_commit": commit, "text_len_by_task": text_len}
    arrays = cache.create(out_dir, n_frames, P, n_text, dim, meta)
    workers = os.cpu_count()
    loader = DataLoader(FrameBatches(files, tasks, bcfg["name"], bcfg["revision"]),
                        batch_size=None, num_workers=workers, prefetch_factor=4)
    t1, done = time.time(), 0
    for i, (idx, batch) in enumerate(loader):
        idx = np.asarray(idx)
        pre, txt, mask = bb.split(bb.forward(batch), batch)
        cache.write_rows(arrays, idx, pre.cpu().numpy(), txt.cpu().numpy(), mask.cpu().numpy(), idx)
        done += len(idx)
        if i % 100 == 0:
            rate = done / (time.time() - t1)
            print(f"{done}/{n_frames} frames, {rate:.1f} frames/s, eta {(n_frames - done) / rate / 60:.1f} min", flush=True)
    for a in arrays.values():
        a.flush()
    del arrays
    build_min = (time.time() - t1) / 60

    # 4. checks on the cache as read back from disk
    meta_r, c = cache.open_cache(out_dir)
    check("every row written, row i holds frame i", np.array_equal(c["frame_index"], np.arange(n_frames)))
    task_of = np.empty(n_frames, dtype=np.int64)
    for f in files:
        t = pq.read_table(f, columns=["index", "task_index"])
        task_of[t["index"].to_numpy()] = t["task_index"].to_numpy()
    expected = np.array([text_len[int(k)] for k in task_of])
    check("text mask length == instruction token count of the row's task (all rows)",
          np.array_equal(c["text_mask"].sum(1), expected))
    finite = all(np.isfinite(c[k][s:s + 2048]).all() for k in ["prefix", "text"] for s in range(0, n_frames, 2048))
    check("no NaN/Inf anywhere in the cache", finite)

    # cached == online (rule approved after R1: fp16 cos >= 0.9999; exact math in fp32 <= 1e-4)
    rng = np.random.default_rng(cfg["seed"])
    wanted = np.sort(rng.choice(n_frames, N_CHECK, replace=False))
    frames = read_frames(files, wanted)
    mains, wrists, instr = [f[0] for f in frames], [f[1] for f in frames], [tasks[f[2]] for f in frames]
    pre_on, txt_on, mask_on = bb.encode(mains, wrists, instr)
    txt_on, mask_on = pad_text(txt_on, mask_on, n_text)
    pre_c = torch.from_numpy(np.array(c["prefix"][wanted])).cuda()
    txt_c = torch.from_numpy(np.array(c["text"][wanted])).cuda()
    mask_c = torch.from_numpy(np.array(c["text_mask"][wanted])).cuda()
    check("cached text masks == online text masks (10 frames)", torch.equal(mask_c, mask_on))
    p_abs, p_cos = compare(pre_c, pre_on)
    t_abs, t_cos = compare(txt_c, txt_on, mask_c)
    scale = pre_on.float().abs().max().item()
    check("fp16: cached == online, prefix tokens, min cos >= 0.9999 (10 frames)", p_cos >= 0.9999,
          f"max abs {p_abs:.2e} (largest |feature| {scale:.1f}), min cos {p_cos:.6f}")
    check("fp16: cached == online, text tokens, min cos >= 0.9999 (10 frames)", t_cos >= 0.9999,
          f"max abs {t_abs:.2e}, min cos {t_cos:.6f}")
    bb32 = Backbone(bcfg["name"], bcfg["revision"], bcfg["num_layers"], dtype=torch.float32)
    alone = bb32.encode(mains[:2], wrists[:2], instr[:2])
    batched = bb32.encode(mains, wrists, instr)
    a_txt, a_mask = pad_text(alone[1], alone[2], n_text)
    b_txt, b_mask = pad_text(batched[1][:2], batched[2][:2], n_text)
    b_abs = max(compare(alone[0], batched[0][:2])[0], compare(a_txt, b_txt, a_mask & b_mask)[0])
    check("fp32: same frames alone vs inside the 10-frame batch, max abs <= 1e-4 (2 frames)", b_abs <= 1e-4,
          f"max abs {b_abs:.2e}")

    # ---------- report ----------
    on_disk = sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1e9
    L = ["# M2 cache build (R2)", "",
         "Generated on Kaggle by `python -m scripts.build_cache` (via `notebooks/runner.ipynb`). "
         "Every number is measured in this run.", "", "## Checks", ""]
    L += ["| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |" for n, ok, d in checks]
    L += ["", "## Cache", "", "| Item | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in [
        ("Location in the version output", f"`/kaggle/working/{out_dir}/`"),
        ("Rows (frames)", n_frames), ("Prefix tokens P", P), ("Text tokens (padded)", n_text), ("Dim", dim),
        ("Size on disk", f"{on_disk:.2f} GB"), ("Code commit", f"`{commit[:12]}`"),
        ("Download minutes", f"{download_min:.1f}"), ("Build minutes", f"{build_min:.1f}"),
        ("Frames per second", f"{n_frames / (build_min * 60):.1f}"),
        ("DataLoader workers (= CPU count)", workers), ("GPU", torch.cuda.get_device_name(0))]]
    L += ["", f"Text tokens per task: {text_len}", ""]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    failed = [n for n, ok, _ in checks if not ok]
    print(f"Checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
