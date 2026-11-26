"""Feature cache on disk: one .npy memmap per array + meta.json. NumPy only (no torch).

Arrays (n_rows = number of cached frames):
  prefix      float16 [n_rows, n_prefix, dim]   image tokens (+ BOS / image wrappers)
  text        float16 [n_rows, n_text, dim]     instruction tokens, zero-padded to n_text
  text_mask   bool    [n_rows, n_text]          True where text is a real token
  frame_index int64   [n_rows]                  global dataset index of each row
"""
import json
from pathlib import Path

import numpy as np

SPECS = {"prefix": np.float16, "text": np.float16, "text_mask": np.bool_, "frame_index": np.int64}


def create(cache_dir, n_rows, n_prefix, n_text, dim, meta: dict) -> dict:
    """Create empty arrays on disk; write meta.json. Returns the writable memmaps."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    shapes = {"prefix": (n_rows, n_prefix, dim), "text": (n_rows, n_text, dim),
              "text_mask": (n_rows, n_text), "frame_index": (n_rows,)}
    arrays = {k: np.lib.format.open_memmap(d / f"{k}.npy", mode="w+", dtype=SPECS[k], shape=shapes[k])
              for k in SPECS}
    arrays["frame_index"][:] = -1  # -1 = row not written yet
    sizes = dict(n_rows=n_rows, n_prefix=n_prefix, n_text=n_text, dim=dim)
    (d / "meta.json").write_text(json.dumps({**meta, **sizes}, indent=2))
    return arrays


def write_rows(arrays, rows, prefix, text, text_mask, frame_index):
    """Write a batch into the given row numbers (batches may arrive in any order).
    Text shorter than n_text is zero-padded."""
    rows = np.asarray(rows)
    t = text.shape[1]
    text_full = np.zeros((len(rows),) + arrays["text"].shape[1:], dtype=np.float16)
    mask_full = np.zeros((len(rows), arrays["text_mask"].shape[1]), dtype=bool)
    text_full[:, :t] = text
    mask_full[:, :t] = text_mask
    arrays["prefix"][rows] = prefix
    arrays["text"][rows] = text_full
    arrays["text_mask"][rows] = mask_full
    arrays["frame_index"][rows] = frame_index


def open_cache(cache_dir):
    """-> (meta dict, read-only memmaps)."""
    d = Path(cache_dir)
    meta = json.loads((d / "meta.json").read_text())
    return meta, {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in SPECS}


def size_gb(n_rows, n_tokens, dim, bytes_per_value=2):
    return n_rows * n_tokens * dim * bytes_per_value / 1e9
