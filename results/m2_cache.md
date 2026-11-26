# M2 cache build (R2)

Generated on Kaggle by `python -m scripts.build_cache` (via `notebooks/runner.ipynb`). Every number is measured in this run.

## Checks

| Check | Result | Detail |
|---|---|---|
| every row written, row i holds frame i | PASS |  |
| text mask length == instruction token count of the row's task (all rows) | PASS |  |
| no NaN/Inf anywhere in the cache | PASS |  |
| cached text masks == online text masks (10 frames) | PASS |  |
| fp16: cached == online, prefix tokens, min cos >= 0.9999 (10 frames) | PASS | max abs 7.81e-02 (largest |feature| 25.2), min cos 0.999990 |
| fp16: cached == online, text tokens, min cos >= 0.9999 (10 frames) | PASS | max abs 6.25e-02, min cos 0.999998 |
| fp32: same frames alone vs inside the 10-frame batch, max abs <= 1e-4 (2 frames) | PASS | max abs 4.77e-05 |

## Cache

| Item | Value |
|---|---|
| Location in the version output | `/kaggle/working/outputs/cache/` |
| Rows (frames) | 52042 |
| Prefix tokens P | 134 |
| Text tokens (padded) | 9 |
| Dim | 960 |
| Size on disk | 14.29 GB |
| Code commit | `e122e301a303` |
| Download minutes | 0.7 |
| Build minutes | 134.9 |
| Frames per second | 6.4 |
| DataLoader workers (= CPU count) | 4 |
| GPU | Tesla T4 |

Text tokens per task: {0: 6, 1: 7, 2: 9, 3: 7, 4: 9, 5: 9, 6: 4, 7: 6, 8: 8, 9: 7}
