# GPU ledger (Kaggle)

One row per Kaggle run (CPU runs too, for the record; they use no GPU quota). Minutes are the notebook's own measured total time unless marked "Kaggle"; weekly budget ~30 h.

| Date (UTC) | Notebook / version | Purpose | Accelerator | Minutes | Results |
|---|---|---|---|---|---|
| 2026-11-25 | 00_feasibility, run 1 | Feasibility: all 7 steps | T4 | 12.0 | `results/feasibility_run1.md` |
| 2026-11-25 | 00_feasibility, run 2 | Rerun steps 3 and 5b | T4 | 3.9 | `results/feasibility_run2.md` |
| 2026-11-26 15:19 | runner R0 (`d645daa`) | Runner smoke test: `pytest -q tests` (6 passed, 4 skipped) | CPU (no GPU quota) | 1.6 | pasted summary |
| 2026-11-26 16:25 | runner R1 (`7bc3edd`) | M2 probe: 6/8 checks PASS (2 FAIL, see results) | T4 | 9.9 | `results/m2_probe.md` |
| 2026-11-26 | runner R1b (`d2b8081`) | M2 probe rerun: 9/9 checks PASS; runner summary cell crashed after the script (path-parsing bug, fixed) | T4 | pending (summary crashed; read from Kaggle version page) | `results/m2_probe.md` |
| 2026-11-26 17:33 | CLI test `vla-cli-test` (`6a5e7bd`) | Kaggle CLI end-to-end test: `pytest -q tests` (8 passed, 4 skipped) | CPU (no GPU quota) | 1.6 | summary via CLI |
| 2026-11-26 19:18 | runner R2 (`e122e30`, notebook7b470429e6) | M2 cache build, all 52,042 frames: 7/7 checks PASS; cache 14.29 GB in the version output (6.4 frames/s, 4 workers) | T4 | 139.0 | `results/m2_cache.md` |

**GPU total used: 164.8 min (2.7 h) + R1b pending** (CPU runs not counted).
