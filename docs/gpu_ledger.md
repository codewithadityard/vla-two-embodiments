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
| 2026-11-26 20:46 | `vla-m34` v1 (`200f696`) | M3+M4 checks: 3/7 PASS (unit-test overfits, real-data overfits and step count S FAIL; see results) | T4 x2 (one used) | 72.5 | `results/m34_checks.md` |
| 2026-11-26 22:03 | `vla-unit` v1 (`491763e`) | torch unit tests with the relative overfit check: 9 passed | CPU (no GPU quota) | 3.0 | summary via CLI |
| 2026-11-26 22:06 | `vla-m34` v2 (`106f576`) | M3+M4 rerun: crashed after the unit tests (old thresholds: 2 FAIL) on `from tests.test_expert import` (an installed `tests` package hides the repo's); fixed by loading the file by path | T4 x2 | 2.5 | summary via CLI |
| 2026-11-26 22:38 | `vla-m34` v3 (`7d3e47d`) | M3+M4 rerun 2: 7/10 PASS; real-data overfits FAIL (C 0.0313, A 0.3694), worker A smoke eval crashed (EGL device id 1 with CUDA_VISIBLE_DEVICES=1) | T4 x2 | 29.8 | `results/m34_checks.md` |
| 2026-11-26 22:57 | `vla-toy` v1 (`a310994`) | Flow toy test (bimodal 2-D): 3/3 PASS (45.7 % / 52.8 %, middle 1.5 %); plot not returned (gitignored) | CPU (no GPU quota) | 2.8 | `results/flow_toy.md` |

| 2026-11-26 23:02 | `vla-toy` v2 (`b5c5114`) | Flow toy test rerun to return the plot: 3/3 PASS (45.5 % / 52.9 %, middle 1.6 %; same training loss as v1, samples differ by 0.1-0.2 points, likely CPU numerics on a different Kaggle machine) | CPU (no GPU quota) | 3.0 | `results/flow_toy.md`, `results/flow_toy.png` |


**GPU total used: 269.6 min (4.5 h) + R1b pending** (CPU runs not counted).
