# GPU ledger (Kaggle)

One row per Kaggle run (CPU runs too, for the record; they use no GPU quota). Minutes are the notebook's own measured total time unless marked "Kaggle"; weekly budget ~30 h.

| Date (UTC) | Notebook / version | Purpose | Accelerator | Minutes | Results |
|---|---|---|---|---|---|
| 2026-11-25 | 00_feasibility, run 1 | Feasibility: all 7 steps | T4 | 12.0 | `results/feasibility_run1.md` |
| 2026-11-25 | 00_feasibility, run 2 | Rerun steps 3 and 5b | T4 | 3.9 | `results/feasibility_run2.md` |
| 2026-11-26 15:19 | runner R0 (`d645daa`) | Runner smoke test: `pytest -q tests` (6 passed, 4 skipped) | CPU (no GPU quota) | 1.6 | pasted summary |

**GPU total used: 15.9 min** (CPU runs not counted).
