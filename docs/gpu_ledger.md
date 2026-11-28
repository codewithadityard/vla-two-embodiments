# GPU ledger (Kaggle)

One row per Kaggle run (CPU runs too, for the record; they use no GPU quota). Minutes are the notebook's own measured total time unless marked "Kaggle"; weekly budget ~30 h.

## Week until 2026-11-28 (quota reset Sat 05:30 IST)

| Date (UTC) | Notebook / version | Purpose | Accelerator | Minutes | Results |
|---|---|---|---|---|---|
| 2026-11-25 | 00_feasibility, run 1 | Feasibility: all 7 steps | T4 | 12.0 | `results/feasibility_run1.md` |
| 2026-11-25 | 00_feasibility, run 2 | Rerun steps 3 and 5b | T4 | 3.9 | `results/feasibility_run2.md` |
| 2026-11-26 15:19 | runner R0 (`d645daa`) | Runner smoke test: `pytest -q tests` (6 passed, 4 skipped) | CPU (no GPU quota) | 1.6 | pasted summary |
| 2026-11-26 16:25 | runner R1 (`7bc3edd`) | M2 probe: 6/8 checks PASS (2 FAIL, see results) | T4 | 9.9 | `results/m2_probe.md` |
| 2026-11-26 | runner R1b (`d2b8081`) | M2 probe rerun: 9/9 checks PASS; runner summary cell crashed after the script (path-parsing bug, fixed) | T4 | unreconciled, previous quota week | `results/m2_probe.md` |
| 2026-11-26 17:33 | CLI test `vla-cli-test` (`6a5e7bd`) | Kaggle CLI end-to-end test: `pytest -q tests` (8 passed, 4 skipped) | CPU (no GPU quota) | 1.6 | summary via CLI |
| 2026-11-26 19:18 | runner R2 (`e122e30`, notebook7b470429e6) | M2 cache build, all 52,042 frames: 7/7 checks PASS; cache 14.29 GB in the version output (6.4 frames/s, 4 workers) | T4 | 139.0 | `results/m2_cache.md` |
| 2026-11-26 20:46 | `vla-m34` v1 (`200f696`) | M3+M4 checks: 3/7 PASS (unit-test overfits, real-data overfits and step count S FAIL; see results) | T4 x2 (one used) | 72.5 | `results/m34_checks.md` |
| 2026-11-26 22:03 | `vla-unit` v1 (`491763e`) | torch unit tests with the relative overfit check: 9 passed | CPU (no GPU quota) | 3.0 | summary via CLI |
| 2026-11-26 22:06 | `vla-m34` v2 (`106f576`) | M3+M4 rerun: crashed after the unit tests (old thresholds: 2 FAIL) on `from tests.test_expert import` (an installed `tests` package hides the repo's); fixed by loading the file by path | T4 x2 | 2.5 | summary via CLI |
| 2026-11-26 22:38 | `vla-m34` v3 (`7d3e47d`) | M3+M4 rerun 2: 7/10 PASS; real-data overfits FAIL (C 0.0313, A 0.3694), worker A smoke eval crashed (EGL device id 1 with CUDA_VISIBLE_DEVICES=1) | T4 x2 | 29.8 | `results/m34_checks.md` |
| 2026-11-26 22:57 | `vla-toy` v1 (`a310994`) | Flow toy test (bimodal 2-D): 3/3 PASS (45.7 % / 52.8 %, middle 1.5 %); plot not returned (gitignored) | CPU (no GPU quota) | 2.8 | `results/flow_toy.md` |

| 2026-11-26 23:02 | `vla-toy` v2 (`b5c5114`) | Flow toy test rerun to return the plot: 3/3 PASS (45.5 % / 52.9 %, middle 1.6 %; same training loss as v1, samples differ by 0.1-0.2 points, likely CPU numerics on a different Kaggle machine) | CPU (no GPU quota) | 3.0 | `results/flow_toy.md`, `results/flow_toy.png` |

| 2026-11-26 23:39 | `vla-m34` v4 (`b5c5114`) | M3+M4 run (c): 13/13 PASS (overfit C 0.0008, A 0.0122; both smoke evals ran); rule S this run 18,831, approved S = 19,047 used | T4 x2 | 29.8 | `results/m34_checks.md` |
| 2026-11-27 01:45 | `vla-pair-s0` v1 (`d79bdeb`) | M5: A_s0 (GPU 0) + C_s0 (GPU 1) at once, S = 19,047: all checks PASS; A_s0 40/50, C_s0 41/50; cache warming 16.7 min (both runs reading at once) | T4 x2 | 124.0 | `results/runs/A_s0.md`, `results/runs/C_s0.md`, `results/eval/` |
| 2026-11-27 13:57 | `vla-p2-r0` v1 (`0a9fbb7`) | P2-R0: disk full while downloading wall_single.zip (uv cache held a second copy of the torch wheels); an import failed after wandb (error not printed); fixed | CPU (no GPU quota) | ~5 (crashed, no summary) | log only |

| 2026-11-27 14:11 | `vla-p2-r0` v2 (`d3f8614`) | P2-R0: wall_single.zip expands to > 20 GB (filled /kaggle/working); wandb 0.13.1 failed with the newest sentry-sdk; fixed (extract to /tmp, environment.yaml pins as constraints) | CPU (no GPU quota) | ~10 (crashed, no summary) | log only |

| 2026-11-27 14:23 | `vla-p2-r0` v3 (`b877a1a`) | P2-R0: plan.py needs mujoco_py (env/__init__ imports PointMaze); checkpoint zip holds macOS `._` files; report not written; fixed (MuJoCo 2.1 + mujoco-py per README, junk filter, report always written) | CPU (no GPU quota) | 10.0 | summary via CLI |

| 2026-11-27 14:40 | `vla-topup-s0` v1 (`9ddb884`) | Top-up eval, start states 5-9, final checkpoints: all checks PASS; A_s0 42/50, C_s0 34/50 (totals now 82/100, 75/100) | T4 x2 | 67.0 | `results/runs/*_ep5-9.md`, `results/eval/` |
| 2026-11-27 14:40 | `vla-p2-r0` v4 (`f498d85`) | P2-R0: 8/10 PASS (venv, mujoco_py, all imports, downloads, checkpoint + its hydra.yaml); dataset listing showed only HF lock files and the compose override clashed with train.yaml's launcher keys (inspection bugs, fixed) | CPU (no GPU quota) | 13.7 | `results/p2_r0.md` |

| 2026-11-27 14:57 | `vla-p2-r0` v5 (`a563fe8`) | P2-R0: 10/10 PASS (venv + mujoco-py, imports, data 1920 x 50, checkpoint epoch 65 + hydra.yaml, config diff, success criterion) | CPU (no GPU quota) | 14.0 | `results/p2_r0.md` |

| 2026-11-27 15:23 | `vla-p2-r1` v1 (`13ad13e`) | P2-R1: probes and planning failed at config composition (hydra-submitit-launcher plugin missing from the package list; fixed). Data check: pixels are float32 0-255 but not integer-valued | T4 x2 | 11.2 | summary via CLI |
| 2026-11-27 15:36 | `vla-p2-r1` v2 (`eaedcb6`) | P2-R1: probes and planning failed building the DINOv2 encoder (torch.hub fetched today's dinov2 main, which needs Python >= 3.10); fixed by pinning the hub cache to dinov2 `e1277af` | T4 x2 | 10.3 | summary via CLI |
| 2026-11-27 16:04 | `vla-p2-r1` v3 (`fdc55a8`) | P2-R1: 1/4 checks PASS. Decoder-on training probe and official planning ran out of T4 memory (16 GB) in the VQ-VAE decoder; decoder-off probe: 8.39 s/step, 554 steps/epoch = 77.5 min/epoch. Stopped for decisions | T4 x2 | 26.3 | `results/p2_r1.md` |
| 2026-11-27 17:26 | `vla-pair-1` v1 (`9ddb884`) | M6 pair 1, S = 19047, 10 episodes/task, final checkpoints: both exit 0; B_s0 5/100, A_s1 76/100 | T4 x2 | 233.8 | `results/runs/{B_s0,A_s1}.md`, `results/eval/` |
| 2026-11-27 18:56 | `vla-p2-r2` v1 (`bb9fe93`) | P2-R2: frame-level reader test FAILED: the one-time conversion (second 59 GB copy in /tmp) did not finish within its 20-min limit, then /tmp became read-only (likely a disk limit) and the run crashed before training. Per the approved rule: no 2x shown within 30 min, so option a (official reader, E = 2) | T4 x2 | 27.9 | summary via CLI |
| 2026-11-27 21:10 | `vla-p2-r2` v2 (`49f89d9`) | P2-R2 option a: training epoch 1 finished (554 steps in 116.8 min = 12.65 s/step, slower than R1's 8.39 probe), then validation ran out of T4 memory (official val loop runs the model forward without torch.no_grad at batch 128); official planning (no decoder) failed at wandb.init (service did not start; WANDB_MODE=disabled is not enough for plan.py's own wandb_logging) | T4 x2 | 125.9 | `results/p2_r2.md` |
| 2026-11-27 21:31 | `vla-pair-2` v1 (`f1a72d7`) | M6 pair 2, S = 19047, 10 episodes/task, final checkpoints: both exit 0; C_s1 75/100, B_s1 4/100 | T4 x2 | 242.8 | `results/runs/{C_s1,B_s1}.md`, `results/eval/` |
| 2026-11-28 02:53 | `vla-p2-r3` v1 (`6c85f08`) | P2-R3: 0/2 checks. CEM failed at config time (Hydra: `~planner.max_iter` cannot delete a null key). MPC-CEM hit our 300-min timeout after 4 MPC iterations (~75 min each on a T4): plan_wall.yaml has max_iter null, so the official loop only stops when all 50 pairs succeed; cumulative successes after iterations 1-4: 27, 44, 46, 46 of 50 (not a reported result). Stopped for decisions | T4 x2 | 307.2 | `results/p2_r3_v1.md` |

**Previous quota week (until Sat 2026-11-28 00:00 UTC = 05:30 IST): 1475.8 min (24.6 h) + R1b (unreconciled, previous quota week)** (CPU runs not counted). `vla-p2-r3` ran across the reset (21:54–02:53 UTC) but Kaggle counted it in the previous week: the new week's quota read 00:13 after the gallery.

## Week of 2026-11-28 (quota 30 h, resets Sat 2026-12-05 00:00 UTC = 05:30 IST)

| Date (UTC) | Notebook / version | Purpose | Accelerator | Minutes | Results |
|---|---|---|---|---|---|
| 2026-11-28 03:08 | `vla-gallery` v1 (`ef0b10c`) | Video gallery, **demo, not part of reported results**: A_s0 on start state 10 of all 10 tasks (8 success, 2 fail), B_s0 on tasks 0-2 (3 fail); exit 0; 13 GIFs, largest 1.61 MB | T4 x2 | 13.5 | `results/video_gallery.md`, `docs/media/gallery/` |
| 2026-11-28 04:38 | `vla-p2-r4` v1 (`a3d9da6`) | P2-R4: open-loop prediction error vs. horizon, official checkpoint, 192 held-out trajectories, decoder off: 3/3 checks PASS | T4 x2 | 10.5 | `results/p2_r4.md` |
| 2026-11-28 09:01 | `vla-p2-r3cem` v1 (`d158e21`) | P2-R3 plain CEM, fixed settings: 1/1 check PASS; 14/50 = 28 % (CI 17.5–41.7), paper 0.74. Queued ~25–85 min on Kaggle before starting | T4 x2 | 198.1 | `results/p2_r3_cem.md`, `results/p2_r3.md` |
| 2026-11-28 12:26 | `vla-p3-data` v1 (`be84612`) | P3-0 data report: crashed before any check (my bug: `pq.concat_tables` instead of `pa.concat_tables`) | CPU (no GPU quota) | 15.5 | summary via CLI |
| 2026-11-28 13:01 | `vla-p3-data` v2 (`a06cdeb`) | P3-0 data report: crashed before any check: `import metaworld` imports MuJoCo's renderer, which needs OpenGL, absent on the CPU session | CPU (no GPU quota) | 33.7 | summary via CLI |
| 2026-11-28 13:14 | `vla-p3-data` v3 (`67292c0`) | P3-0 data report: Kaggle status ERROR after ~10 min. Checked 2026-11-28: `kernels output` returns no files and no log for this session, `kernels logs` is empty, and the status record holds only `ERROR` (no failure message); `kernels files` lists only v1's log.txt and summary.md. So a notebook cell raised before the summary cell (clone, install or summary), but which one is unknown | CPU (no GPU quota) | ~10 (Kaggle) | none |
| 2026-11-28 16:25 | `vla-p3-0a` v1 (`492704e`) | P3-0a dataset-only report (no install step, no simulator code): 17/17 checks PASS; 2,500 episodes, 204,806 frames, 50 tasks, 80 fps, one 480x480 PNG camera, 4-D state and action | CPU (no GPU quota) | 33.3 | `docs/data_report_metaworld.md` |

**This week: 222.1 min (3.7 h)**; Kaggle's quota line on 2026-11-28: "00:13 / 30 hrs".
