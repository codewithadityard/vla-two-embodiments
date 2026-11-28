# P3 plan: Meta-World MT10 as a second embodiment (DRAFT, awaiting approval; nothing launched)

## Question

Does one policy trained on both LIBERO-Goal (Franka, 7-D actions) and Meta-World MT10 (Sawyer, 4-D actions) work on
both? And which action-head design works better for two embodiments?

1. **Shared padded head + embodiment token.** One state input and one action output, padded to 8-D state and 7-D
   action. Meta-World's 4-D state and action are zero-padded, and the loss ignores the padded dimensions. A learned
   embodiment token (one per robot) is appended to the expert's memory, next to the state token.
2. **Separate heads.** A shared expert trunk with a per-embodiment state projection and action output (7-D or 4-D);
   no padding and no embodiment token.

Everything else stays the same as config A (language + flow matching; A vs. C was a null result, so A's head is
kept):
- the frozen SmolVLM2-500M backbone at the same revision, first 16 layers;
- the same fp16 feature-cache format;
- the same 6-layer expert, chunk 50, and the same optimiser and schedule;
- final checkpoint only, never chosen by eval results;
- Wilson 95 % CIs.

## Data (measured from the dataset's `meta/info.json`, everything else checked in P3-0)

`lerobot/metaworld_mt50` (LeRobot v3.0):
- 2,500 episodes in 50 tasks (50 per task), 204,806 frames, 80 fps;
- one camera, 480 × 480;
- `observation.state` is 4-D and `action` is 4-D (x, y, z, gripper).

MT10 uses the 10 MT10 tasks from it. That is about 41 k frames if the tasks are about equally long; P3-0 counts them.

**One camera.** The backbone expects a main and a wrist image. The Meta-World image is the main image and the wrist
image is black, so the prefix stays 134 tokens and the cache format is unchanged. The 64 wrist tokens are masked out
in the expert through the existing `feat_mask`.

## Steps and GPU estimates

Every estimate is scaled from a measured number:
- Cache build: 6.4 frames/s on one T4 with two images per frame (`results/m2_cache.md`).
- Training: 19,047 steps take about 50 min per model, two models at once (`results/runs/`).
- Pair session with 100 LIBERO episodes per model: 233.8 and 242.8 min (ledger). Meta-World episodes are assumed to
  take about as long as LIBERO's (about 100 s each) until P3-0 measures them.

| Step | What | Kaggle | GPU-h (estimate) |
|---|---|---|---|
| P3-0a | Dataset report (split from P3-0 on 2026-11-28, see deviations). Reads only the dataset's parquet and meta files, with Kaggle's preinstalled packages and no install step; no metaworld, LeRobot, MuJoCo or renderer. Counts, episode lengths, fps, camera and image size, state/action dims and ranges, instructions, success flag, task_id. `scripts/data_report_mw.py` -> `docs/data_report_metaworld.md`. | CPU | 0 |
| P3-0b | Simulator checks, the first GPU run (MuJoCo rendering needs EGL, absent on the CPU session), done the same way as R1 (M2 probe) for LIBERO. MT10 task list from metaworld's `MT10_V3`; instruction -> env name by exact text (LeRobot's `TASK_DESCRIPTIONS` at the pinned commit); image orientation (env render vs. dataset frame); env state vs. dataset state; the MT10 split and its train stats; LeRobot's Meta-World eval protocol (max steps, success flag); the reference policy's `n_action_steps` re-check; MuJoCo version; one scripted-policy rollout per task to time an episode. No training. | T4 | not estimated yet (no measured basis) |
| P3-1 | Feature cache for MT10 (~41 k frames), sharded over the two T4s. | T4 x2 | ~1.1 (~2 with one GPU) |
| P3-2 | MT10-only baseline M_s0 and M_s1 (single embodiment), trained at once with 19,047 steps each; eval 10 tasks × 10 episodes each. | T4 x2 | ~4 |
| P3-3 | Joint models, seed 0: J_shared_s0 and J_sep_s0, trained at once with 2 × 19,047 steps each, so that each embodiment gets about as many samples as its single-embodiment model. Batches are half LIBERO, half Meta-World. Eval: 100 Meta-World + 100 LIBERO episodes each. | T4 x2 | ~7.5 |
| P3-4 | Joint models, seed 1 (optional). | T4 x2 | ~7.5 |

- Total: about 12.6 GPU-h without P3-4, or about 20 GPU-h with it.
- This week: 29.75 h at the reset, minus the gallery (0.2), P2-R4 (≤ 0.5) and plain CEM (~4), leaves about 25 h.
- Code before P3-1: Meta-World data loader, two-dataset sampler, the two head designs and a Meta-World eval. Tests
  run on a Kaggle CPU session; nothing heavy runs on the laptop.

## Comparisons (fixed before any eval)

- **Head design:** J_shared vs. J_sep, on each embodiment separately.
- **Transfer or interference:**
  - Meta-World: J_shared and J_sep vs. M.
  - LIBERO: J_shared and J_sep vs. the existing A_s0/A_s1 (same recipe, LIBERO only).
- At 100 episodes per model, a per-model CI is about ±8–10 points (as in Project 1). Only large differences can be
  resolved, and smaller ones will be reported as such.

## Decisions for you

1. Approve the plan, the two head designs and the step list.
2. Joint seeds: seed 0 only (about 12.6 h in total), or seeds 0 and 1 (about 20 h)?
3. Chunk length: 50 actions at 80 fps is about 0.6 s, and the average demo is about 82 frames (204,806 / 2,500). Keep
   chunk 50 and execute it in full (same as LIBERO, required for a shared head), or keep chunk 50 but execute only
   the first part on Meta-World? P3-0 reports episode lengths first.
