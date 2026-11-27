# Project 2 spec: DINO-WM on Wall, reduced budget (draft for approval)

Status: **draft, not approved; nothing launched.** Facts below come from the official repo and the paper as read
on 2026-11-27. "Unknown" items are checked in run P2-R0 before any GPU time is spent.

## 1. Goal and claim

Reproduce DINO-WM's Wall planning result with the official code and data, at a reduced training budget, and
report planning success honestly against the paper.

- **Paper reference:** arXiv 2411.04983 v2, Table 1, Wall, DINO-WM success rate **0.96**. Table 8 shows this is
  the MPC (receding-horizon CEM) planner: CEM alone 0.74, MPC 0.96. The evaluation uses 50 sampled start/goal pairs.
- **What we will claim:** two numbers, each over 50 start/goal pairs with a Wilson 95 % CI:
  1. **Official pretrained Wall checkpoint + official planner** on our hardware. This checks that the planning
     pipeline and the success measurement reproduce.
  2. **Our own model trained from scratch at a reduced budget**, with the same planner and evaluation. This is
     the reproduction of training.
- At n = 50 the CI near 0.9 is roughly ±0.08–0.10, so "matches the paper" means "the paper's 0.96 is inside our
  CI". No other claim.

## 2. Official code and data (pinned)

- **Repo:** github.com/gaoyuezhou/dino_wm @ `0a9492fa12044b852ae9e001cc74604b79c8bb0c` (main, 2025-03-24, "add
  checkpoints").
- **Data:** OSF project bmw48: `wall_single.zip` (1,668,205,895 bytes, link verified in feasibility run 1). Per the
  paper, 1,920 random trajectories of 50 steps. Layout is `$DATASET_DIR/wall_single` (from `conf/env/wall.yaml`,
  which uses a 0.9 random train/val split).
- **Pretrained checkpoints:** on the same OSF project under `checkpoints` (README); the Wall file name is unknown.
  It is loaded from `<ckpt_base_path>/outputs/<model_name>`.
- **Official commands:**
  - Train: README shows `python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3`. For
    Wall we use `env=wall frameskip=5 num_hist=1`. `env=wall` exists as `conf/env/wall.yaml`, and history 1 and
    frameskip 5 come from paper Table 11. This is inferred, not a documented Wall command.
  - Plan: `python plan.py --config-name plan_wall.yaml model_name=<name>`. Settings: MPC-CEM, horizon 5, 300
    samples, top-k 30, 10 CEM steps, 5 actions taken per replan, `goal_H: 5`, `goal_source: random_state`,
    `n_evals: 50`, `seed: 99`. All of these are kept unchanged.
- **Training defaults (`conf/train.yaml`):**
  - 100 epochs, batch 32, image 224, DINOv2 encoder (frozen), ViT predictor, VQ-VAE decoder (trained by default).
  - Learning rates: predictor 5e-4, decoder 3e-4, action encoder 5e-4.
  - Seed 0, checkpoint every epoch.
  - The default launcher is SLURM/submitit; we override it to run locally.

## 3. Environment on Kaggle

- **Official environment:** conda, Python 3.9.19, torch 2.3.0 (cu121), plus many extras (mujoco-py with the
  MuJoCo 2.1 tarball, d4rl, robosuite, TensorFlow, JAX, wandb 0.13.1). Kaggle has Python 3.13 with torch 2.11, so:
  - We use a separate Python 3.9 venv created with `uv`, as in the feasibility notebook.
  - We install torch 2.3.0 + torchvision 0.18.0 (cu121; they support the T4) and only the packages that the
    code paths of train.py, plan.py and the Wall env actually import. P2-R0 finds those by importing them.
  - The full pin list is recorded, and each package we drop is logged in deviations.
  - We do not install mujoco-py, d4rl, robosuite, TensorFlow or JAX unless an import on the Wall path needs them.
- **wandb:** pinned, but no switch was found in the configs. We run with `WANDB_MODE=disabled`, so nothing is
  sent anywhere.

## 4. Runs (Kaggle; one approval for the whole sequence, stop on any failed check)

| Run | Hardware | What it does | Checks (stop if any fails) | Est. time |
|---|---|---|---|---|
| **P2-R0** | CPU, no GPU quota | Clone the pinned repo, build the 3.9 venv, download `wall_single` + the Wall checkpoint (saved as the run's output, attached later like the P1 cache), import train/plan/env, print dataset shapes. Read in the code: the success criterion, whether training uses accelerate/AMP, how wandb is called, and checkpoint file names. | Imports succeed; dataset has 1,920 trajectories × 50 steps; the checkpoint loads; the success criterion is found in code and quoted in the report | ~20 min |
| **P2-R1** | T4 | (a) Official checkpoint + official `plan_wall.yaml`, 50 evals, seed 99. (b) A 300-step training probe (`env=wall frameskip=5 num_hist=1`), measuring s/step with and without the decoder, and on 1 vs 2 GPUs if the code supports it. | Planning finishes 50 evals; probe loss falls; s/step measured | ~1 h |
| **P2-R2** | T4 (×2 if supported) | Train from scratch at the reduced budget: E epochs, E = ⌊(3 h of training) / (measured min/epoch)⌋, capped at 100. All other settings are official. The final epoch's checkpoint is the one planned with; no epoch is picked by planning success. | Trains exactly E epochs; train/val loss logged per epoch; final checkpoint saved | ~3 h |
| **P2-R3** | T4 | Plan with our final checkpoint: same `plan_wall.yaml`, 50 evals, seed 99. | 50 evals ran | ~0.75 h |

- **GPU total:** about 5 h, which is the plan's P2 allotment.
- **Budget so far:** P1 has used 7.1 h. Pairs and top-ups now running or queued add about 6.5 h, giving about
  13.6 h. Adding P2 gives about 18.6 h of the ~30 h weekly budget.
- **Order:** P2-R0 (CPU) can run now alongside the P1 pairs. GPU runs go when a Kaggle slot is free.

## 5. Budget decisions to approve

1. **Decoder off for our training run (`has_decoder=False`).** The planner uses latent distances only, and the
   paper's point is that planning needs no pixel decoder. Turning it off saves training time. The official
   checkpoint keeps whatever it has. If P2-R1 shows the saving is small (< 15 % s/step), we keep the decoder on
   and stay fully official.
2. **Epoch budget E from the P2-R1 measurement** (rule above). With 100 official epochs this is a reduced-budget
   reproduction and is labelled as such everywhere.
3. **No hyperparameter changes** to the planner or evaluation. Only `ckpt_base_path`, the launcher, `DATASET_DIR`,
   `WANDB_MODE`, `has_decoder` (item 1) and `epochs` (item 2) are overridden. All overrides are listed in
   `configs/p2_wall.yaml` and `docs/deviations.md`.

## 6. Outputs

- `results/p2_dinowm_wall.md`:
  - versions and commit
  - the success criterion quoted from code
  - both success rates with Wilson CIs, next to the paper's 0.96 (MPC) and 0.74 (CEM only)
  - the training curve and the time per epoch
  - our run's reduced epochs vs the official 100
- A few planning GIFs from `plan_outputs` (small files only).
- Scripts: `scripts/p2_setup.py` (R0), `scripts/p2_run.py` (R1–R3, a thin wrapper that runs the official
  commands in the venv and writes the reports). The runner notebook is unchanged.

## 7. Risks

- **Dependencies:** the Wall path needs a heavy package (e.g. mujoco-py) → install it in the venv. If that fails
  on Kaggle, stop and report.
- **Time:** training is much slower than estimated → E is smaller. We report the partial-budget result with its
  curve rather than stretch the budget.
- **Gap:** the official checkpoint scores far below 0.96 → the evaluation differs from the paper's. We report it
  and do not tune.
- **Reproducibility:** the official code may not be bit-deterministic on GPU. We use the fixed seeds (train 0,
  plan 99) and report one run, without claiming bitwise repeats.
