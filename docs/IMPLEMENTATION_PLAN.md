# Implementation plan: Project 1 (mini-VLA on LIBERO-Goal) + Project 2 outline

Status: draft for approval, written 2026-11-26. Inputs: `SPEC.md`, `CLAUDE.md`, `results/feasibility_run1.md`, `results/feasibility_run2.md`, and the reference facts in §0.2. Those facts were checked on 2026-11-26 against the pinned LeRobot commit and the Hugging Face Hub.

**About the numbers.** A number tagged *measured* comes from a logged run. A number tagged *verified* comes from a config or metadata file (source given). A number tagged *est.* is planning arithmetic built on measured numbers. Estimates are **not results**: each milestone replaces them with measured values in its results file.

**About "JD skill".** No job description is in the repo. The JD-skill lines below are my wording for a typical robot-learning / VLA research-engineer posting; swap in the JD's exact phrases.

---

## 0. Shared ground

### 0.1 Working rules (from CLAUDE.md, applied to every milestone)
- One milestone at a time. Each one gets its own short plan, needs your approval, ends with its checks passing, and then needs your approval again.
- Seeds come only from config files. The same seed gives the same result, and a determinism test proves it (M3).
- Every reported number comes from a logged run. Results files are written by scripts.
- Every deviation goes into `docs/deviations.md` when it happens (expected ones are listed in §9).
- Failures: stop and report, with no silent workarounds.
- Never commit `data/`, `checkpoints/`, `outputs/`, `mlruns/`.

### 0.2 Facts this plan relies on

| Fact | Value | Source |
|---|---|---|
| Backbone layers used | first **16 of 32** language-model layers, hidden size 960 | verified: `num_vlm_layers: 16` in `lerobot/smolvla_base` and `lerobot/smolvla_libero` config.json; SmolVLM2-500M config (32 layers, 960) |
| Action chunk | `chunk_size: 50`, `n_action_steps: 50` (whole chunk executed before replanning) | verified: `smolvla_libero` config.json |
| Flow-matching steps | `num_steps: 10` | verified: `smolvla_libero` config.json |
| Flow-matching math | t ~ Beta(1.5, 1)·0.999 + 0.001; noise at t = 1; x_t = t·ε + (1−t)·a; target u = ε − a; MSE loss; Euler integration from t = 1 to 0 | verified: `modeling_smolvla.py` at commit `b07adcc5` |
| Attention inside the action chunk | causal (each action token sees earlier ones) | verified: `att_masks += [1] * chunk_size` in `modeling_smolvla.py` |
| State in SmolVLA | one token in the VLM prefix (`state_proj`) | verified: `modeling_smolvla.py` |
| Image tokens | images padded to 512×512; SigLIP patch 16, pixel-shuffle factor 4 → (512/16)² / 4² = **64 tokens per image** | verified: configs (`resize_imgs_with_padding`, `patch_size`, `scale_factor`) |
| smolvla_libero training | fine-tunes the VLM too (`train_expert_only: false`, `freeze_vision_encoder: false`); AdamW lr 1e-4, betas (0.9, 0.95), wd 1e-10, clip 10, 1,000 warm-up steps, cosine decay to 2.5e-6 | verified: `smolvla_libero` config.json |
| Dataset | `lerobot/libero_goal_image`: LeRobot v3.0, **428 episodes, 52,042 frames, 10 tasks**, fps 10 (metadata), cameras `image` + `wrist_image` at 256×256, state 8-D, action 7-D, ~6.3 GB of parquet files (images stored inside the parquet) | verified: `meta/info.json` + Hub file listing |
| State layout mismatch to check | dataset column names say `x, y, z, rx, ry, rz, rw, gripper` (looks like a quaternion); the env's state is eef pos (3) + **axis-angle (3)** + gripper qpos (2) | verified: `meta/info.json`; `sim/integrations/libero.py` |
| LIBERO env (LeRobot) | `libero_goal` step limit **300** (longest demo: 270); 10 no-op steps after reset; images flipped 180°; control at 20 Hz; init state = episode index | verified: `sim/integrations/libero.py`, `envs/configs.py` |
| Frozen-backbone training step | **0.357 s/step** at batch 16, 1.79 GB GPU. Caveats: full 32-layer VLM, **one** 512² image, 73 tokens/sample, random inputs | measured: feasibility run 1 |
| LoRA training step | 0.544 s/step, 3.39 GB | measured: feasibility run 2 |
| One LIBERO episode | **78 s** eval time (1 successful episode of `smolvla_libero`, T4, batch 1); 150.7 s wall time incl. downloads | measured: feasibility run 2 |
| Kaggle session install | ~1 min (0.8–1.0) | measured: runs 1–2 |
| GPU used so far | 12.0 + 3.9 = **15.9 min** | measured: runs 1–2 |

### 0.3 Repo layout (end state)

```
configs/   base.yaml, A_flow_lang.yaml, B_flow_nolang.yaml, C_l1_lang.yaml, eval.yaml,
           runs/{A,B,C}_s{0,1}.yaml           # 3 lines each: which config + seed
minivla/   config.py data.py backbone.py cache.py expert.py heads.py train.py
           policy.py eval_libero.py stats.py
scripts/   data_report.py check_env_vs_data.py build_cache.py make_results_table.py make_figures.py
notebooks/ 00_feasibility.ipynb (exists) 01_features.ipynb 02_train.ipynb 03_eval.ipynb 10_dinowm_wall.ipynb
tests/     test_data.py test_backbone.py test_expert.py test_flow.py test_policy.py test_stats.py
docs/      IMPLEMENTATION_PLAN.md data_report.md gpu_ledger.md deviations.md media/
results/   m2_features.md m3_regression.md m4_flow.md m5_eval_A_s0.md eval/*.csv ablation_table.md
README.md
```

### 0.4 Conventions
- **Configs.** YAML files with an `inherit:` key. A run is named by one file in `configs/runs/`, and that file holds the seed. There are three seeds, all in config files: `split_seed` (train/val split), `seed` (init, data order, noise, time samples), `eval_seed` (env seeds).
- **Run records.** Each run writes `outputs/<run_id>/` (gitignored): the resolved config, the git commit, `metrics.csv`, and the final checkpoint (expert weights + normalization stats + config). Small summaries go into `results/` and are committed.
- **Logging.** Plain CSV, no MLflow: simpler, and every line is readable.
- **GPU ledger.** `docs/gpu_ledger.md` gets one row per Kaggle run: date, notebook version, purpose, accelerator, minutes (Kaggle's figure), link to the results file. Created in M1 with the two feasibility runs.
- **Code on Kaggle.** Each notebook clones `main` of this public repo and logs the exact commit hash it ran, then runs scripts. LeRobot is pinned to `b07adcc5`, the same as feasibility. Work is merged into `main` and pushed before each Kaggle run.
- **Laptop environment.** The full LeRobot/MuJoCo setup stays **on hold** until WSL RAM is raised. For M1–M4 I propose a light Miniforge env (Python 3.12) outside the repo: `pandas pyarrow huggingface_hub numpy pyyaml pillow pytest`, plus `torch` (CPU wheel is enough) from M3. Needs your OK; no sudo.

---

## M1: Data report (LIBERO-Goal)

**1. Goal.** Produce a verified, script-generated description of LIBERO-Goal (counts, lengths, keys, ranges, instructions, normalization stats) and a seeded episode-level train/val split.
JD skill: robot-data pipelines and data QA (LeRobot dataset format).

**2. Files**
| Path | What it does |
|---|---|
| `configs/base.yaml` | First version: dataset repo id + pinned revision, `split_seed`, `val_fraction`. |
| `minivla/config.py` | Loads YAML with `inherit:`, seeds Python/NumPy/torch from config. |
| `minivla/data.py` | Reads the non-image parquet columns, builds the episode-level split (stratified by task), computes normalization stats on the train split. |
| `scripts/data_report.py` | Writes `docs/data_report.md`; every number in it is computed by the script. |
| `tests/test_data.py` | Split and stats tests (below). |
| `docs/data_report.md` | Output. |
| `docs/gpu_ledger.md` | New; starts with feasibility runs 1–2. |

**3. Design decisions**
- **Dataset: `lerobot/libero_goal_image`** (goal-only, 52,042 frames) rather than filtering `HuggingFaceVLA/libero` (273,465 frames over 40 tasks). Reason: 5× smaller, and images sit in the parquet files, so there's no video decoding. Check: its episode counts and lengths match the goal part of `HuggingFaceVLA/libero`, which `smolvla_libero` trained on. Only metadata files are read for this. Pin the dataset revision.
- **Read only the non-image columns** for stats (pyarrow column selection). Reason: fits in the laptop's 3.7 GiB WSL RAM.
- **Split by episode, stratified by task, 10 % val** (≈4 val episodes per task). Reason: frames within one episode are near-duplicates, so a frame-level split leaks training data into validation.
- **Normalization: mean/std** for state and action (SmolVLA's `MEAN_STD`), computed on the **train split only**. Reason: no information leaks from val. The report also lists min/max and 1st/99th percentiles.
- **Camera names:** dataset `image`/`wrist_image` and env `image`/`image2` both map to our names `main`/`wrist`.

**4. Tests and acceptance checks**
- Totals match `info.json`: 428 episodes, 52,042 frames, 10 tasks. Per-task episode counts sum to 428.
- `frame_index` runs 0…len−1 in every episode; no NaN/Inf.
- Split: disjoint, covers every episode, every task appears in val. The same seed gives the same split (hash written in the report); a different seed gives a different split.
- **State-layout verdict:** do columns 3–6 have unit norm (quaternion), or do columns 6–7 look like two mirrored gripper fingers (axis-angle + 2-D gripper, as in the env)? The report states which.
- Actions: per-dimension ranges; is the gripper action binary ±1?
- One decoded image per camera: shape 256×256×3 (saved to `outputs/`, not committed).
- `pytest tests/test_data.py` passes.

**5. Where / GPU.** Laptop, CPU only. **0 GPU min.** The download is 6.3 GB into `data/` (gitignored); I check free disk first.

**6. Risks → fallback**
- Dataset state layout ≠ env layout → a tested conversion function applied to env observations at eval, or switch to the goal subset of `HuggingFaceVLA/libero`. Your call; logged as a deviation.
- Laptop disk/RAM too small → run the same script in a Kaggle **CPU** session (no GPU quota).
- Goal-only dataset differs from the `HuggingFaceVLA/libero` goal subset → report and decide together.

**7. Concepts to explain**
- LeRobot v3.0 layout: `meta/info.json`, episodes metadata, chunked parquet files, and what a frame, an episode and a task index are.
- LIBERO actions: delta end-effector commands for the OSC controller, gripper ±1.
- Quaternion vs axis-angle, and what gripper qpos is.
- Why the split is by episode, why stats come from the train split only, and mean/std vs min/max normalization.
- Dataset fps metadata (10) vs env control rate (20 Hz): we index by frame, and one action is one env step.

---

## M2: Backbone features (cache vs online)

**1. Goal.** Turn each frame (2 cameras + instruction) into per-token SmolVLM2 features from the first 16 LM layers. Decide between caching and online features from measured token counts, storage and speed.
JD skill: using pretrained VLMs / foundation models; ML systems and throughput engineering.

**2. Files**
| Path | What it does |
|---|---|
| `minivla/backbone.py` | Loads SmolVLM2-500M in fp16 and keeps the first 16 LM layers. Builds the input (images first, then instruction) and returns token features `[B, T, 960]`, a mask, and token groups (main / wrist / text). |
| `minivla/cache.py` | Writes/reads the cache: fp16 `np.memmap` for image tokens `[N, T_img, 960]`, text tokens `[N, T_txt, 960]` + mask, and `index.json` (dataset index per row, T values, model revision, dtype). |
| `scripts/check_env_vs_data.py` | Kaggle (LIBERO installed): compares env vs dataset on instruction strings, image orientation (side-by-side PNG) and state layout. |
| `scripts/build_cache.py` | Runs the backbone over all frames in a fixed order, writes the cache, logs throughput. |
| `notebooks/01_features.ipynb` | Kaggle, in this order: install (pinned LeRobot) → clone repo at commit → env check → layer-cut test → cached-vs-online timing → build cache → cached == online test → write `results/m2_features.md`. |
| `tests/test_backbone.py` | Prompt/mask/shape logic on CPU; tests that load the real model are marked `gpu` and run on Kaggle. |
| `results/m2_features.md` | Measured T, storage, s/step (online vs cached), test results, decision. |

**3. Design decisions**
- **Cut the layer list to the first 16 of 32** (verified SmolVLA setting) and apply the LM's final RMSNorm. Reason: half the LM compute and memory, the same features SmolVLA uses, and a stable feature scale. I check the norm placement against `smolvla/smolvlm_with_expert.py` in M2.
- **Images first, then the instruction, with standard causal attention** (plain HF forward pass, no custom masks). Reason: under causal attention, image tokens never see the text, so **one image cache serves A, B and C**. Config B ("empty instruction") is then exactly A with the text tokens removed. SmolVLA uses bidirectional attention over the prefix, so this is a deviation to log.
- **State does not go through the backbone** (it enters the expert in M3). Reason: features then depend only on images + instruction, which makes them cacheable.
- **Inputs:** each 256² image is upscaled to 512², with no image splitting (as in the feasibility test and SmolVLA), giving 64 tokens per image. fp16 compute (the T4 has no fast bf16), fp16 storage, **no pooling** (SPEC).
- **Cache vs online decision rule** (from measurements):
  - **Cache** if storage ≤ 15 GB **and** cache build + 6 × cached training < 6 × online training.
  - *Est.* storage: 52,042 frames × T × 960 × 2 B ≈ **0.1 GB per token**. Feasibility measured 73 tokens for 1 image + a 6-word instruction, so 2 images + instruction gives T ≈ 140–150, i.e. **≈14–15 GB, right at the limit**.
  - **Pre-agreed fallback if > 15 GB:** cache every 2nd frame (≈7 GB). Action-chunk targets still use every action. Logged as a deviation. Not int8: more code and extra error.
  - *Est.* online cost: at 0.357 s/step and batch 16, 1 GPU-hour = ~10k steps = 160k samples ≈ 3.4 epochs per model. Cached training only runs the expert, so it should be much faster; measured in M2.
- **Kaggle storage:** download the dataset to a non-persisted folder. Write the cache to `/kaggle/working`, which Kaggle saves up to 20 GB (checked with `df`), then turn it into a private Kaggle Dataset to attach in later notebooks.

**4. Tests and acceptance checks**
- **Env vs data:** instruction strings identical for all 10 tasks, or the mapping recorded. Image orientation matches (PNG). State layout as decided in M1.
- **Layer cut is right:** the 16-layer model's output equals final_norm(full model `hidden_states[16]`) on 2 samples (max abs diff ≤ 1e-3).
- **Cached == online:** 10 frames chosen with a seed, cache rows vs a fresh forward pass. Same T and mask, max abs diff ≤ 1e-2 (fp16; actual value reported), cosine ≥ 0.9999.
- **B equivalence:** an online pass with an empty instruction equals the cached image tokens (same tolerance).
- **Row alignment:** cache row i holds dataset index i (stored and asserted). No NaN/Inf; per-token norm summary logged.
- **Timing:** s/step for online (batch 16, 16 layers, 2 images) vs cached (batch 16 and 64), mean of 50 steps after 5 warm-up steps.

**5. Where / GPU.** Laptop for the code and CPU tests; Kaggle T4 for everything that loads the model.
*Est.* cache forward pass: 52,042 / 16 × 0.357 s ≈ **19 min**, using the measured step as a stand-in (2 images instead of 1, but 16 layers instead of 32 and no backward). Allow 2× → ≤ 40 min. Install, download, checks and timing add ~25 min. **Allotment: 75 min.**

**6. Risks → fallback**
- Cache > 15 GB → stride-2 cache (above).
- Slow random reads from `/kaggle/input` → copy to local disk or load into RAM (RAM size logged).
- Cached ≠ online → stop and find the cause (padding, batch-size-dependent kernels, dtype); no workaround.
- Env instruction strings differ from the dataset's → cache with the env's strings (eval uses those).
- Making a dataset from the notebook output fails → shard the cache; worst case, online features.

**7. Concepts to explain**
- SmolVLM2 architecture: SigLIP encoder → pixel shuffle (16× fewer tokens) → connector → SmolLM2 language model.
- Why intermediate layers (SmolVLA's half-depth choice: faster, good features).
- Causal vs bidirectional attention and why it decides what can be cached; hidden-state cache vs KV cache.
- fp16 range and precision.
- Storage and throughput arithmetic.
- What a frozen backbone gives up (no image augmentation, no adaptation; `smolvla_libero` fine-tuned its VLM).

---

## M3: Action expert + regression head (config C)

**1. Goal.** Build the action expert and train it with an L1 regression head on cached features. Show that it learns: it overfits a tiny batch and logs val L1.
JD skill: implementing transformer policies from scratch in PyTorch; training infrastructure.

**2. Files**
| Path | What it does |
|---|---|
| `minivla/data.py` | Adds a torch `Dataset`: cache row → features, normalized state, normalized action chunk `[50, 7]` + pad mask. |
| `minivla/expert.py` | `ActionExpert`: backbone tokens → `Linear(960→512)` + learned per-position embedding; state → `Linear(8→512)` as one extra memory token; 6-layer pre-norm `TransformerDecoder` (width 512, 8 heads, FFN 2048; causal self-attention over the 50 action tokens, cross-attention to memory); `Linear(512→7)`. |
| `minivla/heads.py` | `RegressionHead`: 50 learned query embeddings → expert → chunk; masked L1 loss. |
| `minivla/train.py` | Loop: AdamW + warm-up/cosine (SmolVLA preset), fp16 autocast + GradScaler, grad clip 10, fixed step count, val every k steps, `metrics.csv`, final checkpoint (+ a resume checkpoint). |
| `configs/C_l1_lang.yaml`, `configs/runs/C_s{0,1}.yaml` | Config C and its runs. |
| `notebooks/02_train.ipynb` | Kaggle: attach the cache, clone repo at commit, `python -m minivla.train --run configs/runs/<run>.yaml`. |
| `tests/test_expert.py` | Shapes, masks, overfit, determinism, chunk builder. |
| `results/m3_regression.md` | Short-run curves and numbers. |

**3. Design decisions**
- **The expert cross-attends to the final (layer-16) features**, not to SmolVLA's per-layer VLM keys/values. Reason: that's what caching allows, and it's simpler. Width 512 × 6 layers (SPEC) vs SmolVLA's 720 × 16. Deviation logged.
- **State is a memory token** in the expert, not part of the VLM prefix. Reason: the backbone is frozen and cached. Deviation logged.
- **Learned per-position memory embedding.** Reason: the token layout is fixed, so each position always means the same camera patch or text slot. This tells the expert which camera a token came from.
- **Chunk 50 × 7** (verified `smolvla_libero`). Targets are actions t…t+49 of the same episode. Past the episode end, the last action is repeated and **masked out of the loss** (like LeRobot's `action_is_pad`).
- **Causal attention inside the chunk**, as in SmolVLA.
- **Regression queries = 50 learned embeddings** (as in ACT/DETR). Both heads use the same expert, so A vs C isolates the head.
- **L1 in normalized action space** (SPEC). L1 estimates the median: more robust to outliers than L2, but it still averages over distinct modes.
- **Optimizer = SmolVLA preset:** AdamW lr 1e-4, betas (0.9, 0.95), wd 1e-10, clip 10, 1,000 warm-up steps, cosine decay to 2.5e-6 over the run.
- **Batch 64.** Cached features make it cheap; feasibility fit batch 16 *with* the whole backbone in 1.79 GB.
- **Step count S, the same for all 6 models**, fixed at the end of M4 from the measured s/step of both heads: S = min(20,000, steps that fit in 60 min). *Est.* 20,000 steps × 64 ≈ 27 epochs over ~46.8k train frames.
- **Determinism:** `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, seeded DataLoader generator and workers.

**4. Tests and acceptance checks**
- Output shape `[B, 50, 7]`; works with text tokens masked out (the config-B path).
- Masks: changing padded targets doesn't change the loss; padded memory tokens don't change the output.
- Chunk builder: never crosses an episode boundary, and the pad mask is correct at episode ends (toy 3-episode table).
- **Overfit:** 8 samples, ≤ 1,000 steps, train L1 < 0.02 (normalized). Runs on the laptop CPU with random memory and real actions, then again on Kaggle with real features as the notebook's first cell (a gate before the short run).
- **Determinism:** same seed → identical losses after 50 steps (CPU and T4).
- **Short Kaggle run:** 2,000 steps, train/val L1 every 250 steps. Pass if val L1 is below the "always predict the mean chunk" baseline (L1 of zeros in normalized space, logged).

**5. Where / GPU.** Laptop: code, unit tests, CPU overfit. Kaggle: short run. **Allotment: 30 min** (install + attach cache + 2,000 steps + timing).

**6. Risks → fallback**
- Can't overfit → it's a bug (normalization, masks, lr). Stop and report.
- Data loading is the bottleneck → load the cache into RAM.
- A deterministic-mode error from an attention kernel → force PyTorch's math attention backend. Leftover nondeterminism (e.g., laptop vs T4) gets documented.

**7. Concepts to explain**
- Cross-attention (queries from action tokens; keys/values from vision-language tokens), pre-norm transformers, causal and padding masks.
- Action chunking (ACT): fewer decisions means less compounding error and smoother motion, at the cost of open-loop execution.
- L1 vs L2 and mode averaging.
- AdamW, warm-up + cosine, gradient clipping.
- fp16 with loss scaling (and why: no bf16 on T4).
- Overfit-one-batch as a debugging tool; GPU determinism.

---

## M4: Flow-matching head (configs A and B)

**1. Goal.** Add a flow-matching head on the same expert (linear path, velocity prediction, 10 Euler steps at inference) and show that it trains.
JD skill: generative models for control (diffusion / flow-matching policies).

**2. Files**
| Path | What it does |
|---|---|
| `minivla/heads.py` | Adds `FlowMatchingHead`: sample t, build x_t, predict velocity, MSE loss; `sample()` = 10 Euler steps from noise. |
| `minivla/expert.py` | Adds time conditioning: sinusoidal embedding of t (periods 0.004–4.0, as SmolVLA), concatenated with the action embedding → SiLU MLP → queries (SmolVLA's `action_time_mlp`). |
| `configs/A_flow_lang.yaml`, `configs/B_flow_nolang.yaml`, `configs/runs/{A,B}_s{0,1}.yaml` | Configs A and B and their runs. |
| `tests/test_flow.py` | Flow math and overfit tests. |
| `results/m4_flow.md` | Short-run results; fixes S for M5/M6. |

**3. Design decisions**
- **Copy SmolVLA's math exactly** (verified, §0.2): noise at t = 1, x_t = t·ε + (1−t)·a, target u = ε − a, MSE, t ~ Beta(1.5, 1)·0.999 + 0.001 (more weight on noisy t). Reason: fewer deviations to explain, and these settings are known to work on LIBERO.
- **10 Euler steps** (`dt = −0.1`, verified default). Reason: linear paths are close to straight, so few steps suffice. Cost: only the expert runs 10 times, because backbone features are computed once per chunk. An optional cheap offline check reports val L1 at 5/10/20 steps.
- **Validation sampling uses a fixed noise seed.** Val L1 of sampled chunks is then comparable across runs and directly comparable with config C's val L1.
- Loss and time embedding are computed in fp32. Same expert, data and S as C.
- **B = A with the text tokens masked out.** M2's test shows this equals an empty instruction.

**4. Tests and acceptance checks**
- Path endpoints: x_t(0) = a, x_t(1) = ε.
- **Euler with the true velocity** (u = ε − a) turns ε back into a in 10 steps (error ≤ 1e-5). This tests the integrator and the sign convention.
- Time sampler: the mean of 100k samples is within ±0.005 of 0.6004 (Beta(1.5, 1) mean 0.6, scaled); all values in [0.001, 1].
- Shapes. Overfit 8 samples: loss falls, and 10-step samples are within L1 < 0.05 of the targets (normalized).
- Determinism (same test as M3).
- **Short Kaggle run:** 2,000 steps, flow loss + sampled-chunk val L1 logged. Then **S is fixed** for M5/M6 and written into `configs/base.yaml`.

**5. Where / GPU.** Laptop: code and tests. Kaggle: short run. **Allotment: 30 min.**

**6. Risks → fallback**
- Flat loss → check the t convention and time-embedding scale.
- Samples blow up → check the sign of dt.
- Worse val L1 than C is **not** a failure; closed-loop success decides.
- If flow still won't train after debugging → stop and report; we decide together.

**7. Concepts to explain**
- Flow matching vs diffusion (DDPM / score matching): the linear path (rectified flow), conditional vs marginal velocity fields, and why regressing the conditional velocity learns the marginal one.
- ODE integration with Euler, and the step-count trade-off.
- Why Beta time sampling.
- Why a generative head can represent multimodal action distributions while regression averages the modes.
- Sinusoidal time embeddings.

---

## M5: Closed-loop LIBERO eval of config A, seed 0

**1. Goal.** Train A_s0 for the fixed S steps, evaluate its **final** checkpoint closed-loop on all 10 LIBERO-Goal tasks × 5 episodes, measure seconds per episode, and set the final episode budget.
JD skill: simulation-based evaluation of robot policies.

**2. Files**
| Path | What it does |
|---|---|
| `minivla/policy.py` | Inference wrapper: env observation → same preprocessing as training → backbone (online, 16 layers) → expert sample → unnormalize → action queue (execute 50, then replan). |
| `minivla/eval_libero.py` | Eval loop on LeRobot's LIBERO env (pinned commit): fixed init states and seeds; per episode it records success, steps and seconds; writes a CSV and the first k videos. |
| `minivla/stats.py` | Wilson 95 % interval. |
| `configs/eval.yaml` | Suite, task ids, episodes per task, `eval_seed`, `n_action_steps`, number of videos, vectorized-env setting. |
| `notebooks/03_eval.ipynb` | Kaggle: attach the checkpoint, run the eval, write CSV + summary. |
| `tests/test_policy.py` | Train/eval consistency, normalization round-trip, action-queue order, Wilson values. |
| `results/m5_eval_A_s0.md`, `results/eval/A_s0.csv` | Outputs. |

**3. Design decisions**
- **Our own ~100-line eval loop** on LeRobot's env, not wrapping our model as a LeRobot policy for `lerobot-eval`. Reason: far less framework code to explain. We keep `lerobot-eval`'s semantics: same env factory, 300-step limit, success from the env's `check_success`, the env's built-in 10 no-op steps.
- **Execute the full 50-step chunk before replanning** (`n_action_steps = 50`, as `smolvla_libero`). Reason: it matches the reference and caps an episode at 6 inference calls.
- **One fixed eval set for every model:** tasks 0–9, episode e uses init state e, env seeds from `eval_seed`. Reason: paired comparison on identical starts, and it's reproducible.
- **Final checkpoint at step S, and nothing is tuned on eval results.** Fixing a *bug* found in M5 (e.g., a wrong image flip) is not tuning: fix, log, re-run.
- **Measure s/episode in two modes:** 1 env vs 5 vectorized envs (one task's episodes together). Pick one mode and keep it for all models.
- **Episode budget rule** (M6 eval allotment 9 h, §11): with s_ep the measured mean seconds per episode, episodes per task e = ⌊(32,400 / s_ep + 50) / 60⌋, capped at 10. The +50 is A_s0's 50 episodes already run in M5.
  - *Est.* s_ep = 78 → e = 7.
  - e ≥ 5 (SPEC minimum) needs s_ep ≤ 130 s.
  - If e > 5, A_s0 gets the extra init states run later.

**4. Tests and acceptance checks**
- **Train/eval consistency:** a dataset frame sent through the *eval* preprocessing + online backbone gives the same features as its cache row (M2 tolerance). One test catches key mapping, resize and instruction-string bugs.
- Normalize → unnormalize round-trip is exact. The action queue runs actions 0…49 in order, then replans.
- Wilson: k = 5, n = 10 → (0.237, 0.763); k = 0, n = 50 → (0, 0.071) (hand-computed).
- **Eval determinism:** re-running 2 episodes gives the same success and actions (max abs diff logged).
- Run acceptance: 50 CSV rows, 2 videos, s/episode reported (mean, and successes vs failures separately).
- Success rate has no pass threshold. **0 % stops us** for debugging before M6.

**5. Where / GPU.** Laptop: code and unit tests (LIBERO can't run locally while the setup is on hold). Kaggle: training + eval.
*Est.*:
- Training ≤ 60 min.
- Eval: 50 × 78 s = 65 min if every episode costs what the measured success did. Failures run to 300 steps vs the 122-step mean demo (52,042 / 428), so up to ~2.5× → ~160 min.
- Mode timing: 15 min.
- **Allotment: 3.5 h.**

**6. Risks → fallback**
- **0 % success** → checklist: image flip, camera mapping, state layout, gripper sign, normalization, instruction strings, chunk order. Compare predicted chunks on dataset frames vs env frames. Stop and report.
- **Eval too slow** (s_ep > 130 s) → vectorized envs; if still slow, test a Kaggle CPU-only session (`MUJOCO_GL=osmesa`, uses no GPU quota); then the cut list (§12).
- Kaggle session limit → split tasks across runs (per-episode seeds are fixed, so results don't change).

**7. Concepts to explain**
- Closed-loop vs open-loop eval: why low val loss doesn't guarantee success (covariate shift, compounding error).
- LIBERO-Goal's design: one scene, different goals, so language matters.
- Init states and seeding; success predicates.
- Receding-horizon chunk execution.
- What eval time is spent on (physics and rendering vs inference).
- MuJoCo 3.8.1 vs 2.3.7 comparability.

---

## M6: Ablation grid (A/B/C × 2 seeds)

**1. Goal.** Train and evaluate A/B/C × seeds {0, 1} under identical settings. A script builds the results table with Wilson 95 % CIs.
JD skill: experimental design, ablations, statistical reporting.

**2. Files**
| Path | What it does |
|---|---|
| `configs/runs/{A,B,C}_s{0,1}.yaml` | The remaining 5 runs (A_s0 exists). |
| `scripts/make_results_table.py` | `results/eval/*.csv` → `results/ablation_table.md`: per config (seeds pooled), per model, per task; Wilson CIs, n, S, commit hashes. |
| `tests/test_stats.py` | The table script gives known numbers on a toy CSV. |
| `results/eval/*.csv`, `results/ablation_table.md` | Outputs. |

**3. Design decisions**
- Same S, same eval set, final checkpoint for all models. The script **asserts** checkpoint step == S and that the eval-config hash is identical across models.
- Headline CI per config pools both seeds (n = 2 × 10 × e). Per-seed rows show the seed spread.
- **Resolution, stated in the table** (Wilson arithmetic, not results): at n = 100 and p ≈ 0.5 the CI is about ±9.6 points; at n = 50, ±13.4. Differences of ~15–20 points may not be resolvable, and the table says so instead of naming a winner.
- **Run order:** C_s0, B_s0, A_s1, C_s1, B_s1. Reason: one model of each config comes first, so cuts take second seeds.
- **Optional reference row:** `lerobot/smolvla_libero` on the same 50 episodes with our MuJoCo version. Reason: an anchor that sidesteps the MuJoCo 2.3.7 comparability issue. Paid from the reserve; first item on the cut list.
- **Possible saving:** two training runs in one T4×2 session (one per GPU). Used only after a short session confirms T4×2 costs the same quota per hour as a single T4.

**4. Tests and acceptance checks**
- 6 CSVs with the expected row counts.
- Table-script unit test.
- Every number in `ablation_table.md` is generated.
- No checkpoint other than the final one was evaluated (asserted).

**5. Where / GPU.** Kaggle: training + eval. Laptop: table.
*Est.*:
- Training: 5 × ≤ 60 min = **5 h**.
- Eval: 250 episodes × s_ep = 5.4 h at 78 s up to 13.3 h at 192 s. **Allotment: 9 h**; the M5 rule keeps us inside it.
- Optional reference row: 1.1–2.7 h, from the reserve.

**6. Risks → fallback**
- Quota runs short → cut list.
- A session crashes → each model is a separate run, so only that one is re-run.
- Ambiguous result → reported as ambiguous. No extra runs chosen after seeing results.

**7. Concepts to explain**
- What each comparison isolates: A vs B = language, A vs C = action head.
- Seeds vs episodes as sources of variance.
- The Wilson score interval vs the normal approximation, and binomial CI width arithmetic.
- Paired design on shared init states; multiple comparisons.
- Why picking checkpoints on eval results biases them upward.

---

## M7: README

**1. Goal.** A README a reviewer can read in 5 minutes: what was built, results with CIs, 2–3 rollout videos, limitations, next steps.
JD skill: research communication and reproducibility.

**2. Files**
| Path | What it does |
|---|---|
| `README.md` | Overview, setup, how to reproduce, results table, videos, limitations, next steps. |
| `scripts/make_figures.py` | Success-rate bar chart with CIs → `docs/media/` (`results/*.png` is gitignored). |
| `docs/media/*.mp4` or `.gif` | 2–3 rollouts, each < 2 MB. |

**3. Design decisions**
- README numbers are pasted by script from `results/`, not typed.
- **Videos picked by a fixed rule, not by eye:** the first success and first failure of A_s0 in episode order, plus one from B. No cherry-picking.
- **Limitations:** MuJoCo 3.8.1 vs 2.3.7; frozen backbone (vs `smolvla_libero` fine-tuning); small expert; 5 episodes/task; one embodiment; budget-limited training; causal-prefix deviation.
- **Next steps:**
  - Meta-World as a second embodiment.
  - LoRA (measured 3.39 GB, 0.544 s/step).
  - World-model action selection: sample N chunks from the flow head and score them with a learned world model; this links to Project 2.

**4. Tests and acceptance checks.** A script check that README numbers match `ablation_table.md`; links resolve; videos play.

**5. Where / GPU.** Laptop. **0 GPU min** (videos are saved during the M5/M6 evals).

**6. Risks → fallback.** Videos too large → downsampled GIF or trimmed clip.

**7. Concepts to explain**
- Each limitation and how you'd fix it.
- How to add a second embodiment: pad actions/state to a max dimension (SmolVLA uses 32), per-embodiment normalization, optional embodiment token.
- Why a frozen backbone is a fair, cheap baseline.

---

## 8. Project 2 outline: DINO-WM on Wall (days 4–5)

A separate `docs/p2_spec.md` is written first, as SPEC says. This is the skeleton.

- **Goal.** Reproduce DINO-WM's Wall planning result at a reduced budget with the official code and data (`wall_single.zip`, 1.67 GB; OSF link verified in feasibility run 1). Report planning success vs the paper's number, citing the paper's table. No number is filled in here.
- **Steps**
  1. `notebooks/10_dinowm_wall.ipynb`: clone the official repo at a pinned commit, install its dependencies (probably in their own uv venv, as in the feasibility notebook), download the data, and run a few hundred training steps to check the pipeline.
  2. Train at reduced epochs.
  3. Planning eval with the paper's planner (CEM / MPC) on fewer episodes than the paper; success with Wilson CI.
  4. `results/p2_dinowm_wall.md` + deviations.
- **Files.** `docs/p2_spec.md`, `notebooks/10_dinowm_wall.ipynb`, `configs/p2_wall.yaml` (our overrides only), `results/p2_dinowm_wall.md`.
- **GPU.** Allotment 5 h (≈0.5 setup, ≈3 training, ≈1.5 planning eval). Refined in the P2 spec.
- **Risks.** The official dependencies don't install on Kaggle's Python 3.13 → separate venv with the repo's Python version. Training doesn't finish → report the partial result honestly with the training curve.
- **Concepts.**
  - DINOv2 patch features as the world-model state.
  - A ViT predictor with causal attention over time, conditioned on actions.
  - Teacher forcing; frame skip.
  - CEM / MPC planning on latent goal distance.
  - Why planning needs neither a reward model nor a pixel decoder.

---

## 9. Deviations we expect to log (when they happen)

| Expected deviation | Reference | Why |
|---|---|---|
| Backbone fully frozen | `smolvla_libero` fine-tunes the VLM | SPEC; budget |
| Expert attends to final layer-16 features | SmolVLA uses per-layer VLM keys/values | Needed for caching |
| Expert 512 × 6 layers, every layer self + cross | SmolVLA 720 wide, interleaved | SPEC starting size |
| State is an expert memory token | SmolVLA: state token in VLM prefix | Frozen, cached backbone |
| Causal attention over images + text, images first | SmolVLA: bidirectional prefix | One cache serves A/B/C |
| Normalization stats from train split only | LeRobot uses whole-dataset stats | No val leakage |
| `lerobot/libero_goal_image` | `smolvla_libero` trained on `HuggingFaceVLA/libero` | Goal-only, smaller (M1 checks they match) |
| (if needed) Stride-2 cache | SPEC: per-token features for all frames | 15 GB limit |
| Fewer training steps / different batch | SmolVLA presets | GPU budget |

---

## 10. Day-by-day schedule

Day 1 = Thu 2026-11-26. Every gate needs your approval. Kaggle runs are "Save & Run All (Commit)" and run in the background while I write the next code on the laptop.

| Day | Date | Laptop work | Kaggle (planned GPU h) | Gate |
|---|---|---|---|---|
| 1 | Thu 11-26 | Plan approved → light laptop env → **M1** data report, GPU ledger | 0 | M1 |
| 2 | Fri 11-27 | **M2** code + CPU tests; caching decision from the run | `01_features`: ~1.25 | M2 |
| 3 | Sat 11-28 | **M3** then **M4** (code, tests, short runs); fix S; write eval code | `02_train` ×2: ~1.0 | M3, M4 |
| 4 | Sun 11-29 | **M5** (A_s0 train + eval); meanwhile write the P2 spec and P2 notebook | M5 ~3.5; P2 check ~0.5 | M5 → episode budget fixed |
| 5 | Mon 11-30 | **M6** launches (C_s0, B_s0, A_s1, …); **P2** training + planning eval | M6 ~7; P2 ~4.5 | P2 result |
| 6 | Tue 12-01 | M6 remaining runs → table + CIs | M6 ~7 | M6 |
| 7 | Wed 12-02 | **M7** README + videos; buffer | reserve | M7 |

Notes:
- Days 4–6 need ≈7.5 GPU-hours per day of back-to-back runs. If a run is late, the next one moves, never two milestones at once.
- On days 4–5, P2 overlaps M6's background Kaggle runs. If you'd rather keep strictly one thing at a time, P2 moves to days 6–7 and M7 shrinks.

---

## 11. GPU-hour budget (Kaggle T4, weekly quota ~30 h)

| Item | Basis (measured number used) | Estimate (h) | Allotment (h) |
|---|---|---|---|
| Feasibility runs 1–2 | measured: 12.0 + 3.9 min | 0.27 (done) | 0.27 |
| M1 data report | laptop | 0 | 0 |
| M2 features + cache | 52,042/16 × 0.357 s ≈ 19 min forward (×2 margin) + ~25 min overhead | 0.7–1.1 | 1.25 |
| M3 short run | cap | — | 0.5 |
| M4 short run | cap | — | 0.5 |
| M5 train + eval A_s0 | ≤ 60 min train; 50 × 78 s = 65 min … ~160 min if failures run to 300 steps; 15 min timing | 2.3–3.9 | 3.5 |
| M6 training (5 models) | cap 60 min each | ≤ 5.0 | 5.0 |
| M6 eval (250 episodes) | 250 × 78 s … 250 × 192 s | 5.4–13.3 | 9.0 |
| M7 | laptop | 0 | 0 |
| Project 2 | separate spec | — | 5.0 |
| **Subtotal** | | | **25.0** |
| Reserve (re-runs, optional `smolvla_libero` row 1.1–2.7 h) | | | 5.0 |
| **Total** | | | **30.0** |

The biggest unknown is **eval cost per episode**. At the high end, M6 eval alone takes 13 h. M5 measures s_ep before anything is committed, and the rule in M5 sets the episode count to fit the 9 h allotment.

---

## 12. Cut list (cut from the top when we fall behind on GPU or time)

1. Optional `smolvla_libero` reference row.
2. Episodes beyond 5 per task (stay at 5).
3. Optional extras: the 5/10/20-step study (M4) and the CPU-eval experiment (M5).
4. P2 scope: fewer training epochs and a smaller planning eval (still compared honestly to the paper).
5. Second seed of C, then second seed of B. A keeps 2 seeds; the table marks B/C as single-seed.
6. P2 entirely → moves to "next steps", with the data and setup notes we have.
7. Last resort: drop config B or C, choosing with you.

**Never cut:** the fixed step count S, final-checkpoint-only eval, the fixed eval seeds and init states, Wilson CIs, the cached == online test, the GPU ledger, the deviations log. We never go below 5 episodes per task; dropping models comes first.

---

## 13. Decisions (approved 2026-11-26)

1. **Laptop env:** yes, a small Python 3.12 env for M1–M4. The full LeRobot setup is not part of it.
2. **Code on Kaggle:** the GitHub repo is public; Kaggle clones it directly, with no token.
3. **Overlap:** P2 runs on days 4–5 while M6 runs in the background.
4. **Reference row:** `smolvla_libero` row only if the reserve allows; it stays first on the cut list.
5. **Branching:** work is merged into `main` and pushed, so Kaggle always clones `main`. Notebooks still log the exact commit hash.
6. **WSL memory:** raised to 4.8 GB RAM + 16 GB swap (`free -h`).
