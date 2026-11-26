# M2 probe: backbone features, env vs data, cache decision

Generated on Kaggle by `python -m scripts.m2_probe` (via `notebooks/runner.ipynb`). Every number is measured in this run.

- GPU: Tesla T4; torch 2.11.0+cu130; backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `4cde9d208e19`, first 16 LM layers; seed 0
- Env observation keys (after `canonical_observation`): ['observation.images.image', 'observation.images.image2', 'observation.state']

## Checks

| Check | Result | Detail |
|---|---|---|
| env instruction == dataset instruction (all tasks) | **FAIL** |  |
| env images (after LeRobot's flip) match dataset orientation (all tasks, both cameras) | PASS |  |
| env reset state within dataset first-frame range (all tasks, all dims) | PASS | per dim: |env - mean| <= max(3 std, 0.01) over the task's first frames |
| 128 image tokens per frame (2 x 64) | PASS | [128, 128, 128, 128, 128, 128, 128, 128, 128, 128] |
| prompt tokens before the instruction are identical to the empty-instruction prompt | PASS | prefix length 134 |
| full model has 32 LM layers; we keep the first 16 | PASS | full 32, kept 16 |
| 16-layer output == final_norm(full hidden_states[16]) (max abs <= 1e-3) | PASS | max abs 0.00e+00, min cos 1.000000 |
| prefix features with real == with empty instruction (max abs <= 1e-2, cos >= 0.9999) | **FAIL** | max abs 6.25e-02, min cos 0.999992 |

## Env vs data (init state 0, frame 0 of the first episode of each task)

Correlations are between grayscale images; orientation passes if the as-is correlation beats the 180°-flipped one. State ratio = max over dims of |env - mean| / max(3 std, 0.01); must be <= 1. Side-by-side images: `outputs/m2_probe/task*_env_vs_data.png` (env main | data main | env wrist | data wrist).

| Task | Instruction match | Main corr | Main corr (flipped) | Wrist corr | Wrist corr (flipped) | State ratio |
|---|---|---|---|---|---|---|
| 0 | NO: env says 'open the middle drawer of the cabinet' | 0.9405 | -0.2402 | 0.8201 | -0.219 | 0.5832 |
| 1 | NO: env says 'put the bowl on the stove' | 0.9289 | -0.2426 | 0.7877 | -0.06629 | 0.8271 |
| 2 | NO: env says 'put the wine bottle on top of the cabinet' | 0.8926 | -0.237 | 0.8331 | -0.01242 | 0.7192 |
| 3 | NO: env says 'open the top drawer and put the bowl inside' | 0.9151 | -0.2358 | 0.792 | -0.03666 | 0.5242 |
| 4 | NO: env says 'put the bowl on top of the cabinet' | 0.9157 | -0.2269 | 0.8534 | -0.006437 | 0.7615 |
| 5 | yes | 0.9389 | -0.2394 | 0.8585 | -0.08207 | 0.5491 |
| 6 | NO: env says 'put the cream cheese in the bowl' | 0.9069 | -0.2362 | 0.7086 | -0.1073 | 0.7729 |
| 7 | NO: env says 'turn on the stove' | 0.9197 | -0.2332 | 0.8546 | -0.0645 | 0.6925 |
| 8 | NO: env says 'put the bowl on the plate' | 0.938 | -0.234 | 0.8909 | -0.1061 | 0.3153 |
| 9 | NO: env says 'put the wine bottle on the rack' | 0.9071 | -0.2376 | 0.8047 | -0.133 | 0.4319 |

## Token layout

| Item | Value |
|---|---|
| Prefix tokens P (BOS, image wrappers, 2 x 64 image tokens) | 134 |
| Instruction tokens, min / max over the 10 tasks | 4 / 9 |
| Tokens per frame T = P + max instruction | 143 |
| Feature dim | 960 |

## Storage (fp16, all T tokens)

| Frames cached | GB | Within 15 GB? |
|---|---|---|
| all 52042 | 14.29 | yes |
| every 2nd (26021) | 7.144 | yes |

## Timing on Tesla T4 (mean of 50 steps after 5 warm-up unless noted)

| Measurement | Seconds |
|---|---|
| Tokenize/preprocess batch 32 (CPU, one process; 10 calls) | 9.728 |
| Backbone forward batch 32 (16 layers, no grad; 20 calls) | 1.178 |
| Online training step, batch 16 (backbone + stand-in expert fwd/bwd) | 0.6458 |
| Cached training step, batch 16 (stand-in expert only, memory on GPU) | 0.0325 |
| Cached training step, batch 64 | 0.08661 |

Peak GPU memory allocated: 2.24 GB. The stand-in expert has the planned size (6 layers, width 512) but is not the M3 expert. Cached timings exclude reading features from disk.

## Decision

Planning arithmetic from the timings above, for 6 models x 1,280,000 training samples (plan's cap: 20,000 steps x batch 64):

| Option | GPU hours |
|---|---|
| Online: 6 models | 86.11 |
| Cache all frames: build + 6 models | 7.814 |
| Cache every 2nd frame: build + 6 models | 5.35 |

**Rule result: CACHE all 52042 frames (14.3 GB).** Needs your approval before R2.
