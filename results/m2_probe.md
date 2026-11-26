# M2 probe: backbone features, env vs data, cache decision

Generated on Kaggle by `python -m scripts.m2_probe` (via `notebooks/runner.ipynb`). Every number is measured in this run.

- GPU: Tesla T4; torch 2.11.0+cu130; backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `4cde9d208e19`, first 16 LM layers; seed 0
- Env observation keys (after `canonical_observation`): ['observation.images.image', 'observation.images.image2', 'observation.state']

## Checks

| Check | Result | Detail |
|---|---|---|
| env instructions == dataset instructions as a set (exact strings) | PASS | only in env: []; only in data: [] |
| env images (after LeRobot's flip) match dataset orientation (all tasks, both cameras) | PASS |  |
| env reset state within dataset first-frame range (all tasks, all dims) | PASS | per dim: |env - mean| <= max(3 std, 0.01) over the task's first frames |
| 128 image tokens per frame (2 x 64) | PASS | [128, 128, 128, 128, 128, 128, 128, 128, 128, 128] |
| prompt tokens before the instruction are identical to the empty-instruction prompt | PASS | prefix length 134 |
| full model has 32 LM layers; we keep the first 16 | PASS | full 32, kept 16 |
| 16-layer output == final_norm(full hidden_states[16]) (max abs <= 1e-3) | PASS | max abs 0.00e+00, min cos 1.000000 |
| fp16: prefix features with real vs empty instruction, min cos >= 0.9999 | PASS | max abs 6.25e-02 (largest |feature| 24.8; relative 2.5e-03), min cos 0.999992 |
| fp32: prefix features with real vs empty instruction, max abs <= 1e-4 (2 frames) | PASS | max abs 2.03e-05, min cos 0.99999988 |

## Env vs data (init state 0, frame 0 of the first episode of each task)

Dataset and env number the tasks differently; each dataset task is paired with the env task that has the same instruction text. Correlations are between grayscale images; orientation passes if the as-is correlation beats the 180°-flipped one. State ratio = max over dims of |env - mean| / max(3 std, 0.01); must be <= 1. Side-by-side images: `outputs/m2_probe/task*_env_vs_data.png` (env main | data main | env wrist | data wrist).

| Data task | Env task | Instruction | Main corr | Main corr (flipped) | Wrist corr | Wrist corr (flipped) | State ratio |
|---|---|---|---|---|---|---|---|
| 0 | 8 | put the bowl on the plate | 0.9203 | -0.2414 | 0.8312 | -0.1814 | 0.3266 |
| 1 | 9 | put the wine bottle on the rack | 0.9135 | -0.2449 | 0.8318 | -0.08492 | 0.5709 |
| 2 | 3 | open the top drawer and put the bowl inside | 0.903 | -0.2375 | 0.897 | 0.003853 | 0.5203 |
| 3 | 6 | put the cream cheese in the bowl | 0.9144 | -0.242 | 0.8048 | -0.04458 | 0.7413 |
| 4 | 2 | put the wine bottle on top of the cabinet | 0.9223 | -0.229 | 0.9133 | -0.01565 | 0.6925 |
| 5 | 5 | push the plate to the front of the stove | 0.9389 | -0.2394 | 0.8585 | -0.08207 | 0.5491 |
| 6 | 7 | turn on the stove | 0.9273 | -0.2339 | 0.7876 | -0.1589 | 0.7813 |
| 7 | 1 | put the bowl on the stove | 0.9309 | -0.2348 | 0.8509 | -0.04492 | 0.8008 |
| 8 | 4 | put the bowl on top of the cabinet | 0.9042 | -0.2271 | 0.7694 | -0.01521 | 0.7651 |
| 9 | 0 | open the middle drawer of the cabinet | 0.9219 | -0.2356 | 0.8424 | -0.191 | 0.655 |

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
| Tokenize/preprocess batch 32 (CPU, one process; 10 calls) | 9.873 |
| Backbone forward batch 32 (16 layers, no grad; 20 calls) | 1.106 |
| Online training step, batch 16 (backbone + stand-in expert fwd/bwd) | 0.5977 |
| Cached training step, batch 16 (stand-in expert only, memory on GPU) | 0.03091 |
| Cached training step, batch 64 | 0.08222 |

Peak GPU memory allocated: 2.24 GB. The stand-in expert has the planned size (6 layers, width 512) but is not the M3 expert. Cached timings exclude reading features from disk.

## Decision

Planning arithmetic from the timings above, for 6 models x 1,280,000 training samples (plan's cap: 20,000 steps x batch 64):

| Option | GPU hours |
|---|---|
| Online: 6 models | 79.7 |
| Cache all frames: build + 6 models | 7.7 |
| Cache every 2nd frame: build + 6 models | 5.22 |

**Rule result: CACHE all 52042 frames (14.3 GB).** Needs your approval before R2.
