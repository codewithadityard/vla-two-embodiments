# Deviations

Deviations from papers or plans. One entry per deviation.

| Date | What | Plan / paper | Actual | Impact |
|---|---|---|---|---|
| 2026-11-25 | Python version | Laptop setup plan targets Python 3.12 (LeRobot docs) | Kaggle runs Python 3.13 (feasibility runs 1 and 2) | Allowed by LeRobot (>=3.12); the two environments differ |
| 2026-11-26 | Backbone prefix attention | SmolVLA: images + language + state in one prefix with bidirectional attention | Images first, then instruction, standard causal attention (plain HF forward); state not in the backbone | Image features don't depend on the instruction, so one cache serves configs A, B and C (verified in M2 probe); language reaches the expert only through the instruction tokens |
| 2026-11-26 | Backbone training | `lerobot/smolvla_libero` fine-tunes the VLM (`train_expert_only: false`) | Backbone fully frozen; features cached once (fp16, 16 layers) | Much cheaper (86 vs ~8 GPU-h by the M2 probe arithmetic); no image augmentation possible |
| 2026-11-26 | Image preprocessing | SmolVLA: own resize-with-pad to 512 and scaling to [-1, 1] | HF SmolVLM processor (no image splitting): 256 -> 512, its own resampling and normalization | Same 64 tokens per image; pixel values may differ slightly from SmolVLA's path; the same processor is used for cache and eval |

## Two runs at once on Kaggle's T4 x2 (2026-11-26)
- Kaggle's T4 session has two GPUs. Runs go in pairs (`scripts/run_pair.py`), each on its own GPU via CUDA_VISIBLE_DEVICES, sharing one copy of the cache in the OS page cache (two private copies would not fit in RAM). The step count S is measured with two runs going at once (M3+M4 rerun). Seeds, configs and the eval protocol are unchanged.
- Fixes after the first M3+M4 run: the cache was never read into RAM (`np.ascontiguousarray` returns the memmap without copying), so early steps read from disk and the measured s/step (and S) was wrong; the cache is now read once sequentially (page cache) before training. The real-data overfit check now uses the training's mixed-precision mechanics (fp16 autocast + GradScaler + grad clipping); thresholds unchanged.
- Random-data unit tests (`tests/test_expert.py`, approved 2026-11-26): the fixed L1 thresholds (regression < 0.02 after 1000 steps, flow < 0.05 after 3000 steps) are replaced by a relative check: the training loss (mean of the last 100 steps) must be at most 10 % of the first step's loss. Reason: the fixed values were guesses, missed narrowly in fp32 (0.0210, 0.082), and do not detect bugs better than a relative drop. Real-data overfit checks keep their approved thresholds. Launching A_s0 and C_s0 was approved on the condition that only these two tests fail in the M3+M4 rerun, their 3x-step versions pass, and every real-data check passes.
