# Deviations

Deviations from papers or plans. One entry per deviation.

| Date | What | Plan / paper | Actual | Impact |
|---|---|---|---|---|
| 2026-11-25 | Python version | Laptop setup plan targets Python 3.12 (LeRobot docs) | Kaggle runs Python 3.13 (feasibility runs 1 and 2) | Allowed by LeRobot (>=3.12); the two environments differ |
| 2026-11-26 | Backbone prefix attention | SmolVLA: images + language + state in one prefix with bidirectional attention | Images first, then instruction, standard causal attention (plain HF forward); state not in the backbone | Image features don't depend on the instruction, so one cache serves configs A, B and C (verified in M2 probe); language reaches the expert only through the instruction tokens |
| 2026-11-26 | Backbone training | `lerobot/smolvla_libero` fine-tunes the VLM (`train_expert_only: false`) | Backbone fully frozen; features cached once (fp16, 16 layers) | Much cheaper (86 vs ~8 GPU-h by the M2 probe arithmetic); no image augmentation possible |
| 2026-11-26 | Image preprocessing | SmolVLA: own resize-with-pad to 512 and scaling to [-1, 1] | HF SmolVLM processor (no image splitting): 256 -> 512, its own resampling and normalization | Same 64 tokens per image; pixel values may differ slightly from SmolVLA's path; the same processor is used for cache and eval |
