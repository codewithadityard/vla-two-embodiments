# Feasibility check (Kaggle)

- Date (UTC): 2026-11-25 19:41
- Seed: 1000, LeRobot commit: `b07adcc5e3ce42c26a248236dbb3571f84c28c43`
- Steps run: ['step3_libero', 'step5_lora']
- Total notebook time: 3.9 min (budget 30 min: OK)

| Step | Status | Numbers |
|---|---|---|
| 1. Install | PASS | python_used=/usr/bin/python3, driver=580.178.04
580.178.04, torchao_uninstalled=True, install_minutes=0.8 |
| 2. Versions/imports | SKIPPED |  |
| 3. LIBERO eval | PASS | pc_success=100.0, n_episodes=1, eval_s=78.09466361999512, wall_time_s=150.7 |
| 4. Meta-World | SKIPPED |  |
| 5a. Memory: frozen backbone | SKIPPED |  |
| 5b. Memory: LoRA | PASS | setup=lora, trainable_params_M=29.03, backbone_tokens_per_sample=73, peak_gpu_allocated_GB=3.39, peak_gpu_reserved_GB=3.71, peak_ram_GB=3.54, sec_per_step=0.544, final_loss=1.0311 |
| 6. DINO-WM Wall | SKIPPED |  |

## Deviations
- None.

## Notes
- Step 3 PASS means the eval ran end to end; with 1 episode, pc_success is 0 or 100.
- Step 5 sec/step averages steps after the warm-up steps in config.
- Step 5 peak RAM is per process (ru_maxrss).
