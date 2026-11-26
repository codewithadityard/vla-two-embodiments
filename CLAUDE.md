# Project rules
- This is a robot-learning research project (VLA + world models) on a laptop: Windows + WSL2 Ubuntu, RTX 4060 8 GB. Long runs go to Kaggle (Linux, T4/P100).
- Before writing code for any task, propose a short plan and wait for my approval.
- Work one milestone at a time. Don't start the next until its checks pass and I approve.
- Explain each command before running it. If a command needs sudo, tell me and I'll run it myself.
- Seed all randomness from config files. Same seed must give the same result.
- NEVER invent or estimate results. Every number must come from a logged run.
- Log deviations from papers or plans in docs/deviations.md.
- Keep code small and readable. I must be able to explain every line.
- If something fails, stop and report it. Don't silently work around it.
- Never commit data/, checkpoints/, outputs/ or mlruns/.

# Laptop vs Kaggle (from 2026-11-26)
- The laptop is on mobile data with 8 GB RAM. It must not download anything large: no dataset images, no model weights, no CUDA torch, no LeRobot. Laptop work is code editing and git only (plus the tiny conda env "vla" with numpy/pandas/pyarrow/pytest for non-torch tests).
- All data, models, feature caches, training and evaluation run on Kaggle. Only small result files (md, json, small png) come back to the repo.
- Any test that needs torch runs on Kaggle: a CPU session if no GPU is needed (saves GPU quota), otherwise GPU.
- Kaggle runs use notebooks/runner.ipynb: it clones main at a given commit, installs pinned packages, runs one script, and prints a summary to paste back. Push to main before each run.
