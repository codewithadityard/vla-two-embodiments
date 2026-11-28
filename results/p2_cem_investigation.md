# P2: why plain CEM got 14/50 instead of the paper's 0.74 (CPU-only investigation)

No GPU was used. Everything below was compared by reading three sources:
- the paper, arXiv 2411.04983v2: Table 8, Sec. 3.2, 4.1 and 4.3, and Appendix A.5 and A.8–A.9, Tables 10–12;
- the official repo `gaoyuezhou/dino_wm` @ `0a9492f`: `plan.py`, `planning/{cem,mpc,evaluator,objectives}.py`, `conf/plan.yaml`, `conf/plan_wall.yaml`, `conf/planner/{cem,mpc_cem}.yaml`, `env/wall/wall_env_wrapper.py`, README;
- our runs `vla-p2-r3cem` and `vla-p2-r3` (logs; [p2_r3_cem.md](p2_r3_cem.md), [p2_r3.md](p2_r3.md)).

Paper quotes come from a verbatim fetch of the arXiv HTML.

**Result: 14/50 (28%, Wilson 95 % CI 17.5–41.7). The paper reports 0.74.**

## Line-by-line comparison

| Item | Paper | Official repo | What we ran | Same? |
|---|---|---|---|---|
| CEM setting | A.5.3: "CEM denotes the setting where we use CEM to optimize a sequence of actions, and execute those actions in the environment without any correction or replan." | `planning.cem.CEMPlanner`: one plan; `plan.py` executes all of it; `action_len = inf` | `CEMPlanner`, one plan executed in full | yes |
| Samples per iteration | not stated (A.8 times CEM with 100 samples × 10 steps; Table 10: timing only) | `cem.yaml`: `num_samples: 300` | 300 | as repo |
| Top-K (elites) | not stated (A.5.1(e): "top K") | `topk: 30` | 30 | as repo |
| Optimisation steps | not stated (A.5.1(f): "a fixed number of iterations that we set as hyperparameter"; A.8 timing: 10) | `cem.yaml`: `opt_steps: 30`; README example: `planner.opt_steps=30` | 30 | as repo |
| Initial sampling std | not stated (A.5.1(c): "Gaussian") | `var_scale: 1` (normalised actions) | 1 | as repo |
| Planning horizon | not stated (T is generic) | `cem.yaml` `horizon: 5`, overwritten by `plan.py` with `goal_H` | 5 model steps | as repo |
| Frameskip at planning | not stated (Table 11: 5 is the training value) | the checkpoint's `frameskip` (5) | 5 | as repo |
| Evaluation horizon | not stated | open loop: horizon × frameskip = 25 env steps | 25 env steps | as repo |
| Goal source | not stated ("randomly sampled goal state", Sec. 4.1) | `plan_wall.yaml`: `goal_source: 'random_state'`; README example: `goal_source='random_state'`; `plan.yaml` default: `'dset'` | `random_state` | as `plan_wall.yaml` and README |
| Goal distance (`goal_H`) | not stated | `plan_wall.yaml` and README: 5 | 5 | as repo |
| Objective | Eq. 4: MSE at the final step; no proprio weight stated | `objective: alpha 1, mode last` | same | as repo |
| Success rule | not stated (no threshold, no "final vs. any step") | `eval_state`: `‖goal_xy − final_xy‖ < 4.5` at the final executed step | same | as repo |
| Pairs and seed | Sec. 4.3: "we sample 50 initial and goal states" | `plan_wall.yaml`: `n_evals: 50`, `seed: 99` | 50, seed 99 | yes |
| Checkpoint | Table 12: 100 epochs | released Wall checkpoint: epoch 65 | the released checkpoint (epoch 65) | **differs from the paper** |
| Decoder | – | used only for plots (`evaluator.py`) | off | no effect on actions or success |

Our run used the settings above. Two pieces of evidence from the logs:
- The run evaluates after every CEM step (`eval_every: 1`), giving 30 per-step results plus the final one.
- Its first 10 per-step success rates (0.0, 0.02, 0.24, 0.48, 0.46, 0.44, 0.48, 0.52, 0.58, 0.54) are identical to MPC-CEM's first iteration. That planner uses the same CEM with 300 samples, top-30 and horizon 5, so the same samples and pairs were drawn.

## Findings, ranked by how likely they explain the gap

1. **Goal reachability in a 25-step open-loop plan (most likely; cannot be checked against the paper).**
   - `random_state` samples start and goal independently anywhere in the two rooms, and open-loop CEM executes only 25 env steps.
   - In our MPC-CEM run, 27 of 50 pairs succeeded within the first 25 steps. 19 more needed 50 or 75 steps: 17 at iteration 2 and 2 at iteration 3 (cumulative 27, 44, 46, 46 after iterations 1–4).
   - So for about 4 in 10 pairs, the goal was not reached in 25 steps even with feedback.
   - `plan.yaml`'s default `goal_source: 'dset'` instead takes the goal from 5 model steps along a real trajectory, which is reachable in 25 steps by construction. If Table 8 used dataset goals or a longer horizon, it would score higher.
   - The paper does not say which goal source or horizon it used. The repo (`plan_wall.yaml`, README) says `random_state` with `goal_H=5`, which is what we ran.
2. **More optimisation steps lowered real success (measured; not enough alone).**
   - Per-step success peaked at 0.58 (step 9) and fell to 0.28 by step 30. That fits CEM narrowing onto plans the model likes but the simulator doesn't, though we did not log the objective to confirm it.
   - The paper's only stated CEM budget is the timing setup (A.8: 100 samples × 10 steps).
   - Even at step 10 (27/50, CI 40.4–67.0) or at the peak (29/50, CI 44.2–70.6), 0.74 is outside the CI. The step count alone does not explain the gap.
   - We report step 30, the repo's setting. Picking a step by its eval result is not allowed.
3. **Checkpoint: epoch 65 released vs. 100 epochs in Table 12 (a concrete difference, but not fixable).** Open-loop CEM cannot correct model errors and MPC can, so a less-trained model would hurt CEM more than MPC. This fits our MPC-CEM matching the paper (92 % vs. 0.96) while CEM does not. Testing it needs another checkpoint, and our own training was dropped (option c).
4. **Unknown sample count and top-K for Table 8.** The paper states neither, and the 100-sample figure is a timing setup. The repo's 300/30 is what we ran.
5. **Success counted at any step vs. the final step.** The code counts only the final executed state, and the paper does not say. An open-loop plan can pass through the goal and leave it. Our logs keep only the final state, so this cannot be checked without a rerun.
6. **Sampling noise: ruled out.** With 50 pairs, the CI 17.5–41.7 is far below 0.74.

## Conclusion

The paper gives none of the settings that would explain the gap: goal source, horizon, sample count, step count, or success rule. On every setting the repo specifies, our run matches `plan_wall.yaml`, `cem.yaml` and the README example.

The only concrete difference is the checkpoint: the released epoch 65 against the paper's 100 epochs. A rerun with the released checkpoint cannot change that.

**No rerun is proposed.** Changing the goal source, horizon or step count would be our guess at the paper's settings, not a reproduction, and choosing them after seeing results would be tuning on eval.
