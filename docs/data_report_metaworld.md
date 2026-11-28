# Data report: Meta-World MT50 dataset (P3-0a)

Generated on Kaggle (CPU, no install step) by `python -m scripts.data_report_mw`. Every number below is computed from the dataset's files; nothing is typed by hand. No simulator, Meta-World or LeRobot code was used: the MT10 task list, instruction -> env mapping, image orientation, env vs. dataset state, the MT10 split and its stats come in P3-0b (first GPU run).

## Source

| Item | Value |
|---|---|
| Dataset | `lerobot/metaworld_mt50` |
| Hub revision | `b46a1c185a2368660650a82ac5a5083a56722f48` |
| LeRobot format | v3.0 |
| Robot | metaworld |
| How it was read | non-image columns of every file via `HfFileSystem` + pyarrow column selection; images only for episode 0 |
| Versions | numpy 2.1.3, pyarrow 23.0.1, pandas 2.3.3, pillow 13.0.0, huggingface_hub 1.33.0 |
| Code commit | `492704e` |
| Config | `configs/metaworld.yaml` |

## Checks

| Check | Result | Detail |
|---|---|---|
| episodes == info.json total_episodes | PASS | 2500 vs 2500 |
| frames == info.json total_frames | PASS | 204806 vs 204806 |
| tasks == info.json total_tasks | PASS | 50 in tasks.parquet, 50 used, 50 in info.json |
| global index is 0..N-1 | PASS |  |
| frame_index is 0..len-1 in every episode | PASS |  |
| no NaN/Inf in state, action, timestamp | PASS |  |
| timestamp == frame_index / fps | PASS | max abs diff 1.91e-07 s |
| episode lengths match meta/episodes | PASS |  |
| state dim == info.json observation.state shape | PASS | 4 vs 4 |
| action dim == info.json action shape | PASS | 4 vs 4 |
| extra columns cover the same frames (global index 0..N-1) | PASS |  |
| each task_index has exactly one task_id | PASS | 50/50 |
| sample holds exactly the first episode | PASS |  |
| sample state/action == column-only read for the same frames | PASS |  |
| exactly one camera | PASS | ['observation.image'] |
| every decoded observation.image frame of episode 0 is HxWxC == info.json CxHxW | PASS | decoded [(480, 480, 3)], info [3, 480, 480] |
| no metaworld / LeRobot / MuJoCo / OpenGL / gymnasium module imported | PASS | [] |

## Counts (whole dataset)

| Item | Value |
|---|---|
| Episodes | 2500 |
| Frames | 204806 |
| Tasks | 50 |
| Episode length min / mean / median / max (frames) | 14 / 81.9 / 72.0 / 500 |
| fps (metadata) | 80 |
| Mean episode duration at that fps (s) | 1.02 |
| Episodes with `next.success` true at the last frame | 2465/2500 |
| Episodes with `next.success` true at any frame | 2465/2500 |

## Tasks (all 50; MT10 is selected in P3-0b)

| Task index | task_id | Instruction (tasks.parquet) | Episodes | Frames | Length min / mean / max | Success at last frame |
|---|---|---|---|---|---|---|
| 0 | 0 | Pick up a nut and place it onto a peg | 50 | 4469 | 83 / 89.4 / 99 | 50 |
| 1 | 1 | Dunk the basketball into the basket | 50 | 5495 | 81 / 109.9 / 500 | 48 |
| 2 | 2 | Grasp the puck from one bin and place it into another bin | 50 | 6012 | 91 / 120.2 / 247 | 50 |
| 3 | 3 | Grasp the cover and close the box with it | 50 | 8511 | 90 / 170.2 / 500 | 43 |
| 4 | 4 | Press a button from the top | 50 | 3295 | 58 / 65.9 / 76 | 50 |
| 5 | 5 | Bypass a wall and press a button from the top | 50 | 2908 | 52 / 58.2 / 65 | 50 |
| 6 | 6 | Press a button | 50 | 3110 | 58 / 62.2 / 70 | 50 |
| 7 | 7 | Bypass a wall and press a button | 50 | 3690 | 59 / 73.8 / 500 | 49 |
| 8 | 8 | Push a button on the coffee machine | 50 | 1812 | 28 / 36.2 / 43 | 50 |
| 9 | 9 | Pull a mug from a coffee machine | 50 | 3965 | 70 / 79.3 / 87 | 50 |
| 10 | 10 | Push a mug under a coffee machine | 50 | 2747 | 50 / 54.9 / 59 | 50 |
| 11 | 11 | Rotate a dial 180 degrees | 50 | 3592 | 64 / 71.8 / 110 | 50 |
| 12 | 12 | Pick a nut out of a peg | 50 | 6736 | 62 / 134.7 / 228 | 50 |
| 13 | 13 | Close a door with a revolving joint | 50 | 3136 | 53 / 62.7 / 73 | 50 |
| 14 | 14 | Lock the door by rotating the lock clockwise | 50 | 4139 | 56 / 82.8 / 127 | 50 |
| 15 | 15 | Open a door with a revolving joint | 50 | 4988 | 71 / 99.8 / 500 | 48 |
| 16 | 16 | Unlock the door by rotating the lock counter-clockwise | 50 | 2116 | 37 / 42.3 / 61 | 50 |
| 17 | 22 | Insert the gripper into a hole | 50 | 3030 | 51 / 60.6 / 70 | 50 |
| 18 | 17 | Push and close a drawer | 50 | 3923 | 78 / 78.5 / 79 | 50 |
| 19 | 18 | Open a drawer | 50 | 4443 | 86 / 88.9 / 93 | 50 |
| 20 | 20 | Rotate the faucet counter-clockwise | 50 | 2956 | 55 / 59.1 / 64 | 50 |
| 21 | 19 | Rotate the faucet clockwise | 50 | 3264 | 62 / 65.3 / 68 | 50 |
| 22 | 21 | Hammer a screw on the wall | 50 | 3265 | 57 / 65.3 / 79 | 50 |
| 23 | 23 | Press a handle down sideways | 50 | 2059 | 28 / 41.2 / 51 | 50 |
| 24 | 24 | Press a handle down | 50 | 1485 | 14 / 29.7 / 46 | 50 |
| 25 | 25 | Pull a handle up sideways | 50 | 3999 | 65 / 80.0 / 118 | 50 |
| 26 | 26 | Pull a handle up | 50 | 6134 | 92 / 122.7 / 147 | 50 |
| 27 | 27 | Pull a lever down 90 degrees | 50 | 3901 | 63 / 78.0 / 114 | 50 |
| 28 | 32 | Pick a puck, bypass a wall and place the puck | 50 | 4132 | 71 / 82.6 / 116 | 50 |
| 29 | 30 | Pick up a puck from a hole | 50 | 5869 | 105 / 117.4 / 127 | 50 |
| 30 | 31 | Pick and place a puck to a goal | 50 | 2656 | 46 / 53.1 / 59 | 50 |
| 31 | 36 | Slide a plate into a cabinet | 50 | 2555 | 48 / 51.1 / 54 | 50 |
| 32 | 35 | Slide a plate into a cabinet sideways | 50 | 2384 | 45 / 47.7 / 54 | 50 |
| 33 | 34 | Get a plate from the cabinet | 50 | 3260 | 42 / 65.2 / 85 | 50 |
| 34 | 33 | Get a plate from the cabinet sideways | 50 | 2289 | 36 / 45.8 / 55 | 50 |
| 35 | 28 | Insert a peg sideways | 50 | 8332 | 58 / 166.6 / 500 | 42 |
| 36 | 29 | Unplug a peg sideways | 50 | 6338 | 89 / 126.8 / 500 | 49 |
| 37 | 43 | Kick a soccer into the goal | 50 | 5308 | 38 / 106.2 / 500 | 46 |
| 38 | 45 | Grasp a stick and push a box using the stick | 50 | 3766 | 68 / 75.3 / 82 | 50 |
| 39 | 44 | Grasp a stick and pull a box with the stick | 50 | 5469 | 80 / 109.4 / 500 | 48 |
| 40 | 38 | Push the puck to a goal | 50 | 3521 | 52 / 70.4 / 500 | 49 |
| 41 | 39 | Bypass a wall and push a puck to a goal | 50 | 4047 | 64 / 80.9 / 95 | 50 |
| 42 | 40 | Reach a goal position | 50 | 2351 | 31 / 47.0 / 77 | 50 |
| 43 | 41 | Bypass a wall and reach a goal | 50 | 2316 | 35 / 46.3 / 70 | 50 |
| 44 | 42 | Pick and place a puck onto a shelf | 50 | 4925 | 69 / 98.5 / 385 | 50 |
| 45 | 46 | Sweep a puck into a hole | 50 | 4440 | 41 / 88.8 / 500 | 46 |
| 46 | 47 | Sweep a puck off the table | 50 | 4378 | 68 / 87.6 / 109 | 50 |
| 47 | 48 | Push and open a window | 50 | 4358 | 77 / 87.2 / 96 | 50 |
| 48 | 49 | Push and close a window | 50 | 4044 | 76 / 80.9 / 88 | 50 |
| 49 | 37 | Push the puck back toward the robot to the goal | 50 | 8888 | 71 / 177.8 / 500 | 47 |

## Cameras

| Key | Stored as | Encoding | Mode | Decoded shape (H x W x C) | Frames decoded (ep 0) | Sample frame (not committed) |
|---|---|---|---|---|---|---|
| `observation.image` | image | PNG | RGB | 480x480x3 | 88 | `outputs/p3_0a/observation.image.png` |

## Other features (info.json)

| Key | dtype | shape | names |
|---|---|---|---|
| `observation.state` | float32 | [4] | {'axes': None} |
| `action` | float32 | [4] | {'axes': ['x', 'y', 'z', 'gripper']} |
| `next.reward` | float32 | [1] | None |
| `next.success` | bool | [1] | None |
| `observation.environment_state` | float32 | [39] | ['keypoints'] |
| `task_id` | int16 | [1] | None |
| `timestamp` | float32 | [1] | None |
| `frame_index` | int64 | [1] | None |
| `episode_index` | int64 | [1] | None |
| `index` | int64 | [1] | None |
| `task_index` | int64 | [1] | None |

## State (`observation.state`, 4-D): all frames, for information

Normalization stats for training come from the MT10 train split (P3-0b), not from this table.

| Dim | Name (metadata) | Mean | Std | Min | Max | P1 | P99 |
|---|---|---|---|---|---|---|---|
| 0 |  | 0.0140 | 0.1122 | -0.5055 | 0.4869 | -0.3676 | 0.3889 |
| 1 |  | 0.6404 | 0.1033 | 0.3866 | 0.9038 | 0.4016 | 0.8863 |
| 2 |  | 0.1622 | 0.0873 | 0.0434 | 0.5082 | 0.0466 | 0.4072 |
| 3 |  | 0.6139 | 0.2998 | 0.2563 | 1.0000 | 0.2763 | 1.0000 |

## Action (`action`, 4-D): all frames, for information

Normalization stats for training come from the MT10 train split (P3-0b), not from this table.

| Dim | Name (metadata) | Mean | Std | Min | Max | P1 | P99 |
|---|---|---|---|---|---|---|---|
| 0 | x | 0.2547 | 2.0261 | -11.4746 | 19.4557 | -3.9622 | 8.8622 |
| 1 | y | 0.5097 | 2.0075 | -14.9270 | 12.4689 | -3.3909 | 7.9247 |
| 2 | z | -0.1269 | 3.3882 | -17.2790 | 24.6255 | -12.7411 | 10.0000 |
| 3 | gripper | 0.3760 | 0.7099 | -1.0000 | 1.0000 | -1.0000 | 1.0000 |

## Gripper action (last action dim, all frames)

10 distinct values, range -1.0000 to 1.0000.
