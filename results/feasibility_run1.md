# Feasibility check (Kaggle)

- Date (UTC): 2026-11-25 19:17
- Seed: 1000, LeRobot commit: `b07adcc5e3ce42c26a248236dbb3571f84c28c43`
- Total notebook time: 12.0 min (budget 30 min: OK)

| Step | Status | Numbers |
|---|---|---|
| 1. Install | PASS | python_used=/usr/bin/python3, driver=580.178.04
580.178.04, install_minutes=1.0 |
| 2. Versions/imports | PASS | python=3.13.16, torch=2.11.0+cu130, torch_cuda=13.0, cuda_available=True, gpu=Tesla T4, lerobot=0.7.1, mujoco=3.8.1, gymnasium=1.2.0, metaworld=3.0.0, hf-libero=0.1.5, transformers=5.22.2, peft=0.21.2, imports_ok=True |
| 3. LIBERO eval | FAIL |  |
| 4. Meta-World | PASS | task=reach-v3, frame_shape=[480, 480, 3], obs_dim=39 |
| 5a. Memory: frozen backbone | PASS | setup=frozen, trainable_params_M=25.75, backbone_tokens_per_sample=73, peak_gpu_allocated_GB=1.79, peak_gpu_reserved_GB=2.15, peak_ram_GB=3.6, sec_per_step=0.357, final_loss=1.028 |
| 5b. Memory: LoRA | FAIL |  |
| 6. DINO-WM Wall | PASS | file=/datasets/wall_single.zip, download_link=https://osf.io/download/49rnx/?view_only=a56a296ce3b24cceaf408383a175ce28, size_bytes=1668205895, size_GB=1.67 |

## Deviations
- None.

## Notes
- Step 3 PASS means the eval ran end to end; with 1 episode, pc_success is 0 or 100.
- Step 5 sec/step averages steps after the warm-up steps in config.
- Step 5 peak RAM is per process (ru_maxrss).

## Errors
### 3. LIBERO eval
```
sr/local/lib/python3.13/dist-packages/lerobot/configs/parser.py", line 473, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/lerobot/scripts/lerobot_eval.py", line 855, in eval_main
    validate_checkpoint(
    ~~~~~~~~~~~~~~~~~~~^
        descriptor,
        ^^^^^^^^^^^
    ...<3 lines>...
        env_postprocessor=env_postprocessor,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/lerobot/envs/utils.py", line 205, in validate_checkpoint
    raise ValueError(f"Checkpoint feature missing from the simulator (check --rename_map): {name}")
ValueError: Checkpoint feature missing from the simulator (check --rename_map): observation.images.camera1
Traceback (most recent call last):
  File "/kaggle/working/step3_libero.py", line 22, in <module>
    subprocess.run(cmd, input="N\n", text=True, check=True)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.13/subprocess.py", line 578, in run
    raise CalledProcessError(retcode, process.args,
                             output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['/usr/local/bin/lerobot-eval', '--policy.path=lerobot/smolvla_libero', '--policy.device=cuda', '--env.type=libero', '--env.task=libero_goal', '--env.task_ids=[0]', '--eval.batch_size=1', '--eval.n_episodes=1', '--seed=1000', '--output_dir=outputs/feasibility/libero_eval']' returned non-zero exit status 1.

```
### 5b. Memory: LoRA
```
l/lib/python3.13/dist-packages/peft/tuners/tuners_utils.py", line 344, in __init__
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/peft/tuners/tuners_utils.py", line 978, in inject_adapter
    self._create_and_replace(
    ~~~~~~~~~~~~~~~~~~~~~~~~^
        peft_config, adapter_name, target, target_name, parent, current_key=key
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/peft/tuners/lora/model.py", line 389, in _create_and_replace
    new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/peft/tuners/lora/model.py", line 552, in _create_new_module
    new_module = dispatcher(target, adapter_name, config=lora_config, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/peft/tuners/lora/torchao.py", line 160, in dispatch_torchao
    if not is_torchao_available():
           ~~~~~~~~~~~~~~~~~~~~^^
  File "/usr/local/lib/python3.13/dist-packages/peft/import_utils.py", line 162, in is_torchao_available
    raise ImportError(
    ...<2 lines>...
    )
ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported

```
