"""Run several Python modules at once, one per GPU (Kaggle's T4 x2), with tagged live output.

Each child gets MINIVLA_GPU=<index>; setup_determinism() then calls torch.cuda.set_device, so "cuda" means that
GPU for all of its tensors. Not CUDA_VISIBLE_DEVICES: robosuite's EGL renderer then asks for EGL device <index>
while only one device is visible, and crashes (M3+M4 rerun 2). MuJoCo renders on GPU 0 for both runs.
"""
import os
import subprocess
import sys
import threading


def answer_libero_question():
    """LIBERO asks a yes/no question on its first import. Import it here once (the runner answers on our
    stdin), so the config file exists and the children, whose stdin is closed, never ask."""
    from libero.libero import benchmark  # noqa: F401


def run_parallel(jobs):
    """jobs: list of (tag, gpu, [module, *args]). Starts all, streams "[tag] line", waits. Returns {tag: exit code}."""
    answer_libero_question()
    procs, threads = {}, []
    for tag, gpu, argv in jobs:
        env = {**os.environ, "MINIVLA_GPU": str(gpu), "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen([sys.executable, "-m", *argv], env=env, text=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs[tag] = p

        def pump(p=p, tag=tag):
            for line in p.stdout:
                print(f"[{tag}] {line}", end="", flush=True)

        th = threading.Thread(target=pump)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return {tag: p.wait() for tag, p in procs.items()}
