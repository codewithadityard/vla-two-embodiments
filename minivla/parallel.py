"""Run several Python modules at once, one per GPU (Kaggle's T4 x2), with tagged live output.

Each child sees only its own GPU (CUDA_VISIBLE_DEVICES), so inside it the GPU is always cuda:0 and a run
gives the same numbers whether it runs alone or next to another one.
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
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
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
