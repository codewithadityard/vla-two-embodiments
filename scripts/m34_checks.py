"""M3 + M4 in one Kaggle GPU run (T4 x2): torch unit tests, real-feature overfit,
determinism, two short runs at once (one per GPU, as the real runs will go) with a smoke eval each,
and the step-count rule S measured under that load.

Needs the attached inputs: the cache (R2 output) and libero-goal-lowdim. Writes results/m34_checks.md.
Exits 1 if any check fails. Thresholds are the approved ones and are not tuned to the results.
"""
import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from minivla.config import load_config, seed_everything
from minivla.heads import masked_mean
from minivla.parallel import run_parallel
from minivla.train import lr_factor, make_expert_and_head, prepare, setup_determinism, train, zero_baseline_l1

REPORT = Path("results/m34_checks.md")
CFG_A, CFG_C = "configs/A_flow_lang.yaml", "configs/C_l1_lang.yaml"
OVERFIT_LR = 1e-3
# (head, config, steps, L1 threshold). Thresholds are the approved ones; steps and the LR schedule were
# changed after the second M3+M4 run (approved 2026-11-26, see docs/deviations.md).
OVERFIT = [("C (L1)", CFG_C, 3000, 0.02), ("A (flow)", CFG_A, 10000, 0.05)]


def overfit_real(cfg, d, rows, steps, curve_every=500):
    """Fit 8 fixed real samples (dropout 0) with the training's mechanics: fp16 autocast + GradScaler +
    grad clipping, and the training's LR schedule shape (linear warm-up over train.warmup steps, cosine decay
    to the same final/peak ratio), peak lr 1e-3. Returns the loss curve (mean of the last 100 steps, every
    `curve_every` steps) and the masked L1 of the prediction (flow: 10 and 50 Euler steps, fixed noise)."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["dropout"] = 0.0
    tr, flow = cfg["train"], cfg["head"] == "flow"
    torch.manual_seed(0)
    expert, head = make_expert_and_head(cfg, d.n_prefix + d.n_text, d.dim)
    expert = expert.to(d.device).train()
    opt = torch.optim.AdamW(expert.parameters(), lr=OVERFIT_LR)
    final = OVERFIT_LR * tr["final_lr"] / tr["lr"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lr_factor(i, tr["warmup"], steps, OVERFIT_LR, final))
    scaler = torch.amp.GradScaler("cuda")
    feats, mask, state, actions, pad = d.batch(rows)
    losses, curve = [], []
    for step in range(1, steps + 1):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = head.loss(expert, feats, mask, state, actions, pad)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(expert.parameters(), tr["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        if step % curve_every == 0:
            curve.append((step, float(np.mean(losses[-100:]))))
    expert.eval()

    def l1(euler_steps=None):
        g = torch.Generator(device=d.device).manual_seed(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            kw = {"steps": euler_steps} if flow else {}
            pred = head.predict(expert, feats, mask, state, generator=g, **kw)
        return masked_mean((pred.float() - actions).abs(), pad).item()

    return {"first_loss": losses[0], "curve": curve, "l1": l1(), "l1_50": l1(50) if flow else None}


def main():
    setup_determinism()
    cfg_a, cfg_c = load_config(CFG_A), load_config(CFG_C)
    seed_everything(cfg_a["seed"])
    checks, diag = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print("PASS" if ok else "FAIL", name, detail, flush=True)

    # 1. torch unit tests (+ diagnostic: the numbers behind the two random-data overfit tests)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_expert.py"], capture_output=True, text=True)
    check("torch unit tests (tests/test_expert.py)", p.returncode == 0, p.stdout.strip().splitlines()[-1])
    if p.returncode != 0:
        print(p.stdout[-3000:])
    # load by file path: an installed package named `tests` on Kaggle hides the repo's tests/ folder
    spec = importlib.util.spec_from_file_location("test_expert", "tests/test_expert.py")
    test_expert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_expert)
    synthetic_overfit = test_expert.overfit
    for flow, steps in [(False, 1000), (True, 3000)]:
        first, last, l1 = synthetic_overfit(flow=flow, steps=steps)
        diag.append((f"synthetic overfit, {'flow' if flow else 'regression'}, {steps} steps",
                     f"train loss {first:.4f} -> {last:.4f} ({100 * (1 - last / first):.1f} % drop); L1 {l1:.4f}"))

    # 2. data (A and C both use language, so one cache serves both)
    d, train_rows, val_rows, stats = prepare(cfg_a)
    baseline = zero_baseline_l1(d, val_rows)
    cache_info = d.describe()

    # 3. overfit 8 real samples (thresholds unchanged)
    rows8 = np.sort(np.random.default_rng(0).choice(train_rows, 8, replace=False))
    overfit_res = {}
    for name, path, steps, limit in OVERFIT:
        r = overfit_real(load_config(path), d, rows8, steps)
        overfit_res[name] = (steps, limit, r)
        print("overfit", name, r, flush=True)
        check(f"{name} overfits 8 real samples (L1 < {limit} after {steps} steps; training mechanics + LR schedule)",
              r["l1"] < limit, f"L1 {r['l1']:.4f}" + (f" (50 Euler steps: {r['l1_50']:.4f})" if r["l1_50"] else ""))

    # 4. determinism on the GPU: same seed -> identical losses
    h1 = train(cfg_a, d, train_rows, val_rows, 50, "outputs/m34/det1.csv", seed=0, log_every=10)[2]
    h2 = train(cfg_a, d, train_rows, val_rows, 50, "outputs/m34/det2.csv", seed=0, log_every=10)[2]
    same = [r[1] for r in h1] == [r[1] for r in h2]
    check("same seed gives identical training losses on the GPU (50 steps, flow head)", same,
          f"{[round(r[1], 6) for r in h1]} vs {[round(r[1], 6) for r in h2]}" if not same else "")
    del d
    torch.cuda.empty_cache()

    # 5. short runs C and A at the same time, one per GPU, each followed by a 2-episode smoke eval
    n_gpu = torch.cuda.device_count()
    check("two GPUs visible (T4 x2)", n_gpu >= 2, f"{n_gpu} GPU(s)")
    codes = run_parallel([("C", 0, ["scripts.m34_short", CFG_C, "C"]),
                          ("A", 1 if n_gpu >= 2 else 0, ["scripts.m34_short", CFG_A, "A"])])
    res = {}
    for tag in ["C", "A"]:
        f = Path(f"outputs/m34/short_{tag}.json")
        ok = codes[tag] == 0 and f.exists()
        check(f"worker {tag}: short run + smoke eval finished", ok, f"exit code {codes[tag]}")
        if ok:
            res[tag] = json.loads(f.read_text())
    for tag, r in res.items():
        hist = r["history"]
        check(f"{tag}: val L1 after {hist[-1][0]} steps < mean-chunk baseline", hist[-1][2] < r["baseline"],
              f"val L1 {hist[-1][2]:.4f} vs baseline {r['baseline']:.4f}")
        check(f"{tag}: smoke eval ran every episode (2) end to end", len(r["smoke_rows"]) == 2,
              f"{len(r['smoke_rows'])} episodes, {r['smoke_min']:.1f} min")
        check(f"{tag}: online backbone prefix length == cache prefix length",
              r["prefix_len_online"] == r["prefix_len_cache"], f"{r['prefix_len_online']} vs {r['prefix_len_cache']}")

    # 6. step count S (plan rule: S = min(20000, steps that fit in 60 min), slowest head, both runs at once)
    S = s_step = None
    if len(res) == 2:
        s_step = max(r["history"][-1][4] for r in res.values())
        S = min(20000, math.floor(3600 / s_step))

    # report
    L = ["# M3 + M4 checks (one Kaggle run)", "",
         "Generated by `python -m scripts.m34_checks`. Every number is measured in this run.", "",
         f"- GPU: {torch.cuda.get_device_name(0)} x {n_gpu}; {cache_info}",
         f"- Train / val frames: {len(train_rows)} / {len(val_rows)} (fixed val subset); "
         f"mean-chunk baseline val L1: {baseline:.4f}", "", "## Checks", "",
         "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {dt} |" for n, ok, dt in checks]
    for name, (steps, limit, r) in overfit_res.items():
        L += ["", f"## Overfit 8 real samples, {name}: {steps} steps, peak lr {OVERFIT_LR}, threshold L1 < {limit}", "",
              f"Loss at step 1: {r['first_loss']:.4f}. Final L1: {r['l1']:.4f}" +
              (f" (10 Euler steps), {r['l1_50']:.4f} (50 Euler steps)" if r["l1_50"] else "") + ".", "",
              "| Step | Train loss (mean of last 100 steps) |", "|---|---|"]
        L += [f"| {st} | {v:.4f} |" for st, v in r["curve"]]
    L += ["", "## Diagnostics (not checks)", ""] + [f"- {n}: {v}" for n, v in diag]
    for tag, r in res.items():
        L += ["", f"## Short run {tag} (both at once, one per GPU): {r['history'][-1][0]} steps, batch "
              f"{cfg_a['train']['batch']}", "", "| Step | Train loss | Val L1 | LR | s/step |", "|---|---|---|---|---|"]
        L += [f"| {h[0]} | {h[1]:.4f} | {h[2] if h[2] == '' else f'{h[2]:.4f}'} | {h[3]:.2e} | {h[4]:.4f} |"
              for h in r["history"]]
        L += ["", f"Smoke eval {tag} (pipeline test only; successes not used): " +
              "; ".join(f"task {e[1]} ep {e[3]}: {'success' if e[5] else 'fail'}, {e[6]} steps, {e[7]} s"
                        for e in r["smoke_rows"])]
    if S is not None:
        L += ["", f"**Step count rule: S = {S}** ({S * s_step / 60:.0f} min per model at {s_step:.4f} s/step, "
              f"slowest head with both runs at once). Configured now: {cfg_a['train']['steps']}; "
              "set train.steps to S before M5.", ""]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    failed = [n for n, ok, _ in checks if not ok]
    print(f"Checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
