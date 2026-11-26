"""M3 + M4 in one Kaggle GPU run (T4 x2): torch unit tests, real-feature overfit (+ diagnostics),
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
from minivla.train import make_expert_and_head, prepare, setup_determinism, train, zero_baseline_l1

REPORT = Path("results/m34_checks.md")
CFG_A, CFG_C = "configs/A_flow_lang.yaml", "configs/C_l1_lang.yaml"
# How each overfit variant trains. "fp16_scaled" is the official check: the same mixed-precision mechanics
# as minivla/train.py (fp16 autocast + GradScaler + grad clipping). The first M3+M4 run used "fp16"
# (autocast without GradScaler), where small fp16 gradients can underflow to zero.
PRECISIONS = ["fp16_scaled", "fp16", "fp32"]


def overfit_real(cfg, d, rows, steps, precision="fp16_scaled", fixed_noise=False):
    """Fit 8 fixed real samples (dropout 0, lr 1e-3). Returns final train loss (mean of the last 100 steps)
    and the masked L1 of the prediction (flow: 10 and 50 Euler steps, fixed sampling noise).
    fixed_noise (flow only): the same noise and time every step, so a memorizing model drives the loss to ~0."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["dropout"] = 0.0
    flow = cfg["head"] == "flow"
    torch.manual_seed(0)
    expert, head = make_expert_and_head(cfg, d.n_prefix + d.n_text, d.dim)
    expert = expert.to(d.device).train()
    opt = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    scaled, amp = precision == "fp16_scaled", precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=scaled)
    feats, mask, state, actions, pad = d.batch(rows)
    if not amp:
        feats = feats.float()
    extra = {}
    if fixed_noise:
        torch.manual_seed(1)
        extra = {"noise": torch.randn(actions.shape, device=d.device), "t": head.sample_time(len(rows), d.device)}
    losses = []
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = head.loss(expert, feats, mask, state, actions, pad, **extra)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if scaled:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(expert.parameters(), cfg["train"]["grad_clip"])
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    expert.eval()

    def l1(euler_steps=None):
        g = torch.Generator(device=d.device).manual_seed(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            kw = {"steps": euler_steps} if flow else {}
            pred = head.predict(expert, feats, mask, state, generator=g, **kw)
        return masked_mean((pred.float() - actions).abs(), pad).item()

    return {"loss_end": float(np.mean(losses[-100:])), "l1": l1(), "l1_50": l1(50) if flow else None}


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

    # 3. overfit 8 real samples: official check + diagnostic variants
    rows8 = np.sort(np.random.default_rng(0).choice(train_rows, 8, replace=False))
    overfit_rows = []
    for name, cfg, steps in [("C (L1)", cfg_c, 1000), ("A (flow)", cfg_a, 3000)]:
        variants = [(p, False) for p in PRECISIONS] + ([("fp16_scaled", True)] if cfg["head"] == "flow" else [])
        for precision, fixed in variants:
            r = overfit_real(cfg, d, rows8, steps, precision, fixed)
            overfit_rows.append((name, steps, precision + (", fixed noise and t" if fixed else ""), r))
            print("overfit", name, precision, "fixed" if fixed else "", r, flush=True)
    off = {n: r for n, _, v, r in overfit_rows if v == "fp16_scaled"}
    check("regression head overfits 8 real samples (L1 < 0.02 after 1000 steps; fp16 + GradScaler + clip)",
          off["C (L1)"]["l1"] < 0.02, f"L1 {off['C (L1)']['l1']:.4f}")
    check("flow head overfits 8 real samples (10-step sample L1 < 0.05 after 3000 steps; fp16 + GradScaler + clip)",
          off["A (flow)"]["l1"] < 0.05, f"L1 {off['A (flow)']['l1']:.4f}")

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
    L += ["", "## Overfit 8 real samples: all variants (official check = fp16_scaled)", "",
          "| Head | Steps | Training | Final train loss | L1 (10 Euler steps) | L1 (50 Euler steps) |", "|---|---|---|---|---|---|"]
    fmt = lambda x: "–" if x is None else f"{x:.4f}"
    L += [f"| {n} | {s} | {v} | {fmt(r['loss_end'])} | {fmt(r['l1'])} | {fmt(r['l1_50'])} |" for n, s, v, r in overfit_rows]
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
