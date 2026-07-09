"""Headline 5-seed sweep orchestrator — crash-safe & resumable.

Runs, per (frequency, seed): radar-alone, LiDAR-alone, warmstart-fusion (D-015 config)
on the close-range PRIMARY data, plus radar on the O1 far-field anchor. Each step is a
subprocess; a cell is SKIPPED if its checkpoint + metrics JSON already exist, so a crash
+ relaunch resumes. Trains with checkpointing every 10 epochs.

    python scripts/run_sweep.py            # full sweep (overnight)
    python scripts/run_sweep.py --freqs 60 --seeds 11   # subset
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CKPT = os.path.join(_ROOT, "checkpoints")
PROC = os.path.join(_ROOT, "data", "processed")
LOGS = os.path.join(_ROOT, "results", "sweep_logs")


def sh(cmd, log):
    """Run a subprocess, tee to a log; return success bool (never raises)."""
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, log), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def ckpt_path(dtag, branch, seed):
    return os.path.join(CKPT, f"{dtag}_{branch}", f"seed{seed}.pt")


def metrics_path(modality, dtag, seed):
    return os.path.join(_ROOT, "results", "indomain",
                        f"{modality}_{dtag}_seed{seed}_metrics.json")


def ensure_train(cmd, ckpt, log, what):
    if os.path.exists(ckpt):
        print(f"    [skip train] {what} (ckpt exists)", flush=True)
        return True
    print(f"    [train] {what} ...", flush=True)
    t0 = time.time()
    ok = sh(cmd, log)
    print(f"    [train] {what} {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)",
          flush=True)
    return ok and os.path.exists(ckpt)


def ensure_eval(modality, dtag, seed, ckpt, log):
    mp = metrics_path(modality, dtag, seed)
    if os.path.exists(mp):
        print(f"    [skip eval] {modality} {dtag} seed{seed}", flush=True)
        return
    data = os.path.join(PROC, f"{dtag}_full.h5")
    sh([PY, "scripts/eval_headline.py", "--modality", modality, "--seed", str(seed),
        "--data", data, "--ckpt", ckpt], log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=int, nargs="+", default=[60, 140])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    t_start = time.time()

    for f in args.freqs:
        cr = f"closerange_{f}GHz"; o1 = f"synthetic_{f}GHz"
        cr_data = os.path.join(PROC, f"{cr}_full.h5")
        o1_data = os.path.join(PROC, f"{o1}_full.h5")
        for seed in args.seeds:
            print(f"\n=== freq {f}GHz | seed {seed} ===", flush=True)
            ep = ["--epochs", str(args.epochs), "--seed", str(seed)]

            # 1) radar-alone (close-range)
            rk = ckpt_path(cr, "radar", seed)
            ensure_train([PY, "scripts/train_single.py", "--modality", "radar",
                          "--data", cr_data, *ep], rk, f"radar_{cr}_s{seed}.log",
                         f"radar {cr} s{seed}")
            ensure_eval("radar", cr, seed, rk, f"eval_radar_{cr}_s{seed}.log")

            # 2) LiDAR-alone (close-range)
            lk = ckpt_path(cr, "lidar", seed)
            ensure_train([PY, "scripts/train_single.py", "--modality", "lidar",
                          "--data", cr_data, *ep], lk, f"lidar_{cr}_s{seed}.log",
                         f"lidar {cr} s{seed}")
            ensure_eval("lidar", cr, seed, lk, f"eval_lidar_{cr}_s{seed}.log")

            # 3) warmstart fusion (needs the radar ckpt for this seed)
            fk = ckpt_path(cr, "fusion_warmstart", seed)
            if os.path.exists(rk):
                ensure_train([PY, "scripts/train_fusion.py", "--data", cr_data, *ep,
                              "--warmstart-radar", rk, "--freeze-radar"], fk,
                             f"fusion_{cr}_s{seed}.log", f"fusion {cr} s{seed}")
                ensure_eval("fusion", cr, seed, fk, f"eval_fusion_{cr}_s{seed}.log")
            else:
                print("    [skip fusion] radar ckpt missing", flush=True)

            # 4) O1 far-field anchor (radar)
            ok_ = ckpt_path(o1, "radar", seed)
            ensure_train([PY, "scripts/train_single.py", "--modality", "radar",
                          "--data", o1_data, *ep], ok_, f"radar_{o1}_s{seed}.log",
                         f"radar {o1} s{seed}")
            ensure_eval("radar", o1, seed, ok_, f"eval_radar_{o1}_s{seed}.log")

    print(f"\nSWEEP DONE in {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
