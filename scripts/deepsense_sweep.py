"""DeepSense-trained validation sweep (Paper 5 Batch 3, STEP C) — crash-safe.

Trains radar (PRIMARY), LiDAR + warmstart-fusion (SECONDARY) on the real DeepSense
scenario, 5 seeds, trajectory-grouped split. Reuses train_single / train_fusion;
skips a cell whose checkpoint exists (resumable). Real-data angle is the metric (y_r=0).

    python scripts/deepsense_sweep.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DATA = os.path.join(_ROOT, "data", "processed", "deepsense_scenario31_full.h5")
DTAG = "deepsense_scenario31"
SEEDS = [11, 22, 33, 44, 55]
LOGS = os.path.join(_ROOT, "results", "sweep_logs")


def ckpt(branch, seed):
    return os.path.join(_ROOT, "checkpoints", f"{DTAG}_{branch}", f"seed{seed}.pt")


def run(cmd, log, what, target):
    if os.path.exists(target):
        print(f"  [skip] {what}", flush=True); return True
    print(f"  [train] {what} ...", flush=True)
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(LOGS, log), "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT)
    ok = os.path.exists(target)
    print(f"  [train] {what} {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    t0 = time.time()
    ep = ["--epochs", "50", "--data", DATA]
    for seed in SEEDS:
        print(f"\n=== DeepSense seed {seed} ===", flush=True)
        rk = ckpt("radar", seed)
        run([PY, "scripts/train_single.py", "--modality", "radar", "--seed", str(seed), *ep],
            f"ds_radar_s{seed}.log", f"radar s{seed}", rk)
        run([PY, "scripts/train_single.py", "--modality", "lidar", "--seed", str(seed), *ep],
            f"ds_lidar_s{seed}.log", f"lidar s{seed}", ckpt("lidar", seed))
        if os.path.exists(rk):
            run([PY, "scripts/train_fusion.py", "--seed", str(seed), *ep,
                 "--warmstart-radar", rk, "--freeze-radar"],
                f"ds_fusion_s{seed}.log", f"fusion s{seed}", ckpt("fusion_warmstart", seed))
    print(f"\nDEEPSENSE SWEEP DONE in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
