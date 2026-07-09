"""Task A training driver — extend radar + fusion(warm-start) to 20 seeds x 2 freqs.

CRASH-SAFE / RESUMABLE: skips any (radar, fusion) checkpoint that already exists, so the
sweep can be stopped and re-launched and it only trains what is missing. For each
freq {60,140} x seed, trains the radar model first, then the fusion warm-start model
initialized from that radar checkpoint (D-015). Same hyperparameters as the paper
(lr 1e-4, wd 1e-5, 50 epochs, batch 16, fp16).

    python scripts/sweep_taskA.py                 # all 20 seeds, both freqs
    python scripts/sweep_taskA.py --seeds 66 77   # only these new seeds
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOGS = os.path.join(_ROOT, "results", "sweep_logs")

# 5 original + 15 new (step 11) = 20 seeds for the higher-powered comparison.
ALL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165,
             176, 187, 198, 209, 220]
FREQS = ["60", "140"]


def sh(cmd, log):
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, log), "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS)
    args = ap.parse_args()
    t0 = time.time()
    todo = 0
    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        for seed in args.seeds:
            rck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_radar",
                               f"seed{seed}.pt")
            fck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_fusion_warmstart",
                               f"seed{seed}.pt")
            if not os.path.exists(rck):
                todo += 1
                print(f"[train radar] {fghz}GHz seed{seed} ...", flush=True)
                sh([PY, "scripts/train_single.py", "--modality", "radar", "--data", data,
                    "--seed", str(seed), "--epochs", "50"], f"A_radar_{fghz}_s{seed}.log")
            else:
                print(f"[skip radar] {fghz}GHz seed{seed}", flush=True)
            if not os.path.exists(fck):
                todo += 1
                print(f"[train fusion] {fghz}GHz seed{seed} (warmstart) ...", flush=True)
                sh([PY, "scripts/train_fusion.py", "--data", data, "--seed", str(seed),
                    "--epochs", "50", "--warmstart-radar", rck], f"A_fusion_{fghz}_s{seed}.log")
            else:
                print(f"[skip fusion] {fghz}GHz seed{seed}", flush=True)
            print(f"[done] {fghz}GHz seed{seed}", flush=True)
    print(f"\nTASK-A SWEEP: trained {todo} missing runs in {(time.time()-t0)/3600:.2f} h",
          flush=True)


if __name__ == "__main__":
    main()
