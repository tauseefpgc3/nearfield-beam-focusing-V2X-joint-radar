"""E10 sweep: radar-primary B_r ablation (3 B_r x 3 seeds), crash-safe (Paper 5)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOGS = os.path.join(_ROOT, "results", "sweep_logs")
B_RS = [4, 8, 16]
SEEDS = [11, 22, 33]


def sh(cmd, log):
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, log), "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT)


def main():
    t0 = time.time()
    for B_r in B_RS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_60GHz_Br{B_r}_full.h5")
        for seed in SEEDS:
            ckpt = os.path.join(_ROOT, "checkpoints",
                                f"closerange_60GHz_Br{B_r}_radar", f"seed{seed}.pt")
            if not os.path.exists(ckpt):
                print(f"[train] B_r={B_r} seed{seed} ...", flush=True)
                sh([PY, "scripts/train_single.py", "--modality", "radar", "--data", data,
                    "--seed", str(seed), "--epochs", "50"], f"br{B_r}_radar_s{seed}.log")
            else:
                print(f"[skip] B_r={B_r} seed{seed} (ckpt exists)", flush=True)
            res = os.path.join(_ROOT, "results", "ablation", f"br{B_r}_seed{seed}.json")
            if not os.path.exists(res):
                sh([PY, "scripts/ablate_br_eval.py", "--b-r", str(B_r), "--seed", str(seed),
                    "--ckpt", ckpt], f"eval_br{B_r}_s{seed}.log")
            print(f"[done] B_r={B_r} seed{seed}", flush=True)
    print(f"\nABLATION SWEEP DONE in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
