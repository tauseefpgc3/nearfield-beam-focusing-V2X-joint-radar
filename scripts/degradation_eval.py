"""Sensor-degradation eval (E6) on the warmstart-fusion models (Paper 5 Batch 8).

For each seed's close-range 60 GHz fusion checkpoint, evaluate the confidence-weighted
ensemble (the graceful-degradation mechanism, C3) under: clean, radar-out, lidar-out,
both-out, 5deg-misalign. Reports top-1 joint accuracy, mean +/- 95% CI over seeds.

both-out MUST be ~random (~1/512). If >5%, that flags BatchNorm/running-stat leakage
(D9) — we used GroupNorm specifically to avoid it; this verifies it worked.

    python scripts/degradation_eval.py
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.splits import trajectory_grouped_split
from src.eval.statistics import mean_ci
from src.models.sensing_polar import SensingPolarPredictor, confidence_combine

SEEDS = [11, 22, 33, 44, 55]
DATA = os.path.join(_ROOT, "data", "processed", "closerange_60GHz_full.h5")
MODES = ["clean", "radar_out", "lidar_out", "both_out", "misalign5deg"]


@torch.no_grad()
def joint_acc(model, xr, xl, yt, yr, device, mode, bs=256):
    n = yt.shape[0]; correct = 0
    for s in range(0, n, bs):
        e = min(s + bs, n)
        a = xr[s:e].to(device); b = xl[s:e].to(device)
        rd = mode in ("radar_out", "both_out")
        ld = mode in ("lidar_out", "both_out")
        if mode == "misalign5deg":
            b = TF.rotate(b, 5.0)               # 5 deg LiDAR extrinsic misalignment
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p_theta, p_r = confidence_combine(model(a, b, radar_dropout=rd, lidar_dropout=ld))
        ok = (p_theta.argmax(1).cpu() == yt[s:e]) & (p_r.argmax(1).cpu() == yr[s:e])
        correct += int(ok.sum())
    return correct / n


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    with h5py.File(DATA, "r") as hf:
        traj = hf["trajectory_id"][:]
        results = {m: [] for m in MODES}
        for seed in SEEDS:
            ckpt = os.path.join(_ROOT, "checkpoints",
                                "closerange_60GHz_fusion_warmstart", f"seed{seed}.pt")
            if not os.path.exists(ckpt):
                print(f"seed {seed}: ckpt missing, skip"); continue
            _, _, te = trajectory_grouped_split(traj, seed=seed)
            te = np.sort(te)
            xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
            xl = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
            yt = torch.from_numpy(hf["y_theta_gated"][te].astype(np.int64))
            yr = torch.from_numpy(hf["y_r_gated"][te].astype(np.int64))
            model = SensingPolarPredictor().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device,
                                             weights_only=False)["model"])
            model.eval()
            row = {m: joint_acc(model, xr, xl, yt, yr, device, m) for m in MODES}
            for m in MODES:
                results[m].append(row[m])
            print(f"seed {seed}: " + " | ".join(f"{m} {row[m]:.3f}" for m in MODES),
                  flush=True)

    print("\n=== Degradation (mean +/- 95% CI over seeds, top-1 joint) ===")
    summary = {}
    for m in MODES:
        ci = mean_ci(results[m])
        summary[m] = ci
        print(f"  {m:14s}: {ci['mean']:.4f}  [{ci['lo']:.4f}, {ci['hi']:.4f}]  (n={ci['n']})")
    bo = summary["both_out"]["mean"]
    print(f"\n[both-out leakage check] {bo:.4f} "
          f"({'PASS <=5%, no leakage (GroupNorm worked)' if bo <= 0.05 else 'FAIL >5% — D9 LEAKAGE'})"
          f"  | random ref 1/512 = {1/512:.4f}")
    with open(os.path.join(_ROOT, "results", "degradation_closerange60.json"), "w") as f:
        json.dump({m: summary[m] for m in MODES} | {"per_seed": results}, f, indent=2)


if __name__ == "__main__":
    main()
