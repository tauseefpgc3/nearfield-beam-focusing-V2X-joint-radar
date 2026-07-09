"""DeepSense real-data angle eval.

DeepSense is far-field (y_r=0), so the real-data metric is ANGLE: top-1 beam accuracy and
angle top-K beam-selection coverage (does the true mmWave beam land in the model's top-K
angle predictions). Works on radar / lidar / fusion checkpoints.

    # zero-shot: a synthetic-trained radar model applied to DeepSense (whole set)
    python scripts/eval_deepsense.py --modality radar --split all \
        --ckpt checkpoints/closerange_60GHz_radar/seed11.pt
    # DeepSense-trained: evaluate on the held-out trajectory test split
    python scripts/eval_deepsense.py --modality radar --split test --seed 11 \
        --ckpt checkpoints/deepsense_scenario31_radar/seed11.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_single import MODALITY
from src.data.splits import trajectory_grouped_split
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

B_THETA = 64
KS = (1, 2, 4, 8)
DATA = os.path.join(_ROOT, "data", "processed", "deepsense_scenario31_full.h5")


def angle_probs(model, modality, device, **feats):
    if modality == "fusion":
        out = model(feats["radar"], feats["lidar"])
        p = torch.stack([F.softmax(out[f"{b}_theta_logits"].float(), 1) for b in BRANCHES])
        return p[BRANCHES.index("radar")]            # D-015: angle from radar branch
    out = model(feats[modality])
    return out["theta_logits"].float().softmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "lidar", "fusion"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["all", "test"], default="all")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    device = torch.device("cuda")

    with h5py.File(args.data, "r") as hf:
        n = hf["y_theta"].shape[0]
        idx = np.arange(n)
        if args.split == "test":
            _, _, idx = trajectory_grouped_split(hf["trajectory_id"][:], seed=args.seed)
            idx = np.sort(idx)
        yt = torch.from_numpy(hf["y_theta"][idx].astype(np.int64))
        is_fusion = args.modality == "fusion"
        xr = torch.from_numpy(hf["radar_features"][idx].astype(np.float32)) \
            if args.modality in ("radar", "fusion") else None
        xl = torch.from_numpy(hf["lidar_features"][idx].astype(np.float32)) \
            if args.modality in ("lidar", "fusion") else None

    ModelCls = SensingPolarPredictor if is_fusion else MODALITY[args.modality][1]
    model = ModelCls(d_z=128, B_theta=B_THETA, B_r=8).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()

    m = idx.size
    top1 = 0
    cov = {k: 0 for k in KS}
    with torch.no_grad():
        for s in range(0, m, 256):
            e = min(s + 256, m)
            feats = {}
            if xr is not None:
                feats["radar"] = xr[s:e].to(device)
            if xl is not None:
                feats["lidar"] = xl[s:e].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = angle_probs(model, args.modality, device, **feats)
            tk = p.topk(max(KS), 1).indices.cpu()
            y = yt[s:e]
            top1 += int((tk[:, 0] == y).sum())
            for k in KS:
                cov[k] += int((tk[:, :k] == y.unsqueeze(1)).any(1).sum())
    print(f"[{args.modality} {args.split} n={m}] top-1 angle = {top1/m:.4f} | "
          f"angle coverage K=1/2/4/8 = "
          + "/".join(f"{cov[k]/m:.3f}" for k in KS)
          + f"  ({os.path.basename(args.ckpt)})")
    return top1 / m


if __name__ == "__main__":
    main()
