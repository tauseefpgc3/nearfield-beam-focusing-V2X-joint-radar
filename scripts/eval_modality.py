"""Evaluate a trained single-modality checkpoint on the TEST split (Paper 5).

Recomputes overall + near-field-subset range accuracy + range confusion matrix from a
saved checkpoint, on the same trajectory-grouped TEST split (seed). Used for the
apples-to-apples radar-vs-LiDAR comparison on the metric that matters.

    python scripts/eval_modality.py --modality radar \
        --ckpt checkpoints/closerange_60GHz_radar/seed11.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_single import MODALITY, load_dataset, make_loader, require_cuda
from src.data.splits import trajectory_grouped_split
from src.eval.metrics import detailed_eval, print_range_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "lidar"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(
        _ROOT, "data", "processed", "closerange_60GHz_full.h5"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = require_cuda()
    feature_key, ModelCls = MODALITY[args.modality]
    feat, yt, yr, traj, _ = load_dataset(args.data, feature_key)
    _, _, te = trajectory_grouped_split(traj, seed=args.seed)
    loader = make_loader(feat, yt, yr, te, args.batch_size, False)

    model = ModelCls(d_z=128, B_theta=64, B_r=8).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded {args.ckpt} (epoch {state.get('epoch','?')}) | TEST n={len(te)}")
    m = detailed_eval(model, loader, device)
    print_range_report(f"{args.modality} (from checkpoint)", m)


if __name__ == "__main__":
    main()
