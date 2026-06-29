"""Train a single-modality (radar OR lidar) polar classifier (Paper 5 Batch 5/6).

Identical training setup for both modalities (fair comparison): close-range PRIMARY,
gated labels, trajectory-grouped 70/10/20, seed 11, batch 16, fp16. Reports overall AND
near-field-subset range accuracy + the range confusion matrix.

    python scripts/train_single.py --modality lidar            # real run
    python scripts/train_single.py --modality lidar --overfit  # capacity check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.splits import trajectory_grouped_split
from src.eval.metrics import detailed_eval, print_range_report
from src.models.lidar_encoder import LidarPolarModel
from src.models.radar_encoder import RadarPolarModel
from src.train.loss import polar_loss
from src.utils.seeding import seed_everything

MODALITY = {
    "radar": ("radar_features", RadarPolarModel),
    "lidar": ("lidar_features", LidarPolarModel),
}


def require_cuda():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP. (Do not silently train on CPU; docs/10.)")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    return torch.device("cuda")


def safe_args(args):
    """vars(args) with any absolute path under the repo made repo-relative, so no username
    or local absolute path leaks into the saved checkpoint/JSON."""
    out = {}
    for k, v in vars(args).items():
        if (isinstance(v, str) and os.path.isabs(v)
                and os.path.commonpath([os.path.abspath(v), _ROOT]) == _ROOT):
            v = os.path.relpath(v, _ROOT).replace(os.sep, "/")
        out[k] = v
    return out


def load_dataset(path, feature_key):
    with h5py.File(path, "r") as hf:
        feat = torch.from_numpy(hf[feature_key][:].astype(np.float32))
        yt = torch.from_numpy(hf["y_theta_gated"][:].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][:].astype(np.int64))
        traj = hf["trajectory_id"][:]
        b_r = int(hf.attrs.get("B_r", 8))     # B_r-aware (E10 ablation)
    return feat, yt, yr, traj, b_r


def make_loader(feat, yt, yr, idx, bs, shuffle):
    return DataLoader(TensorDataset(feat[idx], yt[idx], yr[idx]),
                      batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0)


def vram_mb():
    return torch.cuda.max_memory_allocated() / 1e6


def overfit_check(model, feat, yt, yr, device, lambda_r, n=100, steps=1500):
    print(f"\n[overfit check] memorizing {n} samples (capacity / label-integrity)...")
    x = feat[:n].to(device); a = yt[:n].to(device); b = yr[:n].to(device)
    model.train()
    # lr=3e-3: lidar's angle signal is diluted by BEV background, so it converges slower
    # than radar; 1e-3 undershoots in 1500 steps (verified). 3e-3 reaches ~1.0 for both.
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda")
    for _ in tqdm(range(steps), desc="overfit", unit="step"):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, _ = polar_loss(model(x), a, b, lambda_r)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x)
    pt = out["theta_logits"].argmax(1); pr = out["range_logits"].argmax(1)
    top1 = float(((pt == a) & (pr == b)).float().mean())
    print(f"[overfit check] joint top-1 on {n} = {top1:.3f} "
          f"(theta {float((pt==a).float().mean()):.3f}, range {float((pr==b).float().mean()):.3f}) "
          f"-> {'PASS' if top1 >= 0.95 else 'LOW'}")


def train(args):
    seed_everything(args.seed)
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats()
    feature_key, ModelCls = MODALITY[args.modality]

    feat, yt, yr, traj, b_r = load_dataset(args.data, feature_key)
    print(f"[{args.modality}] {os.path.relpath(args.data,_ROOT)}: {feature_key} "
          f"{tuple(feat.shape)} | B_r={b_r}")
    model = ModelCls(d_z=128, B_theta=64, B_r=b_r).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"{ModelCls.__name__}: {nparams/1e6:.2f} M params")

    if args.overfit:
        overfit_check(model, feat, yt, yr, device, args.lambda_r)
        print(f"peak VRAM: {vram_mb():.0f} MB")
        return

    dtag = os.path.basename(args.data).replace("_full.h5", "")   # e.g. closerange_60GHz
    tr, va, te = trajectory_grouped_split(traj, seed=args.seed)
    print(f"split (seed {args.seed}): train {len(tr)} | val {len(va)} | test {len(te)}")
    tl = make_loader(feat, yt, yr, tr, args.batch_size, True)
    vl = make_loader(feat, yt, yr, va, args.batch_size, False)
    tel = make_loader(feat, yt, yr, te, args.batch_size, False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    ckpt_dir = os.path.join(_ROOT, "checkpoints", f"{dtag}_{args.modality}")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    print(f"\nTraining {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}, "
          f"lambda_r {args.lambda_r}, fp16\n")
    print(f"[device] model params on {next(model.parameters()).device}")
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0; t0 = time.time()
        for bi, (x, a, b) in enumerate(tqdm(tl, desc=f"ep {ep}/{args.epochs}",
                                            unit="batch", leave=False)):
            x = x.to(device, non_blocking=True); a = a.to(device); b = b.to(device)
            if ep == 1 and bi == 0:
                print(f"[device] live batch: x on {x.device}, model on "
                      f"{next(model.parameters()).device} (GPU-compute confirmed)")
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, _ = polar_loss(model(x), a, b, args.lambda_r)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss.detach()) * x.size(0)
        sched.step()
        v = detailed_eval(model, vl, device, B_r=b_r)
        history.append(dict(epoch=ep, train_loss=run / len(tr), val_theta=v["theta"],
                            val_range=v["range"], val_range_nf=v["range_nf"],
                            val_joint=v["joint"], lr=sched.get_last_lr()[0],
                            vram_mb=vram_mb()))
        print(f"ep {ep:3d} | loss {run/len(tr):.4f} | val theta {v['theta']:.3f} "
              f"range {v['range']:.3f} range_nf {v['range_nf']:.3f} joint {v['joint']:.3f}"
              f" | VRAM {vram_mb():.0f} MB | {time.time()-t0:.1f}s")
        if ep % 10 == 0 or ep == args.epochs:
            torch.save(dict(model=model.state_dict(), epoch=ep, args=safe_args(args)),
                       os.path.join(ckpt_dir, f"seed{args.seed}.pt"))

    test = detailed_eval(model, tel, device, B_r=b_r)
    print(f"\n=== FINAL {args.modality} (seed {args.seed}) ===")
    print_range_report(f"{args.modality} TEST", test)
    print(f"peak VRAM: {vram_mb():.0f} MB")
    res_dir = os.path.join(_ROOT, "results", "indomain")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir,
              f"{args.modality}_{dtag}_seed{args.seed}.json"), "w") as f:
        json.dump(dict(history=history, test=test, peak_vram_mb=vram_mb(),
                       params=nparams, args=safe_args(args)), f, indent=2)
    print("\nepoch | train_loss | val_range_nf | val_joint")
    for h in history:
        print(f"{h['epoch']:5d} | {h['train_loss']:.4f}     | "
              f"{h['val_range_nf']:.3f}        | {h['val_joint']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "lidar"], required=True)
    ap.add_argument("--data", default=os.path.join(
        _ROOT, "data", "processed", "closerange_60GHz_full.h5"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--lambda-r", type=float, default=1.0)
    ap.add_argument("--overfit", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
