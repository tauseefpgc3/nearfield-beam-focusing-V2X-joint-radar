"""Train the radar-only (angle, range) polar classifier (Paper 5 Batch 5).

Single-modality baseline on the close-range PRIMARY dataset. GPU-only (mixed precision),
trajectory-grouped split, gated labels (y_theta_gated / y_r_gated).

    python scripts/train_radar.py --overfit          # 100-sample capacity check
    python scripts/train_radar.py                     # real run, seed 11
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
from src.models.radar_encoder import RadarPolarModel
from src.train.loss import polar_loss
from src.utils.seeding import seed_everything


def require_cuda():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP. (Do not silently train on CPU; docs/10.)")
    torch.backends.cudnn.benchmark = True   # fixed input shape -> autotune fast convs
    name = torch.cuda.get_device_name(0)
    _ = float((torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum())
    print(f"GPU: {name} | VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    return torch.device("cuda")


def load_dataset(path):
    with h5py.File(path, "r") as hf:
        radar = torch.from_numpy(hf["radar_features"][:].astype(np.float32))  # (N,2,256,128)
        yt = torch.from_numpy(hf["y_theta_gated"][:].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][:].astype(np.int64))
        traj = hf["trajectory_id"][:]
    return radar, yt, yr, traj


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ct = cr = cj = n = 0
    for x, yt, yr in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
        pt = out["theta_logits"].argmax(1).cpu()
        pr = out["range_logits"].argmax(1).cpu()
        ct += int((pt == yt).sum()); cr += int((pr == yr).sum())
        cj += int(((pt == yt) & (pr == yr)).sum()); n += yt.numel()
    return dict(theta=ct / n, range=cr / n, joint=cj / n)


def make_loader(radar, yt, yr, idx, bs, shuffle):
    ds = TensorDataset(radar[idx], yt[idx], yr[idx])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0)


def vram_mb():
    return torch.cuda.max_memory_allocated() / 1e6


# ---------------------------------------------------------------------------
def overfit_check(model, radar, yt, yr, device, lambda_r, n=100, steps=1500):
    print(f"\n[overfit check] memorizing {n} samples (capacity / label-integrity)...")
    x = radar[:n].to(device); a = yt[:n].to(device); b = yr[:n].to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    for step in tqdm(range(steps), desc="overfit", unit="step"):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, _ = polar_loss(model(x), a, b, lambda_r)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x)
    pt = out["theta_logits"].argmax(1); pr = out["range_logits"].argmax(1)
    top1 = float(((pt == a) & (pr == b)).float().mean())
    print(f"[overfit check] joint top-1 on {n} samples = {top1:.3f} "
          f"(theta {float((pt==a).float().mean()):.3f}, range {float((pr==b).float().mean()):.3f})")
    print(f"[overfit check] {'PASS' if top1 >= 0.95 else 'LOW'} (target ~1.0)")
    return top1


def train(args):
    seed_everything(args.seed)
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats()

    radar, yt, yr, traj = load_dataset(args.data)
    print(f"Dataset {os.path.relpath(args.data,_ROOT)}: radar {tuple(radar.shape)} | "
          f"labels y_theta[0,{int(yt.max())}] y_r[0,{int(yr.max())}]")

    model = RadarPolarModel(d_z=128, B_theta=64, B_r=8).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"RadarPolarModel: {nparams/1e6:.2f} M params")

    if args.overfit:
        overfit_check(model, radar, yt, yr, device, args.lambda_r)
        print(f"peak VRAM: {vram_mb():.0f} MB")
        return

    tr, va, te = trajectory_grouped_split(traj, seed=args.seed)
    print(f"split (seed {args.seed}): train {len(tr)} | val {len(va)} | test {len(te)}")
    tl = make_loader(radar, yt, yr, tr, args.batch_size, True)
    vl = make_loader(radar, yt, yr, va, args.batch_size, False)
    tel = make_loader(radar, yt, yr, te, args.batch_size, False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = os.path.join(_ROOT, "checkpoints", "closerange_60GHz_radar")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    print(f"\nTraining {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}, "
          f"lambda_r {args.lambda_r}, mixed precision (fp16)\n")
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        t0 = time.time()
        for x, a, b in tqdm(tl, desc=f"epoch {ep}/{args.epochs}", unit="batch", leave=False):
            x = x.to(device, non_blocking=True); a = a.to(device); b = b.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, _ = polar_loss(model(x), a, b, args.lambda_r)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss.detach()) * x.size(0)
        sched.step()
        train_loss = run / len(tr)
        val = evaluate(model, vl, device)
        history.append(dict(epoch=ep, train_loss=train_loss, val_theta=val["theta"],
                            val_range=val["range"], val_joint=val["joint"],
                            lr=sched.get_last_lr()[0], vram_mb=vram_mb()))
        print(f"ep {ep:3d} | train_loss {train_loss:.4f} | val top-1 "
              f"theta {val['theta']:.3f} range {val['range']:.3f} joint {val['joint']:.3f}"
              f" | lr {sched.get_last_lr()[0]:.2e} | VRAM {vram_mb():.0f} MB | "
              f"{time.time()-t0:.1f}s")
        if ep % 10 == 0 or ep == args.epochs:
            torch.save(dict(model=model.state_dict(), epoch=ep, val=val, args=vars(args)),
                       os.path.join(ckpt_dir, f"seed{args.seed}.pt"))

    test = evaluate(model, tel, device)
    print(f"\n=== FINAL (seed {args.seed}) ===")
    print(f"TEST  top-1: theta {test['theta']:.3f} | range {test['range']:.3f} | "
          f"joint {test['joint']:.3f}")
    print(f"peak VRAM: {vram_mb():.0f} MB")
    # save curve + metrics
    res_dir = os.path.join(_ROOT, "results", "indomain")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, f"radar_closerange60_seed{args.seed}.json"), "w") as f:
        json.dump(dict(history=history, test=test, peak_vram_mb=vram_mb(),
                       params=nparams, args=vars(args)), f, indent=2)
    print("\nepoch | train_loss | val_joint")
    for h in history:
        print(f"{h['epoch']:5d} | {h['train_loss']:.4f}     | {h['val_joint']:.3f}")


def main():
    ap = argparse.ArgumentParser()
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
