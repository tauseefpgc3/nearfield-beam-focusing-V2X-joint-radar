"""Train the radar+LiDAR cross-attention fusion predictor (Paper 5 Batch 7).

3-branch (radar / LiDAR / fused) supervision, modality dropout (p=0.15 each, never both,
train only), confidence-weighted ensemble at eval. Same close-range 60 GHz, seed 11,
trajectory 70/10/20 as the single-modality baselines.

    python scripts/train_fusion.py            # real run seed 11
    python scripts/train_fusion.py --overfit  # capacity check
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
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor, confidence_combine
from src.train.loss import polar_loss
from src.utils.seeding import seed_everything

P_DROP = 0.15  # modality dropout prob per modality (never both), §5.3


def require_cuda():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP (no CPU fallback for training).")
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


def load_data(path):
    with h5py.File(path, "r") as hf:
        xr = torch.from_numpy(hf["radar_features"][:].astype(np.float32))
        xl = torch.from_numpy(hf["lidar_features"][:].astype(np.float32))
        yt = torch.from_numpy(hf["y_theta_gated"][:].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][:].astype(np.int64))
        traj = hf["trajectory_id"][:]
    return xr, xl, yt, yr, traj


def loader(xr, xl, yt, yr, idx, bs, shuffle):
    return DataLoader(TensorDataset(xr[idx], xl[idx], yt[idx], yr[idx]),
                      batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0)


def branch_loss(out, yt, yr, lambda_r):
    return sum(polar_loss({"theta_logits": out[f"{b}_theta_logits"],
                           "range_logits": out[f"{b}_range_logits"]}, yt, yr, lambda_r)[0]
               for b in BRANCHES)


def vram_mb():
    return torch.cuda.max_memory_allocated() / 1e6


@torch.no_grad()
def fusion_eval(model, ld, device, B_r=8):
    model.eval()
    YT, PT, YR, PR = [], [], [], []
    wsum = {"theta": np.zeros(3), "range": np.zeros(3)}
    nseen = 0
    for xr, xl, yt, yr in ld:
        xr = xr.to(device); xl = xl.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p_theta, p_r, w = confidence_combine(model(xr, xl), return_weights=True)
        PT.append(p_theta.argmax(1).cpu()); PR.append(p_r.argmax(1).cpu())
        YT.append(yt); YR.append(yr)
        bs = yt.numel(); nseen += bs
        for i, b in enumerate(BRANCHES):
            wsum["theta"][i] += w["theta"][b] * bs
            wsum["range"][i] += w["range"][b] * bs
    YT = torch.cat(YT); PT = torch.cat(PT); YR = torch.cat(YR); PR = torch.cat(PR)
    nf = YR > 0
    cm = np.zeros((B_r, B_r), int)
    for t, p in zip(YR.numpy(), PR.numpy()):
        cm[t, p] += 1
    return dict(
        theta=float((PT == YT).float().mean()),
        range=float((PR == YR).float().mean()),
        joint=float(((PT == YT) & (PR == YR)).float().mean()),
        range_nf=float((PR[nf] == YR[nf]).float().mean()) if int(nf.sum()) else float("nan"),
        n_nf=int(nf.sum()),
        conf_theta={b: wsum["theta"][i] / nseen for i, b in enumerate(BRANCHES)},
        conf_range={b: wsum["range"][i] / nseen for i, b in enumerate(BRANCHES)},
        range_cm=cm.tolist())


def train(args):
    seed_everything(args.seed)
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats()
    xr, xl, yt, yr, traj = load_data(args.data)
    print(f"radar {tuple(xr.shape)} | lidar {tuple(xl.shape)}")
    model = SensingPolarPredictor(d_z=128, B_theta=64, B_r=8).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"SensingPolarPredictor: {nparams/1e6:.2f} M params")

    if args.overfit:
        n = 100
        a = xr[:n].to(device); c = xl[:n].to(device)
        u = yt[:n].to(device); v = yr[:n].to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        for _ in tqdm(range(1500), desc="overfit", unit="step"):
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = branch_loss(model(a, c), u, v, args.lambda_r)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            p_theta, p_r = confidence_combine(model(a, c))
        top1 = float(((p_theta.argmax(1) == u) & (p_r.argmax(1) == v)).float().mean())
        print(f"[overfit] joint top-1 on {n} = {top1:.3f} -> "
              f"{'PASS' if top1 >= 0.9 else 'LOW'} | peak VRAM {vram_mb():.0f} MB")
        return

    tag = "fusion"
    if args.warmstart_radar:
        sd = torch.load(args.warmstart_radar, map_location="cpu",
                        weights_only=False)["model"]
        model.radar_encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
        model.head_radar.load_state_dict(
            {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")})
        print(f"warmstarted radar branch from {os.path.relpath(args.warmstart_radar,_ROOT)}")
        tag = "fusion_warmstart"
        if args.freeze_radar:
            for p in (*model.radar_encoder.parameters(), *model.head_radar.parameters()):
                p.requires_grad_(False)
            print("radar branch FROZEN (encoder + head_radar)")

    tr, va, te = trajectory_grouped_split(traj, seed=args.seed)
    print(f"split (seed {args.seed}): train {len(tr)} | val {len(va)} | test {len(te)}")
    tl = loader(xr, xl, yt, yr, tr, args.batch_size, True)
    vl = loader(xr, xl, yt, yr, va, args.batch_size, False)
    tel = loader(xr, xl, yt, yr, te, args.batch_size, False)
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in train_params)/1e6:.2f} M "
          f"of {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    dtag = os.path.basename(args.data).replace("_full.h5", "")   # e.g. closerange_60GHz
    ckpt_dir = os.path.join(_ROOT, "checkpoints", f"{dtag}_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    history = []
    print(f"\nTraining {args.epochs} epochs, batch {args.batch_size}, modality dropout "
          f"p={P_DROP} (never both), fp16\n")
    print(f"[device] model params on {next(model.parameters()).device}")
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0; t0 = time.time()
        for bi, (xrb, xlb, a, b) in enumerate(tqdm(tl, desc=f"ep {ep}/{args.epochs}",
                                                   unit="batch", leave=False)):
            xrb = xrb.to(device, non_blocking=True); xlb = xlb.to(device, non_blocking=True)
            a = a.to(device); b = b.to(device)
            if ep == 1 and bi == 0:
                print(f"[device] live batch: xr on {xrb.device}, model on "
                      f"{next(model.parameters()).device} (GPU-compute confirmed)")
            u = rng.random()                       # modality dropout (never both)
            rd, ld = (u < P_DROP), (P_DROP <= u < 2 * P_DROP)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = branch_loss(model(xrb, xlb, radar_dropout=rd, lidar_dropout=ld),
                                   a, b, args.lambda_r)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss.detach()) * a.size(0)
        sched.step()
        v = fusion_eval(model, vl, device)
        history.append(dict(epoch=ep, train_loss=run / len(tr), val_theta=v["theta"],
                            val_range=v["range"], val_range_nf=v["range_nf"],
                            val_joint=v["joint"], vram_mb=vram_mb()))
        print(f"ep {ep:3d} | loss {run/len(tr):.4f} | val theta {v['theta']:.3f} "
              f"range {v['range']:.3f} range_nf {v['range_nf']:.3f} joint {v['joint']:.3f}"
              f" | VRAM {vram_mb():.0f} MB | {time.time()-t0:.1f}s")
        if ep % 10 == 0 or ep == args.epochs:
            torch.save(dict(model=model.state_dict(), epoch=ep, args=safe_args(args)),
                       os.path.join(ckpt_dir, f"seed{args.seed}.pt"))

    test = fusion_eval(model, tel, device)
    print(f"\n=== FUSION FINAL (seed {args.seed}) ===")
    print(f"TEST top-1: theta {test['theta']:.3f} | range {test['range']:.3f} | "
          f"range_nf {test['range_nf']:.3f} | joint {test['joint']:.3f}")
    print(f"mean confidence weights (theta): {dict((k,round(x,3)) for k,x in test['conf_theta'].items())}")
    print(f"mean confidence weights (range): {dict((k,round(x,3)) for k,x in test['conf_range'].items())}")
    print(f"peak VRAM: {vram_mb():.0f} MB")
    res_dir = os.path.join(_ROOT, "results", "indomain")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, f"{tag}_{dtag}_seed{args.seed}.json"), "w") as f:
        json.dump(dict(history=history, test=test, peak_vram_mb=vram_mb(),
                       params=nparams, args=safe_args(args)), f, indent=2)
    print("\nepoch | train_loss | val_range_nf | val_joint")
    for h in history:
        print(f"{h['epoch']:5d} | {h['train_loss']:.4f}     | {h['val_range_nf']:.3f}"
              f"        | {h['val_joint']:.3f}")


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
    ap.add_argument("--warmstart-radar", default=None,
                    help="path to Batch-5 radar checkpoint to init the radar branch")
    ap.add_argument("--freeze-radar", action="store_true",
                    help="freeze the warmstarted radar branch (cleanest test)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
