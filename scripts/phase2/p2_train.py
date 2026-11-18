"""Phase 2a (D-029) training: radar-alone CNN or three-branch model, one configuration.

    python scripts/phase2/p2_train.py --model radar  --freq 60 --seed 11
    python scripts/phase2/p2_train.py --model fusion --freq 60 --seed 11   # warm-starts from radar

Adam lr 1e-4, wd 1e-5, cosine over 50 epochs, batch 16, fp16, CE(theta)+CE(r); fusion:
modality dropout p=0.15 each (never both), radar branch warm-started from the same-seed
radar checkpoint and UNFROZEN; final-epoch checkpoint. Train/val users at their assigned SNR
(split.h5 snr_idx); test is never loaded here.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor
from src.train.loss import polar_loss
from src.utils.seeding import seed_everything

P = os.path.join(ROOT, "data", "processed", "phase2")
CK = os.path.join(ROOT, "checkpoints", "phase2")
SNRS = (20, 10, 0)
B_TH, B_R = 256, 8


def mixed_radar(idx, snr_idx):
    """idx sorted ascending; each user's radar tensor at its assigned SNR."""
    out = np.zeros((len(idx), 2, 256, 128), np.float32)
    for s, snr in enumerate(SNRS):
        pos = np.where(snr_idx[idx] == s)[0]
        if len(pos):
            with h5py.File(os.path.join(P, f"radar_snr{snr}.h5"), "r") as hf:
                out[pos] = hf["radar_features"][idx[pos]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["radar", "fusion"], required=True)
    ap.add_argument("--freq", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    seed_everything(a.seed)
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    with h5py.File(os.path.join(P, "split.h5"), "r") as hf:
        tr = np.sort(hf["train"][:]); va = np.sort(hf["val"][:]); snr_idx = hf["snr_idx"][:]
    with h5py.File(os.path.join(P, f"channels_{a.freq}GHz.h5"), "r") as hf:
        yt = hf["y_theta"][:].astype(np.int64); yr = hf["y_r"][:].astype(np.int64)
    Xr = {k: torch.from_numpy(mixed_radar(ix, snr_idx)) for k, ix in (("tr", tr), ("va", va))}
    Y = {k: (torch.from_numpy(yt[ix]), torch.from_numpy(yr[ix])) for k, ix in (("tr", tr), ("va", va))}
    if a.model == "fusion":
        with h5py.File(os.path.join(P, "lidar.h5"), "r") as hf:
            Xl = {k: torch.from_numpy(hf["lidar_features"][ix]) for k, ix in (("tr", tr), ("va", va))}
        model = SensingPolarPredictor(d_z=128, B_theta=B_TH, B_r=B_R).to(dev)
        sd = torch.load(os.path.join(CK, f"radar_{a.freq}GHz_seed{a.seed}.pt"), map_location="cpu", weights_only=False)["model"]
        model.radar_encoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")})
        model.head_radar.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
        ds = TensorDataset(Xr["tr"], Xl["tr"], *Y["tr"]); vds = TensorDataset(Xr["va"], Xl["va"], *Y["va"])
    else:
        model = RadarPolarModel(d_z=128, B_theta=B_TH, B_r=B_R).to(dev)
        ds = TensorDataset(Xr["tr"], *Y["tr"]); vds = TensorDataset(Xr["va"], *Y["va"])
    tl = DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True)
    vl = DataLoader(vds, batch_size=128, shuffle=False, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(a.seed)
    os.makedirs(CK, exist_ok=True)
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for batch in tl:
            batch = [b.to(dev, non_blocking=True) for b in batch]
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if a.model == "fusion":
                    u = rng.random(); rd, ld = u < 0.15, 0.15 <= u < 0.30
                    out = model(batch[0], batch[1], radar_dropout=rd, lidar_dropout=ld)
                    loss = sum(polar_loss({"theta_logits": out[f"{b}_theta_logits"], "range_logits": out[f"{b}_range_logits"]},
                                          batch[2], batch[3])[0] for b in BRANCHES)
                else:
                    loss = polar_loss(model(batch[0]), batch[1], batch[2])[0]
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss.detach()) * batch[0].shape[0]
        sched.step()
        model.eval(); ct = cr = n = 0
        with torch.no_grad():
            for batch in vl:
                batch = [b.to(dev) for b in batch]
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    if a.model == "fusion":
                        o = model(batch[0], batch[1]); th = o["radar_theta_logits"]; rg = o["fused_range_logits"]; yv = batch[2:]
                    else:
                        o = model(batch[0]); th = o["theta_logits"]; rg = o["range_logits"]; yv = batch[1:]
                ct += int((th.argmax(1) == yv[0]).sum()); cr += int((rg.argmax(1) == yv[1]).sum()); n += len(yv[0])
        hist.append(dict(epoch=ep, train_loss=run / len(ds), val_theta=ct / n, val_range=cr / n, sec=time.time() - t0))
        print(f"ep {ep:2d} loss {run/len(ds):.4f} val theta {ct/n:.3f} range {cr/n:.3f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(dict(model=model.state_dict(), epoch=a.epochs, args=vars(a), history=hist),
               os.path.join(CK, f"{a.model}_{a.freq}GHz_seed{a.seed}.pt"))


if __name__ == "__main__":
    main()
