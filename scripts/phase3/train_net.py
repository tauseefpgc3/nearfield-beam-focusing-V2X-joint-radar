"""D-031 location-aware networks: U-Net on the BEV grid with coordinate channels, no global
pooling, one logit map at full resolution.

    python scripts/phase3/train_net.py --sensor {radar,lidar,fused} --head {P,H} --lr 3e-4 --seed 11

P: soft-argmax over the map -> (x, y), smooth-L1 (beta 0.1 m) in metres.
H: cross-entropy vs a Gaussian target (sigma 0.2 m) on the true antenna; estimate =
   soft-argmax within the 5x5 cells around the arg-max.
AdamW wd 1e-4, cosine, 40 epochs, batch 16, fp32, final epoch. Train users, radar 20 dB.
Writes checkpoints/phase3/{tag}.pt and results/phase3/val_pred/{tag}.npy (validation only;
the test set is never loaded here).
"""
import argparse, json, os, sys, time
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from src.utils.seeding import seed_everything
P2 = os.path.join(ROOT, "data", "processed", "phase2"); P3 = os.path.join(ROOT, "data", "processed", "phase3")
CK = os.path.join(ROOT, "checkpoints", "phase3"); VP = os.path.join(ROOT, "results", "phase3", "val_pred")
BX0, BY0, CELL, NX, NY = -3.2, -20.8, 0.2, 128, 208
XC = torch.tensor(BX0 + CELL * (np.arange(NX) + 0.5), dtype=torch.float32)
YC = torch.tensor(BY0 + CELL * (np.arange(NY) + 0.5), dtype=torch.float32)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, w=(32, 64, 128, 256)):
        super().__init__()
        gx, gy = torch.meshgrid(XC / 22.4, YC / 20.8, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy])[None])
        self.enc = nn.ModuleList([block(cin + 2, w[0])] + [block(w[k], w[k + 1]) for k in range(3)])
        self.dec = nn.ModuleList([block(w[k + 1] + w[k], w[k]) for k in reversed(range(3))])
        self.out = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        x = torch.cat([x, self.coords.expand(len(x), -1, -1, -1)], 1); skips = []
        for k, e in enumerate(self.enc):
            x = e(x if k == 0 else F.max_pool2d(x, 2)); skips.append(x)
        for d, s in zip(self.dec, reversed(skips[:-1])):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.out(x)[:, 0]                       # (B, NX, NY) logits


def soft_argmax(lg):
    p = lg.flatten(1).softmax(1).view_as(lg)
    return torch.stack([(p.sum(2) * XC.to(lg.device)).sum(1), (p.sum(1) * YC.to(lg.device)).sum(1)], 1)


def local_soft_argmax(lg, r=2):
    B = len(lg); k = lg.flatten(1).argmax(1); ix, iy = k // NY, k % NY
    out = torch.empty(B, 2, device=lg.device)
    for b in range(B):
        x0, x1 = max(ix[b] - r, 0), min(ix[b] + r + 1, NX); y0, y1 = max(iy[b] - r, 0), min(iy[b] + r + 1, NY)
        w = lg[b, x0:x1, y0:y1].flatten().softmax(0).view(x1 - x0, y1 - y0)
        out[b, 0] = (w.sum(1) * XC[x0:x1].to(lg.device)).sum(); out[b, 1] = (w.sum(0) * YC[y0:y1].to(lg.device)).sum()
    return out


def gauss_target(xy, sig=0.2):
    dx = XC.to(xy.device)[None, :, None] - xy[:, 0, None, None]; dy = YC.to(xy.device)[None, None, :] - xy[:, 1, None, None]
    t = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2)); return t / t.flatten(1).sum(1)[:, None, None]


def load_inputs(sensor, idx):
    parts = []
    if sensor in ("radar", "fused"):
        with h5py.File(os.path.join(P3, "radar_bev20.h5"), "r") as hf: parts.append(hf["bev"][idx])
    if sensor in ("lidar", "fused"):
        with h5py.File(os.path.join(P3, "lidar_bev.h5"), "r") as hf: parts.append(hf["bev"][idx])
    return torch.from_numpy(np.concatenate(parts, 1))   # float16 in RAM


def predict(model, X, head, dev, bs=32):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            lg = model(X[b:b + bs].to(dev).float())
            out.append((soft_argmax(lg) if head == "P" else local_soft_argmax(lg)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["radar", "lidar", "fused"], required=True); ap.add_argument("--head", choices=["P", "H"], required=True)
    ap.add_argument("--lr", type=float, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(); seed_everything(a.seed); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    tag = f"{a.sensor}_{a.head}_lr{a.lr:g}_s{a.seed}"
    with h5py.File(os.path.join(P2, "split.h5"), "r") as hf: tr = np.sort(hf["train"][:]); va = np.sort(hf["val"][:])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as hf: xy = torch.tensor(hf["ant_xyz"][:, :2], dtype=torch.float32)
    Xtr, Xva = load_inputs(a.sensor, tr), load_inputs(a.sensor, va); Ytr = xy[tr]
    model = UNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    # prior level
    if a.head == "P":
        med = Ytr.median(0).values; prior = float(F.smooth_l1_loss(med.expand_as(Ytr), Ytr, beta=0.1, reduction="none").sum(1).mean())
    else:
        acc = torch.zeros(NX, NY)
        for b in range(0, len(Ytr), 256): acc += gauss_target(Ytr[b:b + 256]).sum(0)
        logm = (acc / len(Ytr)).clamp_min(1e-30).log(); ce = 0.0
        for b in range(0, len(Ytr), 256): ce += float(-(gauss_target(Ytr[b:b + 256]) * logm).flatten(1).sum(1).sum())
        prior = ce / len(Ytr)
    g = torch.Generator().manual_seed(a.seed); hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(tr), generator=g); run = 0.0; t0 = time.time()
        for b in range(0, len(tr), 16):
            j = perm[b:b + 16]; x = Xtr[j].to(dev).float(); y = Ytr[j].to(dev)
            lg = model(x)
            if a.head == "P":
                loss = F.smooth_l1_loss(soft_argmax(lg), y, beta=0.1, reduction="none").sum(1).mean()
            else:
                loss = -(gauss_target(y) * lg.flatten(1).log_softmax(1).view_as(lg)).flatten(1).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); run += float(loss.detach()) * len(j)
        sch.step()
        pv = predict(model, Xva, a.head, dev); e = np.hypot(*(pv - xy[va].numpy()).T)
        hist.append(dict(epoch=ep, train_loss=run / len(tr), prior=prior, val_err_med=float(np.median(e)), val_err_p90=float(np.percentile(e, 90)), sec=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
    os.makedirs(CK, exist_ok=True); os.makedirs(VP, exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(a), history=hist, cin=Xtr.shape[1]), os.path.join(CK, tag + ".pt"))
    np.save(os.path.join(VP, tag + ".npy"), pv)


if __name__ == "__main__":
    main()
