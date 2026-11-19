"""D-036 B: multi-car LiDAR net, frozen D-031 design and settings (train_net.py H head: U-Net with
coordinate channels, Gaussian-target CE sigma 0.2 m, AdamW lr 1e-3 wd 1e-4, cosine, 40 epochs,
batch 16, fp32, final epoch). Inputs = multi-car training BEVs after association rule R.

    python scripts/phase4/b_train.py --rule R1 --seed 11   # -> checkpoints/phase4/multicar/lidar_H_lr0.001_s{seed}_{rule}.pt
"""
import argparse, json, os, sys, time
import h5py, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
from src.utils.seeding import seed_everything
from train_net import UNet, gauss_target, predict, NX, NY
D = os.path.join(ROOT, "data", "processed", "phase4", "multicar"); CK = os.path.join(ROOT, "checkpoints", "phase4", "multicar")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rule", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(); seed_everything(a.seed); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True; lr = 1e-3
    tag = f"lidar_H_lr{lr:g}_s{a.seed}_{a.rule}"
    with h5py.File(os.path.join(D, f"train_{a.rule}.h5"), "r") as h: Xtr = torch.from_numpy(h["bev"][:]); Ytr = torch.tensor(h["xy"][:], dtype=torch.float32)
    with h5py.File(os.path.join(D, f"val_{a.rule}_s3.h5"), "r") as h: Xva = torch.from_numpy(h["bev"][:]); yva = h["xy"][:]
    model = UNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    acc = torch.zeros(NX, NY)
    for b in range(0, len(Ytr), 256): acc += gauss_target(Ytr[b:b + 256]).sum(0)
    logm = (acc / len(Ytr)).clamp_min(1e-30).log(); ce = 0.0
    for b in range(0, len(Ytr), 256): ce += float(-(gauss_target(Ytr[b:b + 256]) * logm).flatten(1).sum(1).sum())
    prior = ce / len(Ytr); g = torch.Generator().manual_seed(a.seed); hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); run = 0.0; t0 = time.time()
        for b in range(0, len(Xtr), 16):
            j = perm[b:b + 16]; x = Xtr[j].to(dev).float(); y = Ytr[j].to(dev); lg = model(x)
            loss = -(gauss_target(y) * lg.flatten(1).log_softmax(1).view_as(lg)).flatten(1).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); run += float(loss.detach()) * len(j)
        sch.step(); pv = predict(model, Xva, "H", dev); e = np.hypot(*(pv - yva).T)
        hist.append(dict(epoch=ep, train_loss=run / len(Xtr), prior=prior, val_err_med=float(np.median(e)), val_err_p90=float(np.percentile(e, 90)), sec=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
    os.makedirs(CK, exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(a), history=hist, cin=Xtr.shape[1]), os.path.join(CK, tag + ".pt"))


if __name__ == "__main__":
    main()
