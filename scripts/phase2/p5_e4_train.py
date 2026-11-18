"""D-030 E4 (exploratory): can the radar CNN learn? Seed 11, 60 GHz, all train/val users at
20 dB, fp32, Adam lr in {1e-3, 3e-4}, wd 1e-5, cosine 50 epochs, batch 16. Test unused.
    python scripts/phase2/p5_e4_train.py --lr 1e-3
"""
import argparse, json, os, sys, time
import h5py, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.models.radar_encoder import RadarPolarModel
from src.train.loss import polar_loss
from src.utils.seeding import seed_everything
P = os.path.join(ROOT, "data", "processed", "phase2"); OUT = os.path.join(ROOT, "results", "phase2", "feasibility")


def H(y):
    p = np.bincount(y) / len(y); p = p[p > 0]; return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lr", type=float, required=True); ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args(); seed_everything(11); dev = torch.device("cuda")
    with h5py.File(os.path.join(P, "split.h5"), "r") as hf: tr = np.sort(hf["train"][:]); va = np.sort(hf["val"][:])
    with h5py.File(os.path.join(P, "channels_60GHz.h5"), "r") as hf:
        yt = hf["y_theta"][:].astype(np.int64); yr = hf["y_r"][:].astype(np.int64); ug = hf["u_grid"][:]
    with h5py.File(os.path.join(P, "radar_snr20.h5"), "r") as hf: Xtr = hf["radar_features"][tr]; Xva = hf["radar_features"][va]
    with h5py.File(os.path.join(P, "users.h5"), "r") as hf: ant = hf["ant_xyz"][:]
    rel = ant - np.array([0, 0, 2.0]); rt = np.linalg.norm(rel, axis=1); ut = rel[:, 1] / rt; dh = np.hypot(ant[:, 0], ant[:, 1])
    prior = H(yt[tr]) + H(yr[tr])
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(yt[tr]), torch.from_numpy(yr[tr]))
    tl = DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True)
    Xv = torch.from_numpy(Xva)
    m = RadarPolarModel(d_z=128, B_theta=256, B_r=8).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, weight_decay=1e-5); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    hist = []
    for ep in range(1, a.epochs + 1):
        m.train(); run = 0.0; t0 = time.time()
        for x, y1, y2 in tl:
            x, y1, y2 = x.to(dev), y1.to(dev), y2.to(dev); opt.zero_grad()
            l = polar_loss(m(x), y1, y2)[0]; l.backward(); opt.step(); run += float(l.detach()) * len(x)
        sch.step(); m.eval(); TH, RG = [], []
        with torch.no_grad():
            for b in range(0, len(Xv), 128):
                o = m(Xv[b:b + 128].to(dev)); TH.append(o["theta_logits"].argmax(1).cpu().numpy()); RG.append(o["range_logits"].argmax(1).cpu().numpy())
        th = np.concatenate(TH); rg = np.concatenate(RG)
        side = np.abs(ug[th] - ut[va]) * dh[va]
        h = dict(epoch=ep, train_loss=run / len(ds), prior_entropy=prior, val_theta=float((th == yt[va]).mean()),
                 val_theta_pm1=float((np.abs(th - yt[va]) <= 1).mean()), val_ring=float((rg == yr[va]).mean()),
                 val_side_med=float(np.median(side)), val_side_p90=float(np.percentile(side, 90)), n_pred_classes=int(len(set(th))))
        hist.append(h); print(json.dumps(h), flush=True)
    json.dump(dict(lr=a.lr, history=hist), open(os.path.join(OUT, f"e4_lr{a.lr:g}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
