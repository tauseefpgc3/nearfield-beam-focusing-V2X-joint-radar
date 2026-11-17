"""D-033 C2: location-aware LiDAR network on DeepSense (beam prediction).

    python scripts/phase3c/train_ds.py --scenario 31 --seed 11

Trunk = the frozen D-031 coordinate U-Net (scripts/phase3/train_net.py `block`, widths
32/64/128/256, three max-pools, bilinear up-sampling, skip connections). Only the input grid
(coordinate channels normalised to the scenario grid) and the output change: the output conv
has 1 + B channels; channel 0 -> p(cell) = softmax over cells, channels 1..B -> per-cell beam
logits; log P(b) = logsumexp_cells(log p(cell) + log_softmax_b l(cell, b)). Loss = CE of
log P vs the best beam. AdamW lr 1e-3, wd 1e-4, cosine, 40 epochs, batch 16, fp32, final
epoch. Train split only; validation top-1 is logged; test is never loaded here.
"""
import argparse, json, os, sys, time
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
from src.utils.seeding import seed_everything
from train_net import block
DATA = os.path.join(ROOT, "data", "processed", "phase3c"); CK = os.path.join(ROOT, "checkpoints", "phase3c")


class UNetDS(nn.Module):
    def __init__(self, cin, nx, ny, x0, y0, cell, n_beams, w=(32, 64, 128, 256)):
        super().__init__()
        xc = x0 + cell * (np.arange(nx) + 0.5); yc = y0 + cell * (np.arange(ny) + 0.5)
        gx, gy = np.meshgrid(xc / np.abs(xc).max(), yc / np.abs(yc).max(), indexing="ij")
        self.register_buffer("coords", torch.tensor(np.stack([gx, gy])[None], dtype=torch.float32))
        self.enc = nn.ModuleList([block(cin + 2, w[0])] + [block(w[k], w[k + 1]) for k in range(3)])
        self.dec = nn.ModuleList([block(w[k + 1] + w[k], w[k]) for k in reversed(range(3))])
        self.out = nn.Conv2d(w[0], 1 + n_beams, 1)

    def forward(self, x):
        """-> log P(beam) (B, n_beams)"""
        x = torch.cat([x, self.coords.expand(len(x), -1, -1, -1)], 1); skips = []
        for k, e in enumerate(self.enc):
            x = e(x if k == 0 else F.max_pool2d(x, 2)); skips.append(x)
        for d, s in zip(self.dec, reversed(skips[:-1])):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        o = self.out(x).flatten(2)                                   # (B, 1+nb, cells)
        logp_cell = o[:, 0].log_softmax(1)                          # (B, cells)
        logp_beam = o[:, 1:].log_softmax(1)                         # (B, nb, cells)
        return torch.logsumexp(logp_cell[:, None] + logp_beam, 2)


def build(hf):
    a = hf.attrs
    return UNetDS(3, int(a["NX"]), int(a["NY"]), float(a["x0"]), float(a["y0"]), float(a["cell"]), int(hf["power"].shape[1]))


def predict(model, X, dev, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs): out.append(model(X[b:b + bs].to(dev).float()).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40); a = ap.parse_args()
    seed_everything(a.seed); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    tag = f"ds{a.scenario}_s{a.seed}"
    with h5py.File(os.path.join(DATA, f"ds{a.scenario}.h5"), "r") as hf:
        sp = hf["split"][:]; tr = np.where(sp == 0)[0]; va = np.where(sp == 1)[0]
        Xtr = torch.from_numpy(hf["bev"][tr]); Xva = torch.from_numpy(hf["bev"][va])
        lab = torch.from_numpy(hf["label"][:].astype(np.int64)); model = build(hf).to(dev)
    Ytr, Yva = lab[tr], lab[va]; nb = model.out.out_channels - 1
    pm = torch.bincount(Ytr, minlength=nb).float() / len(Ytr); prior = float(-(pm[Ytr].clamp_min(1e-12).log()).mean())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    g = torch.Generator().manual_seed(a.seed); hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(tr), generator=g); run = 0.0; t0 = time.time()
        for b in range(0, len(tr), 16):
            j = perm[b:b + 16]; lp = model(Xtr[j].to(dev).float()); loss = F.nll_loss(lp, Ytr[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); run += float(loss.detach()) * len(j)
        sch.step()
        pv = predict(model, Xva, dev)
        hist.append(dict(epoch=ep, train_loss=run / len(tr), prior=prior, val_top1=float((pv.argmax(1) == Yva.numpy()).mean()),
                         val_nll=float(-pv[np.arange(len(va)), Yva.numpy()].mean()), sec=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
    os.makedirs(CK, exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(a), history=hist), os.path.join(CK, tag + ".pt"))


if __name__ == "__main__":
    main()
