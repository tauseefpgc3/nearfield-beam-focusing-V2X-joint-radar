"""D-034 B6: direct joint-codeword classifier on the LiDAR BEV (same U-Net backbone as D-031).

    python scripts/phase3d/train_cls.py --freq 60 --head G --lr 1e-3 --seed 11

head G: deepest encoder features (256 ch, 1/8 res) -> global average pool -> linear(2048).
head D: full decoder output (32 ch, full res) -> [global average, global max] -> linear(2048).
Label = genie codeword (channels_{f}GHz.h5 'label', 2048 classes); cross-entropy. Frozen
optimiser settings: AdamW wd 1e-4, cosine, 40 epochs, batch 16, fp32, final epoch.
Validation per epoch: top-1 accuracy and mean loss vs genie when the top-8 codewords by logit
are swept. The test set is never loaded here.
Writes checkpoints/phase3d/cls{f}_{head}_lr{lr}_s{seed}.pt (model, args, history).
"""
import argparse, json, os, sys, time
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
from src.utils.seeding import seed_everything
from train_net import UNet, load_inputs
P2 = os.path.join(ROOT, "data", "processed", "phase2"); CK = os.path.join(ROOT, "checkpoints", "phase3d")
NCLS = 2048


class Cls(nn.Module):
    def __init__(self, cin, head):
        super().__init__(); self.head = head; self.u = UNet(cin)
        self.fc = nn.Linear(256 if head == "G" else 64, NCLS)

    def forward(self, x):
        u = self.u; x = torch.cat([x, u.coords.expand(len(x), -1, -1, -1)], 1); skips = []
        for k, e in enumerate(u.enc):
            x = e(x if k == 0 else F.max_pool2d(x, 2)); skips.append(x)
        if self.head == "G":
            return self.fc(x.mean((2, 3)))
        for d, s in zip(u.dec, reversed(skips[:-1])):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.fc(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))


def logits(model, X, dev, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs): out.append(model(X[b:b + bs].to(dev).float()).cpu())
    return torch.cat(out).numpy()


def topm_loss(G, lg, M):
    top = np.argsort(-lg, 1, kind="stable")[:, :M]
    return 10 * np.log10(G.max(1) / np.take_along_axis(G, top, 1).max(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, choices=[60, 140], required=True); ap.add_argument("--head", choices=["G", "D"], required=True)
    ap.add_argument("--lr", type=float, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(); seed_everything(a.seed); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    tag = f"cls{a.freq}_{a.head}_lr{a.lr:g}_s{a.seed}"
    with h5py.File(os.path.join(P2, "split.h5"), "r") as hf: tr = np.sort(hf["train"][:]); va = np.sort(hf["val"][:])
    with h5py.File(os.path.join(P2, f"channels_{a.freq}GHz.h5"), "r") as hf: lab = hf["label"][:]; Gva = hf["gains"][va]
    Xtr, Xva = load_inputs("lidar", tr), load_inputs("lidar", va); Ytr = torch.tensor(lab[tr], dtype=torch.long)
    pm = np.bincount(lab[tr], minlength=NCLS) / len(tr); prior = float(-(pm[pm > 0] * np.log(pm[pm > 0])).sum())
    model = Cls(Xtr.shape[1], a.head).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    g = torch.Generator().manual_seed(a.seed); hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(tr), generator=g); run = 0.0; t0 = time.time()
        for b in range(0, len(tr), 16):
            j = perm[b:b + 16]; loss = F.cross_entropy(model(Xtr[j].to(dev).float()), Ytr[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); run += float(loss.detach()) * len(j)
        sch.step()
        lv = logits(model, Xva, dev)
        hist.append(dict(epoch=ep, train_loss=run / len(tr), prior=prior, val_top1=float((lv.argmax(1) == lab[va]).mean()),
                         val_loss_M8=float(topm_loss(Gva, lv, 8).mean()), sec=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
    os.makedirs(CK, exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(a), history=hist, cin=Xtr.shape[1]), os.path.join(CK, tag + ".pt"))


if __name__ == "__main__":
    main()
