"""D-034 B6 diagnostic (bug check after the test result): classifier seed 11, 60 GHz, on train / val / test
vs a most-frequent-codeword prior, and the share of each split's genie codewords seen in training.

    python scripts/phase3d/b6_diagnostic.py   # -> results/phase3d/b6_diagnostic.json
"""
import os, sys, json, h5py, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); OUT = {}; sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3d")); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
from train_cls import Cls, logits, topm_loss, P2
from train_net import load_inputs
with h5py.File(os.path.join(P2, "split.h5")) as h: tr, va, te = [np.sort(h[k][:]) for k in ("train", "val", "test")]
with h5py.File(os.path.join(P2, "channels_60GHz.h5")) as h: lab = h["label"][:]; G = h["gains"][:]
OUT["label_equals_genie_argmax"] = float((lab == G.argmax(1)).mean())
freq = np.argsort(-np.bincount(lab[tr], minlength=2048))
for nm, idx in (("train", tr), ("val", va), ("test", te)):
    OUT[nm] = dict(prior_top8_loss_db=float(topm_loss(G[idx], np.tile(-np.arange(2048.)[np.argsort(freq)], (len(idx), 1)), 8).mean()),
                   distinct_codewords=int(len(np.unique(lab[idx]))), share_codeword_seen_in_train=float(np.isin(lab[idx], lab[tr]).mean()))
ck = torch.load("checkpoints/phase3d/cls60_G_lr0.001_s11.pt", map_location="cpu", weights_only=False)
m = Cls(ck["cin"], "G"); m.load_state_dict(ck["model"]); m = m.cuda()
for nm, idx in (("train", tr), ("val", va), ("test", te)):
    lg = logits(m, load_inputs("lidar", idx), "cuda")
    OUT[nm] |= dict(net_top1=float((lg.argmax(1) == lab[idx]).mean()), net_top8_loss_db=float(topm_loss(G[idx], lg, 8).mean()))
json.dump(OUT, open(os.path.join(ROOT, "results", "phase3d", "b6_diagnostic.json"), "w"), indent=1); print(json.dumps(OUT, indent=1))
