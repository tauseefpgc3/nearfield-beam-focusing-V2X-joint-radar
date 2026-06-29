"""DeepSense genie-decomposition + zero-shot transfer, banked as JSON (eval-only, Paper 5).

This re-runs (NO training) two analyses that were previously console-only, writing them to
results/deepsense/ so the figure/table scripts can source them:

  results/deepsense/genie_decomp.json  (Fig 6 — "info present vs extracted")
    Per scenario (31=V2I, 36=V2V), on the trajectory-grouped TEST split, 5 seeds:
      - random   : 1/B_theta chance line (analytic)
      - knn1      : full-radar-feature 1-nearest-neighbour over the TRAIN pool -> predict
                    its beam (how much angle info is locally present in the raw features)
      - cnn       : the trained radar model's top-1 beam (what the model extracts)
    The CNN−1-NN gap is the headline mechanism: it is large for V2I (the CNN exploits
    stable RSU scene geometry) and collapses for V2V (mobile, no stable geometry).
    NOTE: a range-angle-argmax baseline needs the raw 4-RX cubes (not in the processed
    h5), so it is intentionally omitted here — see MISSING note in the audit.

  results/deepsense/transfer.json  (Table 3 extra rows — zero-shot)
    Foreign-checkpoint top-1 beam over the WHOLE target set, 5 seeds:
      - sc31->sc36 (real V2I model on real V2V)
      - synthetic(closerange_60GHz)->sc31 (synthetic-trained on real)

    python scripts/deepsense_decomp.py
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.splits import trajectory_grouped_split
from src.eval.statistics import mean_ci
from src.models.radar_encoder import RadarPolarModel

B_THETA, SEEDS = 64, [11, 22, 33, 44, 55]
KS = (1, 2, 4, 8)
OUT = os.path.join(_ROOT, "results", "deepsense")


def require_cuda():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP. (eval-only, but GPU required; docs/10.)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda")


def data_path(scen):
    return os.path.join(_ROOT, "data", "processed", f"deepsense_scenario{scen}_full.h5")


def load_split(scen, seed):
    with h5py.File(data_path(scen), "r") as hf:
        traj = hf["trajectory_id"][:]
        tr, _, te = trajectory_grouped_split(traj, seed=seed)
        yt = hf["y_theta"][:].astype(np.int64)
        xr_tr = hf["radar_features"][np.sort(tr)].astype(np.float32)
        xr_te = hf["radar_features"][np.sort(te)].astype(np.float32)
    return (np.sort(tr), np.sort(te), yt, xr_tr, xr_te)


def _cnn_forward(ckpt, xr, yt_idx, device, ks=(1,)):
    """Return top-1 and angle top-K coverage (true beam in model's top-K) over xr."""
    model = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=8).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()
    x = torch.from_numpy(xr)
    cov = {k: 0 for k in ks}
    with torch.no_grad():
        for s in range(0, len(x), 256):
            e = min(s + 256, len(x))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = model(x[s:e].to(device))["theta_logits"].float()
            tk = p.topk(max(ks), 1).indices.cpu().numpy()
            y = yt_idx[s:e]
            for k in ks:
                cov[k] += int((tk[:, :k] == y[:, None]).any(1).sum())
    del model
    torch.cuda.empty_cache()
    return cov[1] / len(x), {k: cov[k] / len(x) for k in ks}


def cnn_top1(ckpt, xr, yt_idx, device):
    return _cnn_forward(ckpt, xr, yt_idx, device)[0]


def knn1_top1(xr_tr, ytr, xr_te, yte, device):
    """Full-feature 1-NN: nearest TRAIN sample (Euclidean) -> its beam label."""
    tr = torch.from_numpy(xr_tr.reshape(len(xr_tr), -1)).to(device)  # (Ntr, D) fp32
    tr_sq = (tr * tr).sum(1)                                         # (Ntr,)
    ytr_t = torch.from_numpy(ytr).to(device)
    correct = 0
    for s in range(0, len(xr_te), 128):
        e = min(s + 128, len(xr_te))
        q = torch.from_numpy(xr_te[s:e].reshape(e - s, -1)).to(device)
        d2 = (q * q).sum(1, keepdim=True) + tr_sq[None, :] - 2.0 * (q @ tr.T)
        nn = d2.argmin(1)
        correct += int((ytr_t[nn].cpu().numpy() == yte[s:e]).sum())
    del tr
    torch.cuda.empty_cache()
    return correct / len(xr_te)


def genie_decomposition(device):
    out = {"B_theta": B_THETA, "seeds": SEEDS, "random": 1.0 / B_THETA, "scenarios": {}}
    for scen, link in [(31, "V2I (RSU)"), (36, "V2V")]:
        knn, cnn, gaps = [], [], []
        cov_seed = {k: [] for k in KS}
        for seed in SEEDS:
            tr, te, yt, xr_tr, xr_te = load_split(scen, seed)
            ckpt = os.path.join(_ROOT, "checkpoints", f"deepsense_scenario{scen}_radar",
                                f"seed{seed}.pt")
            c, cov = _cnn_forward(ckpt, xr_te, yt[te], device, ks=KS)
            k = knn1_top1(xr_tr, yt[tr], xr_te, yt[te], device)
            cnn.append(c); knn.append(k); gaps.append(c - k)
            for kk in KS:
                cov_seed[kk].append(cov[kk])
            print(f"  sc{scen} seed{seed}: 1-NN {k:.4f} | CNN {c:.4f} | gap {c-k:+.4f} "
                  f"| cov@8 {cov[8]:.4f} (test n={len(te)})")
        out["scenarios"][str(scen)] = dict(
            link=link, n_test=int(len(te)),
            knn1=dict(per_seed=knn, **mean_ci(knn)),
            cnn=dict(per_seed=cnn, **mean_ci(cnn)),
            cnn_minus_knn1=dict(per_seed=gaps, **mean_ci(gaps)),
            cnn_coverage={str(kk): dict(per_seed=cov_seed[kk], **mean_ci(cov_seed[kk]))
                          for kk in KS})
    return out


def transfer(device):
    """Foreign-checkpoint top-1 over the WHOLE target set (zero-shot)."""
    combos = [
        ("sc31_to_sc36", "deepsense_scenario31_radar", 36, "real V2I -> real V2V"),
        ("synthetic_to_sc31", "closerange_60GHz_radar", 31,
         "synthetic close-range -> real V2I"),
    ]
    out = {"B_theta": B_THETA, "seeds": SEEDS, "random": 1.0 / B_THETA, "combos": {}}
    for key, ckpt_dir, scen, desc in combos:
        with h5py.File(data_path(scen), "r") as hf:
            xr = hf["radar_features"][:].astype(np.float32)
            yt = hf["y_theta"][:].astype(np.int64)
        accs = []
        for seed in SEEDS:
            ckpt = os.path.join(_ROOT, "checkpoints", ckpt_dir, f"seed{seed}.pt")
            accs.append(cnn_top1(ckpt, xr, yt, device))
        out["combos"][key] = dict(desc=desc, target_scenario=scen, n=int(len(yt)),
                                  top1=dict(per_seed=accs, **mean_ci(accs)))
        print(f"  {key}: top-1 {mean_ci(accs)['mean']:.4f} (n={len(yt)}, {desc})")
    return out


def main():
    device = require_cuda()
    os.makedirs(OUT, exist_ok=True)
    print("=== Genie decomposition (Fig 6) ===")
    gd = genie_decomposition(device)
    json.dump(gd, open(os.path.join(OUT, "genie_decomp.json"), "w"), indent=2)
    print("Wrote results/deepsense/genie_decomp.json")
    print("=== Zero-shot transfer (Table 3 extras) ===")
    tr = transfer(device)
    json.dump(tr, open(os.path.join(OUT, "transfer.json"), "w"), indent=2)
    print("Wrote results/deepsense/transfer.json")


if __name__ == "__main__":
    main()
