"""E10 eval: B_r-aware radar metrics on the TEST split (Paper 5).

The COMPARABLE-across-B_r metric is the physical near-field beam-focusing dB-loss vs genie
(K=4/K=8). top-1 range is reported but NOT comparable across B_r (fewer bins = easier).

    python scripts/ablate_br_eval.py --b-r 4 --seed 11 \
        --ckpt checkpoints/closerange_60GHz_Br4_radar/seed11.pt
"""
from __future__ import annotations

import argparse
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
from src.models.radar_encoder import RadarPolarModel

B_THETA = 64
KS = (4, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-r", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    B_r = args.b_r
    device = torch.device("cuda")
    data = os.path.join(_ROOT, "data", "processed", f"closerange_60GHz_Br{B_r}_full.h5")
    cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                              f"polar_codebook_60GHz_Br{B_r}.npz"))["codebook"]
    cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)

    with h5py.File(data, "r") as hf:
        _, _, te = trajectory_grouped_split(hf["trajectory_id"][:], seed=args.seed)
        te = np.sort(te)
        xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
        yt = torch.from_numpy(hf["y_theta_gated"][te].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][te].astype(np.int64))
        chan = hf["channel"][te]

    model = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_r).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()

    n = yt.shape[0]
    pr1 = np.zeros(n, np.int64)
    dbl = {k: np.zeros(n) for k in KS}
    with torch.no_grad():
        for s in range(0, n, 256):
            e = min(s + 256, n)
            h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(xr[s:e].to(device))
                pth = out["theta_logits"].float().softmax(1)
                prg = out["range_logits"].float().softmax(1)
            pr1[s:e] = prg.argmax(1).cpu().numpy()
            joint = (pth.unsqueeze(2) * prg.unsqueeze(1)).reshape(e - s, B_THETA * B_r)
            topk = joint.topk(max(KS), 1).indices
            gains = (cbc @ h.T).abs().pow(2).T
            genie = gains.max(1).values
            for k in KS:
                real = gains.gather(1, topk[:, :k]).max(1).values
                dbl[k][s:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()

    yr_np = yr.numpy(); nf = yr_np > 0
    res = dict(B_r=B_r, seed=args.seed, n=n, n_nf=int(nf.sum()),
               top1_range=float((pr1 == yr_np).mean()),
               top1_range_nf=float((pr1[nf] == yr_np[nf]).mean()) if nf.any() else float("nan"),
               K4_nf_db=float(dbl[4][nf].mean()), K8_nf_db=float(dbl[8][nf].mean()),
               populated_bins=int(len(np.unique(yr_np))),
               y_r_hist=np.bincount(yr_np, minlength=B_r).tolist())
    out_path = os.path.join(_ROOT, "results", "ablation",
                            f"br{B_r}_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"[B_r={B_r} seed{args.seed}] top1_range {res['top1_range']:.3f} | "
          f"K4 NF dB {res['K4_nf_db']:.2f} | K8 NF dB {res['K8_nf_db']:.2f} | "
          f"populated {res['populated_bins']}/{B_r}")


if __name__ == "__main__":
    main()
