"""Top-K joint coverage + realized beam-focusing-gain loss (Paper 5 C1/E5).

From a trained single-modality checkpoint (no retraining), on the close-range 60 GHz
TEST split:
  1. Top-K JOINT (theta,r) coverage at K=1,2,4,8 (overall + near-field subset y_r>0):
     does the model's target label land in the top-K joint-prob shortlist?
  2. Beam-focusing-gain ratio (the headline E5 metric): for each sample, take the model's
     top-K shortlist, pick the codeword with the best ACTUAL channel gain in it (the RF
     verification sweep), and report dB-loss vs the genie (best of all 512). Matched
     filter |f^H h|^2 (D-005).

    python scripts/eval_topk_gain.py --modality radar \
        --ckpt checkpoints/closerange_60GHz_radar/seed11.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_single import MODALITY, require_cuda
from src.data.splits import trajectory_grouped_split
from src.models.sensing_polar import SensingPolarPredictor, confidence_combine

B_THETA, B_R = 64, 8
KS = (1, 2, 4, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "lidar", "fusion"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(
        _ROOT, "data", "processed", "closerange_60GHz_full.h5"))
    ap.add_argument("--codebook", default=os.path.join(
        _ROOT, "data", "codebooks", "polar_codebook_60GHz.npz"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    device = require_cuda()
    is_fusion = args.modality == "fusion"

    with h5py.File(args.data, "r") as hf:
        traj = hf["trajectory_id"][:]
        _, _, te = trajectory_grouped_split(traj, seed=args.seed)
        te = np.sort(te)
        if is_fusion:
            feat_r = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
            feat_l = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
        else:
            feat = torch.from_numpy(hf[MODALITY[args.modality][0]][te].astype(np.float32))
        yt = torch.from_numpy(hf["y_theta_gated"][te].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][te].astype(np.int64))
        chan = hf["channel"][te]                       # (n,16384) complex128

    cb = np.load(args.codebook)["codebook"]            # (512,16384) unit-norm
    cb_conj = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)

    ModelCls = SensingPolarPredictor if is_fusion else MODALITY[args.modality][1]
    model = ModelCls(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()

    n = yt.shape[0]
    true_flat = (yt * B_R + yr).numpy()
    cover = {k: np.zeros(n, bool) for k in KS}
    dbloss = {k: np.zeros(n) for k in KS}
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)  # (b,16384)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if is_fusion:
                    pth, prg = confidence_combine(
                        model(feat_r[s:e].to(device), feat_l[s:e].to(device)))
                    pth, prg = pth.float(), prg.float()
                else:
                    out = model(feat[s:e].to(device))
                    pth = out["theta_logits"].float().softmax(1)
                    prg = out["range_logits"].float().softmax(1)
            joint = (pth.unsqueeze(2) * prg.unsqueeze(1)).reshape(e - s, B_THETA * B_R)
            topk = joint.topk(max(KS), dim=1).indices          # (b,8) flat codeword idx
            # actual channel gains for all 512 codewords (matched filter)
            gains = (cb_conj @ h.T).abs().pow(2).T             # (b,512)
            genie = gains.max(1).values                        # (b,)
            tk = torch.from_numpy(true_flat[s:e]).to(device)
            for k in KS:
                idx_k = topk[:, :k]                            # (b,k)
                cover[k][s:e] = (idx_k == tk.unsqueeze(1)).any(1).cpu().numpy()
                realized = gains.gather(1, idx_k).max(1).values
                dbloss[k][s:e] = (10 * torch.log10(genie / realized.clamp_min(1e-30))
                                  ).cpu().numpy()

    nf = (yr > 0).numpy()
    n_nf = int(nf.sum())
    print(f"\n=== {args.modality} | TEST n={n} (near-field y_r>0: n={n_nf}) ===")
    print(f"{'K':>3} | {'cover(all)':>10} {'cover(NF)':>10} | "
          f"{'dBloss(all)':>11} {'dBloss(NF)':>11}")
    for k in KS:
        print(f"{k:>3} | {cover[k].mean():>10.3f} {cover[k][nf].mean():>10.3f} | "
              f"{dbloss[k].mean():>11.3f} {dbloss[k][nf].mean():>11.3f}")
    print(f"\n[headline] near-field beam-focusing dB-loss vs genie: "
          f"K=4 {dbloss[4][nf].mean():.2f} dB | K=8 {dbloss[8][nf].mean():.2f} dB")


if __name__ == "__main__":
    main()
