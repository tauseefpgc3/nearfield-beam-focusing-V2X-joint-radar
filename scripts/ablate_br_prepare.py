"""E10 prep: per-B_r polar codebook + RECOMPUTED gain-gated labels (Paper 5).

For a given B_r, regenerate the polar codebook (N_side=128, beta_delta=1.2, matched
filter D-005) and recompute the labels FROM THE STORED CHANNEL (the range bins differ, so
labels differ — never reuse B_r=8 labels). Then apply the 3 dB gain gate (D-013). Writes
a per-B_r dataset (radar + channel + B_r-specific labels) for the radar-primary ablation.

    python scripts/ablate_br_prepare.py --b-r 4
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

from src.codebook.polar import fraunhofer_distance, generate_polar_codebook

F_GHZ, N, B_THETA, BETA, SPACING, N_SIDE = 60, 16384, 64, 1.2, 0.5, 128
GATE_DB = 3.0
SRC = os.path.join(_ROOT, "data", "processed", "closerange_60GHz_full.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-r", type=int, required=True)
    ap.add_argument("--batch", type=int, default=400)
    args = ap.parse_args()
    B_r = args.b_r
    device = torch.device("cuda")
    f_c = F_GHZ * 1e9

    cb, angles, ranges = generate_polar_codebook(N, f_c, B_theta=B_THETA, B_r=B_r,
                                                 beta_delta=BETA, spacing=SPACING)
    d_f = fraunhofer_distance(f_c, N_side=N_SIDE, spacing=SPACING)
    cb_path = os.path.join(_ROOT, "data", "codebooks", f"polar_codebook_60GHz_Br{B_r}.npz")
    np.savez(cb_path, codebook=cb, angles=angles, ranges=ranges, d_F=np.float64(d_f),
             f_c=np.float64(f_c), N=np.int64(N), N_side=np.int64(N_SIDE),
             B_theta=np.int64(B_THETA), B_r=np.int64(B_r), beta_delta=np.float64(BETA),
             spacing=np.float64(SPACING))
    cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)   # (B_theta*B_r, N)

    with h5py.File(SRC, "r") as hf:
        n = hf["y_theta"].shape[0]
        radar = hf["radar_features"][:]
        dist = hf["distance"][:]
        traj = hf["trajectory_id"][:]
        yt = np.zeros(n, np.int32); yr = np.zeros(n, np.int32)
        ytg = np.zeros(n, np.int32); yrg = np.zeros(n, np.int32)
        adv = np.zeros(n, np.float32)
        out = os.path.join(_ROOT, "data", "processed", f"closerange_60GHz_Br{B_r}_full.h5")
        of = h5py.File(out, "w")
        of.create_dataset("channel", shape=(n, N), dtype=np.complex128, chunks=True)
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            h = hf["channel"][s:e]
            of["channel"][s:e] = h
            ht = torch.from_numpy(h).to(device, dtype=torch.complex64).T          # (N,b)
            g = (cbc @ ht).abs().pow(2).T                                         # (b, B_theta*B_r)
            flat = g.argmax(1)
            y_th = (flat // B_r).cpu().numpy(); y_r = (flat % B_r).cpu().numpy()
            dft = g[:, ::B_r]                                                     # j=0 block (b, B_theta)
            polar_gain = g.max(1).values; dft_gain = dft.max(1).values
            a = (10 * torch.log10(polar_gain / dft_gain.clamp_min(1e-30))).cpu().numpy()
            dft_theta = dft.argmax(1).cpu().numpy()
            keep = a >= GATE_DB
            yt[s:e] = y_th; yr[s:e] = y_r; adv[s:e] = a
            yrg[s:e] = np.where(keep, y_r, 0)
            ytg[s:e] = np.where(keep, y_th, dft_theta)
        for k, v in (("radar_features", radar), ("distance", dist),
                     ("trajectory_id", traj.astype(np.int32)), ("y_theta", yt),
                     ("y_r", yr), ("y_theta_gated", ytg), ("y_r_gated", yrg),
                     ("polar_adv_db", adv)):
            of.create_dataset(k, data=v)
        of.attrs.update(dict(f_c=f_c, d_F=d_f, N=N, N_side=N_SIDE, B_theta=B_THETA,
                             B_r=B_r, spacing=SPACING, gain_gate_db=GATE_DB))
        of.close()

    pop = sorted(set(yrg.tolist()))
    print(f"B_r={B_r}: wrote {os.path.basename(out)} | codebook {cb.shape} | "
          f"gated y_r hist {np.bincount(yrg, minlength=B_r).tolist()} | "
          f"populated bins (after gate) {len([b for b in range(B_r) if (yrg==b).sum()>0])}/{B_r} "
          f"(j>0: {sorted(b for b in pop if b>0)})")


if __name__ == "__main__":
    main()
