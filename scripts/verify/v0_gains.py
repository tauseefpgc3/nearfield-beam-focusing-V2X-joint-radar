"""Independent verification (Phase 1 audit check), step 0: codebook gain matrix.

G[u, k] = |f_k^H h_u|^2 for all 6000 close-range users and all 512 polar codewords,
computed twice: complex64 on GPU (the precision the banked evals use) and complex128 on
GPU (reference). Saved to results/verify/gains_{f}GHz.npz. Written from scratch; does
not import scripts/revision.
"""
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "verify")
os.makedirs(OUT, exist_ok=True)

for f in (60, 140):
    cb = np.load(os.path.join(ROOT, "data", "codebooks", f"polar_codebook_{f}GHz.npz"))["codebook"]
    assert cb.shape == (512, 16384)
    dev = torch.device("cuda")
    c64 = torch.from_numpy(np.conj(cb)).to(dev, torch.complex64)
    c128 = torch.from_numpy(np.conj(cb)).to(dev, torch.complex128)
    G32 = np.zeros((6000, 512), np.float32)
    G64 = np.zeros((6000, 512), np.float64)
    with h5py.File(os.path.join(ROOT, "data", "processed", f"closerange_{f}GHz_full.h5"), "r") as hf:
        n = hf["channel"].shape[0]
        for s in range(0, n, 250):
            h = hf["channel"][s:s + 250]
            G32[s:s + 250] = (c64 @ torch.from_numpy(h).to(dev, torch.complex64).T).abs().pow(2).T.cpu().numpy()
            G64[s:s + 250] = (c128 @ torch.from_numpy(h).to(dev).T).abs().pow(2).T.cpu().numpy()
        meta = {k: hf[k][:] for k in ("y_theta", "y_r", "y_theta_gated", "y_r_gated",
                                      "polar_adv_db", "distance", "trajectory_id")}
    np.savez(os.path.join(OUT, f"gains_{f}GHz.npz"), G32=G32, G64=G64, **meta)
    print(f, "done", G32.shape, float(np.abs(G32 - G64).max() / G64.max()))
