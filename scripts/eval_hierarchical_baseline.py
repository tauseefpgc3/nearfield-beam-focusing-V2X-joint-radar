"""Task B — quantitative gain-vs-overhead baseline (eval-only; no training, no sensor).

A measurement-only two-stage HIERARCHICAL near-field beam sweep on the controlled
close-range test set, for comparison against the exhaustive polar sweep and our
sensor-driven predictor:

  Stage 1 (coarse angle): measure the 64 far-field (j=0, DFT) codewords -> pick best angle i*.
  Stage 2 (range refine) : at i*, measure the B_r range codewords (i*, j=1..B_r-1) -> pick j*.
  Selected beam = argmax over {(i*, j): j=0..B_r-1}; #RF measurements = 64 + (B_r-1).

Reported per (freq, seed) on the near-field subset (y_r_gated>0): realized beam-focusing
dB-loss vs the polar genie (max over all 512 codewords) and the measurement count. This is
NOT a specific published method (we do not have Wu et al. TWC 2024 to reproduce exactly);
it is the standard two-stage angle->range hierarchical sweep the task offered as the
alternative, implemented on our own codebook. Same |f^H h|^2 / genie math as eval_headline.

    python scripts/eval_hierarchical_baseline.py
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

B_THETA, B_R = 64, 8
SEEDS = [11, 22, 33, 44, 55]
FREQS = ["60", "140"]


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_meas = 64 + (B_R - 1)                       # stage1 (64 angles) + stage2 (B_r-1 ranges)
    out = {"metric": "near-field beam-focusing dB-loss vs polar genie (lower=better)",
           "seeds": SEEDS, "freqs": FREQS, "n_measurements": n_meas, "results": {}}

    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                                  f"polar_codebook_{fghz}GHz.npz"))["codebook"]
        cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)
        per_seed = []
        for seed in SEEDS:
            with h5py.File(data, "r") as hf:
                traj = hf["trajectory_id"][:]
                _, _, te = trajectory_grouped_split(traj, seed=seed)
                te = np.sort(te)
                yr = hf["y_r_gated"][te].astype(np.int64)
                chan = hf["channel"][te]
            nf = yr > 0
            n = len(te)
            dbl = np.zeros(n)
            with torch.no_grad():
                for s in range(0, n, 256):
                    e = min(s + 256, n)
                    h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
                    gains = (cbc @ h.T).abs().pow(2).T          # (b,512) |f^H h|^2
                    genie = gains.max(1).values
                    g = gains.reshape(e - s, B_THETA, B_R)      # (b, 64, 8) [angle, range]
                    i_star = g[:, :, 0].argmax(1)               # stage 1: best DFT angle
                    row = g[torch.arange(e - s), i_star, :]     # (b, 8) ranges at i*
                    real = row.max(1).values                    # stage 2: best range at i*
                    dbl[s:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
            per_seed.append(float(dbl[nf].mean()))
            print(f"  {fghz}GHz seed{seed}: hierarchical NF dB-loss={per_seed[-1]:.2f} "
                  f"({n_meas} meas, nf n={int(nf.sum())})")
        out["results"][fghz] = dict(per_seed=per_seed, **mean_ci(per_seed))

    od = os.path.join(_ROOT, "results", "baselines")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "hierarchical_sweep.json"), "w"), indent=2)
    print(f"Wrote results/baselines/hierarchical_sweep.json ({n_meas} measurements/user)")
    for fghz in FREQS:
        print(f"  [{fghz}GHz] hierarchical NF dB-loss = {out['results'][fghz]['mean']:.2f} dB")


if __name__ == "__main__":
    main()
