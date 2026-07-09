"""Gain-vs-overhead at n=20: hierarchical baseline + our predictor (eval-only).

Banks results/gain_overhead_n20.json:
  - hierarchical (measurement-only two-stage angle->range sweep, 64 + (B_r-1) = 71 meas),
    NF K=8-equivalent realized dB-loss vs genie, 20-seed mean+/-95%CI (GPU sweep).
  - exhaustive polar sweep: 512 meas, 0 dB (genie) by definition.
  - our predictors (soft-avg C3 and radar-alone) NF dB-loss at K in {1,2,4,8}, pulled from
    the banked results/indomain_n20/close_n20.json (per-seed, 20 seeds) -> mean+/-95%CI.
Does NOT overwrite the n=5 hierarchical_sweep.json. GPU required.

    python scripts/eval_gain_overhead_n20.py
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
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165,
         176, 187, 198, 209, 220]
FREQS = ["60", "140"]
N_MEAS = 64 + (B_R - 1)   # 71


def hierarchical_db(fghz, device):
    """Two-stage angle->range measurement sweep NF dB-loss per seed (20 seeds)."""
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
        n = len(te); dbl = np.zeros(n)
        with torch.no_grad():
            for s in range(0, n, 256):
                e = min(s + 256, n)
                h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
                gains = (cbc @ h.T).abs().pow(2).T
                genie = gains.max(1).values
                g = gains.reshape(e - s, B_THETA, B_R)
                i_star = g[:, :, 0].argmax(1)                     # stage 1: best DFT angle
                real = g[torch.arange(e - s), i_star, :].max(1).values  # stage 2: best range
                dbl[s:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
        per_seed.append(float(dbl[nf].mean()))
    return per_seed


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    close = json.load(open(os.path.join(_ROOT, "results", "indomain_n20", "close_n20.json")))
    out = {"n_seeds": len(SEEDS), "hier_measurements": N_MEAS, "exhaustive_measurements": 512,
           "seeds": SEEDS, "results": {}}
    for fghz in FREQS:
        hier = hierarchical_db(fghz, device)
        res = {"hierarchical": dict(per_seed=hier, **mean_ci(hier))}
        for pred in ("softavg", "radar"):
            res[pred] = {}
            for k in (1, 2, 4, 8):
                v = close["results"][fghz][pred][f"db{k}_nf"]
                res[pred][f"K{k}"] = dict(measurements=k, **mean_ci(v))
        out["results"][fghz] = res
        print(f"  {fghz}GHz: hier({N_MEAS} meas)={res['hierarchical']['mean']:.2f} | "
              f"softavg K4={res['softavg']['K4']['mean']:.2f} K8={res['softavg']['K8']['mean']:.2f}")
    json.dump(out, open(os.path.join(_ROOT, "results", "gain_overhead_n20.json"), "w"), indent=2)
    print("Wrote results/gain_overhead_n20.json")


if __name__ == "__main__":
    main()
