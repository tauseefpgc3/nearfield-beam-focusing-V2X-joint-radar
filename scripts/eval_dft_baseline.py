"""Bank the sensor-driven DFT-only baseline vs polar (Paper 5 appendix; eval-only).

Our radar model is evaluated two ways on the SAME near-field test subset (y_r_gated>0),
5 seeds x 2 freqs, reporting realized K=8 near-field beam-focusing dB-loss vs the polar
genie (max over all 512 codewords):
  - POLAR : top-8 shortlist over the full 512 angle x range polar codebook (= the paper's
            radar result; recomputed here for an apples-to-apples comparison).
  - DFT   : the SAME model's angle distribution, but the shortlist is restricted to the 64
            far-field DFT codewords (the j=0 polar slice; range = infinity). This isolates
            what a sensor-driven *angle-only DFT* codebook leaves on the table in near field.
Both use the identical |f^H h|^2 / genie math as eval_headline.py. NO retraining.
Writes results/dft_baseline/dft_sweep.json.

    python scripts/eval_dft_baseline.py
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

B_THETA, B_R, K = 64, 8, 8
SEEDS = [11, 22, 33, 44, 55]
FREQS = ["60", "140"]


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    dft_cols = torch.arange(0, B_THETA * B_R, B_R, device=device)   # j=0 -> 64 DFT codewords
    out = {"K": K, "metric": "near-field K=8 realized dB-loss vs polar genie (lower=better)",
           "seeds": SEEDS, "freqs": FREQS, "results": {}}

    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                                  f"polar_codebook_{fghz}GHz.npz"))["codebook"]
        cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)
        polar_db, dft_db = [], []
        for seed in SEEDS:
            with h5py.File(data, "r") as hf:
                traj = hf["trajectory_id"][:]
                _, _, te = trajectory_grouped_split(traj, seed=seed)
                te = np.sort(te)
                xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
                yr = hf["y_r_gated"][te].astype(np.int64)
                chan = hf["channel"][te]
            nf = yr > 0
            ckpt = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_radar",
                                f"seed{seed}.pt")
            model = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device,
                                             weights_only=False)["model"])
            model.eval()

            n = len(te)
            pdb = np.zeros(n); ddb = np.zeros(n)
            with torch.no_grad():
                for s in range(0, n, 256):
                    e = min(s + 256, n)
                    h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        o = model(xr[s:e].to(device))
                        p_theta = o["theta_logits"].float().softmax(1)
                        p_r = o["range_logits"].float().softmax(1)
                    gains = (cbc @ h.T).abs().pow(2).T               # (b,512)
                    genie = gains.max(1).values                     # polar near-field optimum
                    # POLAR: top-8 over full 512 joint codebook
                    joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(e - s, 512)
                    pidx = joint.topk(K, 1).indices
                    preal = gains.gather(1, pidx).max(1).values
                    pdb[s:e] = (10 * torch.log10(genie / preal.clamp_min(1e-30))).cpu().numpy()
                    # DFT: top-8 ANGLES -> the 64 j=0 codewords only
                    a_top = p_theta.topk(K, 1).indices              # (b,8) angle indices
                    didx = dft_cols[a_top]                          # -> flat DFT codeword idx
                    dreal = gains.gather(1, didx).max(1).values
                    ddb[s:e] = (10 * torch.log10(genie / dreal.clamp_min(1e-30))).cpu().numpy()
            polar_db.append(float(pdb[nf].mean())); dft_db.append(float(ddb[nf].mean()))
            print(f"  {fghz}GHz seed{seed}: polar={polar_db[-1]:.2f} dft={dft_db[-1]:.2f} dB "
                  f"(nf n={int(nf.sum())})")
        out["results"][fghz] = dict(
            polar=dict(per_seed=polar_db, **mean_ci(polar_db)),
            dft=dict(per_seed=dft_db, **mean_ci(dft_db)),
            dft_minus_polar=dict(per_seed=[d - p for d, p in zip(dft_db, polar_db)],
                                 **mean_ci([d - p for d, p in zip(dft_db, polar_db)])))

    od = os.path.join(_ROOT, "results", "dft_baseline")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "dft_sweep.json"), "w"), indent=2)
    print("Wrote results/dft_baseline/dft_sweep.json")
    for fghz in FREQS:
        r = out["results"][fghz]
        print(f"  [{fghz}GHz K=8 NF dB-loss] polar={r['polar']['mean']:.2f} "
              f"dft={r['dft']['mean']:.2f} (DFT costs +{r['dft_minus_polar']['mean']:.2f} dB)")


if __name__ == "__main__":
    main()
