"""n=20 in-domain metrics for the close-range predictors (Fig 4 CDF + Tables 1/2, eval-only).

For every seed in the 20 D-018 seeds x {60,140} GHz, computes on the seed's own NF test set,
for BOTH predictors:
  - radar   : the radar-only model (radar_closerange_{f}GHz_radar).
  - softavg : the multimodal C3 soft-average (fusion_warmstart), the D-018 recommended rule.
Banks per-seed top-1 theta, NF range top-1, NF coverage@{1,2,4,8}, NF dB-loss@{1,2,4,8}, and
the per-sample K=4 NF gain ratio (for the CDF). Same |f^H h|^2 / genie math as eval_headline.
Writes results/indomain_n20/close_n20.json (does NOT touch the banked n=5 files). GPU.

    python scripts/eval_indomain_n20.py
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.eval_fusion_rules import combine
from src.data.splits import trajectory_grouped_split
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

B_THETA, B_R = 64, 8
KS = (1, 2, 4, 8)
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165,
         176, 187, 198, 209, 220]
FREQS = ["60", "140"]


def metrics_from_probs(p_theta, p_r, yt, yr, gains, genie, nf):
    """top1/coverage/dbloss/gain-ratio for a predictor's marginals on one batch set."""
    B_ = p_theta.shape[0]
    joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(B_, B_THETA * B_R)
    topk = joint.topk(max(KS), 1).indices
    true_flat = torch.from_numpy(yt * B_R + yr).to(joint.device)
    out = dict(pt1=p_theta.argmax(1).cpu().numpy(), pr1=p_r.argmax(1).cpu().numpy())
    for k in KS:
        idx = topk[:, :k]
        out[f"cov{k}"] = (idx == true_flat.unsqueeze(1)).any(1).cpu().numpy()
        real = gains.gather(1, idx).max(1).values
        out[f"db{k}"] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
        if k == 4:
            out["gr4"] = (real / genie.clamp_min(1e-30)).cpu().numpy()
    return out


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = {"seeds": SEEDS, "freqs": FREQS, "predictors": ["radar", "softavg"], "results": {}}

    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                                  f"polar_codebook_{fghz}GHz.npz"))["codebook"]
        cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)
        acc = {pred: {m: [] for m in ("top1_theta", "range_nf")}
               | {f"cov{k}_nf": [] for k in KS} | {f"db{k}_nf": [] for k in KS}
               | {"gain_ratio_k4_nf": []} for pred in ("radar", "softavg")}
        for seed in SEEDS:
            with h5py.File(data, "r") as hf:
                traj = hf["trajectory_id"][:]
                _, _, te = trajectory_grouped_split(traj, seed=seed)
                te = np.sort(te)
                xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
                xl = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
                yt = hf["y_theta_gated"][te].astype(np.int64)
                yr = hf["y_r_gated"][te].astype(np.int64)
                chan = hf["channel"][te]
            nf = yr > 0
            rmodel = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
            rmodel.load_state_dict(torch.load(os.path.join(_ROOT, "checkpoints",
                f"closerange_{fghz}GHz_radar", f"seed{seed}.pt"), map_location=device,
                weights_only=False)["model"]); rmodel.eval()
            fmodel = SensingPolarPredictor(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
            fmodel.load_state_dict(torch.load(os.path.join(_ROOT, "checkpoints",
                f"closerange_{fghz}GHz_fusion_warmstart", f"seed{seed}.pt"),
                map_location=device, weights_only=False)["model"]); fmodel.eval()

            n = len(te)
            store = {pred: {key: np.zeros(n, bool) if key.startswith("cov")
                            else np.zeros(n) for key in
                            (["gr4"] + [f"cov{k}" for k in KS] + [f"db{k}" for k in KS])}
                     for pred in ("radar", "softavg")}
            pt1 = {"radar": np.zeros(n, np.int64), "softavg": np.zeros(n, np.int64)}
            pr1 = {"radar": np.zeros(n, np.int64), "softavg": np.zeros(n, np.int64)}
            with torch.no_grad():
                for s in range(0, n, 256):
                    e = min(s + 256, n)
                    h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
                    gains = (cbc @ h.T).abs().pow(2).T; genie = gains.max(1).values
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        ro = rmodel(xr[s:e].to(device))
                        rpt = ro["theta_logits"].float().softmax(1)
                        rpr = ro["range_logits"].float().softmax(1)
                        fo = fmodel(xr[s:e].to(device), xl[s:e].to(device))
                        pth = torch.stack([F.softmax(fo[f"{b}_theta_logits"].float(), 1) for b in BRANCHES])
                        prg = torch.stack([F.softmax(fo[f"{b}_range_logits"].float(), 1) for b in BRANCHES])
                    cth, crg = pth.max(-1).values, prg.max(-1).values
                    spt, spr = combine("soft_avg", pth, prg, cth, crg)
                    for pred, (pt, pr) in (("radar", (rpt, rpr)), ("softavg", (spt, spr))):
                        r = metrics_from_probs(pt, pr, yt[s:e], yr[s:e], gains, genie,
                                               nf[s:e])
                        pt1[pred][s:e] = r["pt1"]; pr1[pred][s:e] = r["pr1"]
                        for k in KS:
                            store[pred][f"cov{k}"][s:e] = r[f"cov{k}"]
                            store[pred][f"db{k}"][s:e] = r[f"db{k}"]
                        store[pred]["gr4"][s:e] = r["gr4"]
            for pred in ("radar", "softavg"):
                acc[pred]["top1_theta"].append(float((pt1[pred] == yt).mean()))
                acc[pred]["range_nf"].append(float((pr1[pred][nf] == yr[nf]).mean()))
                for k in KS:
                    acc[pred][f"cov{k}_nf"].append(float(store[pred][f"cov{k}"][nf].mean()))
                    acc[pred][f"db{k}_nf"].append(float(store[pred][f"db{k}"][nf].mean()))
                acc[pred]["gain_ratio_k4_nf"].extend(store[pred]["gr4"][nf].tolist())
            print(f"  {fghz}GHz seed{seed}: radar db8={acc['radar']['db8_nf'][-1]:.2f} "
                  f"softavg db8={acc['softavg']['db8_nf'][-1]:.2f}")
        out["results"][fghz] = acc

    od = os.path.join(_ROOT, "results", "indomain_n20")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "close_n20.json"), "w"), indent=2)
    print("Wrote results/indomain_n20/close_n20.json")


if __name__ == "__main__":
    main()
