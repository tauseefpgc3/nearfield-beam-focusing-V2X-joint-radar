"""Bank the fusion combination-rule comparison as JSON (Paper 5 appendix; eval-only).

For each freq {60,140} x seed {11,22,33,44,55}, evaluates the trained fusion warm-start
checkpoint under every inference combination rule and records the **K=8 near-field
beam-focusing dB-loss vs genie** on the trajectory-grouped TEST split's near-field subset
(y_r_gated>0). Rules:
  radar_alone     -- the radar-only model (pulled from its banked metrics, the paper's
                     radar-primary reference)
  soft_avg        -- C3 confidence-weighted ensemble (confidence_combine)
  argmax_conf     -- hard most-confident branch, both heads
  radar_ang+bestR -- D-015 headline rule (radar angle + most-confident range)
  radar_both      -- angle AND range from the radar branch
  temp4.0/temp8.0 -- temperature-scaled C3
Gain math mirrors eval_headline.py exactly (|f^H h|^2; genie=max over 512; dB-loss=
10log10(genie/best-of-top-8)). NO retraining. Writes results/fusion_rules/rule_sweep.json.

    python scripts/eval_fusion_rules_sweep.py
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

from scripts.eval_fusion_rules import combine            # the audited rule definitions
from src.data.splits import trajectory_grouped_split
from src.eval.statistics import mean_ci
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

B_THETA, B_R, K = 64, 8, 8
SEEDS = [11, 22, 33, 44, 55]
FREQS = ["60", "140"]
FUSION_RULES = ["soft_avg", "argmax_conf", "radar_ang+bestR", "radar_both",
                "temp4.0", "temp8.0"]
# JSON storage keys must be dot-free (the audit path-navigator splits on '.').
STORE_KEY = {"temp4.0": "temp4", "temp8.0": "temp8"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--out", default="rule_sweep.json",
                    help="output filename under results/fusion_rules/")
    args = ap.parse_args()
    seeds = args.seeds
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | seeds={seeds} -> {args.out}")
    stored_rules = ["radar_alone"] + [STORE_KEY.get(r, r) for r in FUSION_RULES]
    out = {"K": K, "metric": "near-field K=8 beam-focusing dB-loss vs genie (lower=better)",
           "seeds": seeds, "freqs": FREQS, "rules": stored_rules, "results": {}}

    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                                  f"polar_codebook_{fghz}GHz.npz"))["codebook"]
        cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)
        per_rule = {r: [] for r in FUSION_RULES}
        radar_alone = []
        for seed in seeds:
            with h5py.File(data, "r") as hf:
                traj = hf["trajectory_id"][:]
                _, _, te = trajectory_grouped_split(traj, seed=seed)
                te = np.sort(te)
                xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
                xl = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
                yr = hf["y_r_gated"][te].astype(np.int64)
                chan = hf["channel"][te]
            nf = yr > 0
            fck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_fusion_warmstart",
                               f"seed{seed}.pt")
            model = SensingPolarPredictor(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
            model.load_state_dict(torch.load(fck, map_location=device,
                                             weights_only=False)["model"])
            model.eval()
            # radar-alone reference: the standalone radar model, computed the SAME way as
            # eval_headline (self-contained -> works for seeds without a banked _metrics.json).
            rck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_radar",
                               f"seed{seed}.pt")
            rmodel = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
            rmodel.load_state_dict(torch.load(rck, map_location=device,
                                              weights_only=False)["model"])
            rmodel.eval()

            n = len(te)
            dbl = {r: np.zeros(n) for r in FUSION_RULES}
            dbl_radar = np.zeros(n)
            with torch.no_grad():
                for s in range(0, n, 256):
                    e = min(s + 256, n)
                    h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        o = model(xr[s:e].to(device), xl[s:e].to(device))
                        pth = torch.stack([F.softmax(o[f"{b}_theta_logits"].float(), 1)
                                           for b in BRANCHES])      # (3,B,64)
                        prg = torch.stack([F.softmax(o[f"{b}_range_logits"].float(), 1)
                                           for b in BRANCHES])      # (3,B,8)
                        ro = rmodel(xr[s:e].to(device))
                        rpt = ro["theta_logits"].float().softmax(1)
                        rpr = ro["range_logits"].float().softmax(1)
                    cth, crg = pth.max(-1).values, prg.max(-1).values
                    gains = (cbc @ h.T).abs().pow(2).T              # (b,512) |f^H h|^2
                    genie = gains.max(1).values
                    for r in FUSION_RULES:
                        p_theta, p_r = combine(r, pth, prg, cth, crg)
                        joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(e - s, 512)
                        idx = joint.topk(K, 1).indices
                        real = gains.gather(1, idx).max(1).values
                        dbl[r][s:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
                    rjoint = (rpt.unsqueeze(2) * rpr.unsqueeze(1)).reshape(e - s, 512)
                    rreal = gains.gather(1, rjoint.topk(K, 1).indices).max(1).values
                    dbl_radar[s:e] = (10 * torch.log10(genie / rreal.clamp_min(1e-30))).cpu().numpy()
            radar_alone.append(float(dbl_radar[nf].mean()))
            for r in FUSION_RULES:
                per_rule[r].append(float(dbl[r][nf].mean()))
            print(f"  {fghz}GHz seed{seed}: radar_alone={radar_alone[-1]:.2f} " +
                  " ".join(f"{r}={per_rule[r][-1]:.2f}" for r in FUSION_RULES))

        res = {"radar_alone": dict(per_seed=radar_alone, **mean_ci(radar_alone))}
        for r in FUSION_RULES:
            res[STORE_KEY.get(r, r)] = dict(per_seed=per_rule[r], **mean_ci(per_rule[r]))
        out["results"][fghz] = res

    od = os.path.join(_ROOT, "results", "fusion_rules")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, args.out), "w"), indent=2)
    print(f"Wrote results/fusion_rules/{args.out}")
    for fghz in FREQS:
        print(f"  [{fghz}GHz K=8 NF dB-loss mean] " +
              " ".join(f"{r}={out['results'][fghz][r]['mean']:.2f}" for r in stored_rules))


if __name__ == "__main__":
    main()
