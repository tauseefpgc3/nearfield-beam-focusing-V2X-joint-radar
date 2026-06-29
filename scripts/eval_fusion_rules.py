"""Inference-time combination-rule variants for the trained fusion checkpoint (Paper 5).

The C3 soft-average ensemble was worse than its own branches, suggesting the COMBINATION
RULE (not the fusion concept) is the problem. This evaluates alternative inference-time
combiners on the EXISTING checkpoint (no retraining, no tuning toward a target):

  1. soft_avg        : C3 confidence-weighted average (baseline)
  2. argmax_conf     : per head, take the single most-confident branch
  3. radar_ang+bestR : angle from the radar branch; range from the most-confident branch
  4. temp_T{4,8}     : confidence weights raised to power T before normalizing

For each: top-1 theta (overall) + near-field (y_r>0) K=4 / K=8 beam-focusing dB-loss.
Reference: radar-alone = theta 0.768, K4 1.34 dB, K8 0.50 dB.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.splits import trajectory_grouped_split
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

B_THETA, B_R = 64, 8
RADAR_IDX = BRANCHES.index("radar")
KS = (4, 8)
CKPT = os.path.join(_ROOT, "checkpoints", "closerange_60GHz_fusion", "seed11.pt")


def combine(rule, pth, prg, cth, crg):
    """pth:(3,B,64) prg:(3,B,8) cth/crg:(3,B). Returns p_theta(B,64), p_r(B,8)."""
    B = pth.shape[1]
    ar = torch.arange(B, device=pth.device)

    def wavg(p, w):                                  # w:(3,B) -> normalized
        w = w / w.sum(0, keepdim=True).clamp_min(1e-12)
        return (w.unsqueeze(-1) * p).sum(0)

    if rule == "soft_avg":
        return wavg(pth, cth), wavg(prg, crg)
    if rule == "argmax_conf":
        return pth[cth.argmax(0), ar], prg[crg.argmax(0), ar]
    if rule == "radar_ang+bestR":
        return pth[RADAR_IDX], prg[crg.argmax(0), ar]
    if rule == "radar_both":                          # Q4: angle AND range from radar branch
        return pth[RADAR_IDX], prg[RADAR_IDX]
    if rule.startswith("temp"):
        T = float(rule[4:])
        return wavg(pth, cth ** T), wavg(prg, crg ** T)
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "| ckpt:", os.path.relpath(args.ckpt, _ROOT))
    with h5py.File(os.path.join(_ROOT, "data", "processed",
                                "closerange_60GHz_full.h5"), "r") as hf:
        traj = hf["trajectory_id"][:]
        _, _, te = trajectory_grouped_split(traj, seed=11)
        te = np.sort(te)
        xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
        xl = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
        yt = torch.from_numpy(hf["y_theta_gated"][te].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][te].astype(np.int64))
        chan = hf["channel"][te]
    cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                              "polar_codebook_60GHz.npz"))["codebook"]
    cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)

    model = SensingPolarPredictor().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()

    rules = ["soft_avg", "argmax_conf", "radar_ang+bestR", "radar_both", "temp4.0", "temp8.0"]
    n = yt.shape[0]
    nf = (yr > 0).numpy()
    th_ok = {r: np.zeros(n, bool) for r in rules}
    dbl = {r: {k: np.zeros(n) for k in KS} for r in rules}
    with torch.no_grad():
        for s in range(0, n, 256):
            e = min(s + 256, n)
            h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(xr[s:e].to(device), xl[s:e].to(device))
            pth = torch.stack([F.softmax(out[f"{b}_theta_logits"].float(), 1) for b in BRANCHES])
            prg = torch.stack([F.softmax(out[f"{b}_range_logits"].float(), 1) for b in BRANCHES])
            cth = pth.max(-1).values                 # (3,b)
            crg = prg.max(-1).values
            gains = (cbc @ h.T).abs().pow(2).T        # (b,512)
            genie = gains.max(1).values
            ytb = yt[s:e].to(device)
            for r in rules:
                p_theta, p_r = combine(r, pth, prg, cth, crg)
                th_ok[r][s:e] = (p_theta.argmax(1) == ytb).cpu().numpy()
                joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(e - s, 512)
                topk = joint.topk(max(KS), 1).indices
                for k in KS:
                    real = gains.gather(1, topk[:, :k]).max(1).values
                    dbl[r][k][s:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()

    print(f"\nclose-range 60 GHz TEST | near-field n={int(nf.sum())} | reference "
          f"radar-alone: theta 0.768, K4 1.34 dB, K8 0.50 dB\n")
    print(f"{'rule':>16} | {'top1_theta':>10} | {'K4 NF dB':>9} | {'K8 NF dB':>9} | verdict")
    print("-" * 70)
    for r in rules:
        th = th_ok[r].mean()
        k4, k8 = dbl[r][4][nf].mean(), dbl[r][8][nf].mean()
        win = (k8 <= 0.50 + 1e-9) and (k4 < 1.34) and (th >= 0.768)
        verdict = "BEATS radar" if win else ("K8<=0.50 " if k8 <= 0.50 else "") + \
            ("K4<1.34 " if k4 < 1.34 else "") + ("ang>=0.768" if th >= 0.768 else "")
        print(f"{r:>16} | {th:>10.3f} | {k4:>9.2f} | {k8:>9.2f} | {verdict or 'no'}")


if __name__ == "__main__":
    main()
