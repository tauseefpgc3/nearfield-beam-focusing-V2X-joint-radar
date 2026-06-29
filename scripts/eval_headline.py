"""Full headline metrics for one trained checkpoint (Paper 5 Batch 8).

top-1 theta/range (overall + near-field), K=1,2,4,8 joint coverage + beam-focusing
dB-loss vs genie (overall + near-field), and the near-field gain-ratio array (E5 CDF
data). Fusion uses the D-015 combiner: angle from the radar branch, range from the
most-confident branch. Writes results/indomain/{modality}_{dtag}_seed{seed}_metrics.json.

    python scripts/eval_headline.py --modality fusion --seed 22 \
        --data data/processed/closerange_60GHz_full.h5 \
        --ckpt checkpoints/closerange_60GHz_fusion_warmstart/seed22.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_single import MODALITY
from src.data.splits import trajectory_grouped_split
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

B_THETA, B_R = 64, 8
RADAR_IDX = BRANCHES.index("radar")
KS = (1, 2, 4, 8)


def _joint_radar_lidar(model, feat, device):
    out = model(feat)
    pth = out["theta_logits"].float().softmax(1)
    prg = out["range_logits"].float().softmax(1)
    return pth, prg


def _joint_fusion(model, xr, xl, device):
    out = model(xr, xl)
    pth = torch.stack([F.softmax(out[f"{b}_theta_logits"].float(), 1) for b in BRANCHES])
    prg = torch.stack([F.softmax(out[f"{b}_range_logits"].float(), 1) for b in BRANCHES])
    crg = prg.max(-1).values                        # (3,B)
    ar = torch.arange(pth.shape[1], device=device)
    p_theta = pth[RADAR_IDX]                         # angle from radar branch (D-015)
    p_r = prg[crg.argmax(0), ar]                     # range from most-confident branch
    return p_theta, p_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "lidar", "fusion"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")

    dtag = os.path.basename(args.data).replace("_full.h5", "")
    f_ghz = int(re.search(r"(\d+)GHz", dtag).group(1))
    cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                              f"polar_codebook_{f_ghz}GHz.npz"))["codebook"]
    cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)

    is_fusion = args.modality == "fusion"
    with h5py.File(args.data, "r") as hf:
        traj = hf["trajectory_id"][:]
        _, _, te = trajectory_grouped_split(traj, seed=args.seed)
        te = np.sort(te)
        if is_fusion:
            xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
            xl = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
        else:
            feat = torch.from_numpy(hf[MODALITY[args.modality][0]][te].astype(np.float32))
        yt = torch.from_numpy(hf["y_theta_gated"][te].astype(np.int64))
        yr = torch.from_numpy(hf["y_r_gated"][te].astype(np.int64))
        chan = hf["channel"][te]

    ModelCls = SensingPolarPredictor if is_fusion else MODALITY[args.modality][1]
    model = ModelCls(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()

    n = yt.shape[0]
    true_flat = (yt * B_R + yr).numpy()
    pt1 = np.zeros(n, np.int64); pr1 = np.zeros(n, np.int64)
    cover = {k: np.zeros(n, bool) for k in KS}
    dbl = {k: np.zeros(n) for k in KS}
    gain_ratio_k4 = np.zeros(n)
    polar_vs_dft = np.zeros(n)        # genie polar advantage over genie DFT (dB)
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            h = torch.from_numpy(chan[s:e]).to(device, dtype=torch.complex64)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if is_fusion:
                    p_theta, p_r = _joint_fusion(model, xr[s:e].to(device),
                                                 xl[s:e].to(device), device)
                else:
                    p_theta, p_r = _joint_radar_lidar(model, feat[s:e].to(device), device)
            pt1[s:e] = p_theta.argmax(1).cpu().numpy()
            pr1[s:e] = p_r.argmax(1).cpu().numpy()
            joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(e - s, B_THETA * B_R)
            topk = joint.topk(max(KS), 1).indices
            gains = (cbc @ h.T).abs().pow(2).T
            genie = gains.max(1).values
            dft_genie = gains[:, 0::B_R].max(1).values     # best far-field (DFT) codeword
            polar_vs_dft[s:e] = (10 * torch.log10(genie / dft_genie.clamp_min(1e-30))).cpu().numpy()
            tk = torch.from_numpy(true_flat[s:e]).to(device)
            for k in KS:
                idx = topk[:, :k]
                cover[k][s:e] = (idx == tk.unsqueeze(1)).any(1).cpu().numpy()
                real = gains.gather(1, idx).max(1).values
                dbl[k][s:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
                if k == 4:
                    gain_ratio_k4[s:e] = (real / genie.clamp_min(1e-30)).cpu().numpy()

    yt_np, yr_np = yt.numpy(), yr.numpy()
    nf = yr_np > 0
    def acc(p, y, m=None):
        return float((p[m] == y[m]).mean()) if m is not None else float((p == y).mean())
    metrics = dict(
        modality=args.modality, dtag=dtag, seed=args.seed, n=n, n_nf=int(nf.sum()),
        top1_theta=acc(pt1, yt_np), top1_range=acc(pr1, yr_np),
        top1_theta_nf=acc(pt1, yt_np, nf), range_nf=acc(pr1, yr_np, nf),
        top1_joint=float(((pt1 == yt_np) & (pr1 == yr_np)).mean()),
        coverage={str(k): dict(all=float(cover[k].mean()), nf=float(cover[k][nf].mean()))
                  for k in KS},
        dbloss={str(k): dict(all=float(dbl[k].mean()), nf=float(dbl[k][nf].mean()))
                for k in KS},
        polar_vs_dft_genie_db=dict(all=float(polar_vs_dft.mean()),
                                   nf=float(polar_vs_dft[nf].mean()) if nf.any() else float("nan"),
                                   far=float(polar_vs_dft[~nf].mean()) if (~nf).any() else float("nan")),
        gain_ratio_k4_nf=gain_ratio_k4[nf].tolist())
    out_dir = os.path.join(_ROOT, "results", "indomain")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.modality}_{dtag}_seed{args.seed}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{args.modality} {dtag} seed{args.seed}] theta {metrics['top1_theta']:.3f} "
          f"range_nf {metrics['range_nf']:.3f} | K4 NF dB {metrics['dbloss']['4']['nf']:.2f}"
          f" K8 NF dB {metrics['dbloss']['8']['nf']:.2f} -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
