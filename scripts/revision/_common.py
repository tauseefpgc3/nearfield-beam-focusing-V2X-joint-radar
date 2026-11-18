"""Shared helpers for the TVT-revision Phase-1 diagnostics (D-021..D-028). Eval-only."""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.statistics import mean_ci                       # noqa: E402
from src.models.radar_encoder import RadarPolarModel          # noqa: E402
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor  # noqa: E402

ROOT = _ROOT
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165,
         176, 187, 198, 209, 220]
FREQS = ["60", "140"]
B_THETA, B_R, NCW = 64, 8, 512
KS = (1, 2, 4, 8)
SETS = ("all", "r_lt_dF", "nf_ge3dB", "complement")
OUT = os.path.join(ROOT, "results", "revision")


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"wrote results/revision/{name}", flush=True)


def ckpt(kind, f, s):
    d = {"radar": f"closerange_{f}GHz_radar", "fusion": f"closerange_{f}GHz_fusion_warmstart"}[kind]
    return os.path.join(ROOT, "checkpoints", d, f"seed{s}.pt")


def load_model(kind, f, s, device):
    m = (RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R) if kind == "radar"
         else SensingPolarPredictor(d_z=128, B_theta=B_THETA, B_r=B_R)).to(device)
    m.load_state_dict(torch.load(ckpt(kind, f, s), map_location=device, weights_only=False)["model"])
    return m.eval()


def load_freq(f, device, need_lidar=True):
    """Whole close-range file in RAM + gains (6000,512) computed once on GPU."""
    path = os.path.join(ROOT, "data", "processed", f"closerange_{f}GHz_full.h5")
    cb = np.load(os.path.join(ROOT, "data", "codebooks", f"polar_codebook_{f}GHz.npz"))
    cbc = torch.from_numpy(np.conj(cb["codebook"])).to(device, dtype=torch.complex64)
    with h5py.File(path, "r") as hf:
        D = dict(traj=hf["trajectory_id"][:], dist=hf["distance"][:].astype(np.float64),
                 yt=hf["y_theta_gated"][:].astype(np.int64), yr=hf["y_r_gated"][:].astype(np.int64),
                 adv=hf["polar_adv_db"][:].astype(np.float64), dF=float(hf.attrs["d_F"]),
                 xr=hf["radar_features"][:])
        if need_lidar:
            D["xl"] = hf["lidar_features"][:]
        n = D["yt"].shape[0]
        G = np.empty((n, NCW), np.float32)
        for s in range(0, n, 256):
            e = min(s + 256, n)
            h = torch.from_numpy(hf["channel"][s:e]).to(device, dtype=torch.complex64)
            G[s:e] = (cbc @ h.T).abs().pow(2).T.cpu().numpy()
    D["G"] = G
    D["ranges"] = cb["ranges"]
    return D


def masks(D, idx):
    yr, dist = D["yr"][idx], D["dist"][idx]
    return {"all": np.ones(len(idx), bool), "r_lt_dF": dist < D["dF"],
            "nf_ge3dB": yr > 0, "complement": yr == 0}


@torch.no_grad()
def infer_radar(model, xr, device, bs=256):
    PT, PR = [], []
    for s in range(0, len(xr), bs):
        x = torch.from_numpy(xr[s:s + bs].astype(np.float32)).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            o = model(x)
            PT.append(o["theta_logits"].float().softmax(1).cpu())
            PR.append(o["range_logits"].float().softmax(1).cpu())
    return torch.cat(PT), torch.cat(PR)


@torch.no_grad()
def infer_fusion(model, xr, xl, device, bs=256, rd=False, ld=False, zero_inputs=False):
    """Per-branch probabilities: pth (3,n,64), prg (3,n,8)."""
    PT, PR = [], []
    for s in range(0, len(xr), bs):
        a = torch.from_numpy(xr[s:s + bs].astype(np.float32)).to(device)
        b = torch.from_numpy(xl[s:s + bs].astype(np.float32)).to(device)
        if zero_inputs:
            a = torch.zeros_like(a); b = torch.zeros_like(b)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            o = model(a, b, radar_dropout=rd, lidar_dropout=ld)
            PT.append(torch.stack([F.softmax(o[f"{br}_theta_logits"].float(), 1) for br in BRANCHES]).cpu())
            PR.append(torch.stack([F.softmax(o[f"{br}_range_logits"].float(), 1) for br in BRANCHES]).cpu())
    return torch.cat(PT, 1), torch.cat(PR, 1)


def joint_rank(p_theta, p_r, k=8):
    n = p_theta.shape[0]
    joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(n, NCW)
    return joint.topk(k, 1).indices.numpy()


def losses_from_rank(G, rank, ks=KS):
    """rank (n,8) flat codeword indices in priority order -> {K: (n,) dB loss}."""
    genie = G.max(1)
    out = {}
    for k in ks:
        real = np.take_along_axis(G, rank[:, :k], 1).max(1)
        out[k] = 10 * np.log10(genie / np.maximum(real, 1e-30))
    return out


def agg(per_seed_vals, pooled=None):
    d = dict(per_seed=[float(v) for v in per_seed_vals], **mean_ci(per_seed_vals))
    if pooled is not None and len(pooled):
        d["pooled_median"] = float(np.median(pooled)); d["pooled_n"] = int(len(pooled))
    return d
