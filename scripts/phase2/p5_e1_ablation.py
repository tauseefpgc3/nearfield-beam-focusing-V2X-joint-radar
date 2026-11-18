"""D-030 E1 (exploratory): three-branch model with the radar (or LiDAR) input zeroed."""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2"))
import p3_eval as E
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor


def infer(freq, seed, xr, xl):
    m = SensingPolarPredictor(d_z=128, B_theta=256, B_r=8)
    m.load_state_dict(torch.load(os.path.join(E.CK, f"fusion_{freq}GHz_seed{seed}.pt"), map_location="cpu", weights_only=False)["model"])
    m = m.to(E.dev).eval(); PT, PR = [], []
    with torch.no_grad():
        for b in range(0, len(xr), 128):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                o = m(torch.from_numpy(xr[b:b + 128]).to(E.dev), torch.from_numpy(xl[b:b + 128]).to(E.dev))
                pth = torch.stack([F.softmax(o[f"{x}_theta_logits"].float(), 1) for x in BRANCHES])
                prg = torch.stack([F.softmax(o[f"{x}_range_logits"].float(), 1) for x in BRANCHES])
                wt = pth.max(-1).values; wt = wt / wt.sum(0, keepdim=True).clamp_min(1e-12)
                wr = prg.max(-1).values; wr = wr / wr.sum(0, keepdim=True).clamp_min(1e-12)
                PT.append((wt[..., None] * pth).sum(0).cpu().numpy()); PR.append((wr[..., None] * prg).sum(0).cpu().numpy())
    return np.concatenate(PT), np.concatenate(PR)


def main():
    with h5py.File(os.path.join(E.P, "split.h5"), "r") as hf: te = np.sort(hf["test"][:])
    with h5py.File(os.path.join(E.P, "lidar.h5"), "r") as hf: XL = hf["lidar_features"][te]
    res = {}
    for freq in (60, 140):
        with h5py.File(os.path.join(E.P, f"channels_{freq}GHz.h5"), "r") as hf: G = hf["gains"][:][te]
        for snr in E.SNRS:
            X = E.load_radar(snr, te)[0]
            for cond, xr, xl in (("normal", X, XL), ("radar_zeroed", np.zeros_like(X), XL), ("lidar_zeroed", X, np.zeros_like(XL))):
                per = []
                for s in E.SEEDS:
                    sl = E.topk_joint(*infer(freq, s, xr, xl)); per.append([float(E.loss_of(G, sl[:, :k]).mean()) for k in E.KS])
                per = np.array(per); res[f"{freq}|{snr}|{cond}"] = dict(mean=per.mean(0).tolist(), per_seed=per.tolist())
                print(freq, snr, cond, np.round(per.mean(0), 2), flush=True)
    json.dump(res, open(os.path.join(ROOT, "results", "phase2", "feasibility", "e1_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
