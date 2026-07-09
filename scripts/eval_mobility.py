"""D-020 Stage 2 eval: future-beam accuracy vs prediction horizon (DeepSense radar).

Evaluates mobility checkpoints (trained by train_single on mobility_sc{scen}_delta{D}_full.h5)
per (scenario, Delta, seed): top-1 beam accuracy + angle top-K coverage @ {4,8} on the
trajectory-grouped TEST split. Asserts no trajectory is in both train and test.

Modes:
  --anchor-check : evaluate Delta=0 only and verify the 5-seed mean top-1 reproduces the
    banked DeepSense accuracy WITHIN its 5-seed 95% CI (Sc31 23.05%, Sc36 17.14%). Exit 4
    (HARD-FAIL) if outside — the chain must stop before training Delta in {1,2,3}.
  default        : evaluate --deltas and bank results/mobility/future_beam.json.

    python scripts/eval_mobility.py --deltas 0 --anchor-check
    python scripts/eval_mobility.py --deltas 0 1 2 3
"""
from __future__ import annotations

import argparse
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

B_THETA, B_R = 64, 8
KS = (1, 2, 4, 8)
SEEDS = [11, 22, 33, 44, 55]
SCEN = [31, 36]


def mob_h5(scen, delta):
    return os.path.join(_ROOT, "data", "processed", "mobility",
                        f"mobility_sc{scen}_delta{delta}_full.h5")


def eval_one(scen, delta, seed, device):
    data = mob_h5(scen, delta)
    with h5py.File(data, "r") as hf:
        traj = hf["trajectory_id"][:]
        tr, _, te = trajectory_grouped_split(traj, seed=seed)
        assert not (set(traj[tr].tolist()) & set(traj[te].tolist())), "train/test trajectory overlap!"
        te = np.sort(te)
        xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
        yt = hf["y_theta_gated"][te].astype(np.int64)
    ck = os.path.join(_ROOT, "checkpoints", f"mobility_sc{scen}_delta{delta}_radar",
                      f"seed{seed}.pt")
    if not os.path.exists(ck):
        print(f"*** MOBILITY HARD-FAIL: missing checkpoint {os.path.relpath(ck,_ROOT)} ***")
        sys.exit(3)
    model = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    model.eval()
    n = len(te); top1 = 0; cov = {k: 0 for k in KS}
    with torch.no_grad():
        for s in range(0, n, 256):
            e = min(s + 256, n)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = model(xr[s:e].to(device))["theta_logits"].float()
            tk = p.topk(max(KS), 1).indices.cpu().numpy()
            y = yt[s:e]
            top1 += int((tk[:, 0] == y).sum())
            for k in KS:
                cov[k] += int((tk[:, :k] == y[:, None]).any(1).sum())
    return top1 / n, {k: cov[k] / n for k in KS}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--anchor-check", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    out = {"seeds": SEEDS, "scenarios": SCEN, "deltas": args.deltas, "results": {}}
    for scen in SCEN:
        out["results"][str(scen)] = {}
        for delta in args.deltas:
            with h5py.File(mob_h5(scen, delta), "r") as hf:
                horizon = float(hf.attrs["horizon_s"]); disp = float(hf.attrs["mean_disp_m"])
            t1, covs = [], {k: [] for k in KS}
            for seed in SEEDS:
                a, c, n = eval_one(scen, delta, seed, device)
                t1.append(a)
                for k in KS:
                    covs[k].append(c[k])
            ct1 = mean_ci(t1)
            out["results"][str(scen)][str(delta)] = dict(
                horizon_s=horizon, mean_disp_m=disp,
                top1=dict(per_seed=t1, **ct1),
                coverage={str(k): dict(**mean_ci(covs[k])) for k in KS})
            print(f"  sc{scen} d{delta} ({horizon:.1f}s): top1={ct1['mean']*100:.2f}"
                  f"±{(ct1['hi']-ct1['mean'])*100:.2f}% cov@4={mean_ci(covs[4])['mean']*100:.1f}% "
                  f"cov@8={mean_ci(covs[8])['mean']*100:.1f}%")

    if args.anchor_check:
        # banked DeepSense 5-seed top-1 CI (from the committed checkpoints' test.theta)
        for scen in SCEN:
            banked = [json.load(open(os.path.join(_ROOT, "results", "indomain",
                      f"radar_deepsense_scenario{scen}_seed{s}.json")))["test"]["theta"]
                      for s in SEEDS]
            bc = mean_ci(banked)
            m = out["results"][str(scen)]["0"]["top1"]["mean"]
            ok = bc["lo"] <= m <= bc["hi"]
            print(f"  ANCHOR sc{scen}: delta0 mean {m*100:.2f}% vs banked "
                  f"[{bc['lo']*100:.2f},{bc['hi']*100:.2f}]% -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(f"*** MOBILITY ANCHOR HARD-FAIL: sc{scen} delta0 {m*100:.2f}% outside "
                      f"banked 5-seed 95% CI [{bc['lo']*100:.2f},{bc['hi']*100:.2f}]% ***")
                sys.exit(4)
        print("  ANCHOR: PASS (both scenarios reproduce banked within 95% CI)")
        return   # anchor mode does not bank the partial (delta=0-only) result

    od = os.path.join(_ROOT, "results", "mobility")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "future_beam.json"), "w"), indent=2)
    print("Wrote results/mobility/future_beam.json")


if __name__ == "__main__":
    main()
