"""B4 (pre-registered D-024): mobility persistence on DeepSense Sc31/Sc36. Eval-only.

Fixed sample set per (scenario, seed) shared by all horizons; beam-change rate, persistence
baseline, and the Delta-trained models' accuracy on changed vs unchanged frames.

    python scripts/revision/b4_mobility.py
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, agg, dump                              # noqa: E402
from src.data.splits import trajectory_grouped_split              # noqa: E402
from src.models.radar_encoder import RadarPolarModel              # noqa: E402

SEEDS = [11, 22, 33, 44, 55]
SCEN = [31, 36]
DELTAS = [1, 2, 3]


def pairs(traj, delta):
    """Exactly build_mobility_pairs.build(): within-trajectory, sorted by source index."""
    src, tgt = [], []
    for tt in np.unique(traj):
        gi = np.sort(np.where(traj == tt)[0])
        if len(gi) > delta:
            src.extend(gi[:len(gi) - delta].tolist()); tgt.extend(gi[delta:].tolist())
    src, tgt = np.asarray(src), np.asarray(tgt)
    o = np.argsort(src, kind="stable")
    return src[o], tgt[o]


def persist_cov(y0, K, nb=64):
    b = np.arange(nb)
    out = np.empty((len(y0), K), int)
    for n, y in enumerate(y0):
        out[n] = b[np.lexsort((b, np.abs(b - y)))][:K]
    return out


@torch.no_grad()
def predict(ck, X, dev):
    m = RadarPolarModel(d_z=128, B_theta=64, B_r=8).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"]); m.eval()
    tk = []
    for s in range(0, len(X), 256):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p = m(torch.from_numpy(X[s:s + 256].astype(np.float32)).to(dev))["theta_logits"].float()
        tk.append(p.topk(8, 1).indices.cpu().numpy())
    return np.concatenate(tk)


def main():
    dev = torch.device("cuda")
    out = {"seeds": SEEDS, "prereg": "docs/decisions.md D-024, release commit 99b5cda",
           "frame_dt_s_nominal": 0.1, "results": {}}
    for scen in SCEN:
        src_h5 = os.path.join(ROOT, "data", "processed", f"deepsense_scenario{scen}_full.h5")
        with h5py.File(src_h5, "r") as hf:
            traj = hf["trajectory_id"][:]; y = hf["y_theta_gated"][:].astype(np.int64)
            X = hf["radar_features"][:]
        P = {d: pairs(traj, d) for d in [0] + DELTAS}
        # consistency with the banked mobility files
        chk = {}
        for d in [0] + DELTAS:
            with h5py.File(os.path.join(ROOT, "data", "processed", "mobility",
                                        f"mobility_sc{scen}_delta{d}_full.h5"), "r") as mf:
                ok_y = bool(np.array_equal(mf["y_theta_gated"][:], y[P[d][1]]))
                ok_t = bool(np.array_equal(mf["trajectory_id"][:], traj[P[d][0]]))
                ok_x = bool(np.array_equal(mf["radar_features"][:3], X[P[d][0][:3]]))
            chk[str(d)] = dict(labels=ok_y, traj=ok_t, radar_rows_spot=ok_x)
        # whole-file change rate over all within-trajectory pairs
        whole = {str(d): float((y[P[d][0]] != y[P[d][1]]).mean()) for d in DELTAS}
        R = {"consistency_with_banked_mobility_h5": chk, "whole_file_change_rate": whole,
             "n_frames": int(len(y)), "n_trajectories": int(len(np.unique(traj))), "per_delta": {}}
        acc = {d: {k: [] for k in ("change_rate", "persist_top1", "ours_top1", "ours_top1_changed",
                                   "ours_top1_unchanged", "persist_cov4", "persist_cov8", "ours_cov4",
                                   "ours_cov8", "delta0_model_top1", "n_changed", "n_unchanged")}
               for d in DELTAS}
        sizes, d0n = [], []
        pos = {}                                              # frame -> (traj, position)
        for tt in np.unique(traj):
            gi = np.sort(np.where(traj == tt)[0])
            for k, g in enumerate(gi):
                pos[g] = (tt, k, gi)
        for seed in SEEDS:
            test_traj = []
            for d in DELTAS:
                _, _, te = trajectory_grouped_split(traj[P[d][0]], seed=seed)
                test_traj.append(set(traj[P[d][0]][te].tolist()))
            common_t = set.intersection(*test_traj)
            fixed = np.array([t for t in P[3][0] if traj[t] in common_t])      # has t+3 in traj
            _, _, te0 = trajectory_grouped_split(traj[P[0][0]], seed=seed)
            t0_traj = set(traj[te0].tolist())
            sizes.append(int(len(fixed)))
            y0 = y[fixed]
            Xf = X[fixed]
            in0 = np.array([traj[t] in t0_traj for t in fixed]); d0n.append(int(in0.sum()))
            tk0 = predict(os.path.join(ROOT, "checkpoints", f"mobility_sc{scen}_delta0_radar", f"seed{seed}.pt"), Xf, dev)
            for d in DELTAS:
                yd = np.array([pos[t][2][pos[t][1] + d] for t in fixed]); yd = y[yd]
                ch = yd != y0
                tk = predict(os.path.join(ROOT, "checkpoints", f"mobility_sc{scen}_delta{d}_radar", f"seed{seed}.pt"), Xf, dev)
                hit1 = tk[:, 0] == yd
                a = acc[d]
                a["change_rate"].append(float(ch.mean())); a["persist_top1"].append(float((~ch).mean()))
                a["ours_top1"].append(float(hit1.mean()))
                a["ours_top1_changed"].append(float(hit1[ch].mean())); a["ours_top1_unchanged"].append(float(hit1[~ch].mean()))
                a["n_changed"].append(int(ch.sum())); a["n_unchanged"].append(int((~ch).sum()))
                for K in (4, 8):
                    a[f"persist_cov{K}"].append(float((persist_cov(y0, K) == yd[:, None]).any(1).mean()))
                    a[f"ours_cov{K}"].append(float((tk[:, :K] == yd[:, None]).any(1).mean()))
                a["delta0_model_top1"].append(float((tk0[:, 0] == yd)[in0].mean()))
            print(f"  sc{scen} seed{seed}: fixed n={len(fixed)} | " + " | ".join(
                f"D{d} chg {acc[d]['change_rate'][-1]:.3f} ours {acc[d]['ours_top1'][-1]:.3f} "
                f"(chg {acc[d]['ours_top1_changed'][-1]:.3f} / unchg {acc[d]['ours_top1_unchanged'][-1]:.3f})"
                for d in DELTAS), flush=True)
        R["fixed_set_size_per_seed"] = sizes
        R["delta0_model_eval_subset_size_per_seed"] = d0n
        R["delta0_model_note"] = ("Delta=0 model scored only on fixed-set frames whose trajectory is also in "
                                  "the Delta=0 test split (no train leakage)")
        R["per_delta"] = {str(d): {k: (v if k.startswith("n_") else agg(v)) for k, v in acc[d].items()}
                          for d in DELTAS}
        out["results"][str(scen)] = R
    dump("B4_mobility.json", out)


if __name__ == "__main__":
    main()
