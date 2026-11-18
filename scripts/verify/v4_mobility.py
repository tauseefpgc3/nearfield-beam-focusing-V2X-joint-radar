"""Independent verification, step 4: V7 mobility (DeepSense Sc31 / Sc36).

Own pairing from data/processed/deepsense_scenario{31,36}_full.h5 (row order within a
seq_index trajectory), checked against the banked mobility_sc*_delta*_full.h5 files.
Beam-change rate, label-noise diagnostics, persistence vs the banked Delta models on each
Delta dataset's own trajectory-grouped test split. Output results/verify/v4_mobility.json
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.splits import trajectory_grouped_split
from src.models.radar_encoder import RadarPolarModel

dev = torch.device("cuda")
SEEDS = [11, 22, 33, 44, 55]
res = {}
hand = {}
for sc in (31, 36):
    with h5py.File(os.path.join(ROOT, "data", "processed", f"deepsense_scenario{sc}_full.h5"), "r") as hf:
        traj = hf["trajectory_id"][:]; y = hf["y_theta_gated"][:].astype(int)
        bb = hf["best_beam"][:].astype(int); dist = hf["distance"][:]
    r = dict(n_frames=int(len(y)), n_traj=int(len(np.unique(traj))),
             label_eq_power_argmax=float((y == bb).mean()),
             label_within1_power_argmax=float((np.abs(y - bb) <= 1).mean()))
    # trajectories contiguous in row order?
    r["traj_contiguous"] = bool(all(np.all(np.diff(np.where(traj == t)[0]) == 1) for t in np.unique(traj)))
    pairs = {}
    for D in (1, 2, 3):
        s_, t_ = [], []
        for t in np.unique(traj):
            gi = np.where(traj == t)[0]
            s_ += list(gi[:len(gi) - D]); t_ += list(gi[D:])
        pairs[D] = (np.array(s_), np.array(t_))
    s1, t1 = pairs[1]
    dd = dist[t1] - dist[s1]
    r["mean_abs_range_step_m"] = float(np.abs(dd).mean())
    r["frac_range_step_gt_5m"] = float((np.abs(dd) > 5).mean())
    # sign persistence of range change (a temporally ordered drive changes sign rarely)
    s2, t2 = pairs[2]
    same_traj_3 = np.isin(s1, s2)
    r["range_step_sign_flip_rate"] = float(np.mean(np.sign(dd[:-1]) != np.sign(dd[1:])))
    for D in (1, 2, 3):
        s_, t_ = pairs[D]
        db = y[t_] - y[s_]
        r[f"D{D}"] = dict(n_pairs=int(len(s_)), change_rate=float((db != 0).mean()),
                          change_ge2=float((np.abs(db) >= 2).mean()),
                          abs_change_hist={int(k): int(v) for k, v in zip(*np.unique(np.minimum(np.abs(db), 10), return_counts=True))},
                          change_rate_power_argmax=float((bb[t_] != bb[s_]).mean()))
    # flip-back A->B->A at lag 1 (jitter signature) vs B->C continuing
    a, b, c = [], [], []
    for t in np.unique(traj):
        gi = np.where(traj == t)[0]
        a += list(y[gi[:-2]]); b += list(y[gi[1:-1]]); c += list(y[gi[2:]])
    a, b, c = map(np.array, (a, b, c))
    ch = a != b
    r["flip_back_rate_given_change"] = float((c[ch] == a[ch]).mean())
    r["continue_same_direction_given_change"] = float((np.sign(c[ch] - b[ch]) == np.sign(b[ch] - a[ch])).mean())
    # whole-trajectory net drift vs sum of |steps| (label noise inflates the path length)
    net, path = [], []
    for t in np.unique(traj):
        g = y[traj == t]
        if len(g) > 5:
            net.append(abs(int(g[-1]) - int(g[0]))); path.append(int(np.abs(np.diff(g)).sum()))
    r["traj_net_beam_drift_mean"] = float(np.mean(net)); r["traj_sum_abs_steps_mean"] = float(np.mean(path))

    # ---- alignment with the banked mobility files
    for D in (1, 2, 3):
        with h5py.File(os.path.join(ROOT, "data", "processed", "mobility", f"mobility_sc{sc}_delta{D}_full.h5"), "r") as mf:
            my_s, my_t = pairs[D]
            o = np.argsort(my_s, kind="stable"); my_s, my_t = my_s[o], my_t[o]
            lab_ok = bool(np.array_equal(mf["y_theta_gated"][:], y[my_t]))
            tr_ok = bool(np.array_equal(mf["trajectory_id"][:], traj[my_s]))
            with h5py.File(os.path.join(ROOT, "data", "processed", f"deepsense_scenario{sc}_full.h5"), "r") as hf:
                rf_ok = all(np.array_equal(mf["radar_features"][q], hf["radar_features"][my_s[q]]) for q in (0, 100, len(my_s) - 1))
        r[f"D{D}"]["banked_pairs_match_mine"] = dict(labels=lab_ok, traj=tr_ok, radar_rows=rf_ok)

    # ---- persistence vs models on each Delta dataset's own test split
    with h5py.File(os.path.join(ROOT, "data", "processed", f"deepsense_scenario{sc}_full.h5"), "r") as hf:
        XR = hf["radar_features"][:]
    rows = {}
    for D in (0, 1, 2, 3):
        rows[D] = []
        if D == 0:
            s_, t_ = np.arange(len(y)), np.arange(len(y))
        else:
            s_, t_ = pairs[D]; o = np.argsort(s_, kind="stable"); s_, t_ = s_[o], t_[o]
        for seed in SEEDS:
            _, _, te = trajectory_grouped_split(traj[s_], seed=seed); te = np.sort(te)
            m = RadarPolarModel().to(dev).eval()
            m.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"mobility_sc{sc}_delta{D}_radar", f"seed{seed}.pt"),
                                         map_location="cpu", weights_only=False)["model"])
            preds = []
            with torch.no_grad():
                for b in range(0, len(te), 256):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        preds.append(m(torch.from_numpy(XR[s_[te[b:b + 256]]]).to(dev))["theta_logits"].float().argmax(1).cpu().numpy())
            p = np.concatenate(preds)
            yt_, ys_ = y[t_[te]], y[s_[te]]
            chg = yt_ != ys_
            rows[D].append(dict(seed=seed, n=len(te), model_top1=float((p == yt_).mean()),
                                persistence_top1=float((ys_ == yt_).mean()),
                                persistence_within1=float((np.abs(ys_ - yt_) <= 1).mean()),
                                model_within1=float((np.abs(p - yt_) <= 1).mean()),
                                model_top1_on_changed=float((p[chg] == yt_[chg]).mean()) if chg.any() else None,
                                model_top1_vs_current=float((p == ys_).mean())))
        r[f"eval_D{D}"] = {k: float(np.mean([x[k] for x in rows[D] if x[k] is not None])) for k in rows[D][0] if k != "seed"}
    # hand examples: first trajectory, frames 0..5
    t0 = np.unique(traj)[0]; gi = np.where(traj == t0)[0][:8]
    hand[str(sc)] = dict(traj=int(t0), rows=[dict(row=int(i), beam=int(y[i]), power_argmax=int(bb[i]), dist_m=round(float(dist[i]), 2)) for i in gi])
    res[str(sc)] = r
    print(sc, json.dumps(r, indent=1))
    del XR
res["hand"] = hand
print(json.dumps(hand, indent=1))
json.dump(res, open(os.path.join(ROOT, "results", "verify", "v4_mobility.json"), "w"), indent=1)
