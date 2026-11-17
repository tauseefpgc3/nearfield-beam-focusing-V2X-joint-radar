"""A5 (EXPLORATORY): causes of the LiDAR network's worst 5% of test users.
Per-user loss = mean over the 5 seeds and both frequencies at M = 8 (shape chosen on validation
per seed). Each tail user is tagged (tags can overlap):
  front_back  : 5-seed mean prediction within 0.6 m of the flipped-heading antenna position
                (vehicle centre + 1.3 m * forward) and > 1 m from the true antenna;
  pole        : the RSU pole (0, 0) lies inside the vehicle footprint (sensor inside the car);
  close       : horizontal distance < 2 m;
  few_returns : fewer vehicle returns than the 5th percentile over test users;
  large_error : 5-seed mean prediction error > 0.3 m (any cause);
  small_error : error < 0.1 m (loss not explained by position error: sweep/codebook limit).
Rates are compared with the remaining 95% of test users."""
import json, os, sys
import h5py, numpy as np
from common import *
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
import lidar_v2 as SIM

S = split(); va, te = S["val"], S["test"]; ant, psi = users(); xy = ant[:, :2]
P = dict(np.load(os.path.join(OUT, "preds.npz"))); CH = {f: channels(f) for f in (60, 140)}
L = np.mean([sweep_with_val_shape(CH[f], va, te, P[f"lidar_s{s}_val"], P[f"lidar_s{s}_test"], 8)[0] for s in SEEDS for f in (60, 140)], 0)
pm = np.mean([P[f"lidar_s{s}_test"] for s in SEEDS], 0); err = np.hypot(*(pm - xy[te]).T)
cen = SIM.vehicle_centre(ant, psi)[te]; fwd = np.c_[np.cos(psi), np.sin(psi)][te]
flip = cen + 1.3 * fwd
with h5py.File(os.path.join(P3, "lidar_bev.h5"), "r") as h: nv = h["n_vehicle_returns"][:][te]; pole = h["pole_in_footprint"][:][te]
d = np.hypot(*xy[te].T)
tags = dict(front_back=(np.hypot(*(pm - flip).T) < 0.6) & (err > 1.0), pole=pole, close=d < 2.0,
            few_returns=nv < np.percentile(nv, 5), large_error=err > 0.3, small_error=err < 0.1)
k = int(np.ceil(0.05 * len(te))); order = np.argsort(-L); tail = np.zeros(len(te), bool); tail[order[:k]] = True
adv = {f: CH[f]["adv"][te] for f in CH}
res = dict(n_tail=k, tail_loss_threshold_db=float(L[order[k - 1]]),
           share_of_total_loss_from_tail=float(L[tail].sum() / L.sum()),
           mean_loss_all=float(L.mean()), mean_loss_without_tail=float(L[~tail].mean()),
           tags={t: dict(tail=float(v[tail].mean()), rest=float(v[~tail].mean()), n_tail=int(v[tail].sum())) for t, v in tags.items()},
           tail_unexplained=int((tail & ~(tags["front_back"] | tags["pole"] | tags["close"] | tags["few_returns"] | tags["large_error"])).sum()),
           distance_median_m=dict(tail=float(np.median(d[tail])), rest=float(np.median(d[~tail]))),
           err_median_m=dict(tail=float(np.median(err[tail])), rest=float(np.median(err[~tail]))),
           adv60_median_db=dict(tail=float(np.median(adv[60][tail])), rest=float(np.median(adv[60][~tail]))),
           returns_median=dict(tail=float(np.median(nv[tail])), rest=float(np.median(nv[~tail]))),
           n_pole_test=int(pole.sum()), n_close_test=int((d < 2).sum()))
json.dump(res, open(os.path.join(OUT, "a5_tail.json"), "w"), indent=1); print(json.dumps(res, indent=1))
