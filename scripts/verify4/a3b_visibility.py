"""D-034 A3 follow-up: receiver visibility in the LiDAR for every test frame (scenarios 31, 36).

Uses the rotation fitted in a3_ds{S}.json (lag 0) and the GPS projection of
a3_deepsense_alignment.py. Per test frame: inside the BEV grid; visible = >= 3 non-ground
points within 2 m of the projected receiver; receiver range. Network top-1 accuracy (20-seed
mean) and the baselines' top-1 are split by visibility.

    python scripts/verify4/a3b_visibility.py 31 36   # -> results/verify4/a3b_ds{S}.json
"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds_prepare as DP, a3_deepsense_alignment as A3, train_ds as T


def run(s):
    a3 = json.load(open(os.path.join(ROOT, "results", "verify4", f"a3_ds{s}.json")))
    df, lid, _ = DP.index(s); rel, ok = A3.gps(s, df)
    with h5py.File(os.path.join(T.DATA, f"ds{s}.h5"), "r") as h:
        split = h["split"][:]; lab = h["label"][:].astype(int); P = h["power"][:]
        g = {k: h.attrs[k] for k in ("x0", "y0", "cell", "NX", "NY", "z_ground", "R_clip")}
        te = np.where((split == 2) & np.isfinite(P).all(1))[0]; Xte = torch.from_numpy(h["bev"][te])
        th = np.deg2rad(a3["geometry_fit"]["theta_at_lag0_deg"]); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        p = (Rm @ rel[te].T).T
        with Pool(8) as pool: Qs = pool.map(A3.ng_points, [(s, lid[i], float(g["z_ground"])) for i in te], chunksize=16)
        near = np.array([(np.hypot(*(Q - q).T) <= 2).sum() for Q, q in zip(Qs, p)])
        inside = ((p[:, 0] - g["x0"]) / g["cell"] >= 0) & ((p[:, 0] - g["x0"]) / g["cell"] < g["NX"]) & \
                 ((p[:, 1] - g["y0"]) / g["cell"] >= 0) & ((p[:, 1] - g["y0"]) / g["cell"] < g["NY"]) & (np.hypot(*p.T) <= g["R_clip"])
        vis = (near >= 3) & ok[te]
        dev = torch.device("cuda"); hit = []
        for sd in range(11, 221, 11):
            ck = torch.load(os.path.join(T.CK, f"ds{s}_s{sd}.pt"), map_location="cpu", weights_only=False)
            m = T.build(h); m.load_state_dict(ck["model"], strict=True); hit.append(T.predict(m.to(dev), Xte, dev).argmax(1) == lab[te])
    hit = np.mean(hit, 0); rng_ = np.hypot(*p.T)
    R = dict(scenario=s, n_test=int(len(te)), gps_valid=int(ok[te].sum()), inside_grid=float(inside[ok[te]].mean()), visible=float(vis.mean()),
             receiver_range_m=dict(median=float(np.median(rng_[ok[te]])), p90=float(np.percentile(rng_[ok[te]], 90))),
             network_top1=dict(visible=float(hit[vis].mean()) if vis.any() else None, not_visible=float(hit[~vis].mean()) if (~vis).any() else None),
             visible_frames=int(vis.sum()))
    json.dump(R, open(os.path.join(ROOT, "results", "verify4", f"a3b_ds{s}.json"), "w"), indent=1); print(json.dumps(R, indent=1), flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
