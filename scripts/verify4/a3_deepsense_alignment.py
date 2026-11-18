"""D-034 A3: DeepSense sanity for scenarios 31 and 36 (GPS projected into the LiDAR frame).

    python scripts/verify4/a3_deepsense_alignment.py 31 36

Per scenario:
  * receiver position relative to the LiDAR from GPS (local ENU, equirectangular about the
    first unit-1 fix); s36: in unit 1's heading frame (heading from unit 1's GPS track,
    central difference over +-2 frames, frames with < 0.5 m displacement dropped);
  * rotation theta (0..359 deg) and lag k (frames, within a sequence): score = mean number of
    non-ground LiDAR points within 2 m of R(theta) * rel(t + k); fitted on training frames
    (s31 every 2nd, s36 every 3rd frame). Lag range +-20 (deviation from +-10: the s36 file
    timestamps show constant per-device offsets up to 1.3 s, recorded before this ran);
  * label lag: mutual information between the beam label at t and the GPS bearing bin (5 deg)
    at t + k, over all frames;
  * per-sequence theta at k = 0 (circular SD); 10 test frames (default_rng(34)) rendered on the
    stored BEV with the projected receiver, inside-grid check. Images -> results/verify4/a3_local/
    (derived from licensed data: not committed).
-> results/verify4/a3_ds{S}.json
"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); import ds_prepare as DP
OUT = os.path.join(ROOT, "results", "verify4"); LOC = os.path.join(OUT, "a3_local"); os.makedirs(LOC, exist_ok=True)
KMAX = 20; RE = 6378137.0


def enu(lat, lon, lat0, lon0):
    return np.c_[np.deg2rad(lon - lon0) * np.cos(np.deg2rad(lat0)) * RE, np.deg2rad(lat - lat0) * RE]


def gps(s, df):
    if s == 31:
        root = DP.SC[31]["root"]; u1 = np.loadtxt(os.path.join(root, "unit1", "GPS_data", "gps_location.txt"))
        u2 = np.array([np.loadtxt(os.path.join(root, p)) for p in df.unit2_loc])
        rel = enu(u2[:, 0], u2[:, 1], u1[0], u1[1]); return rel, np.ones(len(df), bool)
    p1 = enu(df.unit1_gps1_lat.to_numpy(), df.unit1_gps1_lon.to_numpy(), df.unit1_gps1_lat.iloc[0], df.unit1_gps1_lon.iloc[0])
    p2 = enu(df.unit2_gps1_lat.to_numpy(), df.unit2_gps1_lon.to_numpy(), df.unit1_gps1_lat.iloc[0], df.unit1_gps1_lon.iloc[0])
    seq = df.seq_index.to_numpy(); n = len(df); hd = np.full(n, np.nan)
    for i in range(n):
        a, b = max(i - 2, 0), min(i + 2, n - 1)
        while seq[a] != seq[i]: a += 1
        while seq[b] != seq[i]: b -= 1
        d = p1[b] - p1[a]
        if np.hypot(*d) >= 0.5: hd[i] = np.arctan2(d[1], d[0])
    r = p2 - p1; c, s_ = np.cos(-hd), np.sin(-hd)
    rel = np.c_[c * r[:, 0] - s_ * r[:, 1], s_ * r[:, 0] + c * r[:, 1]]; return rel, np.isfinite(hd)


def ng_points(args):
    s, path, zg = args
    P = DP.load_points((s, path)); m = (P[:, 2] > zg + 0.3) & (P[:, 2] < zg + 3.0) & (np.hypot(P[:, 0], P[:, 1]) < 60)
    Q = P[m, :2]
    if len(Q) > 4000: Q = Q[np.random.default_rng(0).choice(len(Q), 4000, replace=False)]
    return Q.astype(np.float32)


def ring_counts(Q, r):
    """C[a] = #points within 2 m of r*(cos a, sin a), a = 0..359 deg."""
    rho = np.hypot(Q[:, 0], Q[:, 1]); beta = np.rad2deg(np.arctan2(Q[:, 1], Q[:, 0]))
    m = np.abs(rho - r) <= 2.0; rho, beta = rho[m], beta[m]; C = np.zeros(361)
    if not len(rho) or r < 0.5: return C[:360]
    w = np.rad2deg(np.arccos(np.clip((rho ** 2 + r ** 2 - 4.0) / (2 * rho * r), -1, 1)))
    lo = np.floor(beta - w).astype(int); hi = np.floor(beta + w).astype(int)
    for a, b in zip(lo, hi):
        if b - a >= 359: C[0] += 1; C[360] -= 1; continue
        a %= 360; b %= 360
        if a <= b: C[a] += 1; C[b + 1] -= 1
        else: C[a] += 1; C[360] -= 1; C[0] += 1; C[b + 1] -= 1
    return np.cumsum(C)[:360]


def mi(x, y):
    J = np.zeros((x.max() + 1, y.max() + 1)); np.add.at(J, (x, y), 1); J /= J.sum()
    px, py = J.sum(1, keepdims=True), J.sum(0, keepdims=True); nz = J > 0
    return float((J[nz] * np.log(J[nz] / (px @ py)[nz])).sum())


def run(s):
    df, lid, _ = DP.index(s); seq = df.seq_index.to_numpy(); n = len(df)
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase3c", f"ds{s}.h5"), "r") as h:
        split = h["split"][:]; lab = h["label"][:].astype(int); g = {k: h.attrs[k] for k in ("x0", "y0", "cell", "NX", "NY", "z_ground", "R_clip")}
    rel, ok = gps(s, df); R = dict(scenario=s, frames=n, gps_valid=int(ok.sum()))
    step = 2 if s == 31 else 3; fit = np.where((split == 0) & ok)[0][::step]
    with Pool(8) as pool: Qs = pool.map(ng_points, [(s, lid[i], float(g["z_ground"])) for i in fit], chunksize=16)
    lags = np.arange(-KMAX, KMAX + 1); S = np.zeros((len(lags), 360)); N = np.zeros(len(lags))
    per_seq = {}
    for Q, t in zip(Qs, fit):
        for j, k in enumerate(lags):
            u = t + k
            if u < 0 or u >= n or seq[u] != seq[t] or not ok[u]: continue
            r = float(np.hypot(*rel[u])); phi = int(np.round(np.rad2deg(np.arctan2(rel[u, 1], rel[u, 0])))) % 360
            C = ring_counts(Q, r); sh = C[(phi + np.arange(360)) % 360]       # sh[theta] = count at rotation theta
            S[j] += sh; N[j] += 1
            if k == 0: per_seq.setdefault(int(seq[t]), np.zeros(360)).__iadd__(sh)
    M = S / np.maximum(N, 1)[:, None]; jb, tb = np.unravel_index(M.argmax(), M.shape); j0 = int(np.where(lags == 0)[0][0])
    th0 = int(M[j0].argmax())
    R["geometry_fit"] = dict(frames_used=int(len(fit)), best_lag_frames=int(lags[jb]), best_theta_deg=int(tb), score_best=float(M[jb, tb]),
                             theta_at_lag0_deg=th0, score_lag0=float(M[j0, th0]), score_lag0_median_over_theta=float(np.median(M[j0])),
                             best_lag_within_10=int(lags[np.argmax(M[np.abs(lags) <= 10].max(1)) + (KMAX - 10)]),
                             score_by_lag=[round(float(x), 3) for x in M.max(1)])
    ths = np.deg2rad([int(v.argmax()) for v in per_seq.values()])
    Rbar = np.hypot(np.cos(ths).mean(), np.sin(ths).mean())
    R["per_sequence_theta"] = dict(n_sequences=len(ths), theta_deg=[int(np.rad2deg(t)) for t in ths], circular_sd_deg=float(np.rad2deg(np.sqrt(-2 * np.log(max(Rbar, 1e-12))))))
    # label lag: MI(label(t), bearing bin(t+k)) over all frames
    bb = (np.round(np.rad2deg(np.arctan2(rel[:, 1], rel[:, 0])) / 5).astype(int)) % 72
    mis = []
    for k in lags:
        t = np.arange(n); u = t + k; v = (u >= 0) & (u < n); t, u = t[v], u[v]; v = (seq[u] == seq[t]) & ok[u] & ok[t]; t, u = t[v], u[v]
        mis.append(mi(lab[t], bb[u]))
    R["label_lag"] = dict(best_lag_frames=int(lags[int(np.argmax(mis))]), mi_by_lag=[round(x, 4) for x in mis], mi_lag0=mis[j0])
    # 10 test frames with the fitted rotation (lag 0) on the stored BEV
    te = np.where((split == 2) & ok)[0]; pick = np.sort(np.random.default_rng(34).choice(te, 10, replace=False))
    th = np.deg2rad(th0); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase3c", f"ds{s}.h5"), "r") as h: B = h["bev"][pick][:, 0].astype(np.float32)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = []
    for k, i in enumerate(pick):
        p = Rm @ rel[i]; ix = (p[0] - g["x0"]) / g["cell"]; iy = (p[1] - g["y0"]) / g["cell"]
        inside = bool(0 <= ix < g["NX"] and 0 <= iy < g["NY"] and np.hypot(*p) <= g["R_clip"])
        Q = ng_points((s, lid[i], float(g["z_ground"]))); near = int((np.hypot(*(Q - p).T) <= 2).sum())
        rr = np.hypot(*p); ang = np.random.default_rng(int(i)).uniform(-np.pi, np.pi, 50)
        bgd = float(np.median([(np.hypot(*(Q - rr * np.array([np.cos(a), np.sin(a)])).T) <= 2).sum() for a in ang]))
        rows.append(dict(row=int(i), seq=int(seq[i]), receiver_lidar_xy=p.round(2).tolist(), inside_grid=inside, points_within_2m=near,
                         median_points_within_2m_same_range=bgd, label=int(lab[i])))
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(B[k].T, origin="lower", cmap="Greys", extent=[g["x0"], g["x0"] + g["NX"] * g["cell"], g["y0"], g["y0"] + g["NY"] * g["cell"]])
        ax.plot(p[0], p[1], "o", mfc="none", mec="#eb6834", ms=12, mew=2); ax.plot(0, 0, "+", color="#2a78d6", ms=10)
        ax.set_title(f"s{s} row {i} seq {seq[i]} beam {lab[i]}", fontsize=8); fig.savefig(os.path.join(LOC, f"ds{s}_row{i}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    R["test_frames"] = rows; R["inside_grid_count"] = int(sum(r["inside_grid"] for r in rows))
    lag_ok = abs(R["geometry_fit"]["best_lag_frames"]) <= 1 and abs(R["label_lag"]["best_lag_frames"]) <= 1
    R["verdict"] = "PASS" if (R["inside_grid_count"] >= 9 and lag_ok and R["per_sequence_theta"]["circular_sd_deg"] < 10) else "FAIL"
    json.dump(R, open(os.path.join(OUT, f"a3_ds{s}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "test_frames"}, indent=1)); [print(r) for r in rows]


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
