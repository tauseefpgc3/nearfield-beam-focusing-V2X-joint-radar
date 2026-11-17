"""A1 leakage tests and A2 precision tests (evaluation only, no retraining).
A1a  radar scene: re-draw the radar's 6 vehicle scatterers (documented draw order in
     scripts/phase2/p1_radar.py, rng 20261117+idx) and measure where they sit relative to the
     antenna and to the LiDAR vehicle body; measure the radar network's bias along heading.
A1b  LiDAR counterfactual: re-render test users with the car moved out of the scene (same
     ground/background/noise seeds); LiDAR net and geometric estimator must lose the user.
A1c  fused net with the radar or the LiDAR channels zeroed.
A1d  stored BEV == BEV of a fresh render; outputs invariant to point order.
A2   shift test: translate the whole vehicle (and antenna) by sub-cell offsets and regress
     the LiDAR net's output shift on the true shift; error vs distance; returns on vehicle."""
import json, os, sys
import h5py, numpy as np
from common import *
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
import lidar_v2 as SIM, lidar_estimator as GEO

S = split(); va, te, tr = S["val"], S["test"], S["train"]; ant, psi = users(); xy = ant[:, :2]
P = dict(np.load(os.path.join(OUT, "preds.npz"))); CH = {f: channels(f) for f in (60, 140)}
dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json")))
prm = dict(GEO.DEFAULT, **dl["best"]["params"]); fb = np.array(dl["fallback_xy"])
fwd = np.c_[np.cos(psi), np.sin(psi)]; lat = np.c_[-np.sin(psi), np.cos(psi)]
body = SIM.vehicle_centre(ant, psi)                  # LiDAR body centre (1.3 m ahead of antenna)
lid = [load_net("lidar", s)[0] for s in SEEDS]
res = {}


def render_bev(ix, A):
    out, cl = [], []
    for b in range(0, len(ix), 25):
        pcs, _ = SIM.render(ix[b:b + 25], A, psi); cl += pcs; out += [SIM.bev(p) for p in pcs]
    return np.stack(out), cl


def mean_pred(models, X):
    return np.mean([predict(m, X) for m in models], 0)


# ---- A1a radar scatterers
off = []
for i in te:
    r = np.random.default_rng(20261117 + int(i)); sc = []
    for _ in range(6):
        xi = r.uniform(-2.25, 2.25); eta = r.uniform(-1, 1); r.uniform(0.3, 1.5); a = r.uniform(0.3, 1.0); r.uniform(0, 2 * np.pi)
        sc.append((xi, eta, a))
    sc = np.array(sc); w = sc[:, 2] ** 2 / (sc[:, 2] ** 2).sum()
    off.append([(w * sc[:, 0]).sum(), (w * sc[:, 1]).sum()])
off = np.array(off)
rn = np.mean([P[f"radar_s{s}_test"] for s in SEEDS], 0); e = rn - xy[te]
res["A1a_radar"] = dict(
    scatterer_power_centroid_rel_antenna_vehicle_frame_m=dict(mean_xi=float(off[:, 0].mean()), mean_eta=float(off[:, 1].mean()), sd_xi=float(off[:, 0].std())),
    lidar_body_centre_rel_antenna_xi_m=1.3,
    radar_net=dict(mean_err_along_heading_m=float((e * fwd[te]).sum(1).mean()), mean_err_lateral_m=float((e * lat[te]).sum(1).mean()),
                   median_dist_to_antenna_m=float(np.median(np.hypot(*e.T))),
                   median_dist_to_lidar_body_centre_m=float(np.median(np.hypot(*(rn - body[te]).T)))),
    reading="radar vehicle scatterers are drawn uniformly over a 4.5 x 2 m box centred on the true antenna; "
            "a radar consistent with the LiDAR scene would see the body centred 1.3 m ahead of the antenna")
print("A1a", json.dumps(res["A1a_radar"]), flush=True)

# ---- A1b car removed (moved 1 km away, outside the ROI); same per-user seeds
far = ant.copy(); far[:, 0] += 1000.0
sub = te[::3]
Bn, Cn = render_bev(sub, far)
pn = mean_pred(lid, Bn)
gn = [GEO.estimate(c, prm, fb) for c in Cn]
gpos = np.array([g[0] for g in gn]); gfb = float(np.mean([g[1]["fallback"] for g in gn]))
cl = {}
for f in (60, 140):
    for M in (8, 64):
        Ln = sweep_with_val_shape(CH[f], va, sub, P["lidar_s11_val"], pn, M)[0]
        Lp = sweep_with_val_shape(CH[f], va, sub, P["prior_val"], np.repeat(P["prior_test"][:1], len(sub), 0), M)[0]
        cl[f"{f}GHz_M{M}"] = dict(lidar_net_no_car=float(Ln.mean()), prior=float(Lp.mean()))
res["A1b_car_removed"] = dict(
    n_users=len(sub), lidar_net_err_median_m=float(np.median(np.hypot(*(pn - xy[sub]).T))),
    lidar_net_corr_pred_true_x=float(np.corrcoef(pn[:, 0], xy[sub, 0])[0, 1]), lidar_net_corr_pred_true_y=float(np.corrcoef(pn[:, 1], xy[sub, 1])[0, 1]),
    lidar_net_pred_sd_m=[float(pn[:, 0].std()), float(pn[:, 1].std())],
    geometric_fallback_rate=gfb, geometric_err_median_m=float(np.median(np.hypot(*(gpos - xy[sub]).T))), loss=cl)
print("A1b", json.dumps(res["A1b_car_removed"]), flush=True)

# ---- A1c fused with one sensor zeroed (channel 0 = radar, 1..3 = LiDAR)
Xf = bev_inputs("fused", te); fz = {}
fused = [load_net("fused", s)[0] for s in SEEDS]
for nm, sl in (("none_zeroed", slice(0, 0)), ("radar_zeroed", slice(0, 1)), ("lidar_zeroed", slice(1, 4))):
    X = Xf.copy(); X[:, sl] = 0
    pz = [predict(m, X) for m in fused]
    row = dict(err_median_m=float(np.median(np.hypot(*(np.concatenate(pz) - np.tile(xy[te], (5, 1))).T))))
    for f in (60, 140):
        for M in (8, 64):
            row[f"{f}GHz_M{M}"] = float(np.mean([sweep_with_val_shape(CH[f], va, te, P[f"fused_s{s}_val"], p, M)[0].mean() for s, p in zip(SEEDS, pz)]))
    fz[nm] = row
res["A1c_fused_sensor_zeroed"] = fz
print("A1c", json.dumps(fz), flush=True)

# ---- A1d stored BEV vs fresh render; point-order invariance
chk = te[:60]; Bf, Cf = render_bev(chk, ant)
with h5py.File(os.path.join(P3, "lidar_bev.h5"), "r") as h: Bst = h["bev"][chk]
rng = np.random.default_rng(5)
perm_bev = max(float(np.abs(SIM.bev(c[rng.permutation(len(c))]).astype(np.float32) - SIM.bev(c).astype(np.float32)).max()) for c in Cf)
perm_geo = max(float(np.abs(GEO.estimate(c[rng.permutation(len(c))], prm, fb)[0] - GEO.estimate(c, prm, fb)[0]).max()) for c in Cf)
res["A1d"] = dict(stored_vs_fresh_bev_max_abs_diff=float(np.abs(Bf.astype(np.float32) - Bst.astype(np.float32)).max()),
                  point_order_bev_max_abs_diff=perm_bev, point_order_geometric_max_abs_diff_m=perm_geo,
                  bev_grid_world_fixed=dict(x0=SIM.BX0, y0=SIM.BY0, cell=SIM.CELL, nx=SIM.NX, ny=SIM.NY))
print("A1d", json.dumps(res["A1d"]), flush=True)

# ---- A2 shift test
sub2 = te[::6]
shifts = [(-0.1, 0), (-0.05, 0), (0.05, 0), (0.1, 0), (0, -0.1), (0, -0.05), (0, 0.05), (0, 0.1)]
B0, _ = render_bev(sub2, ant); p0 = mean_pred(lid, B0)
td, od = [], []
for dx, dy in shifts:
    A = ant.copy(); A[:, 0] += dx; A[:, 1] += dy
    Bs_, _ = render_bev(sub2, A); td.append(np.tile([dx, dy], (len(sub2), 1))); od.append(mean_pred(lid, Bs_) - p0)
td = np.concatenate(td); od = np.concatenate(od); sx = td[:, 0] != 0; sy = td[:, 1] != 0
slope = lambda t, o: float((t * o).sum() / (t * t).sum())
d = np.hypot(*xy[te].T)
el = np.hypot(*(np.mean([P[f"lidar_s{s}_test"] for s in SEEDS], 0) - xy[te]).T)
el1 = np.hypot(*(P["lidar_s11_test"] - xy[te]).T); eg = np.hypot(*(P["geo_test"] - xy[te]).T)
with h5py.File(os.path.join(P3, "lidar_bev.h5"), "r") as h: nv = h["n_vehicle_returns"][:][te]
bins = [1, 5, 10, 15, 20.01]
pr = P["lidar_s11_test"]; fx = (pr[:, 0] - GX0) / GC; fy = (pr[:, 1] - GY0) / GC
res["A2"] = dict(
    shift_slope_x=slope(td[sx, 0], od[sx, 0]), shift_slope_y=slope(td[sy, 1], od[sy, 1]),
    shift_residual_rms_m=float(np.sqrt(np.mean((od - td) ** 2))), shift_n_users=len(sub2),
    decoding="arg-max cell then softmax-weighted mean of cell centres in a 5x5 window (sub-cell)",
    pred_offset_from_cell_centre_mean_abs_cells=dict(x=float(np.abs(fx - np.floor(fx) - 0.5).mean()), y=float(np.abs(fy - np.floor(fy) - 0.5).mean()), uniform=0.25),
    err_vs_distance=[dict(d=f"{bins[k]}-{bins[k+1]:.0f} m", n=int(m.sum()), lidar_net_seed11_med_m=float(np.median(el1[m])),
                          lidar_net_5seed_mean_pred_med_m=float(np.median(el[m])), geometric_med_m=float(np.median(eg[m])),
                          vehicle_returns_med=float(np.median(nv[m])), azimuth_spacing_m=float(np.deg2rad(0.2) * (bins[k] + bins[k + 1]) / 2))
                     for k in range(4) for m in [(d >= bins[k]) & (d < bins[k + 1])]],
    vehicle_returns_test=dict(median=float(np.median(nv)), p5=float(np.percentile(nv, 5)), min=int(nv.min())))
print("A2", json.dumps(res["A2"]), flush=True)
json.dump(res, open(os.path.join(OUT, "a1_a2.json"), "w"), indent=1)
