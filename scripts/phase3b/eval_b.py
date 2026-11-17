"""D-032 Part B evaluation (B1-B4, items 3a-3c). Uses the audited independent evaluation code in
scripts/verify3/common.py (network class, decoder, sweep, loss); the simulator and the frozen
geometric estimator are called as black boxes. Writes results/phase3b/results_b.json."""
import json, os, sys, time
import h5py, numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "verify3")); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2")); sys.path.insert(0, HERE)
from common import (split, users, channels, load_net, predict, beam_set, loss_db, shape_options, P3, CK)
import lidar_estimator as GEO, lidar_v2 as LV, lidar_var as LR
import p3_eval as E
P3B = os.path.join(ROOT, "data", "processed", "phase3b"); OUTD = os.path.join(ROOT, "results", "phase3b"); os.makedirs(OUTD, exist_ok=True)
SEEDS20 = tuple(range(11, 56, 11)) + tuple(range(66, 221, 11)); SEEDS5 = SEEDS20[:5]
MS = (8, 16, 32, 64); FR = (60, 140)
S = split(); tr, va, te = S["train"], S["val"], S["test"]; ant, psi = users(); xy = ant[:, :2]
CH = {f: channels(f) for f in FR}
dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json")))
PRM = dict(GEO.DEFAULT, **dl["best"]["params"]); FB = np.array(dl["fallback_xy"])
LB = os.path.join(P3, "lidar_bev.h5"); LBV = os.path.join(P3B, "lidar_bev_var.h5")
RB = {snr: os.path.join(P3B, f"radar_bev{snr}.h5") for snr in (20, 10, 0)}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def h5(path, idx):
    with h5py.File(path, "r") as h: return h["bev"][idx]


def inputs(sensor, idx, radar=20, lidar=None):
    """lidar: None -> stored standard BEV; a path -> that file; an array -> given BEVs (len idx)."""
    parts = []
    if sensor in ("radar", "fused"): parts.append(h5(RB[radar], idx))
    if sensor in ("lidar", "fused"):
        parts.append(h5(LB, idx) if lidar is None else (h5(lidar, idx) if isinstance(lidar, str) else lidar))
    return np.concatenate(parts, 1)


def net_preds(sensor, suffix, seeds, X):
    out = []
    for sd in seeds:
        m, ck = load_net(sensor, sd) if not suffix else _load(sensor, sd, suffix)
        out.append(predict(m, X))
    return np.array(out)


def _load(sensor, sd, suffix):
    import torch
    from common import Net, DEV
    ck = torch.load(os.path.join(CK, f"{sensor}_H_lr0.001_s{sd}{suffix}.pt"), map_location="cpu", weights_only=False)
    m = Net(ck["cin"]); m.load_state_dict(ck["model"], strict=True); return m.to(DEV).eval(), ck


def geo(clouds):
    out = [GEO.estimate(c, PRM, FB) for c in clouds]
    return np.array([o[0] for o in out]), float(np.mean([o[1]["fallback"] for o in out]))


# ---------------------------------------------------------------- radar picker (fixed radar)
def picker_raw(snr, idx):
    with h5py.File(os.path.join(P3B, f"radar_snr{snr}.h5"), "r") as h:
        X = h["radar_features"][idx]; lo = h["ra_db_min"][idx]; hi = h["ra_db_max"][idx]
    return E.estimate(E.ra_db(X, lo, hi), "centroid")


def picker_bias():
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase2", "split.h5"), "r") as h: si = h["snr_idx"][:]
    with h5py.File(os.path.join(P3B, "radar_geometry.h5"), "r") as h: Rt = h["R_true"][:]; St = h["sin_true"][:]
    Rh = np.empty(len(tr)); Sh = np.empty(len(tr))
    for s, snr in enumerate((20, 10, 0)):
        pos = np.where(si[tr] == s)[0]
        for b in range(0, len(pos), 400):
            p = pos[b:b + 400]; Rh[p], Sh[p] = picker_raw(snr, tr[p])
    return float(np.median(Rt[tr] - Rh)), float(np.median(St[tr] - Sh))


def picker_xy(snr, idx, bias):
    R, Sn = picker_raw(snr, idx); R = R + bias[0]; Sn = np.clip(Sn + bias[1], -1, 1)
    py = R * Sn; px = np.sqrt(np.maximum(R ** 2 - py ** 2 - (1.5 - E.RADAR[2]) ** 2, 0))
    return np.c_[px, py + E.RADAR[1]]


# ---------------------------------------------------------------- sweep helpers
def choose_shape(f, est_va, M):
    G = CH[f]["G"][va]
    sc = {sh: loss_db(G, beam_set(est_va, CH[f]["u"], CH[f]["rings"], *sh)).mean() for sh in shape_options(M)}
    k = min(sc, key=sc.get); return k, float(sc[k])


def test_loss(f, est_te, shape, idx=None):
    idx = te if idx is None else idx
    return loss_db(CH[f]["G"][idx], beam_set(est_te, CH[f]["u"], CH[f]["rings"], *shape))


def ff_loss(f, est_te, M, idx=None):
    """B3: the M nearest angles, far-field ring (ring 0) only."""
    idx = te if idx is None else idx
    rel = np.c_[est_te, np.full(len(est_te), -0.5)]; u = rel[:, 1] / np.linalg.norm(rel, axis=1)
    ang = np.argsort(np.abs(u[:, None] - CH[f]["u"][None]), axis=1, kind="mergesort")[:, :M]
    return loss_db(CH[f]["G"][idx], ang * 8)


def evaluate(va_runs, te_runs, shape_from=None):
    """va_runs/te_runs (n_runs, n, 2). Returns dict[(f, M)] -> dict(L (runs, n_test), shape list, val score)."""
    out = {}
    for f in FR:
        for M in MS:
            Ls, shp, vs = [], [], []
            for r in range(len(te_runs)):
                sh, v = choose_shape(f, (shape_from if shape_from is not None else va_runs)[r], M)
                shp.append(sh); vs.append(v); Ls.append(test_loss(f, te_runs[r], sh))
            out[(f, M)] = dict(L=np.array(Ls), shapes=[f"{a}x{b}" for a, b in shp], val=float(np.mean(vs)))
    return out


def ci_t(x):
    x = np.asarray(x, float)
    if len(x) < 2: return dict(mean=float(x.mean()))
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h))


def boot(d, B=10000):
    rng = np.random.default_rng(0); n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), verdict="network better" if hi < 0 else ("network worse" if lo > 0 else "no detectable difference"))


def err_stats(est, true):
    e = est - true; rad = true / np.linalg.norm(true, axis=1, keepdims=True); perp = np.c_[-rad[:, 1], rad[:, 0]]
    return {nm: dict(median=float(np.median(np.abs(v))), p90=float(np.percentile(np.abs(v), 90)))
            for nm, v in (("range", (e * rad).sum(1)), ("sideways", (e * perp).sum(1)))}


def main():
    t0 = time.time(); R = dict(seeds=list(SEEDS20))
    # ---------- predictions, clean
    P = {}
    bias = picker_bias(); R["picker_bias_fixed_radar"] = bias; log("bias", bias)
    P["radar_picker"] = (picker_xy(20, va, bias)[None], picker_xy(20, te, bias)[None])
    vz = np.load(os.path.join(ROOT, "results", "verify3", "preds.npz"))
    P["lidar_geometric"] = (vz["geo_val"][None], vz["geo_test"][None])
    for s, suf in (("radar", "_r2"), ("lidar", ""), ("fused", "_r2")):
        P[f"{s}_net"] = (net_preds(s, suf, SEEDS20, inputs(s, va)), net_preds(s, suf, SEEDS20, inputs(s, te))); log(s, "preds")
    np.savez(os.path.join(OUTD, "preds_clean.npz"), **{f"{k}_{w}": v[i] for k, v in P.items() for i, w in enumerate(("val", "test"))})
    EV = {k: evaluate(*v) for k, v in P.items()}; log("sweeps")
    # ---------- subsets
    d2 = np.hypot(*xy[te].T); r3 = np.linalg.norm(ant[te] - np.array([0, 0, 2.0]), axis=1)
    dF = {f: 127 ** 2 * (299792458.0 / (f * 1e9)) / 2 for f in FR}; R["d_F_m"] = dF
    def subsets(f):
        base = dict(all=np.ones(len(te), bool), near_field=r3 < dF[f], adv_ge3=CH[f]["adv"][te] >= 3)
        out = dict(base); out.update({k + "|d_ge_2m": v & (d2 >= 2) for k, v in base.items()}); return out
    R["subset_sizes"] = {f: {k: int(v.sum()) for k, v in subsets(f).items()} for f in FR}
    # ---------- B2 tables + B3 far-field only
    T, FF = {}, {}
    for name, (pv, pt) in P.items():
        for f in FR:
            sub = subsets(f)
            for M in MS:
                e = EV[name][(f, M)]; L = e["L"]
                Lff = np.array([ff_loss(f, pt[r], M) for r in range(len(pt))])
                for k, m in sub.items():
                    cell = ci_t(L[:, m].mean(1)); cell["within_1dB"] = float((L[:, m] <= 1).mean())
                    T[f"{name}|{f}|{M}|{k}"] = cell
                    FF[f"{name}|{f}|{M}|{k}"] = dict(ff_only=ci_t(Lff[:, m].mean(1)), delta_focus=ci_t((Lff[:, m] - L[:, m]).mean(1)))
                T[f"{name}|{f}|{M}|shapes"] = sorted(set(e["shapes"]))
    for f in FR:
        sub = subsets(f); G = CH[f]["G"][te]
        sw = np.concatenate([np.arange(256)[None].repeat(len(te), 0) * 8, G[:, 0::8].argmax(1)[:, None] * 8 + np.arange(1, 8)[None]], 1)
        L263 = loss_db(G, sw)
        for k, m in sub.items():
            T[f"sweep263|{f}|{k}"] = float(L263[m].mean()); FF[f"genie_ff_only|{f}|{k}"] = float(CH[f]["adv"][te][m].mean())
    R["tables"] = T; R["far_field_only"] = FF
    R["position_error"] = {name: {k: err_stats(pt[:, m].reshape(-1, 2), np.tile(xy[te][m], (len(pt), 1)))
                                  for k, m in (("all", np.ones(len(te), bool)), ("d_ge_2m", d2 >= 2))} for name, (pv, pt) in P.items()}
    # ---------- B1 primary endpoint
    prim, pvals = {}, []
    for f in FR:
        for M in (8, 16):
            Ln = EV["lidar_net"][(f, M)]["L"]; Lg = EV["lidar_geometric"][(f, M)]["L"][0]
            ds = Ln.mean(1) - Lg.mean(); tt = stats.ttest_1samp(ds, 0.0); w = stats.wilcoxon(ds)
            prim[f"{f}|{M}"] = dict(bootstrap=boot(Ln.mean(0) - Lg), seed_delta=ci_t(ds), t=float(tt.statistic), p_t=float(tt.pvalue), p_wilcoxon=float(w.pvalue))
            pvals.append(float(tt.pvalue))
    order = np.argsort(pvals); adj = np.empty(4); run = 0.0
    for j, i in enumerate(order): run = max(run, min(1.0, (4 - j) * pvals[i])); adj[i] = run
    for (k, v), a in zip(prim.items(), adj): v["p_t_holm"] = float(a)
    better = all(v["bootstrap"]["hi"] < 0 and v["p_t_holm"] < 0.05 and v["seed_delta"]["mean"] < 0 for v in prim.values())
    worse = all(v["bootstrap"]["lo"] > 0 and v["p_t_holm"] < 0.05 and v["seed_delta"]["mean"] > 0 for v in prim.values())
    R["primary"] = dict(cells=prim, verdict="LiDAR net better" if better else ("LiDAR net worse" if worse else "no detectable difference"))
    log("primary", R["primary"]["verdict"])
    # ---------- bootstraps (secondary)
    BS = {}
    for f in FR:
        sub = subsets(f)
        for M in MS:
            single = ["radar_picker", "lidar_geometric", "radar_net", "lidar_net"]
            best = min(single, key=lambda n: EV[n][(f, M)]["val"])
            mu = {n: EV[n][(f, M)]["L"].mean(0) for n in P}
            for k, m in sub.items():
                BS[f"{f}|{M}|{k}"] = dict(radar_net_vs_radar_picker=boot(mu["radar_net"][m] - mu["radar_picker"][m]),
                                          lidar_net_vs_lidar_geometric=boot(mu["lidar_net"][m] - mu["lidar_geometric"][m]),
                                          fused_net_vs_best_single=dict(boot(mu["fused_net"][m] - mu[best][m]), best_single=best))
    R["bootstrap"] = BS
    json.dump(R, open(os.path.join(OUTD, "results_b.json"), "w"), indent=1); log("B1-B3 written")
    # ---------- B4 stress (shape from clean validation of the same model/seed)
    def stress_cell(name, pt, pv_clean):
        out = {}
        for f in FR:
            for M in MS:
                Ls = [test_loss(f, pt[r], choose_shape(f, pv_clean[r], M)[0]) for r in range(len(pt))]
                out[f"{f}|{M}"] = ci_t([L.mean() for L in Ls])
        out["err_median_m"] = float(np.median(np.hypot(*(pt.reshape(-1, 2) - np.tile(xy[te], (len(pt), 1))).T)))
        return out
    ST = {}
    std = LR.standard_vehicles(len(ant))
    for nm, kw in (("lidar_32ch", dict(ch_step=2)), ("lidar_16ch", dict(ch_step=4)), ("lidar_drop30", dict(p_drop=0.30)), ("lidar_noise5cm", dict(sig_r=0.05))):
        Bt, Ct = LR.render_bev(te, ant, psi, std, **kw)
        g, fbr = geo(Ct)
        ST[nm] = dict(lidar_geometric=dict(stress_cell("g", g[None], P["lidar_geometric"][0]), fallback=fbr),
                      lidar_net=stress_cell("l", net_preds("lidar", "", SEEDS20, Bt), P["lidar_net"][0]),
                      fused_net=stress_cell("f", net_preds("fused", "_r2", SEEDS20, inputs("fused", te, lidar=Bt)), P["fused_net"][0]))
        log("stress", nm)
    for snr in (10, 0):
        ST[f"radar_{snr}dB"] = dict(radar_picker=stress_cell("p", picker_xy(snr, te, bias)[None], P["radar_picker"][0]),
                                    radar_net=stress_cell("r", net_preds("radar", "_r2", SEEDS20, inputs("radar", te, radar=snr)), P["radar_net"][0]),
                                    fused_net=stress_cell("f", net_preds("fused", "_r2", SEEDS20, inputs("fused", te, radar=snr)), P["fused_net"][0]))
        log("stress radar", snr)
    Xf = inputs("fused", te)
    for nm, sl in (("fused_radar_removed", slice(0, 1)), ("fused_lidar_removed", slice(1, 4))):
        X = Xf.copy(); X[:, sl] = 0; ST[nm] = dict(fused_net=stress_cell("f", net_preds("fused", "_r2", SEEDS20, X), P["fused_net"][0]))
    R["stress"] = ST; json.dump(R, open(os.path.join(OUTD, "results_b.json"), "w"), indent=1); log("B4 written")
    # ---------- 3a varied vehicles
    VV = {}
    var = LR.varied_vehicles(len(ant))
    _, Cv = LR.render_bev(te, ant, psi, var); g, fbr = geo(Cv)
    VV["lidar_geometric_frozen"] = dict(stress_cell("g", g[None], P["lidar_geometric"][0]), fallback=fbr)
    VV["lidar_net_frozen_single_vehicle"] = stress_cell("l", net_preds("lidar", "", SEEDS20, inputs("lidar", te, lidar=LBV)), P["lidar_net"][0])
    pv5 = net_preds("lidar", "_var", SEEDS5, inputs("lidar", va, lidar=LBV)); pt5 = net_preds("lidar", "_var", SEEDS5, inputs("lidar", te, lidar=LBV))
    VV["lidar_net_varied_trained"] = stress_cell("v", pt5, pv5)
    VV["lidar_net_varied_trained_on_standard_test"] = stress_cell("v", net_preds("lidar", "_var", SEEDS5, inputs("lidar", te)), pv5)
    VV["reference_single_vehicle_lidar_net_on_standard_test"] = {k: T[f"lidar_net|{k}|all"] for k in [f"{f}|{M}" for f in FR for M in MS]}
    VV["reference_single_vehicle_geometric_on_standard_test"] = {k: T[f"lidar_geometric|{k}|all"] for k in [f"{f}|{M}" for f in FR for M in MS]}
    R["varied_vehicles"] = VV; json.dump(R, open(os.path.join(OUTD, "results_b.json"), "w"), indent=1); log("3a written")
    # ---------- 3b antenna-offset error
    OFF = {}
    for dlt in (-0.5, -0.2, -0.1, 0.1, 0.2, 0.5):
        veh = dict(std); veh["ant_xi"] = np.full(len(ant), LV.ANT_OFF[0] - dlt)   # true antenna 1.3+dlt behind centre
        Bo, Co = LR.render_bev(te, ant, psi, veh); g, fbr = geo(Co)
        OFF[f"{dlt:+.1f}"] = dict(lidar_geometric=stress_cell("g", g[None], P["lidar_geometric"][0]),
                                  lidar_net=stress_cell("l", net_preds("lidar", "", SEEDS20, Bo), P["lidar_net"][0]))
        log("offset", dlt)
    R["antenna_offset_error"] = OFF
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(OUTD, "results_b.json"), "w"), indent=1); log("done")


if __name__ == "__main__":
    main()
