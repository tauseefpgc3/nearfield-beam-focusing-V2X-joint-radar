"""D-033 C1: varied vehicles at 20 seeds. Reuses the D-032 evaluation helpers (scripts/phase3b/eval_b.py,
which uses the audited scripts/verify3 code). Writes results/phase3c/c1.json and preds_c1.npz."""
import json, os, sys, time
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3b"))
import eval_b as B
OUT = os.path.join(ROOT, "results", "phase3c"); os.makedirs(OUT, exist_ok=True)
te, va, xy, ant, psi, FR, MS, CH = B.te, B.va, B.xy, B.ant, B.psi, B.FR, B.MS, B.CH
SEEDS20 = B.SEEDS20
log = B.log


def boot_ci(d, B_=10000):
    rng = np.random.default_rng(0); n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B_)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi))


def verdict(c):
    return "network better" if c["hi"] < 0 else ("network worse" if c["lo"] > 0 else "no detectable difference")


def pos_err(pt, m):
    """pt (runs, n, 2) on test users; m mask. Median/p90 of Euclidean, range, sideways (pooled over runs)."""
    est = pt[:, m].reshape(-1, 2); tru = np.tile(xy[te][m], (len(pt), 1))
    out = B.err_stats(est, tru); e = np.hypot(*(est - tru).T)
    out["euclid"] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90))); return out


def main():
    t0 = time.time(); R = dict(seeds=list(SEEDS20))
    pc = np.load(os.path.join(ROOT, "results", "phase3b", "preds_clean.npz"))
    geo_va, geo_te = pc["lidar_geometric_val"], pc["lidar_geometric_test"]
    sv_va, sv_te = pc["lidar_net_val"], pc["lidar_net_test"]
    Xva_var, Xte_var = B.inputs("lidar", va, lidar=B.LBV), B.inputs("lidar", te, lidar=B.LBV)
    vn_va = B.net_preds("lidar", "_var", SEEDS20, Xva_var); log("varied val")
    vn_te_var = B.net_preds("lidar", "_var", SEEDS20, Xte_var); log("varied test (varied)")
    vn_te_std = B.net_preds("lidar", "_var", SEEDS20, B.inputs("lidar", te)); log("varied net on standard test")
    sv_te_var = B.net_preds("lidar", "", SEEDS20, Xte_var); log("single-vehicle net on varied test")
    var = B.LR.varied_vehicles(len(ant)); _, Cv = B.LR.render_bev(te, ant, psi, var); g_var, fb = B.geo(Cv); log("geometric on varied")
    R["geometric_fallback_rate_varied"] = fb
    np.savez(os.path.join(OUT, "preds_c1.npz"), varied_net_val=vn_va, varied_net_test_varied=vn_te_var, varied_net_test_standard=vn_te_std,
             single_net_test_varied=sv_te_var, geometric_test_varied=g_var)
    # (test predictions, validation predictions used for the shape)
    SETS = dict(
        varied=dict(lidar_geometric=(g_var[None], geo_va), single_vehicle_net=(sv_te_var, sv_va), varied_trained_net=(vn_te_var, vn_va)),
        standard=dict(lidar_geometric=(geo_te, geo_va), single_vehicle_net=(sv_te, sv_va), varied_trained_net=(vn_te_std, vn_va)))
    d2 = np.hypot(*xy[te].T)
    def subsets(f): return dict(all=np.ones(len(te), bool), adv_ge3=CH[f]["adv"][te] >= 3, d_ge_2m=d2 >= 2)
    R["subset_sizes"] = {f: {k: int(v.sum()) for k, v in subsets(f).items()} for f in FR}
    T, L_all, POS = {}, {}, {}
    for ts, meths in SETS.items():
        for name, (pt, pv) in meths.items():
            POS[f"{ts}|{name}"] = {k: pos_err(pt, m) for k, m in subsets(60).items()}
            for f in FR:
                for M in MS:
                    L = np.array([B.test_loss(f, pt[r], B.choose_shape(f, pv[r], M)[0]) for r in range(len(pt))])
                    L_all[(ts, name, f, M)] = L
                    for k, m in subsets(f).items():
                        c = B.ci_t(L[:, m].mean(1)) if len(L) > 1 else boot_ci(L[0, m])
                        c["within_1dB"] = float((L[:, m] <= 1).mean()); T[f"{ts}|{name}|{f}|{M}|{k}"] = c
        log("tables", ts)
    R["tables"] = T; R["position_error"] = POS
    # headline: varied-trained vs geometric, varied test, all users, M in {8,16}
    H, pv_ = {}, []
    for f in FR:
        for M in (8, 16):
            Ln = L_all[("varied", "varied_trained_net", f, M)]; Lg = L_all[("varied", "lidar_geometric", f, M)][0]
            ds = Ln.mean(1) - Lg.mean(); tt = stats.ttest_1samp(ds, 0.0); w = stats.wilcoxon(ds)
            b = boot_ci(Ln.mean(0) - Lg); b["verdict"] = verdict(b)
            H[f"{f}|{M}"] = dict(bootstrap=b, seed_delta=B.ci_t(ds), p_t=float(tt.pvalue), p_wilcoxon=float(w.pvalue)); pv_.append(float(tt.pvalue))
    order = np.argsort(pv_); adj = np.empty(4); run = 0.0
    for j, i in enumerate(order): run = max(run, min(1.0, (4 - j) * pv_[i])); adj[i] = run
    for (k, v), a in zip(H.items(), adj): v["p_t_holm"] = float(a)
    better = all(v["bootstrap"]["hi"] < 0 and v["p_t_holm"] < 0.05 and v["seed_delta"]["mean"] < 0 for v in H.values())
    worse = all(v["bootstrap"]["lo"] > 0 and v["p_t_holm"] < 0.05 and v["seed_delta"]["mean"] > 0 for v in H.values())
    R["headline"] = dict(cells=H, verdict="varied-trained net better" if better else ("varied-trained net worse" if worse else "no detectable difference"))
    log("headline", R["headline"]["verdict"])
    # secondary, every cell
    S = {}
    for ts in SETS:
        for f in FR:
            for M in MS:
                Lv = L_all[(ts, "varied_trained_net", f, M)]; Ls = L_all[(ts, "single_vehicle_net", f, M)]; Lg = L_all[(ts, "lidar_geometric", f, M)][0]
                for k, m in subsets(f).items():
                    c = {}
                    b = boot_ci(Lv.mean(0)[m] - Ls.mean(0)[m]); b["verdict"] = verdict(b).replace("network", "varied-trained")
                    b["welch_p"] = float(stats.ttest_ind(Lv[:, m].mean(1), Ls[:, m].mean(1), equal_var=False).pvalue); c["varied_vs_single"] = b
                    for nm, Lx in (("varied_vs_geometric", Lv), ("single_vs_geometric", Ls)):
                        b = boot_ci(Lx.mean(0)[m] - Lg[m]); b["verdict"] = verdict(b); c[nm] = b
                    S[f"{ts}|{f}|{M}|{k}"] = c
    R["secondary"] = S; R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(OUT, "c1.json"), "w"), indent=1); log("done")


if __name__ == "__main__":
    main()
