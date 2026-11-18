"""D-034 Part B evaluation: B1 (1-NN), B2 (radar ensembles), B3 (sweep shapes + fixed shape),
B5 (fresh test set, evaluated once) and C3 (varied test set, adv >= 3 dB). B6 classifiers are
included when their checkpoints exist. Reuses the Phase 3b evaluation helpers (eval_b) for the
sweep, shape choice and model loading (audited in verify3 and reproduced in verify4/A5).

    python scripts/phase3d/eval_d.py      # -> results/phase3d/results_d.json, preds_d.npz
"""
import glob, json, os, sys, time
import h5py, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("phase3b", "phase3", "phase2", "verify3"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
sys.path.insert(0, HERE)
import eval_b as B
from common import logits as LOGITS, decode_local, DEV
import p3_eval as E
OUTD = os.path.join(ROOT, "results", "phase3d"); FR3 = os.path.join(ROOT, "data", "processed", "phase3d", "fresh")
FR, MS, SEEDS20 = B.FR, B.MS, B.SEEDS20
TRIPLES = [SEEDS20[3 * i:3 * i + 3] for i in range(6)]
log = B.log
tr, va, te = B.tr, B.va, B.te; ant = B.ant; xy = B.xy


def ci_seed(Ls, m):
    x = np.array([L[m].mean() for L in Ls]); return B.ci_t(x) | dict(n_runs=len(x))


def ci_user(L, m, B_=10000):
    rng = np.random.default_rng(0); d = L[m]; n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B_)])
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_runs=1)


def pooled_feat(X):
    return F.normalize(F.avg_pool2d(torch.from_numpy(np.ascontiguousarray(X[:, :1])).float(), 4).flatten(1), dim=1)


def one_nn(Ftr, Fq):
    out = []
    for b in range(0, len(Fq), 512): out.append(torch.cdist(Fq[b:b + 512], Ftr).argmin(1).numpy())
    return xy[tr][np.concatenate(out)]


def ens_xy(triple, X):
    acc = None
    for sd in triple:
        m, _ = B._load("radar", sd, "_r2"); lg = LOGITS(m, X); lg = lg.reshape(len(lg), -1)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); acc = p if acc is None else acc + p
    return decode_local(np.log(acc / len(triple) + 1e-30).reshape(len(X), B.LV.NX, B.LV.NY).astype(np.float32))


def cls_topm(f, seed, X, M):
    from train_cls import Cls
    sel = json.load(open(os.path.join(OUTD, "cls_selection.json")))[str(f)]["chosen"]; h, lr = sel.split("|")
    ck = torch.load(os.path.join(ROOT, "checkpoints", "phase3d", f"cls{f}_{h}_lr{float(lr):g}_s{seed}.pt"), map_location="cpu", weights_only=False)
    m = Cls(ck["cin"], h); m.load_state_dict(ck["model"], strict=True); m = m.to(DEV).eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), 64): out.append(m(torch.from_numpy(X[b:b + 64]).to(DEV).float()).cpu().numpy())
    lg = np.concatenate(out); return {M_: np.argsort(-lg, 1, kind="stable")[:, :M_] for M_ in MS}, ck["history"][-1]


def main():
    t0 = time.time(); R = dict(seeds=list(SEEDS20), triples=[list(t) for t in TRIPLES])
    pc = np.load(os.path.join(ROOT, "results", "phase3b", "preds_clean.npz")); c1 = np.load(os.path.join(ROOT, "results", "phase3c", "preds_c1.npz"))
    rb = json.load(open(os.path.join(ROOT, "results", "phase3b", "results_b.json"))); bias = rb["picker_bias_fixed_radar"]
    # ---------------- fresh set
    with h5py.File(os.path.join(FR3, "users.h5"), "r") as h: fant = h["ant_xyz"][:]
    fxy = fant[:, :2]; nf = len(fxy); FCH = {}
    for f in FR:
        with h5py.File(os.path.join(FR3, f"channels_{f}GHz.h5"), "r") as h: FCH[f] = dict(G=h["gains"][:], adv=h["adv_db"][:])
    with h5py.File(os.path.join(FR3, "lidar_bev.h5"), "r") as h: FL = h["bev"][:]
    with h5py.File(os.path.join(FR3, "radar_bev20.h5"), "r") as h: FRB = h["bev"][:]
    with h5py.File(os.path.join(FR3, "radar_snr20.h5"), "r") as h: X_ = h["radar_features"][:]; lo = h["ra_db_min"][:]; hi = h["ra_db_max"][:]
    Rr, Sn = E.estimate(E.ra_db(X_, lo, hi), "centroid"); Rr = Rr + bias[0]; Sn = np.clip(Sn + bias[1], -1, 1); del X_
    py = Rr * Sn; px = np.sqrt(np.maximum(Rr ** 2 - py ** 2 - (1.5 - E.RADAR[2]) ** 2, 0)); f_picker = np.c_[px, py + E.RADAR[1]]
    cz = np.load(os.path.join(FR3, "lidar_clouds.npz")); f_geo, f_fb = B.geo([cz[f"arr_{i}"] for i in range(nf)]); log("fresh geo, fallback", f_fb)
    FX = dict(radar=FRB, lidar=FL, fused=np.concatenate([FRB, FL], 1))
    # ---------------- predictions: name -> dict(val, test, varied, fresh) arrays (runs, n, 2)
    P = {}
    P["radar_picker"] = dict(val=pc["radar_picker_val"], test=pc["radar_picker_test"], varied=pc["radar_picker_test"], fresh=f_picker[None])
    P["lidar_geometric"] = dict(val=pc["lidar_geometric_val"], test=pc["lidar_geometric_test"], varied=c1["geometric_test_varied"][None], fresh=f_geo[None])
    P["radar_net"] = dict(val=pc["radar_net_val"], test=pc["radar_net_test"], varied=pc["radar_net_test"], fresh=B.net_preds("radar", "_r2", SEEDS20, FX["radar"]))
    P["lidar_net"] = dict(val=pc["lidar_net_val"], test=pc["lidar_net_test"], varied=c1["single_net_test_varied"], fresh=B.net_preds("lidar", "", SEEDS20, FX["lidar"]))
    P["fused_net"] = dict(val=pc["fused_net_val"], test=pc["fused_net_test"], varied=B.net_preds("fused", "_r2", SEEDS20, B.inputs("fused", te, lidar=B.LBV)),
                          fresh=B.net_preds("fused", "_r2", SEEDS20, FX["fused"]))
    P["varied_trained_net"] = dict(val=c1["varied_net_val"], test=c1["varied_net_test_standard"], varied=c1["varied_net_test_varied"],
                                   fresh=B.net_preds("lidar", "_var", SEEDS20, FX["lidar"]))
    log("frozen model preds")
    # B1 1-NN
    for sensor, src, fX in (("lidar", B.LB, FL), ("radar", B.RB[20], FRB)):
        Ftr = pooled_feat(B.h5(src, tr))
        var_src = B.LBV if sensor == "lidar" else src
        P[f"one_nn_{sensor}"] = dict(val=one_nn(Ftr, pooled_feat(B.h5(src, va)))[None], test=one_nn(Ftr, pooled_feat(B.h5(src, te)))[None],
                                     varied=one_nn(Ftr, pooled_feat(B.h5(var_src, te)))[None], fresh=one_nn(Ftr, pooled_feat(fX))[None])
    log("1-NN")
    # B2 radar ensembles (radar is not affected by vehicle variation: varied = standard)
    Xv, Xt = B.inputs("radar", va), B.inputs("radar", te)
    ev = [ens_xy(t, Xv) for t in TRIPLES]; et = [ens_xy(t, Xt) for t in TRIPLES]; ef = [ens_xy(t, FX["radar"]) for t in TRIPLES]
    P["radar_ensemble3"] = dict(val=np.array(ev), test=np.array(et), varied=np.array(et), fresh=np.array(ef)); log("ensembles")
    np.savez(os.path.join(OUTD, "preds_d.npz"), **{f"{k}_{w}": v[w] for k, v in P.items() for w in v})
    # ---------------- losses
    sets = dict(test=(te, {f: B.CH[f]["G"][te] for f in FR}, {f: B.CH[f]["adv"][te] for f in FR}, xy[te]),
                varied=(te, {f: B.CH[f]["G"][te] for f in FR}, {f: B.CH[f]["adv"][te] for f in FR}, xy[te]),
                fresh=(None, {f: FCH[f]["G"] for f in FR}, {f: FCH[f]["adv"] for f in FR}, fxy))
    def subsets(adv, pxy):
        d2 = np.hypot(*pxy.T); return dict(all=np.ones(len(pxy), bool), adv_ge3=adv >= 3, d_ge_2m=d2 >= 2)
    T, SH, FIX, PER = {}, {}, {}, {}
    for name, pr in P.items():
        for f in FR:
            for M in MS:
                shapes = [B.choose_shape(f, pr["val"][r], M)[0] for r in range(len(pr["val"]))]
                SH[f"{name}|{f}|{M}"] = {f"{a}x{b}": shapes.count((a, b)) for a, b in sorted(set(shapes))}
                for sname, (_, Gs, advs, pxy) in sets.items():
                    G = Gs[f]; runs = pr[sname]
                    Ls = [B.loss_db(G, B.beam_set(runs[r], B.CH[f]["u"], B.CH[f]["rings"], *shapes[r])) for r in range(len(runs))]
                    PER[f"{name}|{sname}|{f}|{M}"] = np.mean(Ls, 0)
                    for k, m in subsets(advs[f], pxy).items():
                        T[f"{sname}|{name}|{f}|{M}|{k}"] = (ci_seed(Ls, m) if len(Ls) > 1 else ci_user(Ls[0], m)) | dict(within_1dB=float(np.mean([(L[m] <= 1).mean() for L in Ls])))
                # B3 fixed shape (M/8 angles x 8 rings) on the original test set
                Lf = [B.test_loss(f, pr["test"][r], (M // 8, 8)) for r in range(len(pr["test"]))]
                FIX[f"{name}|{f}|{M}"] = ci_seed(Lf, np.ones(len(te), bool)) if len(Lf) > 1 else ci_user(Lf[0], np.ones(len(te), bool))
    # B6 classifiers (top-M codewords; no shape)
    cls_files = glob.glob(os.path.join(ROOT, "checkpoints", "phase3d", "cls*_s55.pt"))
    if os.path.exists(os.path.join(OUTD, "cls_selection.json")) and len(cls_files) >= 2:
        learn = []
        Xin = dict(test=B.inputs("lidar", te), varied=B.inputs("lidar", te, lidar=B.LBV), fresh=FL)
        for f in FR:
            per = {s: {M: [] for M in MS} for s in sets}
            for sd in SEEDS20[:5]:
                for sname in sets:
                    tops, hist = cls_topm(f, sd, Xin[sname], 8)
                    G = sets[sname][1][f]
                    for M in MS: per[sname][M].append(10 * np.log10(G.max(1) / np.take_along_axis(G, tops[M], 1).max(1)))
                learn.append(dict(freq=f, seed=sd, final_train_loss=hist["train_loss"], prior=hist["prior"], learned=hist["train_loss"] < 0.95 * hist["prior"], val_top1=hist["val_top1"]))
            for sname, (_, Gs, advs, pxy) in sets.items():
                for M in MS:
                    Ls = per[sname][M]; PER[f"codeword_classifier|{sname}|{f}|{M}"] = np.mean(Ls, 0)
                    for k, m in subsets(advs[f], pxy).items():
                        T[f"{sname}|codeword_classifier|{f}|{M}|{k}"] = ci_seed(Ls, m) | dict(within_1dB=float(np.mean([(L[m] <= 1).mean() for L in Ls])))
        R["codeword_classifier"] = dict(selection=json.load(open(os.path.join(OUTD, "cls_selection.json"))), learning_check=learn)
    else:
        R["codeword_classifier"] = "not available"
    # full sweep
    for sname, (_, Gs, advs, pxy) in sets.items():
        for f in FR:
            G = Gs[f]; Sw = np.c_[np.arange(256)[None].repeat(len(G), 0) * 8, G[:, 0::8].argmax(1)[:, None] * 8 + np.arange(1, 8)[None]]
            L = B.loss_db(G, Sw)
            for k, m in subsets(advs[f], pxy).items(): T[f"{sname}|sweep263|{f}|263|{k}"] = dict(mean=float(L[m].mean()), n=int(m.sum()))
    R["tables"] = T; R["shapes_chosen"] = SH; R["fixed_shape_baseline_test"] = FIX
    R["subset_sizes"] = {s: {f: {k: int(v.sum()) for k, v in subsets(sets[s][2][f], sets[s][3]).items()} for f in FR} for s in sets}
    # paired comparisons (per-user bootstrap of method means), read by CI
    def cmp(a, b, s, f, M, m):
        return B.boot(PER[f"{a}|{s}|{f}|{M}"][m] - PER[f"{b}|{s}|{f}|{M}"][m])
    C = {}
    for s in sets:
        for f in FR:
            al = np.ones(len(sets[s][3]), bool)
            for M in MS:
                C[f"{s}|{f}|{M}|radar_ensemble3_vs_radar_net"] = cmp("radar_ensemble3", "radar_net", s, f, M, al)
                C[f"{s}|{f}|{M}|one_nn_lidar_vs_lidar_net"] = cmp("one_nn_lidar", "lidar_net", s, f, M, al)
                C[f"{s}|{f}|{M}|one_nn_radar_vs_radar_net"] = cmp("one_nn_radar", "radar_net", s, f, M, al)
                if f"codeword_classifier|{s}|{f}|{M}" in PER: C[f"{s}|{f}|{M}|codeword_classifier_vs_lidar_net"] = cmp("codeword_classifier", "lidar_net", s, f, M, al)
    R["comparisons"] = C
    # position error (Euclidean, horizontal), all users
    def pe(pt, ref):
        e = np.hypot(*(pt.reshape(-1, 2) - np.tile(ref, (len(pt), 1))).T); return dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)))
    R["position_error_euclid"] = {f"{s}|{n}": pe(P[n][s], sets[s][3]) for n in P for s in sets}
    R["fresh"] = dict(users=nf, geometric_fallback=f_fb); R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(OUTD, "results_d.json"), "w"), indent=1); log("done")


if __name__ == "__main__":
    main()
