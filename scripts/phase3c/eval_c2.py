"""D-033 C2 evaluation on the DeepSense test split (run once, after training).

    python scripts/phase3c/eval_c2.py 8 31 36

Methods: LiDAR net (20 seeds), most common beam, 1-NN on LiDAR features, geometric estimator
(scenario 31 only; see the design addendum). Metrics: top-K accuracy, power loss vs the best
beam (mean, median) at K in {1, 3, 5, 8}; beam-change rate. Comparisons: sequence-cluster
bootstrap (10,000, default_rng(0)). Writes results/phase3c/c2.json (merged per scenario)."""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
import train_ds as T, ds_prepare as DP
SEEDS20 = tuple(range(11, 221, 11)); KS = (1, 3, 5, 8)
RES = os.path.join(ROOT, "results", "phase3c"); OUTF = os.path.join(RES, "c2.json")


def metrics(rank, P, lab):
    """rank (n, >=8) beam ids best-first; P (n, B) linear power."""
    best = P.max(1); out = {}
    for K in KS:
        top = rank[:, :K]; hit = (top == lab[:, None]).any(1)
        got = np.take_along_axis(P, top, 1).max(1); loss = 10 * np.log10(best / np.maximum(got, 1e-30))
        out[K] = dict(hit=hit.astype(float), loss=loss)
    return out


def change_rate(beam, seq, row):
    o = np.lexsort((row, seq)); b, s = beam[o], seq[o]; same = s[1:] == s[:-1]
    return float((b[1:] != b[:-1])[same].mean())


def summary(per):
    return {f"top{K}_acc": float(per[K]["hit"].mean()) for K in KS} | {f"loss{K}_mean_db": float(per[K]["loss"].mean()) for K in KS} | \
           {f"loss{K}_median_db": float(np.median(per[K]["loss"])) for K in KS}


def ci_t(x):
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h))


def cluster_boot(d, seq, higher_better, B=10000):
    rng = np.random.default_rng(0); us = np.unique(seq); groups = [np.where(seq == u)[0] for u in us]
    sums = np.array([d[g].sum() for g in groups]); cnts = np.array([len(g) for g in groups]); bs = np.empty(B)
    for b in range(B):
        k = rng.integers(0, len(us), len(us)); bs[b] = sums[k].sum() / cnts[k].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5]); m = float(d.mean())
    good = lo > 0 if higher_better else hi < 0; bad = hi < 0 if higher_better else lo > 0
    return dict(mean=m, lo=float(lo), hi=float(hi), verdict="network better" if good else ("network worse" if bad else "no detectable difference"))


def geometric_s31(hf, tr, te, lab, n_beams):
    """Frozen D-031 estimator on ground-shifted clouds; beam angles and one LiDAR-to-GPS
    azimuth offset calibrated on training frames (design addendum)."""
    import pandas as pd, lidar_estimator as GEO
    dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json"))); PRM = dict(GEO.DEFAULT, **dl["best"]["params"])
    df, lid, _ = DP.index(31); root = DP.SC[31]["root"]
    bs = np.loadtxt(os.path.join(root, df.unit1_loc[0]))
    ue = np.stack([np.loadtxt(os.path.join(root, p)) for p in df.unit2_loc])
    lat0 = np.deg2rad(bs[0]); east = np.deg2rad(ue[:, 1] - bs[1]) * np.cos(lat0) * 6371e3; north = np.deg2rad(ue[:, 0] - bs[0]) * 6371e3
    phi_gps = np.arctan2(north, east); zg = float(hf.attrs["z_ground"])
    est = np.full(len(df), np.nan); fb = np.zeros(len(df), bool)
    for i in np.r_[tr, te]:
        pc = DP.load_points((31, lid[i])); pc = np.c_[pc[:, 0], pc[:, 1], pc[:, 2] - zg]
        xy, info = GEO.estimate(pc, PRM, np.array([np.nan, np.nan])); fb[i] = info["fallback"]
        if not fb[i]: est[i] = np.arctan2(xy[1], xy[0])
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    cmed = lambda a: float(np.angle(np.median(np.cos(a)) + 1j * np.median(np.sin(a))))
    theta = np.full(n_beams, np.nan)
    for b in range(n_beams):
        m = tr[lab[tr] == b]
        if len(m): theta[b] = cmed(phi_gps[m])
    ok = tr[~fb[tr]]; off = cmed(wrap(est[ok] - phi_gps[ok]))
    freq = np.argsort(-np.bincount(lab[tr], minlength=n_beams), kind="stable")
    rank = np.empty((len(te), n_beams), int)
    for j, i in enumerate(te):
        if fb[i]: rank[j] = freq; continue
        d = np.abs(wrap(theta - (est[i] - off))); d[np.isnan(d)] = np.inf; rank[j] = np.argsort(d, kind="stable")
    cal = dict(offset_rad=off, fallback_rate_train=float(fb[tr].mean()), fallback_rate_test=float(fb[te].mean()),
               beams_with_angle=int(np.isfinite(theta).sum()),
               train_azimuth_error_median_deg=float(np.rad2deg(np.median(np.abs(wrap(est[ok] - off - phi_gps[ok]))))),
               test_azimuth_error_median_deg=float(np.rad2deg(np.median(np.abs(wrap(est[te][~fb[te]] - off - phi_gps[te][~fb[te]]))))))
    return rank, cal


def run(s):
    dev = torch.device("cuda")
    with h5py.File(os.path.join(T.DATA, f"ds{s}.h5"), "r") as hf:
        P = hf["power"][:].astype(np.float64); lab = hf["label"][:].astype(int); seq = hf["seq"][:]; row = hf["row"][:]
        ok = np.isfinite(P).all(1)                     # raw DeepSense power vectors with NaN entries are excluded from scoring
        sp = hf["split"][:]; tr = np.where((sp == 0) & ok)[0]; te = np.where((sp == 2) & ok)[0]
        Xte = torch.from_numpy(hf["bev"][te]); nb = P.shape[1]
        feat = lambda X: F.normalize(F.avg_pool2d(torch.from_numpy(X[:, :1]).float(), 4).flatten(1), dim=1)
        Ftr = feat(hf["bev"][tr]); Fte = feat(hf["bev"][te])
        R = dict(scenario=s, excluded_nonfinite_power=dict(train=int(((sp == 0) & ~ok).sum()), test=int(((sp == 2) & ~ok).sum())), n_test=int(len(te)), test_sequences=int(len(np.unique(seq[te]))), beams=nb)
        # ---- network
        runs, learn = [], []
        for sd in SEEDS20:
            ck = torch.load(os.path.join(T.CK, f"ds{s}_s{sd}.pt"), map_location="cpu", weights_only=False)
            m = T.build(hf); m.load_state_dict(ck["model"], strict=True); m = m.to(dev)
            runs.append(T.predict(m, Xte, dev)); h = ck["history"][-1]
            learn.append(dict(seed=sd, final_train_loss=h["train_loss"], prior=h["prior"], below_prior=h["train_loss"] < h["prior"], relative_drop=1 - h["train_loss"] / h["prior"], learned=h["train_loss"] < 0.95 * h["prior"], val_top1=h["val_top1"]))
        geo_rank, geo_cal = (geometric_s31(hf, tr, te, lab, nb) if s == 31 else (None, None))
    R["learning_check"] = learn
    Pt, Lt, St, Rt = P[te], lab[te], seq[te], row[te]
    net_per = [metrics(np.argsort(-lp, 1, kind="stable"), Pt, Lt) for lp in runs]
    R["network"] = {k: ci_t([summary(p)[k] for p in net_per]) for k in summary(net_per[0])}
    R["network"]["beam_change_rate"] = ci_t([change_rate(lp.argmax(1), St, Rt) for lp in runs])
    R["ground_truth_beam_change_rate"] = change_rate(Lt, St, Rt)
    # ---- baselines
    freq = np.argsort(-np.bincount(lab[tr], minlength=nb), kind="stable")
    nn_idx = torch.cdist(Fte, Ftr).argmin(1).numpy()
    B = dict(most_common=np.tile(freq, (len(te), 1)), one_nn=np.argsort(-P[tr][nn_idx], 1, kind="stable"))
    if geo_rank is not None: B["geometric"] = geo_rank
    R["baselines"] = {}
    for name, rk in B.items():
        per = metrics(rk, Pt, Lt); R["baselines"][name] = summary(per) | dict(beam_change_rate=change_rate(rk[:, 0], St, Rt))
        mean_hit = np.mean([p[1]["hit"] for p in net_per], 0)
        cmp = dict(top1_acc=cluster_boot(mean_hit - per[1]["hit"], St, True))
        for K in (1, 3):
            cmp[f"loss{K}_db"] = cluster_boot(np.mean([p[K]["loss"] for p in net_per], 0) - per[K]["loss"], St, False)
        R["baselines"][name]["network_minus_baseline"] = cmp
    R["geometric"] = geo_cal if geo_cal else dict(applicable=False, reason=("2-D scan: no height, the frozen estimator needs a 3-D cloud" if s == 8 else
                                                  "V2V: the LiDAR moves with the receiving car; no heading for a LiDAR-to-GPS azimuth offset, and four arrays"))
    return R


def main():
    out = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
    for s in map(int, sys.argv[1:]):
        out[f"ds{s}"] = run(s); json.dump(out, open(OUTF, "w"), indent=1); print(s, json.dumps(out[f"ds{s}"]["network"]["top1_acc"]), flush=True)


if __name__ == "__main__":
    main()
