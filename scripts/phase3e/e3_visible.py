"""D-035 E3 (exploratory, post-hoc): DeepSense scenario 31, existing networks and baselines on
the test frames where the receiving car is visible in the LiDAR (A3b definition), next to the
all-frames result. Paired network-minus-baseline: G sequences, cluster bootstrap, cluster-t.

    python scripts/phase3e/e3_visible.py   # -> results/phase3e/e3_visible.json
"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("phase3c", "verify4"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import ds_prepare as DP, a3_deepsense_alignment as A3, train_ds as T, eval_c2 as E
from a6_cluster_coverage import t_ci
S = 31


def main():
    a3 = json.load(open(os.path.join(ROOT, "results", "verify4", f"a3_ds{S}.json")))
    c2 = json.load(open(os.path.join(ROOT, "results", "phase3c", "c2.json")))[f"ds{S}"]
    df, lid, _ = DP.index(S); rel, okg = A3.gps(S, df); dev = torch.device("cuda")
    with h5py.File(os.path.join(T.DATA, f"ds{S}.h5"), "r") as hf:
        P = hf["power"][:].astype(np.float64); ok = np.isfinite(P).all(1); lab = hf["label"][:].astype(int); seq = hf["seq"][:]
        sp = hf["split"][:]; tr = np.where((sp == 0) & ok)[0]; te = np.where((sp == 2) & ok)[0]; nb = P.shape[1]
        g = {k: hf.attrs[k] for k in ("z_ground",)}
        th = np.deg2rad(a3["geometry_fit"]["theta_at_lag0_deg"]); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        p = (Rm @ rel[te].T).T
        with Pool(8) as pool: Qs = pool.map(A3.ng_points, [(S, lid[i], float(g["z_ground"])) for i in te], chunksize=16)
        near = np.array([(np.hypot(*(Q - q).T) <= 2).sum() for Q, q in zip(Qs, p)])
        vis = (near >= 3) & okg[te]
        runs = []
        for sd in E.SEEDS20:
            ck = torch.load(os.path.join(T.CK, f"ds{S}_s{sd}.pt"), map_location="cpu", weights_only=False)
            m = T.build(hf); m.load_state_dict(ck["model"], strict=True); runs.append(T.predict(m.to(dev), torch.from_numpy(hf["bev"][te]), dev))
        feat = lambda Z: F.normalize(F.avg_pool2d(torch.from_numpy(Z[:, :1]).float(), 4).flatten(1), dim=1)
        nn_idx = torch.cdist(feat(hf["bev"][te]), feat(hf["bev"][tr])).argmin(1).numpy()
        geo = E.geometric_s31(hf, tr, te, lab, nb)[0]
    freq = np.argsort(-np.bincount(lab[tr], minlength=nb), kind="stable")
    BASE = dict(most_common=np.tile(freq, (len(te), 1)), one_nn=np.argsort(-P[tr][nn_idx], 1, kind="stable"), geometric=geo)
    out = dict(scenario=S, label="exploratory, post-hoc (subset defined after A3b)", n_test=int(len(te)), n_visible=int(vis.sum()),
               visibility_rule="A3b: rotation at lag 0 from a3_ds31.json; >= 3 non-ground points within 2 m of the GPS-projected receiver")
    for sub, msk in (("all", np.ones(len(te), bool)), ("visible", vis), ("not_visible", ~vis)):
        Pt, Lt, St = P[te][msk], lab[te][msk], seq[te][msk]
        net = [E.metrics(np.argsort(-lp[msk], 1, kind="stable"), Pt, Lt) for lp in runs]
        R = dict(n=int(msk.sum()), G=int(len(np.unique(St))), network={k: E.ci_t([E.summary(q)[k] for q in net]) for k in E.summary(net[0])}, baselines={})
        for bn, rk in BASE.items():
            per = E.metrics(rk[msk], Pt, Lt); R["baselines"][bn] = E.summary(per); cmp = {}
            diffs = dict(top1_acc=(np.mean([q[1]["hit"] for q in net], 0) - per[1]["hit"], True))
            for K in (1, 3): diffs[f"loss{K}_db"] = (np.mean([q[K]["loss"] for q in net], 0) - per[K]["loss"], False)
            for k, (d, hb) in diffs.items():
                cb = E.cluster_boot(d, St, hb); lo, hi = t_ci(d, St)
                cmp[k] = dict(mean=cb["mean"], boot_lo=cb["lo"], boot_hi=cb["hi"], verdict_boot=cb["verdict"], t_lo=float(lo), t_hi=float(hi))
            R["baselines"][bn]["network_minus_baseline"] = cmp
        out[sub] = R
    out["all_matches_c2_top1"] = abs(out["all"]["network"]["top1_acc"]["mean"] - c2["network"]["top1_acc"]["mean"])
    json.dump(out, open(os.path.join(ROOT, "results", "phase3e", "e3_visible.json"), "w"), indent=1)
    for sub in ("all", "visible", "not_visible"):
        r = out[sub]; print(sub, r["n"], "G", r["G"], "net top1 %.3f" % r["network"]["top1_acc"]["mean"],
                            {b: "%.3f" % v["top1_acc"] for b, v in r["baselines"].items()}, flush=True)
    print("c2 match", out["all_matches_c2_top1"])


if __name__ == "__main__":
    main()
