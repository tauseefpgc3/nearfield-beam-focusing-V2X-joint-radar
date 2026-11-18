"""D-034 A6 follow-up: the real-data paired comparisons (network minus baseline) on the DeepSense
test split, with the number of test sequences G, the percentile cluster bootstrap (as banked in
results/phase3c/c2.json, recomputed here with the same code and seed) and the cluster-robust
t-interval (G - 1 df). Per the A6 rule, the t-interval is reported next to the bootstrap where
results/verify4/a6.json says rule_switch_to_t (scenarios 8 and 31).

    python scripts/verify4/a6b_realdata_ci.py   # -> results/verify4/a6b_realdata_ci.json
"""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); sys.path.insert(0, os.path.join(ROOT, "scripts", "verify4"))
import train_ds as T, eval_c2 as E
from a6_cluster_coverage import t_ci


def main():
    dev = torch.device("cuda"); c2 = json.load(open(os.path.join(ROOT, "results", "phase3c", "c2.json")))
    a6 = json.load(open(os.path.join(ROOT, "results", "verify4", "a6.json"))); out = {}
    for s in (8, 31, 36):
        with h5py.File(os.path.join(T.DATA, f"ds{s}.h5"), "r") as hf:
            P = hf["power"][:].astype(np.float64); ok = np.isfinite(P).all(1); lab = hf["label"][:].astype(int); seq = hf["seq"][:]
            sp = hf["split"][:]; tr = np.where((sp == 0) & ok)[0]; te = np.where((sp == 2) & ok)[0]; nb = P.shape[1]
            runs = []
            for sd in E.SEEDS20:
                ck = torch.load(os.path.join(T.CK, f"ds{s}_s{sd}.pt"), map_location="cpu", weights_only=False)
                m = T.build(hf); m.load_state_dict(ck["model"], strict=True); runs.append(T.predict(m.to(dev), torch.from_numpy(hf["bev"][te]), dev))
            feat = lambda Z: F.normalize(F.avg_pool2d(torch.from_numpy(Z[:, :1]).float(), 4).flatten(1), dim=1)
            nn_idx = torch.cdist(feat(hf["bev"][te]), feat(hf["bev"][tr])).argmin(1).numpy()
            geo = E.geometric_s31(hf, tr, te, lab, nb)[0] if s == 31 else None
        Pt, Lt, St = P[te], lab[te], seq[te]; G = int(len(np.unique(St)))
        net = [E.metrics(np.argsort(-lp, 1, kind="stable"), Pt, Lt) for lp in runs]
        freq = np.argsort(-np.bincount(lab[tr], minlength=nb), kind="stable")
        B = dict(most_common=np.tile(freq, (len(te), 1)), one_nn=np.argsort(-P[tr][nn_idx], 1, kind="stable"))
        if geo is not None: B["geometric"] = geo
        res = {}
        for bn, rk in B.items():
            per = E.metrics(rk, Pt, Lt)
            diffs = dict(top1_acc=(np.mean([p[1]["hit"] for p in net], 0) - per[1]["hit"], True))
            for K in (1, 3): diffs[f"loss{K}_db"] = (np.mean([p[K]["loss"] for p in net], 0) - per[K]["loss"], False)
            for k, (d, hb) in diffs.items():
                cb = E.cluster_boot(d, St, hb); lo, hi = t_ci(d, St); banked = c2[f"ds{s}"]["baselines"][bn]["network_minus_baseline"][k]
                res[f"{bn}|{k}"] = dict(mean=float(d.mean()), boot_lo=cb["lo"], boot_hi=cb["hi"], t_lo=float(lo), t_hi=float(hi),
                                        verdict_boot=cb["verdict"], banked_max_abs_diff=max(abs(cb[q] - banked[q]) for q in ("mean", "lo", "hi")))
        out[f"ds{s}"] = dict(G=G, n_test=int(len(te)), report_t=bool(a6[f"ds{s}"]["rule_switch_to_t"]), cells=res)
        print(s, json.dumps(out[f"ds{s}"], indent=1), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "verify4", "a6b_realdata_ci.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
