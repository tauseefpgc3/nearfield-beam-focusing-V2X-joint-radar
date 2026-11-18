"""D-034 A6: validity of the sequence-cluster bootstrap with few test sequences.

Population per scenario = validation + test frames (finite power). Per frame: the 20-seed mean
network top-1 hit minus each baseline's hit (most common beam, 1-NN), and the same for the
K = 1 power loss. Population mean = pooled mean. 1,000 replicates (default_rng(36)): draw G
sequences without replacement (G = the scenario's test-sequence count), compute the percentile
cluster-bootstrap 95% CI (2,000 resamples, as eval_c2 but fewer resamples) and the
cluster-robust t-interval (G - 1 df); coverage = share of CIs containing the population mean.

    python scripts/verify4/a6_cluster_coverage.py   # -> results/verify4/a6.json
"""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); import train_ds as T
G_TEST = {8: 16, 31: 8, 36: 24}


def boot_ci(d, seq, B, rng):
    us = np.unique(seq); sums = np.array([d[seq == u].sum() for u in us]); cnt = np.array([(seq == u).sum() for u in us])
    k = rng.integers(0, len(us), (B, len(us))); bs = sums[k].sum(1) / cnt[k].sum(1); return np.percentile(bs, [2.5, 97.5])


def t_ci(d, seq):
    us = np.unique(seq); G = len(us); m = d.mean(); cnt = np.array([(seq == u).sum() for u in us])
    r = np.array([(d[seq == u] - m).sum() for u in us]); se = np.sqrt(G / (G - 1) * (r ** 2).sum()) / len(d)
    h = stats.t.ppf(0.975, G - 1) * se; return m - h, m + h


def main():
    dev = torch.device("cuda"); out = {}
    for s, G in G_TEST.items():
        with h5py.File(os.path.join(T.DATA, f"ds{s}.h5"), "r") as h:
            P = h["power"][:].astype(np.float64); ok = np.isfinite(P).all(1); sp = h["split"][:]; lab = h["label"][:].astype(int); seq = h["seq"][:]
            tr = np.where((sp == 0) & ok)[0]; pop = np.where((sp > 0) & ok)[0]; X = torch.from_numpy(h["bev"][pop])
            hits, l1 = [], []
            for sd in range(11, 221, 11):
                ck = torch.load(os.path.join(T.CK, f"ds{s}_s{sd}.pt"), map_location="cpu", weights_only=False)
                m = T.build(h); m.load_state_dict(ck["model"], strict=True); top = T.predict(m.to(dev), X, dev).argmax(1)
                hits.append(top == lab[pop]); l1.append(10 * np.log10(P[pop].max(1) / P[pop, top]))
            feat = lambda Z: F.normalize(F.avg_pool2d(torch.from_numpy(Z[:, :1]).float(), 4).flatten(1), dim=1)
            nn_ = torch.cdist(feat(h["bev"][pop]), feat(h["bev"][tr])).argmin(1).numpy()
        mc = np.bincount(lab[tr], minlength=P.shape[1]).argmax(); nn_top = P[tr][nn_].argmax(1)
        net_hit = np.mean(hits, 0); net_l1 = np.mean(l1, 0)
        base = dict(most_common=(lab[pop] == mc, 10 * np.log10(P[pop].max(1) / P[pop, mc])),
                    one_nn=(lab[pop] == nn_top, 10 * np.log10(P[pop].max(1) / P[pop, nn_top])))
        sq = seq[pop]; us = np.unique(sq); rng = np.random.default_rng(36); res = {}
        for bn, (bh, bl) in base.items():
            for metric, d in (("top1", net_hit - bh), ("loss1_db", net_l1 - bl)):
                mu = d.mean(); cb = ct = 0
                for _ in range(1000):
                    pick = rng.choice(us, G, replace=False); m = np.isin(sq, pick)
                    lo, hi = boot_ci(d[m], sq[m], 2000, rng); cb += lo <= mu <= hi
                    lo, hi = t_ci(d[m], sq[m]); ct += lo <= mu <= hi
                res[f"{bn}|{metric}"] = dict(population_mean=float(mu), coverage_cluster_bootstrap=cb / 1000, coverage_cluster_t=ct / 1000)
        out[f"ds{s}"] = dict(G=G, population_sequences=int(len(us)), population_frames=int(len(pop)), cells=res,
                             rule_switch_to_t=bool(min(v["coverage_cluster_bootstrap"] for v in res.values()) < 0.93))
        print(s, json.dumps(out[f"ds{s}"], indent=1), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "verify4", "a6.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
