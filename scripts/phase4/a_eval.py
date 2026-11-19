"""D-036 Experiment A evaluation: DeepSense 32-34, grouped 5-fold sequence CV, 5 seeds per fold.

    python scripts/phase4/a_eval.py 32 33 34      # -> results/phase4/a_results.json

Run j = seed j in every fold; each test frame is predicted once per run (by its fold's model).
Metrics as eval_c2 (top-1/3/5/8, loss K = 1/3/5/8 mean and median, beam-change rate). Baselines
fitted per fold on its training frames: most-common beam, 1-NN on the eval_c2 LiDAR feature.
CIs: network metrics t-interval over the 5 runs and, on the per-frame run means, cluster
bootstrap (10,000, default_rng(0)) and cluster-t over all G sequences; network minus baseline:
paired per frame, cluster bootstrap and cluster-t. Learning check per seed and fold.
"""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("phase3c", "verify4"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import train_ds as T, eval_c2 as E
from a6_cluster_coverage import t_ci
SEEDS = (11, 22, 33, 44, 55); KS = E.KS


def clus(d, seq, hb):
    cb = E.cluster_boot(d, seq, hb); lo, hi = t_ci(d, seq)
    return dict(mean=cb["mean"], boot_lo=cb["lo"], boot_hi=cb["hi"], verdict_boot=cb["verdict"], t_lo=float(lo), t_hi=float(hi), G=int(len(np.unique(seq))))


def run(s, dev):
    runs = [[] for _ in SEEDS]; base = dict(most_common=[], one_nn=[]); Pt, Lt, St, Rt, learn = [], [], [], [], []
    for k in range(5):
        with h5py.File(os.path.join(ROOT, "data", "processed", "phase4", "cv", f"f{k}", f"ds{s}.h5"), "r") as hf:
            P = hf["power"][:].astype(np.float64); lab = hf["label"][:].astype(int); seq = hf["seq"][:]; row = hf["row"][:]
            ok = np.isfinite(P).all(1); sp = hf["split"][:]; tr = np.where((sp == 0) & ok)[0]; te = np.where((sp == 2) & ok)[0]; nb = P.shape[1]
            Xte = torch.from_numpy(hf["bev"][te])
            feat = lambda X: F.normalize(F.avg_pool2d(torch.from_numpy(X[:, :1]).float(), 4).flatten(1), dim=1)
            Ftr, Fte = feat(hf["bev"][tr]), feat(hf["bev"][te])
            for j, sd in enumerate(SEEDS):
                ck = torch.load(os.path.join(ROOT, "checkpoints", "phase4", "cv", f"f{k}", f"ds{s}_s{sd}.pt"), map_location="cpu", weights_only=False)
                m = T.build(hf); m.load_state_dict(ck["model"], strict=True); runs[j].append(T.predict(m.to(dev), Xte, dev)); h = ck["history"][-1]
                learn.append(dict(fold=k, seed=sd, final_train_loss=h["train_loss"], prior=h["prior"], learned=bool(h["train_loss"] < 0.95 * h["prior"]), val_top1=h["val_top1"]))
        freq = np.argsort(-np.bincount(lab[tr], minlength=nb), kind="stable")
        base["most_common"].append(np.tile(freq, (len(te), 1))); base["one_nn"].append(np.argsort(-P[tr][torch.cdist(Fte, Ftr).argmin(1).numpy()], 1, kind="stable"))
        Pt.append(P[te]); Lt.append(lab[te]); St.append(seq[te]); Rt.append(row[te])
    Pt, Lt, St, Rt = map(np.concatenate, (Pt, Lt, St, Rt)); runs = [np.concatenate(r) for r in runs]; base = {b: np.concatenate(v) for b, v in base.items()}
    net = [E.metrics(np.argsort(-lp, 1, kind="stable"), Pt, Lt) for lp in runs]
    R = dict(scenario=s, protocol="grouped 5-fold sequence CV, 5 seeds per fold", n_test=int(len(Lt)), G=int(len(np.unique(St))), seeds=list(SEEDS),
             learning_check=learn, learned_count=int(sum(x["learned"] for x in learn)), runs=len(learn))
    R["network"] = {k: E.ci_t([E.summary(p)[k] for p in net]) for k in E.summary(net[0])}
    R["network"]["beam_change_rate"] = E.ci_t([E.change_rate(lp.argmax(1), St, Rt) for lp in runs])
    hit = {K: np.mean([p[K]["hit"] for p in net], 0) for K in KS}; los = {K: np.mean([p[K]["loss"] for p in net], 0) for K in KS}
    R["network_cluster"] = {f"top{K}_acc": clus(hit[K], St, True) for K in KS} | {f"loss{K}_mean_db": clus(los[K], St, False) for K in KS}
    R["ground_truth_beam_change_rate"] = E.change_rate(Lt, St, Rt); R["baselines"] = {}
    for bn, rk in base.items():
        per = E.metrics(rk, Pt, Lt); B = E.summary(per) | dict(beam_change_rate=E.change_rate(rk[:, 0], St, Rt))
        B["network_minus_baseline"] = {f"top{K}_acc": clus(hit[K] - per[K]["hit"], St, True) for K in KS} | {f"loss{K}_db": clus(los[K] - per[K]["loss"], St, False) for K in KS}
        R["baselines"][bn] = B
    def better(c, hb): return (c["boot_lo"] > 0 and c["t_lo"] > 0) if hb else (c["boot_hi"] < 0 and c["t_hi"] < 0)
    R["works"] = bool(all(better(R["baselines"][b]["network_minus_baseline"]["top1_acc"], True) for b in base))
    return R


if __name__ == "__main__":
    dev = torch.device("cuda"); out = {}
    p = os.path.join(ROOT, "results", "phase4", "a_results.json")
    if os.path.exists(p): out = json.load(open(p))
    for a in sys.argv[1:]:
        out[f"ds{a}"] = run(int(a), dev); r = out[f"ds{a}"]
        print(a, "n", r["n_test"], "G", r["G"], "learned", r["learned_count"], "/", r["runs"], "net top1 %.3f" % r["network"]["top1_acc"]["mean"],
              {b: "%.3f" % v["top1_acc"] for b, v in r["baselines"].items()}, "works", r["works"], flush=True)
        json.dump(out, open(p, "w"), indent=1)
