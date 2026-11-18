"""Independent verification, step 3.

(a) V8 follow-up: is the shortlist's advantage over "top-1 angle + all 8 ranges" due to the
    range head, or to angle diversity? Replace p_r by (i) the seed's train-split y_r_gated
    prior, (ii) a uniform p_r; keep p_theta. Same loss as v2.
(b) V13: paired tests from MY per-seed NF K=8 values (results/verify/v2_models.json),
    not from banked JSON.
Output results/verify/v3_rangehead_stats.json
"""
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.splits import trajectory_grouped_split
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import SensingPolarPredictor

OUT = os.path.join(ROOT, "results", "verify")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220]
dev = torch.device("cuda")
res = {"rangehead": {}, "stats": {}}


def loss_of(G, sl):
    return 10 * np.log10(G.max(1) / np.take_along_axis(G, sl, 1).max(1))


def topk(pt, pr, K=8):
    j = (pt[:, :, None] * pr[:, None, :]).reshape(len(pt), 512)
    return torch.from_numpy(j).topk(K, 1).indices.numpy()


for f in (60, 140):
    z = np.load(os.path.join(OUT, f"gains_{f}GHz.npz"))
    G = z["G32"].astype(np.float64); yrg = z["y_r_gated"]; traj = z["trajectory_id"]
    with h5py.File(os.path.join(ROOT, "data", "processed", f"closerange_{f}GHz_full.h5"), "r") as hf:
        XR = hf["radar_features"][:]; XL = hf["lidar_features"][:]
    rows = []
    for s in SEEDS:
        tr, _, te = trajectory_grouped_split(traj, seed=s); te = np.sort(te)
        prior = np.bincount(yrg[tr], minlength=8) / len(tr)
        nf = yrg[te] > 0; Gt = G[te]
        rm = RadarPolarModel().to(dev).eval()
        rm.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"closerange_{f}GHz_radar", f"seed{s}.pt"),
                                      map_location="cpu", weights_only=False)["model"])
        fm = SensingPolarPredictor().to(dev).eval()
        fm.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"closerange_{f}GHz_fusion_warmstart", f"seed{s}.pt"),
                                      map_location="cpu", weights_only=False)["model"])
        PT = {"radar_alone": [], "soft_avg": []}; PR = {"radar_alone": [], "soft_avg": []}
        with torch.no_grad():
            for b in range(0, len(te), 128):
                idx = te[b:b + 128]
                xr = torch.from_numpy(XR[idx]).to(dev); xl = torch.from_numpy(XL[idx]).to(dev)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = rm(xr); fo = fm(xr, xl)
                PT["radar_alone"].append(o["theta_logits"].float().softmax(1).cpu().numpy())
                PR["radar_alone"].append(o["range_logits"].float().softmax(1).cpu().numpy())
                pth = torch.stack([F.softmax(fo[f"{x}_theta_logits"].float(), 1) for x in ("radar", "lidar", "fused")])
                prg = torch.stack([F.softmax(fo[f"{x}_range_logits"].float(), 1) for x in ("radar", "lidar", "fused")])
                wt = pth.max(-1).values; wt = wt / wt.sum(0, keepdim=True)
                wr = prg.max(-1).values; wr = wr / wr.sum(0, keepdim=True)
                PT["soft_avg"].append((wt[..., None] * pth).sum(0).cpu().numpy())
                PR["soft_avg"].append((wr[..., None] * prg).sum(0).cpu().numpy())
        rec = dict(seed=s)
        for m in PT:
            pt = np.concatenate(PT[m]); pr = np.concatenate(PR[m])
            own = loss_of(Gt, topk(pt, pr))
            pri = loss_of(Gt, topk(pt, np.repeat(prior[None], len(te), 0)))
            # uniform p_r: joint ties within an angle -> take top-1 angle x 8 ranges then next angle
            uni = loss_of(Gt, topk(pt, np.ones((len(te), 8)) / 8 + 1e-9 * np.arange(8)[None]))
            ia = np.argsort(-pt, 1)
            top2x4 = np.concatenate([ia[:, :2, None] * 8 + np.argsort(-pr, 1)[:, None, :4]], 1).reshape(len(te), 8)
            rec[m] = dict(own_K8_nf=float(own[nf].mean()), prior_K8_nf=float(pri[nf].mean()),
                          uniform_K8_nf=float(uni[nf].mean()), top2ang_x_top4rng_nf=float(loss_of(Gt, top2x4)[nf].mean()),
                          own_K8_all=float(own.mean()), prior_K8_all=float(pri.mean()))
        rows.append(rec)
        print(f, s, {m: {k: round(v, 3) for k, v in rec[m].items()} for m in PT}, flush=True)
    res["rangehead"][str(f)] = dict(per_seed=rows, mean={m: {k: float(np.mean([r[m][k] for r in rows])) for k in rows[0][m]}
                                                          for m in ("radar_alone", "soft_avg")})
    del XR, XL

# ---------------- V13 stats from my own per-seed values ---------------------------------
v2 = json.load(open(os.path.join(OUT, "v2_models.json")))
new15 = SEEDS[5:]


def tests(a, b):
    d = np.asarray(a) - np.asarray(b)            # radar - other  (positive = other better)
    n = len(d)
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    dz = d.mean() / d.std(ddof=1)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lo = stats.t.sf((d.mean() + 0.5) / se, n - 1)     # H0: diff <= -0.5
    p_hi = stats.t.cdf((d.mean() - 0.5) / se, n - 1)    # H0: diff >= +0.5
    rng = np.random.default_rng(0)
    bs = [(lambda x: x.mean() / x.std(ddof=1))(d[rng.integers(0, n, n)]) for _ in range(10000)]
    return dict(n=n, mean_diff=float(d.mean()), t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue), d_z=float(dz),
                d_z_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                tost_p=float(max(p_lo, p_hi)), tost_equiv=bool(max(p_lo, p_hi) < 0.05),
                bonferroni12_t_p=float(min(1, 12 * t.pvalue)))


for f in ("60", "140"):
    ps = {p["seed"]: p for p in v2[f]["per_seed"]}
    for setname, ss in (("new15", new15), ("all20", SEEDS), ("orig5", SEEDS[:5])):
        ra = [ps[s]["loss"]["radar_alone"]["8"]["nf"] for s in ss]
        for other in ("soft_avg", "radar_primary"):
            ob = [ps[s]["loss"][other]["8"]["nf"] for s in ss]
            res["stats"][f"{f}_{setname}_radar_vs_{other}"] = tests(ra, ob)
for k, v in res["stats"].items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
json.dump(res, open(os.path.join(OUT, "v3_rangehead_stats.json"), "w"), indent=1)
