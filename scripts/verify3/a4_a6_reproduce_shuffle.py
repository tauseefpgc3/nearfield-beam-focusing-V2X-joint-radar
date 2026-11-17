"""A4: reproduce the Phase 3a headline numbers from the checkpoints with the independent code in
common.py (own network class loaded strict, own decoder, own sweep and loss).
A6: shuffle the LiDAR inputs across test users (and across validation users, for the
shape choice) and confirm the LiDAR network collapses to the label-prior level.
Also caches all per-seed predictions in results/verify3/preds.npz for A1/A5."""
import json, os, sys, time
import numpy as np
from common import *
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
import lidar_v2 as SIM, lidar_estimator as GEO          # objects under audit (black boxes)

S = split(); va, te, tr = S["val"], S["test"], S["train"]; ant, psi = users(); xy = ant[:, :2]
CH = {f: channels(f) for f in (60, 140)}
pred = {}
for s in ("radar", "lidar", "fused"):
    Xv, Xt = bev_inputs(s, va), bev_inputs(s, te)
    for sd in SEEDS:
        m, ck = load_net(s, sd); assert ck["args"]["head"] == "H" and abs(ck["args"]["lr"] - 1e-3) < 1e-12
        pred[f"{s}_s{sd}_val"] = predict(m, Xv); pred[f"{s}_s{sd}_test"] = predict(m, Xt)
        if s == "lidar":                                  # A6 shuffles (one fixed permutation each)
            pv = np.random.default_rng(606).permutation(len(va)); pt = np.random.default_rng(607).permutation(len(te))
            pred[f"lidarshuf_s{sd}_val"] = predict(m, Xv[pv]); pred[f"lidarshuf_s{sd}_test"] = predict(m, Xt[pt])
        print(s, sd, "done", flush=True)
# geometric estimator, frozen at revision-design-v2
dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json")))
prm = dict(GEO.DEFAULT, **dl["best"]["params"]); fb = np.array(dl["fallback_xy"])
assert np.allclose(fb, np.median(xy[tr], 0)), "fallback is not the training median"
def geo(ix):
    out = []
    for b in range(0, len(ix), 25):
        for pc in SIM.render(ix[b:b + 25], ant, psi)[0]: out.append(GEO.estimate(pc, prm, fb)[0])
    return np.array(out)
t0 = time.time(); pred["geo_val"] = geo(va); pred["geo_test"] = geo(te); print("geo", time.time() - t0, flush=True)
pred["prior_val"] = np.repeat(np.median(xy[tr], 0)[None], len(va), 0); pred["prior_test"] = np.repeat(np.median(xy[tr], 0)[None], len(te), 0)
np.savez(os.path.join(OUT, "preds.npz"), **pred)

def method_loss(name, f, M):
    runs = [k[:-4] for k in pred if k.startswith(name) and k.endswith("_val")]
    runs = [r for r in runs if r == name or r.startswith(name + "_s")]
    L = np.array([sweep_with_val_shape(CH[f], va, te, pred[r + "_val"], pred[r + "_test"], M)[0] for r in runs])
    return L
res = {}
for name in ("lidar", "geo", "radar", "fused", "lidarshuf", "prior"):
    e = np.concatenate([np.hypot(*(pred[k] - xy[te]).T) for k in pred if k.startswith(name) and k.endswith("_test") and (k == name + "_test" or k.startswith(name + "_s"))])
    res[name] = dict(err_median_m=float(np.median(e)), err_p90_m=float(np.percentile(e, 90)))
    for f in (60, 140):
        for M in (8, 16, 32, 64):
            L = method_loss(name, f, M)
            res[name][f"{f}GHz_M{M}"] = dict(mean=float(L.mean()), per_seed=[float(x) for x in L.mean(1)], within_1dB=float((L <= 1).mean()))
ref = json.load(open(os.path.join(ROOT, "results", "phase3", "test_results.json")))["loss"]
mp = dict(lidar="lidar_net", geo="lidar_geometric", radar="radar_net", fused="fused_net")
res["comparison_with_phase3a"] = {f"{n}|{f}|{M}": dict(verify3=res[n][f"{f}GHz_M{M}"]["mean"], phase3a=ref[f"{mp[n]}|{f}|{M}"]["all"]["mean"])
                                  for n in mp for f in (60, 140) for M in (8, 16, 32, 64)}
json.dump(res, open(os.path.join(OUT, "a4_a6.json"), "w"), indent=1)
for k, v in res["comparison_with_phase3a"].items(): print(k, "verify3 %.3f  phase3a %.3f" % (v["verify3"], v["phase3a"]))
for n in ("lidar", "lidarshuf", "prior", "geo"):
    print(n, res[n]["err_median_m"], res[n]["err_p90_m"], {k: round(v["mean"], 2) for k, v in res[n].items() if "GHz" in k})
