"""Equivalence check: b_world.clusters (grid/union-find) vs clusters_pairs (original) on every validation frame
and the first 120 test frames at v = 0 and v = 20 m/s (current + earlier frame). Frames whose original pair list
would exceed 2e8 pairs are compared only on run time feasibility and counted as skipped."""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import b_world as W
from scipy.spatial import cKDTree
D = "data/processed/phase4/multicar"; Mt = np.load(os.path.join(D, "meta.npz")); ant, psi = Mt["ant"], Mt["psi"]
res = dict(compared=0, identical=0, skipped=0, max_ng=0, t_new=[], t_old=[])
def chk(pc):
    ng = int((pc[:, 2] > 0.3).sum()); res["max_ng"] = max(res["max_ng"], ng)
    t = time.perf_counter(); a = W.clusters(pc); res["t_new"].append(time.perf_counter() - t)
    if ng >= 10 and cKDTree(pc[pc[:, 2] > 0.3, :2]).count_neighbors(cKDTree(pc[pc[:, 2] > 0.3, :2]), 0.8) > 4e8: res["skipped"] += 1; return
    t = time.perf_counter(); b = W.clusters_pairs(pc); res["t_old"].append(time.perf_counter() - t)
    res["compared"] += 1; res["identical"] += int(np.array_equal(a, b))
for i in Mt["val"]:
    sc = W.scene(int(i), ant[i], psi[i]); chk(W.render(W.at_time(sc, 0.0, 0.0), W.NSEED + int(i), None)[0])
for k, i in enumerate(Mt["test"][:120]):
    sc = W.scene(int(i), ant[i], psi[i])
    for v in (0.0, 20.0):
        chk(W.render(W.at_time(sc, v, 0.0), W.NSEED + int(i), None)[0]); chk(W.render(W.at_time(sc, v, -W.DT), W.NSEED + int(i), W.PSEED + 1000 * 2 + int(i))[0])
out = dict(compared=res["compared"], identical=res["identical"], skipped_too_large_for_original=res["skipped"], max_above_ground_points=res["max_ng"],
           median_ms_new=1e3 * float(np.median(res["t_new"])), median_ms_original=1e3 * float(np.median(res["t_old"])))
print(out); json.dump(out, open("results/phase4/b_cluster_check.json", "w"), indent=1)
