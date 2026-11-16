"""D-031 design step (train/val only): tune the geometric LiDAR estimator thresholds.
Criterion: mean validation local-sweep loss over M in {8,16,32,64} and both frequencies
(sweep shape chosen per M on the same validation users, as everywhere in D-031)."""
import itertools, json, os, sys, time
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2"))
import lidar_v2 as LV, lidar_estimator as LE
from p5_e23_sweep import sweep_loss, shapes
P2 = LV.P2


def clouds(idx, ant, psi):
    out = []
    for b in range(0, len(idx), 25):
        pcs, _ = LV.render(idx[b:b + 25], ant, psi); out += [p[p[:, 2] >= 0.1] for p in pcs]
    return out


def sweep_score(est, G, ug, rings):
    sc = []
    for f in (60, 140):
        for M in (8, 16, 32, 64):
            sc.append(min(sweep_loss(G[f], est, ug[f], rings[f], A, R).mean() for A, R in shapes(M)))
    return float(np.mean(sc))


def err(est, xy):
    e = np.hypot(*(est - xy).T); return float(np.median(e)), float(np.percentile(e, 90))


def main():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as hf: tr = np.sort(hf["train"][:]); va = np.sort(hf["val"][:])
    ant, psi = LV.users(); xy = ant[:, :2]
    fb = np.median(xy[tr], 0)
    G, ug, rings = {}, {}, {}
    for f in (60, 140):
        with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as hf: G[f] = hf["gains"][:][va]; ug[f] = hf["u_grid"][:]; rings[f] = hf["rings"][:]
    t0 = time.time(); C = clouds(va, ant, psi); print("clouds", time.time() - t0, flush=True)
    grid = dict(AMBIG=["centre", "near"], Z_CAB=[1.1, 1.2], N_CAB=[3, 10], DELTA=[0.05, 0.15], TOL=[0.2, 0.4], Z_SIDE=[0.6, 0.95])
    res = []
    for vals in itertools.product(*grid.values()):
        p = dict(LE.DEFAULT, **dict(zip(grid, vals)))
        out = [LE.estimate(c, p, fb) for c in C]; est = np.array([o[0] for o in out])
        s = sweep_score(est, G, ug, rings); m, p90 = err(est, xy[va])
        res.append(dict(params=dict(zip(grid, vals)), val_sweep=s, err_med=m, err_p90=p90,
                        fallback=float(np.mean([o[1]["fallback"] for o in out])),
                        resolved=float(np.mean([o[1].get("resolved", False) for o in out]))))
        print(json.dumps(res[-1]), flush=True)
    best = min(res, key=lambda r: r["val_sweep"])
    print("BEST", json.dumps(best))
    os.makedirs(os.path.join(ROOT, "results", "phase3"), exist_ok=True)
    json.dump(dict(fallback_xy=fb.tolist(), grid=res, best=best), open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
