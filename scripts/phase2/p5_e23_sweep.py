"""D-030 E2/E3 (exploratory): local sweep of M beams seeded by a position estimate.
E2: true antenna position + Gaussian error (sigma per horizontal axis). E3: radar picker
(centroid + training bias, 20 dB) and a geometric LiDAR estimator on the raw point cloud.
Shape (A angles x R rings, A*R = M) chosen per (freq, source, M) on validation."""
import json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2"))
import p3_eval as E
from p1_lidar_split import point_cloud
OUT = os.path.join(ROOT, "results", "phase2", "feasibility")
SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0); MS = (8, 16, 32, 64); ARR = np.array([0, 0, 2.0])


def shapes(M):
    return [(A, M // A) for A in (M // 8, M // 4, M // 2, M) if M // A <= 8 and A >= 1]


def sweep_loss(G, xy, ug, rings, A, R):
    p = np.c_[xy, np.full(len(xy), 1.5)] - ARR; r = np.linalg.norm(p, axis=1); u = p[:, 1] / r
    oa = np.argsort(np.abs(u[:, None] - ug[None]), axis=1, kind="stable")[:, :A]
    inv = np.where(np.isinf(rings), 0.0, 1 / rings)
    orr = np.argsort(np.abs(inv[None] - 1 / r[:, None]), axis=1, kind="stable")[:, :R]
    S = (oa[:, :, None] * 8 + orr[:, None, :]).reshape(len(xy), -1)
    return E.loss_of(G, S)


def lidar_est(i, ant, psi):
    pc = point_cloud(ant[i, 0], ant[i, 1], psi[i], 20261118 + i)
    q = pc[(pc[:, 2] >= -1.8) & (pc[:, 2] <= 0.0), :2]
    H, xe, ye = np.histogram2d(q[:, 0], q[:, 1], bins=[np.arange(-20, 40.01, 0.5), np.arange(-20, 20.01, 0.5)])
    a, b = np.unravel_index(H.argmax(), H.shape); c = np.array([xe[a] + 0.25, ye[b] + 0.25])
    for _ in range(3):                         # collect, then 2 re-centrings on the median
        cl = q[np.linalg.norm(q - c, axis=1) <= 3.0]; c = np.median(cl, 0)
    cl = q[np.linalg.norm(q - c, axis=1) <= 3.0]
    mu = cl.mean(0); _, _, Vt = np.linalg.svd(cl - mu, full_matrices=False); pr = (cl - mu) @ Vt.T
    mid = (pr.max(0) + pr.min(0)) / 2
    return mu + mid @ Vt                       # box centre = antenna (offset 0, D-029)


def err_stats(est, true):
    e = est - true; rad = true / np.linalg.norm(true, axis=1, keepdims=True); perp = np.c_[-rad[:, 1], rad[:, 0]]
    out = {}
    for nm, v in (("range", (e * rad).sum(1)), ("sideways", (e * perp).sum(1))):
        out[nm] = dict(median_abs=float(np.median(np.abs(v))), p90_abs=float(np.percentile(np.abs(v), 90)), rms=float(np.sqrt(np.mean(v ** 2))))
    return out


def main():
    with h5py.File(os.path.join(E.P, "split.h5"), "r") as hf: va = np.sort(hf["val"][:]); te = np.sort(hf["test"][:])
    with h5py.File(os.path.join(E.P, "users.h5"), "r") as hf: ant = hf["ant_xyz"][:]; psi = hf["psi"][:]
    xy = ant[:, :2]; idx = dict(va=va, te=te)
    noise = np.random.default_rng(2032).standard_normal((len(SIGMAS), len(ant), 2))
    bias = json.load(open(os.path.join(ROOT, "results", "phase2", "gate_results.json")))["bias"]["centroid"]
    est = {}
    for nm, ix in idx.items():
        X, lo, hi = E.load_radar(20, ix); R, S = E.estimate(E.ra_db(X, lo, hi), "centroid")
        R = R + bias[0]; S = np.clip(S + bias[1], -1, 1); py = R * S; px = np.sqrt(np.maximum(R ** 2 - py ** 2 - (1.5 - 1.2) ** 2, 0))
        est[("radar20", nm)] = np.c_[px, py + E.RADAR[1]]
        est[("lidar", nm)] = np.array([lidar_est(i, ant, psi) for i in ix])
        for s, sg in enumerate(SIGMAS):
            est[(f"sigma{sg:g}", nm)] = xy[ix] + sg * noise[s, ix]
    sources = [f"sigma{sg:g}" for sg in SIGMAS] + ["radar20", "lidar"]
    res = dict(errors={src: err_stats(est[(src, "te")], xy[te]) for src in sources}, freqs={})
    for freq in (60, 140):
        with h5py.File(os.path.join(E.P, f"channels_{freq}GHz.h5"), "r") as hf: G = hf["gains"][:]; ug = hf["u_grid"][:]; rings = hf["rings"][:]
        fr = {}
        for src in sources:
            fr[src] = {}
            for M in MS:
                vs = {f"{A}x{R}": float(sweep_loss(G[va], est[(src, "va")], ug, rings, A, R).mean()) for A, R in shapes(M)}
                best = min(vs, key=vs.get); A, R = map(int, best.split("x"))
                L = sweep_loss(G[te], est[(src, "te")], ug, rings, A, R)
                fr[src][str(M)] = dict(shape=best, val=vs, test_mean=float(L.mean()), test_median=float(np.median(L)), test_p90=float(np.percentile(L, 90)),
                                       frac_within_1dB=float((L <= 1).mean()))
            print(freq, src, " ".join(f"M{M}:{fr[src][str(M)]['test_mean']:.2f}({fr[src][str(M)]['shape']})" for M in MS), flush=True)
        res["freqs"][str(freq)] = fr
    print(json.dumps(res["errors"], indent=0))
    json.dump(res, open(os.path.join(OUT, "e23_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
