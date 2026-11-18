"""D-034 C1: regenerate one simulated test-set table end to end in a fresh clone.

Run from the clone root after the data steps (see scripts/repro/run_fresh_clone.sh):
    python scripts/repro/c1_table.py --ckpt <dir with lidar_H_lr0.001_s{seed}.pt>

Computes the standard test set, 60 GHz, all users, M = 8..64 loss vs genie for the
single-vehicle LiDAR net (20 seeds, shape chosen on validation per seed) and the geometric
estimator (on freshly rendered clouds), and compares with the banked results_b.json.
-> results/repro/c1_table.json (PASS if every cell is within 0.005 dB)
"""
import argparse, json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "verify4")); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3b"))
import a5_reproduce as A


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
    P2 = os.path.join(ROOT, "data", "processed", "phase2")
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: va = np.sort(h["val"][:]); te = np.sort(h["test"][:])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: ant = h["ant_xyz"][:]; psi = h["psi"][:]
    with h5py.File(os.path.join(P2, "channels_60GHz.h5"), "r") as h: G, u, rings = h["gains"][:], h["u_grid"][:], h["rings"][:]
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase3", "lidar_bev.h5"), "r") as h: Xva, Xte = h["bev"][va], h["bev"][te]
    PV = [A.net_xy(os.path.join(a.ckpt, f"lidar_H_lr0.001_s{sd}.pt"), Xva) for sd in A.SEEDS]
    PT = [A.net_xy(os.path.join(a.ckpt, f"lidar_H_lr0.001_s{sd}.pt"), Xte) for sd in A.SEEDS]
    import lidar_estimator as GEO, lidar_var as LR
    dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json")))
    prm = dict(GEO.DEFAULT, **dl["best"]["params"]); fb = np.array(dl["fallback_xy"]); std = LR.standard_vehicles(len(ant))
    def geo(idx): return np.array([GEO.estimate(c, prm, fb)[0] for b in range(0, len(idx), 25) for c in LR.render(idx[b:b + 25], ant, psi, std)])
    gva, gte = geo(va), geo(te)
    b = json.load(open(os.path.join(ROOT, "results", "phase3b", "results_b.json"))); cells = {}
    for M in A.MS:
        pick = lambda ev: min(A.shapes(M), key=lambda s: A.loss(G[va], A.beams(ev, u, rings, *s)).mean())
        ln = float(np.mean([A.loss(G[te], A.beams(PT[r], u, rings, *pick(PV[r]))).mean() for r in range(len(A.SEEDS))]))
        lg = float(A.loss(G[te], A.beams(gte, u, rings, *pick(gva))).mean())
        cells[f"lidar_net|60|{M}"] = dict(clone=ln, banked=b["tables"][f"lidar_net|60|{M}|all"]["mean"])
        cells[f"geometric|60|{M}"] = dict(clone=lg, banked=b["tables"][f"lidar_geometric|60|{M}|all"]["mean"])
    worst = max(abs(v["clone"] - v["banked"]) for v in cells.values())
    out = dict(cells=cells, max_abs_diff_db=worst, verdict="PASS" if worst <= 0.005 else "FAIL")
    os.makedirs(os.path.join(ROOT, "results", "repro"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "repro", "c1_table.json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
