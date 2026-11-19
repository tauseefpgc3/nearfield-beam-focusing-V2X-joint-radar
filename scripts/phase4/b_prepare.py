"""D-036 B: multi-car training / validation inputs (v = 0, frame t0) for association rules R0, R1, R2.

    python scripts/phase4/b_prepare.py
Train: sigma per user from default_rng(2037) (1 or 3 m); val: both sigma. GPS report = true antenna
+ sigma * e, e ~ N(0, I2) from default_rng(20270501 + i). Writes data/processed/phase4/multicar/
{train_R*, val_R*_s*}.h5 (bev, xy) and meta.npz; also the val geometric-estimator clouds are
not stored (re-rendered deterministically in b_eval).
"""
import json, os, sys, time
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import b_world as W
OUT = os.path.join(ROOT, "data", "processed", "phase4", "multicar"); os.makedirs(OUT, exist_ok=True)
RULES = ("R0", "R1", "R2")


def gps_err(i): return np.random.default_rng(W.GSEED + i).normal(0, 1, 2)


def main():
    t0 = time.time(); d, th, psi, ant = W.users(); S = W.split(d, th, ant)
    sig_all = np.where(np.random.default_rng(2037).random(W.N_USERS) < 0.5, 1.0, 3.0)
    np.savez(os.path.join(OUT, "meta.npz"), d=d, th=th, psi=psi, ant=ant, train=S["train"], val=S["val"], test=S["test"], sigma_train=sig_all,
             guard_dropped=S["guard_dropped"])
    jobs = [("train", S["train"], None)] + [("val", S["val"], s) for s in (1.0, 3.0)]
    for name, idx, sfix in jobs:
        files = {r: h5py.File(os.path.join(OUT, f"{name}_{r}" + ("" if sfix is None else f"_s{int(sfix)}") + ".h5"), "w") for r in RULES}
        for r, f in files.items(): f.create_dataset("bev", (len(idx), 3, W.LV.NX, W.LV.NY), np.float16, chunks=(1, 3, W.LV.NX, W.LV.NY)); f["xy"] = ant[idx, :2]; f["idx"] = idx
        fbs = {r: 0 for r in RULES}
        for j, i in enumerate(idx):
            sc = W.scene(int(i), ant[i], psi[i]); pc, ids, _, _ = W.render(W.at_time(sc, 0.0, 0.0), W.NSEED + int(i)); lab = W.clusters(pc)
            s = sig_all[i] if sfix is None else sfix; gps = ant[i, :2] + s * gps_err(int(i))
            for r in RULES:
                c, _, fb = W.associate(pc, gps, s, r, lab); files[r]["bev"][j] = W.LV.bev(c); fbs[r] += fb
            if j % 500 == 0: print(name, sfix, j, time.time() - t0, flush=True)
        for f in files.values(): f.close()
        print(name, sfix, "no-candidate rate", {r: fbs[r] / len(idx) for r in RULES}, flush=True)


if __name__ == "__main__":
    main()
