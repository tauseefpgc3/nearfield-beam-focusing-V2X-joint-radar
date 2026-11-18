"""D-034 A3 follow-up (scenario 36): does a mirrored LiDAR frame or a longer heading baseline
explain the weak GPS-to-LiDAR fit? Lag 0, 1,500 training frames (default_rng(5)), frames with
unit-1 speed >= 2 m/s. Score contrast = best-theta score / median-over-theta score.
-> results/verify4/a3c_ds36.json"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds_prepare as DP, a3_deepsense_alignment as A3

def main():
    s = 36; df, lid, _ = DP.index(s); seq = df.seq_index.to_numpy(); n = len(df)
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase3c", "ds36.h5"), "r") as h: split = h["split"][:]; zg = float(h.attrs["z_ground"])
    p1 = A3.enu(df.unit1_gps1_lat.to_numpy(), df.unit1_gps1_lon.to_numpy(), df.unit1_gps1_lat.iloc[0], df.unit1_gps1_lon.iloc[0])
    p2 = A3.enu(df.unit2_gps1_lat.to_numpy(), df.unit2_gps1_lon.to_numpy(), df.unit1_gps1_lat.iloc[0], df.unit1_gps1_lon.iloc[0])
    def heading(w):
        hd = np.full(n, np.nan); sp = np.zeros(n)
        for i in range(n):
            a, b = max(i - w, 0), min(i + w, n - 1)
            while seq[a] != seq[i]: a += 1
            while seq[b] != seq[i]: b -= 1
            d = p1[b] - p1[a]; dt = max(b - a, 1) * 0.1; sp[i] = np.hypot(*d) / dt
            if np.hypot(*d) >= 0.5: hd[i] = np.arctan2(d[1], d[0])
        return hd, sp
    cand = np.where(split == 0)[0]; pick = np.sort(np.random.default_rng(5).choice(cand, 1500, replace=False))
    with Pool(8) as pool: Qs = dict(zip(pick, pool.map(A3.ng_points, [(s, lid[i], zg) for i in pick], chunksize=16)))
    out = {}
    for w in (2, 5):
        hd, sp = heading(w)
        for mirror in (False, True):
            r = p2 - p1; c, s_ = np.cos(-hd), np.sin(-hd); rel = np.c_[c * r[:, 0] - s_ * r[:, 1], s_ * r[:, 0] + c * r[:, 1]]
            if mirror: rel[:, 1] *= -1
            S = np.zeros(360); m = 0
            for i in pick:
                if not np.isfinite(hd[i]) or sp[i] < 2: continue
                rr = float(np.hypot(*rel[i])); phi = int(np.round(np.rad2deg(np.arctan2(rel[i, 1], rel[i, 0])))) % 360
                S += A3.ring_counts(Qs[i], rr)[(phi + np.arange(360)) % 360]; m += 1
            S /= max(m, 1); k = f"heading_pm{w}|mirror={mirror}"
            out[k] = dict(frames=m, best_theta=int(S.argmax()), score_best=float(S.max()), score_median=float(np.median(S)), contrast=float(S.max() / max(np.median(S), 1e-9)))
            print(k, out[k], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "verify4", "a3c_ds36.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
