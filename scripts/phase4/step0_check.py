"""D-036 Step 0: data check for DeepSense scenarios 32-34 (run before any training).

    python scripts/phase4/step0_check.py 32 33 34

1. ds_prepare (Phase 3c pipeline, loader adapted only) -> data/processed/phase3c/ds{S}.h5 and
   results/phase3c/c2_design_ds{S}.json (labels = arg-max of power; CSV beam agreement).
2. A3 alignment (scripts/verify4/a3_deepsense_alignment.py) -> results/verify4/a3_ds{S}.json.
3. Visible share on test frames (A3b rule: rotation at lag 0 from a3_ds{S}.json; >= 3 non-ground
   points within 2 m of the GPS-projected receiver).
-> results/phase4/step0_ds{S}.json
"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("phase3c", "verify4"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import ds_prepare as DP, a3_deepsense_alignment as A3
OUT = os.path.join(ROOT, "results", "phase4"); os.makedirs(OUT, exist_ok=True)


def visible(s):
    a3 = json.load(open(os.path.join(ROOT, "results", "verify4", f"a3_ds{s}.json")))
    df, lid, _ = DP.index(s); rel, okg = A3.gps(s, df)
    with h5py.File(os.path.join(DP.OUT, f"ds{s}.h5"), "r") as h:
        P = h["power"][:]; ok = np.isfinite(P).all(1); sp = h["split"][:]; seq = h["seq"][:]; zg = float(h.attrs["z_ground"])
    te = np.where((sp == 2) & ok)[0]; th = np.deg2rad(a3["geometry_fit"]["theta_at_lag0_deg"])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); p = (Rm @ rel[te].T).T
    with Pool(8) as pool: Qs = pool.map(A3.ng_points, [(s, lid[i], zg) for i in te], chunksize=16)
    near = np.array([(np.hypot(*(Q - q).T) <= 2).sum() for Q, q in zip(Qs, p)]); vis = (near >= 3) & okg[te]
    return dict(n_test=int(len(te)), test_sequences=int(len(np.unique(seq[te]))), n_visible=int(vis.sum()), share_visible=float(vis.mean()),
                visible_sequences=int(len(np.unique(seq[te][vis]))), finite_power_frames=int(ok.sum()))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        root = DP.SC[s]["root"]
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)
        counts = {m: len(os.listdir(os.path.join(root, "unit1", m))) for m in ("lidar_data", "mmWave_data")} | {"unit2_GPS": len(os.listdir(os.path.join(root, "unit2", "GPS_data")))}
        DP.main(s); A3.run(s)
        R = dict(scenario=s, extracted_size_gb=size / 1e9, file_counts=counts,
                 design=json.load(open(os.path.join(DP.RES, f"c2_design_ds{s}.json"))),
                 a3={k: v for k, v in json.load(open(os.path.join(A3.OUT, f"a3_ds{s}.json"))).items() if k != "test_frames"},
                 visible=visible(s))
        json.dump(R, open(os.path.join(OUT, f"step0_ds{s}.json"), "w"), indent=1); print("STEP0", s, json.dumps(R["visible"]), flush=True)
