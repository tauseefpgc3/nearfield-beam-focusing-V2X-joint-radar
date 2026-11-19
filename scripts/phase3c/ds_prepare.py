"""D-033 C2: DeepSense 6G scenarios 8, 31, 36 -> labels, sequence split, LiDAR BEV.

    python scripts/phase3c/ds_prepare.py {8|31|36}

Grid rule (design addendum, from training frames only): horizontal range clip
R_c = min(99th percentile of training-point horizontal range, 51.2 m); ROI = the [0.5, 99.5]
percentile box of training x and y within R_c; cell = smallest of CELLS with both sides
<= 128 cells; sides rounded up to a multiple of 8. Height channel: z above z_ground (2nd
percentile of training z within R_c), clipped at 0, / 2. Intensity / its training 99th
percentile, clipped to 1. 2-D scan: height and intensity channels are 0.
Writes data/processed/phase3c/ds{S}.h5 and results/phase3c/c2_design_ds{S}.json.
"""
import json, os, sys
from multiprocessing import Pool
import h5py, numpy as np, pandas as pd, scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "deepsense_raw")
OUT = os.path.join(ROOT, "data", "processed", "phase3c"); RES = os.path.join(ROOT, "results", "phase3c")
CELLS = (0.2, 0.25, 0.4, 0.5, 0.8, 1.0, 1.25); MAXSIDE = 128; SPLIT_SEED = 20261118
SC = {8: dict(root=os.path.join(RAW, "ex", "s8", "DEV[95%]"), csv="scenario8.csv"),
      31: dict(root=os.path.join(RAW, "ex", "s31", "scenario31"), csv="scenario31_dev.csv"),
      36: dict(root=os.path.join(RAW, "ex", "s36", "scenario36"), csv=os.path.join(RAW, "Scenario36", "scenario36.csv")),
      # D-036 (loader only): scenarios 32-34, same testbed and file format as 31
      32: dict(root=os.path.join(RAW, "ex", "s32", "scenario32"), csv="scenario32_dev.csv"),
      33: dict(root=os.path.join(RAW, "ex", "s33", "scenario33"), csv="scenario33_dev.csv"),
      34: dict(root=os.path.join(RAW, "ex", "s34", "scenario34"), csv="scenario34.csv")}
S31_FMT = (31, 32, 33, 34)


def index(s):
    c = SC[s]; df = pd.read_csv(os.path.join(c["root"], c["csv"]))
    if s == 8:
        lid = df.unit1_lidar; pw = [[p] for p in df.unit1_pwr_60ghz]
    elif s in S31_FMT:
        lid = df.unit1_lidar; pw = [[p] for p in df.unit1_pwr_60ghz]
    else:
        lid = df.unit1_lidar1; pw = [[df[f"unit1_pwr{k}"][i] for k in range(1, 5)] for i in range(len(df))]
    return df, [os.path.join(c["root"], p) for p in lid], [[os.path.join(c["root"], q) for q in p] for p in pw]


def load_points(args):
    """-> (N, 4) float32: x, y, z (m, sensor frame, z up), intensity (raw)."""
    s, path = args
    if s == 8:
        d = sio.loadmat(path)["data"]; r, a = d[:, 0], d[:, 1]; k = r > 0
        return np.c_[r[k] * np.cos(a[k]), r[k] * np.sin(a[k]), np.zeros(k.sum()), np.zeros(k.sum())].astype(np.float32)
    if s in S31_FMT:
        with open(path) as f:
            n = 0
            while True:
                l = f.readline()
                if l.startswith("element vertex"): n = int(l.split()[-1])
                if l.startswith("end_header"): break
            return np.loadtxt(f, max_rows=n, dtype=np.float32).reshape(-1, 4)
    d = pd.read_csv(path, comment=None, header=0).to_numpy(np.float64)   # TIMESTAMP, RANGE, SIGNAL, NEAR_IR, REFLECTIVITY, X, Y, Z (mm)
    k = d[:, 1] > 0
    return np.c_[d[k, 5:8] / 1000.0, d[k, 4]].astype(np.float32)


def load_power(paths):
    return np.concatenate([np.loadtxt(p, dtype=np.float64).ravel() for p in paths])


def split_by_sequence(seq):
    rng = np.random.default_rng(SPLIT_SEED); u = np.unique(seq); rng.shuffle(u)
    cnt = {q: int((seq == q).sum()) for q in u}; n = len(seq); out = np.empty(n, np.int8); cum = 0
    for q in u:
        f = cum / n; out[seq == q] = 0 if f < 0.6 else (1 if f < 0.8 else 2); cum += cnt[q]
    return out


def design(clouds, two_d):
    P = np.concatenate(clouds); hr = np.hypot(P[:, 0], P[:, 1])
    Rc = float(min(np.percentile(hr, 99), 51.2)); Q = P[hr <= Rc]
    x0, x1 = np.percentile(Q[:, 0], [0.5, 99.5]); y0, y1 = np.percentile(Q[:, 1], [0.5, 99.5])
    for cell in CELLS:
        nx = int(np.ceil((x1 - x0) / cell)); ny = int(np.ceil((y1 - y0) / cell))
        nx, ny = 8 * int(np.ceil(nx / 8)), 8 * int(np.ceil(ny / 8))
        if nx <= MAXSIDE and ny <= MAXSIDE: break
    zg = 0.0 if two_d else float(np.percentile(Q[:, 2], 2)); imax = 1.0 if two_d else float(max(np.percentile(Q[:, 3], 99), 1e-6))
    return dict(R_clip=Rc, x0=float(x0), y0=float(y0), cell=cell, NX=nx, NY=ny, z_ground=zg, i99=imax, two_d=two_d)


def bev(pc, g):
    ix = np.floor((pc[:, 0] - g["x0"]) / g["cell"]).astype(int); iy = np.floor((pc[:, 1] - g["y0"]) / g["cell"]).astype(int)
    m = (ix >= 0) & (ix < g["NX"]) & (iy >= 0) & (iy < g["NY"]) & (np.hypot(pc[:, 0], pc[:, 1]) <= g["R_clip"])
    ix, iy, q = ix[m], iy[m], pc[m]; N = g["NX"] * g["NY"]; f = ix * g["NY"] + iy
    n = np.bincount(f, minlength=N); h = np.zeros(N)
    if not g["two_d"]:
        np.maximum.at(h, f, np.clip(q[:, 2] - g["z_ground"], 0, None))
        it = np.bincount(f, weights=np.clip(q[:, 3] / g["i99"], 0, 1), minlength=N); it = np.where(n > 0, it / np.maximum(n, 1), 0)
    else:
        it = np.zeros(N)
    return np.stack([np.log1p(n), h / 2.0, it]).reshape(3, g["NX"], g["NY"]).astype(np.float16)


def main(s):
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    df, lid, pw = index(s); n = len(df)
    seq = df.seq_index.to_numpy(); split = split_by_sequence(seq)
    with Pool(8) as pool:
        P = np.stack(pool.map(load_power, pw, chunksize=64))
        print("power", P.shape, flush=True)
        tr = np.where(split == 0)[0]
        sample = pool.map(load_points, [(s, lid[i]) for i in tr[::5]], chunksize=16)
        g = design(sample, s == 8); print("grid", g, flush=True)
        with h5py.File(os.path.join(OUT, f"ds{s}.h5"), "w") as hf:
            D = hf.create_dataset("bev", (n, 3, g["NX"], g["NY"]), np.float16, chunks=(1, 3, g["NX"], g["NY"]))
            npts = np.zeros(n, np.int32)
            for b in range(0, n, 512):
                cl = pool.map(load_points, [(s, lid[i]) for i in range(b, min(b + 512, n))], chunksize=16)
                D[b:b + len(cl)] = np.stack([bev(c, g) for c in cl]); npts[b:b + len(cl)] = [len(c) for c in cl]
                print(b, flush=True)
            hf["power"] = P.astype(np.float32); hf["label"] = P.argmax(1).astype(np.int16); hf["seq"] = seq; hf["split"] = split
            hf["row"] = np.arange(n); hf["n_points"] = npts
            for k, v in g.items(): hf.attrs[k] = v
    lab = P.argmax(1)
    info = dict(scenario=s, frames=n, sequences=int(len(np.unique(seq))), beams=int(P.shape[1]),
                split_frames={k: int((split == i).sum()) for i, k in enumerate(("train", "val", "test"))},
                split_sequences={k: int(len(np.unique(seq[split == i]))) for i, k in enumerate(("train", "val", "test"))},
                grid=g, points_per_frame_median=float(np.median(npts)),
                csv_beam_vs_power_argmax=(None if s == 36 else float(np.mean(df["unit1_beam_index" if s == 8 else "unit1_beam"].to_numpy() - 1 == lab))),
                labels_used=int(len(np.unique(lab[split == 0]))))
    if s == 36: info["csv_overall_beam_vs_power_argmax"] = float(np.mean(df["unit1_overall-beam"].to_numpy() == lab))
    json.dump(info, open(os.path.join(RES, f"c2_design_ds{s}.json"), "w"), indent=1); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
