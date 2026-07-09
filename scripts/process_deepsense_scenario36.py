"""Process DeepSense6G Scenario 36 (V2V, 4-array format) -> HDF5.

Option A (per-array 64-beam, apples-to-apples with scenario 31): use radar1 + pwr1 only.
Radar cube axis VERIFIED for scenario 36: (rx=4, chirps=256, samples=128) — range is the
LAST axis (range-FFT sharpness 18.6 vs 14.0), same convention as the scenario-31 fix.

  y_theta = unit1_pwr1_best-beam       (64-beam; scenario 36's best-beam is 0-INDEXED,
                                        verified argmax==csv 40/40 — unlike scenario 31's
                                        1-indexed column. The label gate caught this.)
  y_r     = 0                          (far-field; DeepSense has no near-field info)
  best_beam = argmax(pwr1)             (independent power-file argmax for the G0.4 check)
  distance  = haversine(unit1_gps1, unit2_gps1)   (BOTH per-sample — mobile V2V)
  radar1 (4,256,128) -> (2,256,128) RD/RA, same representation as scenario 31. No LiDAR.

    python scripts/process_deepsense_scenario36.py --stride 3
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.radar_processing import radar_preprocessing

BASE = os.path.join(_ROOT, "data", "deepsense", "Scenario36")
SCEN = os.path.join(BASE, "scenario36")          # extracted root (paths are relative to it)
EARTH_R = 6_371_000.0


def haversine_m(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(x))


def read_gps(path):
    v = [float(x) for x in open(path).read().split()]
    return v[0], v[1]


def radar_feature(mat_path):
    c = np.asarray(sio.loadmat(mat_path)["data"])     # (4,256,128)=(rx,chirps,samples)
    if c.shape != (4, 256, 128):
        raise ValueError(f"unexpected radar shape {c.shape} in {mat_path}")
    maps = radar_preprocessing(c, n_rx=4, n_chirps=256, n_samples=128)  # range over last axis
    feat = np.zeros((2, 256, 128), dtype=np.float32)
    feat[0] = maps["RD"]
    feat[1, :, :maps["RA"].shape[1]] = maps["RA"]
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3, help="process every Nth csv row")
    ap.add_argument("--out", default=os.path.join(_ROOT, "data", "processed",
                                                  "deepsense_scenario36_full.h5"))
    args = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(BASE, "scenario36.csv"))))[::args.stride]
    rp = lambda p: os.path.join(SCEN, p.lstrip("./").replace("/", os.sep))

    radar, yt, yr, dist, traj, best = [], [], [], [], [], []
    n_skip = 0
    for r in tqdm(rows, desc="scenario36", unit="frame"):
        try:
            pwr = np.loadtxt(rp(r["unit1_pwr1"]))
            rf = radar_feature(rp(r["unit1_radar1"]))
            u1 = read_gps(rp(r["unit1_gps1"]))
            u2 = read_gps(rp(r["unit2_gps1"]))
        except Exception as e:
            n_skip += 1
            if n_skip <= 3:
                print(f"  skip {r['abs_index']}: {e!r}")
            continue
        radar.append(rf)
        yt.append(int(r["unit1_pwr1_best-beam"]))   # 0-indexed in scenario 36 (gate-verified)
        yr.append(0)
        best.append(int(np.argmax(pwr)))
        dist.append(haversine_m(u1[0], u1[1], u2[0], u2[1]))
        traj.append(int(r.get("seq_index", "0") or 0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    yt_a = np.array(yt, np.int32); yr_a = np.array(yr, np.int32)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("radar_features", data=np.stack(radar))
        hf.create_dataset("y_theta", data=yt_a)
        hf.create_dataset("y_r", data=yr_a)
        hf.create_dataset("y_theta_gated", data=yt_a)
        hf.create_dataset("y_r_gated", data=yr_a)
        hf.create_dataset("best_beam", data=np.array(best, np.int32))
        hf.create_dataset("distance", data=np.array(dist, np.float32))
        hf.create_dataset("trajectory_id", data=np.array(traj, np.int32))
        hf.attrs.update(dict(source="deepsense6g_scenario36_v2v", B_theta=64,
                             has_channel=False, label_trust="low",
                             note="radar1+pwr1 per-array 64-beam; y_r=0 far-field; mobile V2V"))
    print(f"\nwrote {os.path.relpath(args.out,_ROOT)}: {len(yt)} frames ({n_skip} skipped) | "
          f"distance {np.min(dist):.1f}-{np.max(dist):.1f} m | distinct seq {len(set(traj))}")


if __name__ == "__main__":
    main()
