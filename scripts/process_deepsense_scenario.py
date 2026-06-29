"""Process a DeepSense6G *development-format* scenario -> HDF5 (Paper 5 Batch 3).

Targets the scenario31-style layout (single 60 GHz power array, single radar/lidar,
`unit1_beam` best-beam label, `unit1_loc`/`unit2_loc` GPS). DeepSense has NO channel, so
polar labels are RECONSTRUCTED (low-trust, G0.4):

  - y_theta = unit1_beam - 1   (the trustworthy part: the max-power DFT beam, 64 beams
              -> 64 angle bins). `best_beam` = argmax of the raw power file (independent
              of the CSV) is stored too, so the gate can check power<->label agreement.
  - y_r     = 0  (far-field). DeepSense is far-field with no near-field range info; this
              is EXPECTED and means DeepSense validates angle + far-field beam selection,
              not near-field gain. Flagged low-trust.

Sensors -> the SAME representation as synthetic (so the synthetic-trained radar model
transfers): radar (4,256,250) ADC -> (2,256,128) RD/RA; LiDAR .ply -> (2,150,100) BEV.
NO `channel` dataset is written.

    python scripts/process_deepsense_scenario.py \
        --scenario data/deepsense/scenario31_new/scenario31 \
        --csv scenario31_dev.csv --out data/processed/deepsense_scenario31.h5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.lidar_processing import lidar_preprocessing
from src.data.radar_processing import radar_preprocessing

B_THETA = 64
EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * EARTH_R * np.arcsin(np.sqrt(a)))


def read_gps(path):
    vals = [float(x) for x in open(path).read().split()]
    return vals[0], vals[1]                      # lat, lon


def read_ply_xyzi(path) -> np.ndarray:
    with open(path) as f:
        line = ""
        while not line.startswith("end_header"):
            line = f.readline()
            if not line:
                raise ValueError(f"no end_header in {path}")
        pts = np.loadtxt(f)                       # (M, 4) x y z intensity
    return pts.reshape(-1, 4) if pts.size else np.zeros((0, 4))


def radar_feature(npy_path) -> np.ndarray:
    # DeepSense cube axis order CONFIRMED by range-FFT sharpness (Batch-3 diagnosis):
    # (rx=4, chirps=256, samples=250) — range/fast-time is the LAST axis (250 samples).
    # (The earlier transpose ran the range FFT over chirps -> structureless RA: a bug.)
    r = np.load(npy_path)                         # (4, 256, 250) = (rx, chirps, samples)
    if r.shape != (4, 256, 250):
        raise ValueError(f"unexpected radar shape {r.shape} in {npy_path}")
    maps = radar_preprocessing(r, n_rx=4, n_chirps=256, n_samples=250)
    feat = np.zeros((2, 256, 128), dtype=np.float32)
    feat[0] = maps["RD"]
    feat[1, :, :maps["RA"].shape[1]] = maps["RA"]
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="scenario root dir")
    ap.add_argument("--csv", required=True, help="dev csv filename (relative to scenario)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    root = args.scenario
    rows = list(csv.DictReader(open(os.path.join(root, args.csv))))
    if args.max_samples:
        rows = rows[:args.max_samples]
    rp = lambda p: os.path.join(root, p.lstrip("./").replace("/", os.sep))

    u1_lat, u1_lon = read_gps(rp(rows[0]["unit1_loc"]))   # RSU static position
    radar, lidar, yt, yr, dist, traj, best = [], [], [], [], [], [], []
    n_skip = 0
    for r in tqdm(rows, desc="deepsense", unit="frame"):
        try:
            pwr = np.loadtxt(rp(r["unit1_pwr_60ghz"]))
            rf = radar_feature(rp(r["unit1_radar"]))
            lf = lidar_preprocessing(read_ply_xyzi(rp(r["unit1_lidar"])))
            u2_lat, u2_lon = read_gps(rp(r["unit2_loc"]))
        except Exception as e:                    # no silent fallback: count + report
            n_skip += 1
            if n_skip <= 3:
                print(f"  skip frame {r['index']}: {e!r}")
            continue
        radar.append(rf); lidar.append(lf)
        yt.append(int(r["unit1_beam"]) - 1)       # official best beam (0-indexed)
        yr.append(0)                              # far-field (low-trust, D-016)
        best.append(int(np.argmax(pwr)))          # power-file argmax (independent check)
        dist.append(haversine_m(u1_lat, u1_lon, u2_lat, u2_lon))
        traj.append(int(r.get("seq_index", "0") or 0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("radar_features", data=np.stack(radar))
        hf.create_dataset("lidar_features", data=np.stack(lidar))
        yt_arr = np.array(yt, np.int32); yr_arr = np.array(yr, np.int32)
        hf.create_dataset("y_theta", data=yt_arr)
        hf.create_dataset("y_r", data=yr_arr)
        # gated == raw for DeepSense (no gain gate; no channel) — for train_single.
        hf.create_dataset("y_theta_gated", data=yt_arr)
        hf.create_dataset("y_r_gated", data=yr_arr)
        hf.create_dataset("best_beam", data=np.array(best, np.int32))
        hf.create_dataset("distance", data=np.array(dist, np.float32))
        hf.create_dataset("trajectory_id", data=np.array(traj, np.int32))
        hf.attrs.update(dict(source="deepsense6g_dev", scenario=os.path.basename(root),
                             B_theta=B_THETA, has_channel=False, label_trust="low",
                             note="y_r=0 far-field (no near-field info); angle=best beam"))
    print(f"\nwrote {os.path.relpath(args.out,_ROOT)}: {len(yt)} frames "
          f"({n_skip} skipped) | distance {np.min(dist):.1f}-{np.max(dist):.1f} m | "
          f"distinct seq {len(set(traj))}")


if __name__ == "__main__":
    main()
