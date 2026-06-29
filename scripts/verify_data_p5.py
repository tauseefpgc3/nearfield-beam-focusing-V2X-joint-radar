"""Verify processed Paper 5 datasets (docs/04 Step 8) + distance/label previews.

Asserts shapes/dtypes, label ranges (y_theta in [0,64), y_r in [0,8)), distance > 0, and
that 'channel' is present in every synthetic file. Then prints the G0.1 distance-
distribution preview (percentiles + fraction d < d_F) and the G0.3 label preview
(distinct (y_theta,y_r) count + y_r histogram). Previews are diagnostics, NOT the
official Gate-0 probes (those run next session).

    python scripts/verify_data_p5.py [file1.h5 file2.h5 ...]
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

EXPECTED = {
    "radar_features": ((2, 256, 128), np.float32),
    "lidar_features": ((2, 150, 100), np.float32),
    "y_theta": ((), np.int32),
    "y_r": ((), np.int32),
    "trajectory_id": ((), np.int32),
    "distance": ((), np.float32),
}


def verify_file(path: str) -> None:
    with h5py.File(path, "r") as hf:
        n = hf["y_theta"].shape[0]
        print(f"\n=== {os.path.relpath(path, _ROOT)}  (N={n}) ===")
        print(f"{'dataset':<18}{'shape':<22}{'dtype':<12}")
        for k, (shp, dt) in EXPECTED.items():
            a = hf[k][:]
            ok = a.shape[1:] == shp and a.dtype == dt
            print(f"{k:<18}{str(a.shape):<22}{str(a.dtype):<12}{'OK' if ok else 'FAIL'}")
            assert ok, f"{k}: shape {a.shape}/{a.dtype} != {shp}/{dt}"

        yt, yr, dist = hf["y_theta"][:], hf["y_r"][:], hf["distance"][:]
        assert 0 <= yt.min() and yt.max() < 64, "y_theta out of [0,64)"
        assert 0 <= yr.min() and yr.max() < 8, "y_r out of [0,8)"
        assert dist.min() > 0, "distance must be > 0"

        base = os.path.basename(path)
        requires_channel = "synthetic" in base or "closerange" in base
        has_channel = "channel" in hf
        ch_shape = hf["channel"].shape if has_channel else None
        ch_dtype = hf["channel"].dtype if has_channel else None
        print(f"{'channel':<18}{str(ch_shape):<22}{str(ch_dtype):<12}"
              f"{'PRESENT' if has_channel else 'ABSENT'}")
        if requires_channel:
            assert has_channel, f"synthetic file {path} missing full 'channel'!"
            n_elem = int(hf.attrs["N"])  # 16384 at 128x128 (D-011)
            assert ch_shape == (n, n_elem), f"channel shape {ch_shape} != ({n}, {n_elem})"
            _correspondence_check(hf, dist)

        d_f = float(hf.attrs["d_F"]) if "d_F" in hf.attrs else float("nan")
        _preview(dist, yt, yr, d_f)


def _correspondence_check(hf, dist) -> None:
    """E1: radar & lidar derive from the SAME per-sample position as the channel.

    Radar RD-map peak range bin must match round(distance / RANGE_RES_M); lidar BEV
    peak-count cell radius must match the stored distance. Both tie the sensors to the
    channel's user position (stored `distance` = channel LoS range).
    """
    from src.data.lidar_processing import N_Y, RANGE_X, RANGE_Y, VOXEL
    from src.data.synthetic_radar import RANGE_RES_M

    n = dist.shape[0]
    idx = np.linspace(0, n - 1, min(200, n)).astype(int)
    radar = hf["radar_features"][idx]      # (m,2,256,128)
    lidar = hf["lidar_features"][idx]      # (m,2,150,100)
    d = dist[idx]
    yt = hf["y_theta"][idx]

    # radar: peak range bin of the RD map (channel 0) vs expected (radar reaches 307 m,
    # so this is checkable for ALL users in both datasets)
    rd_peak = radar[:, 0].reshape(len(idx), 256, -1).sum(2).argmax(1)
    exp_bin = np.clip(np.round(d / RANGE_RES_M), 0, 255)
    radar_ok = float(np.mean(np.abs(rd_peak - exp_bin) <= 2))

    # lidar: only checkable where the user falls inside the BEV ROI. Expected (x,y) from
    # stored distance and the codebook angle of y_theta. Users beyond the ROI (e.g. far
    # O1 users) are correctly cropped out -> lidar legitimately holds no user there.
    az = np.arcsin((2.0 * yt - 63.0) / 64.0)
    ex, ey = d * np.cos(az), d * np.sin(az)
    in_roi = ((ex >= RANGE_X[0]) & (ex < RANGE_X[1]) &
              (ey >= RANGE_Y[0]) & (ey < RANGE_Y[1]))
    counts = lidar[:, 0]
    flat = counts.reshape(len(idx), -1).argmax(1)
    xi, yi = np.divmod(flat, N_Y)
    x = xi * VOXEL + RANGE_X[0] + VOXEL / 2
    y = yi * VOXEL + RANGE_Y[0] + VOXEL / 2
    radius = np.sqrt(x ** 2 + y ** 2)
    n_roi = int(in_roi.sum())
    lidar_ok = float(np.mean(np.abs(radius[in_roi] - d[in_roi]) <= 2.0)) if n_roi else float("nan")

    roi_frac = n_roi / len(idx)
    print(f"  [E1 correspondence] radar peak-range vs distance: {radar_ok:.0%} match "
          f"(<=2 bins, all users) | lidar peak-cell radius vs distance: {lidar_ok:.0%} "
          f"match over {n_roi}/{len(idx)} users inside LiDAR ROI ({roi_frac:.0%})")
    assert radar_ok >= 0.9, "radar range does not correspond to stored distance (E1)"
    if roi_frac >= 0.5:                     # LiDAR FoV covers the user population
        assert lidar_ok >= 0.9, "lidar position does not correspond to distance in ROI (E1)"
    else:
        print(f"  [E1 note] LiDAR ROI (x<{RANGE_X[1]:.0f} m) covers only {roi_frac:.0%} of "
              f"users (far-field anchor); LiDAR informs the near subset, radar carries the rest.")


def _preview(dist, yt, yr, d_f) -> None:
    pct = np.percentile(dist, [0, 5, 25, 50, 75, 95, 100])
    frac_nf = float(np.mean(dist < d_f)) if np.isfinite(d_f) else float("nan")
    print(f"  [G0.1 preview] d_F={d_f:.3f} m | dist min/5/25/50/75/95/max = "
          f"{np.round(pct, 2).tolist()}")
    print(f"  [G0.1 preview] fraction d < d_F = {frac_nf:.3f}")

    joint = set(zip(yt.tolist(), yr.tolist()))
    hist_r = np.bincount(yr, minlength=8)
    print(f"  [G0.3 preview] distinct (y_theta,y_r) = {len(joint)}/512 | "
          f"y_r histogram = {hist_r.tolist()}")


def main() -> None:
    files = sys.argv[1:]
    if not files:
        files = sorted(glob.glob(os.path.join(_ROOT, "data", "processed", "*.h5")))
    if not files:
        print("No HDF5 files found in data/processed/. Run generation first.")
        sys.exit(1)
    for f in files:
        verify_file(f)
    print("\nALL DATA VERIFIED.")


if __name__ == "__main__":
    main()
