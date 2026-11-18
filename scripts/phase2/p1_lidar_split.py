"""Phase 2a (D-029) step 1c: LiDAR with heights + the fixed rotated split + SNR assignment.

LiDAR: the existing simulate_point_cloud model (400 car / 1500 ground / 600 background, same
intensities and draw order), sensor at (0,0,2.0): ground z = -2.0, car box 4.5x2.0x1.5 m on
the ground, rotated by the vehicle heading psi; rng = default_rng(20261118 + idx);
lidar_preprocessing unchanged.
Split: D-029 amendment 1 rotation + 0.5 m guard on training users. SNR assignment for
train/val users: default_rng(2030).integers(0, 3, 6000) -> index into (20, 10, 0).
Re-runs the angle-class / ring coverage check on the real labels.
"""
import json
import os
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.lidar_processing import lidar_preprocessing

OUT = os.path.join(ROOT, "data", "processed", "phase2")
BANDS = [1.0, 5.75, 10.5, 15.25, 20.0]


def point_cloud(ux, uy, psi, seed):
    rng = np.random.default_rng(seed)
    lx, ly, lz = 4.5, 2.0, 1.5
    car = np.empty((400, 4))
    xi = rng.uniform(-lx / 2, lx / 2, 400); eta = rng.uniform(-ly / 2, ly / 2, 400)
    c, s = np.cos(psi), np.sin(psi)
    car[:, 0] = ux + c * xi - s * eta; car[:, 1] = uy + s * xi + c * eta
    car[:, 2] = -2.0 + rng.uniform(0.0, lz, 400); car[:, 3] = rng.uniform(0.4, 1.0, 400)
    gr = np.empty((1500, 4))
    gr[:, 0] = rng.uniform(-20, 40, 1500); gr[:, 1] = rng.uniform(-20, 20, 1500)
    gr[:, 2] = -2.0 + rng.normal(0, 0.05, 1500); gr[:, 3] = rng.uniform(0, 0.2, 1500)
    bg = np.empty((600, 4))
    bg[:, 0] = rng.uniform(-20, 40, 600); bg[:, 1] = rng.uniform(-20, 20, 600)
    bg[:, 2] = rng.uniform(-2, 2, 600); bg[:, 3] = rng.uniform(0, 0.5, 600)
    return np.concatenate([car, gr, bg])


def roles_table():
    R = np.zeros((4, 15), int)
    for b in range(4):
        R[b, [(b + 5 * k) % 15 for k in range(3)]] = 2
        R[b, [(b + 3 + 5 * k) % 15 for k in range(2)]] = 1
    return R


def main():
    with h5py.File(os.path.join(OUT, "users.h5"), "r") as hf:
        d = hf["d"][:]; th = hf["theta"][:]; psi = hf["psi"][:]; ant = hf["ant_xyz"][:]
    n = len(d)
    L = np.stack([lidar_preprocessing(point_cloud(ant[i, 0], ant[i, 1], psi[i], 20261118 + i)) for i in range(n)])
    with h5py.File(os.path.join(OUT, "lidar.h5"), "w") as hf:
        hf["lidar_features"] = L.astype(np.float32)
    sec = np.minimum(((np.rad2deg(th) + 60) // 8).astype(int), 14)
    band = np.minimum(np.searchsorted(BANDS, d, side="right") - 1, 3)
    role = roles_table()[band, sec]
    xy = ant[:, :2]
    test = np.where(role == 2)[0]
    near = cKDTree(xy[test]).query(xy, distance_upper_bound=0.5)[0] < 0.5
    train = np.where((role == 0) & ~near)[0]; val = np.where(role == 1)[0]
    dropped = int(((role == 0) & near).sum())
    snr_idx = np.random.default_rng(2030).integers(0, 3, n)
    with h5py.File(os.path.join(OUT, "split.h5"), "w") as hf:
        hf["train"] = train; hf["val"] = val; hf["test"] = test; hf["role"] = role
        hf["snr_idx"] = snr_idx; hf.attrs["snrs"] = [20, 10, 0]
    rep = dict(n_train=len(train), n_val=len(val), n_test=len(test), guard_dropped_train=dropped,
               min_test_to_train_m=float(cKDTree(xy[train]).query(xy[test])[0].min()))
    for f in (60, 140):
        with h5py.File(os.path.join(OUT, f"channels_{f}GHz.h5"), "r") as hf:
            yt = hf["y_theta"][:]; yr = hf["y_r"][:]; adv = hf["adv_db"][:]
        rep[str(f)] = dict(
            angle_test_only=sorted(set(yt[test].tolist()) - set(yt[train].tolist())),
            n_test_users_with_test_only_angle=int(np.isin(yt[test], list(set(yt[test]) - set(yt[train]))).sum()),
            ring_test_only=sorted(set(yr[test].tolist()) - set(yr[train].tolist())),
            n_angle_test=len(set(yt[test])), n_angle_train=len(set(yt[train])),
            rings_test=sorted(set(yr[test].tolist())), rings_train=sorted(set(yr[train].tolist())),
            frac_adv_ge3_test=float((adv[test] >= 3).mean()), n_adv_ge3_test=int((adv[test] >= 3).sum()),
            genie_dft_mean_test=float(adv[test].mean()), n_test_d_ge_1p6=int((d[test] >= 1.6).sum()))
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(ROOT, "results", "phase2", "checks_split.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
