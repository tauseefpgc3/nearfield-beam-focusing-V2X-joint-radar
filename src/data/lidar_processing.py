"""LiDAR feature extraction: point cloud -> 2-channel BEV (Paper 5).

Implements ``02_SYSTEM_MODEL.md`` §2.2: crop ROI, voxelize at 0.4 m, build a
``(2, 150, 100)`` bird's-eye-view with channel 0 = point count, channel 1 = mean
intensity. ROI / voxel constants in ``docs/decisions.md`` D-009. Signatures match
``05_REPO_STRUCTURE.md``.
"""
from __future__ import annotations

import numpy as np

# --- ROI / BEV grid (D-009) -------------------------------------------------
VOXEL = 0.4
RANGE_X = (-20.0, 40.0)            # -> 150 bins
RANGE_Y = (-20.0, 20.0)            # -> 100 bins
RANGE_Z = (-2.0, 2.0)
N_X = int(round((RANGE_X[1] - RANGE_X[0]) / VOXEL))   # 150
N_Y = int(round((RANGE_Y[1] - RANGE_Y[0]) / VOXEL))   # 100


def crop_roi(point_cloud: np.ndarray, x_range, y_range, z_range) -> np.ndarray:
    """Keep only points inside the axis-aligned ROI box ``[lo, hi)`` per axis.

    Parameters
    ----------
    point_cloud : np.ndarray
        ``(M, >=3)`` array with columns ``[x, y, z, ...]``.
    x_range, y_range, z_range : tuple
        ``(lo, hi)`` bounds in metres.

    Returns
    -------
    np.ndarray
        Filtered point cloud (same column layout).
    """
    pc = np.asarray(point_cloud)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"point_cloud must be (M, >=3); got {pc.shape}")
    m = ((pc[:, 0] >= x_range[0]) & (pc[:, 0] < x_range[1]) &
         (pc[:, 1] >= y_range[0]) & (pc[:, 1] < y_range[1]) &
         (pc[:, 2] >= z_range[0]) & (pc[:, 2] < z_range[1]))
    return pc[m]


def lidar_preprocessing(point_cloud: np.ndarray) -> np.ndarray:
    """Convert a point cloud to a ``(2, 150, 100)`` BEV (count, mean intensity).

    Parameters
    ----------
    point_cloud : np.ndarray
        ``(M, 4)`` array ``[x, y, z, intensity]``.

    Returns
    -------
    np.ndarray
        ``(2, N_X, N_Y)`` float32 BEV; channel 0 = point count, channel 1 = mean
        intensity per voxel. Non-negative.
    """
    pc = crop_roi(point_cloud, RANGE_X, RANGE_Y, RANGE_Z)
    bev = np.zeros((2, N_X, N_Y), dtype=np.float32)
    if pc.shape[0] == 0:
        return bev

    xi = ((pc[:, 0] - RANGE_X[0]) / VOXEL).astype(np.int64)
    yi = ((pc[:, 1] - RANGE_Y[0]) / VOXEL).astype(np.int64)
    np.clip(xi, 0, N_X - 1, out=xi)
    np.clip(yi, 0, N_Y - 1, out=yi)
    flat = xi * N_Y + yi

    counts = np.bincount(flat, minlength=N_X * N_Y).astype(np.float32)
    intensity_sum = np.bincount(flat, weights=pc[:, 3], minlength=N_X * N_Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_int = np.where(counts > 0, intensity_sum / np.maximum(counts, 1), 0.0)

    bev[0] = counts.reshape(N_X, N_Y)
    bev[1] = mean_int.reshape(N_X, N_Y).astype(np.float32)
    return bev
