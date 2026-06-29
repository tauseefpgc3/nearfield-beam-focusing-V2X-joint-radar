"""Tests for synthetic sensor generation + near-field channel reconstruction.

Covers docs/07 A3 (radar RD peak at expected range bin), A4 (LiDAR BEV non-zero at
expected location), and the near-field channel reconstruction (docs/decisions D-008).
None of these require a DeepMIMO scenario download — they use fabricated inputs.

    pytest tests/test_data_synthesis.py -v
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.polar import compute_optimal_label, fraunhofer_distance, generate_polar_codebook
from src.data.deepmimo_loader import reconstruct_near_field_channel
from src.data.lidar_processing import RANGE_X, RANGE_Y, VOXEL, crop_roi, lidar_preprocessing
from src.data.radar_processing import RANGE_RES_M, radar_preprocessing
from src.data.synthetic_lidar import simulate_point_cloud
from src.data.synthetic_radar import simulate_radar_iq

N_RX, N_CHIRPS, N_SAMPLES = 4, 256, 256


# ============================================================================
# A3 — synthetic radar: RD map peak at the expected range bin
# ============================================================================
@pytest.mark.parametrize("distance", [12.0, 30.6, 60.0])
def test_radar_rd_peak_at_expected_range_bin(distance):
    iq = simulate_radar_iq(distance=distance, velocity=0.0, azimuth_rad=0.1,
                           n_rx=N_RX, n_chirps=N_CHIRPS, n_samples=N_SAMPLES,
                           seed=123, snr_db=30.0)
    assert iq.shape == (N_RX, N_CHIRPS, N_SAMPLES)
    maps = radar_preprocessing(iq, n_rx=N_RX, n_chirps=N_CHIRPS, n_samples=N_SAMPLES)
    rd, ra = maps["RD"], maps["RA"]
    assert rd.shape == (256, 128) and ra.shape == (256, 64)
    assert np.isfinite(rd).all() and np.isfinite(ra).all()
    assert rd.min() >= 0.0 and rd.max() <= 1.0  # normalized maps
    expected_bin = int(round(distance / RANGE_RES_M))
    peak_bin = int(np.argmax(rd.sum(axis=1)))  # collapse Doppler, find range peak
    assert abs(peak_bin - expected_bin) <= 1, f"peak {peak_bin} != expected {expected_bin}"


def test_radar_clutter_is_seeded_deterministic():
    a = simulate_radar_iq(20.0, 0.0, 0.0, N_RX, N_CHIRPS, N_SAMPLES, seed=7)
    b = simulate_radar_iq(20.0, 0.0, 0.0, N_RX, N_CHIRPS, N_SAMPLES, seed=7)
    c = simulate_radar_iq(20.0, 0.0, 0.0, N_RX, N_CHIRPS, N_SAMPLES, seed=8)
    assert np.array_equal(a, b), "same seed must be deterministic"
    assert not np.array_equal(a, c), "different seed must differ"


# ============================================================================
# A4 — synthetic LiDAR: BEV non-zero near the user location
# ============================================================================
def test_lidar_bev_nonzero_at_expected_location():
    pc = simulate_point_cloud(user_xy=(10.0, 5.0), user_z=0.0, seed=42)
    assert pc.shape[1] == 4  # x, y, z, intensity
    bev = lidar_preprocessing(pc)
    assert bev.shape == (2, 150, 100)
    assert (bev >= 0).all()
    # Expected voxel for (10, 5): x->(10 - RANGE_X[0])/VOXEL, y->(5 - RANGE_Y[0])/VOXEL
    xi = int((10.0 - RANGE_X[0]) / VOXEL)
    yi = int((5.0 - RANGE_Y[0]) / VOXEL)
    window = bev[0, max(0, xi - 4):xi + 5, max(0, yi - 4):yi + 5]
    assert window.sum() > 0, "point-count channel must be non-zero around the user"


def test_lidar_seeded_deterministic():
    a = simulate_point_cloud((8.0, -3.0), 0.0, seed=5)
    b = simulate_point_cloud((8.0, -3.0), 0.0, seed=5)
    assert a.shape == b.shape and np.array_equal(a, b)


def test_crop_roi_filters_outside_points():
    pc = np.array([[0, 0, 0, 1.0],          # inside
                   [100, 0, 0, 1.0],         # outside x
                   [0, 100, 0, 1.0],         # outside y
                   [0, 0, 50, 1.0]], float)  # outside z
    out = crop_roi(pc, RANGE_X, RANGE_Y, (-2.0, 2.0))
    assert out.shape[0] == 1


# ============================================================================
# Near-field channel reconstruction (docs/decisions D-008)
# ============================================================================
def test_reconstruct_single_path_matches_steering_label():
    """A LoS near-field path -> label matches the codebook point and j>0."""
    N, f_c = 16384, 60e9
    cb, angles, ranges = generate_polar_codebook(N, f_c)
    d_f = fraunhofer_distance(f_c, N_side=128)
    i, j = 30, 4
    assert ranges[j] < d_f  # near-field
    # one dominant path exactly at codebook point (theta_i, r_j)
    h = reconstruct_near_field_channel(
        azimuths=np.array([angles[i]]), ranges=np.array([ranges[j]]),
        gains=np.array([1.0 + 0j]), N=N, f_c=f_c)
    assert h.shape == (N,)
    (yt, yr), _ = compute_optimal_label(h, cb, 64, 8)
    assert yt == i and yr == j and yr > 0


def test_reconstruct_far_path_gives_j0():
    N, f_c = 16384, 60e9
    cb, angles, ranges = generate_polar_codebook(N, f_c)
    far = 500.0  # >> d_F
    h = reconstruct_near_field_channel(np.array([angles[20]]), np.array([far]),
                                       np.array([1.0 + 0j]), N=N, f_c=f_c)
    (yt, yr), _ = compute_optimal_label(h, cb, 64, 8)
    assert yr == 0, "a far path should focus to the far-field bin"


def test_reconstruct_multipath_superposition():
    """Channel is the linear superposition of per-path near-field responses."""
    N, f_c = 16384, 60e9
    az = np.array([0.1, -0.3])
    rng = np.array([3.0, 8.0])
    g = np.array([1.0 + 0j, 0.4 - 0.2j])
    h = reconstruct_near_field_channel(az, rng, g, N=N, f_c=f_c)
    from src.codebook.steering import spherical_steering
    manual = (g[0] * spherical_steering(az[0], rng[0], N_side=128, f_c=f_c)
              + g[1] * spherical_steering(az[1], rng[1], N_side=128, f_c=f_c))
    assert np.allclose(h, manual, atol=1e-9)
