"""Synthetic LiDAR point-cloud generation (Paper 5, proof-of-concept sensor).

Implements the recipe in ``04_DATASET_SETUP.md`` §6: seeded points on a bounding box
around the user (a vehicle) + a ground plane + background scatter. Parameters in
``docs/decisions.md`` D-009/D-010. All randomness is seeded.
"""
from __future__ import annotations

import numpy as np

# --- vehicle / scene model (D-009) ------------------------------------------
CAR_SIZE_M = (4.5, 2.0, 1.5)        # length (x), width (y), height (z)
N_CAR_POINTS = 400
N_GROUND_POINTS = 1500
N_BACKGROUND_POINTS = 600
GROUND_Z = -1.8                     # ground plane height rel. to sensor
SCENE_X = (-20.0, 40.0)             # background scatter extent (matches ROI)
SCENE_Y = (-20.0, 20.0)


def simulate_point_cloud(user_xy: tuple, user_z: float = 0.0, seed: int = 0
                         ) -> np.ndarray:
    """Simulate a LiDAR point cloud for one frame with a vehicle at ``user_xy``.

    Parameters
    ----------
    user_xy : tuple
        ``(x, y)`` of the user vehicle centre in metres (sensor frame).
    user_z : float
        Vehicle base height in metres.
    seed : int
        RNG seed (deterministic; D-010).

    Returns
    -------
    np.ndarray
        ``(M, 4)`` array of points ``[x, y, z, intensity]`` (float64).
    """
    rng = np.random.default_rng(seed)
    ux, uy = float(user_xy[0]), float(user_xy[1])
    lx, ly, lz = CAR_SIZE_M

    # Vehicle: points scattered over a box centred at (ux, uy), base at user_z.
    car = np.empty((N_CAR_POINTS, 4))
    car[:, 0] = ux + rng.uniform(-lx / 2, lx / 2, N_CAR_POINTS)
    car[:, 1] = uy + rng.uniform(-ly / 2, ly / 2, N_CAR_POINTS)
    car[:, 2] = user_z + rng.uniform(0.0, lz, N_CAR_POINTS)
    car[:, 3] = rng.uniform(0.4, 1.0, N_CAR_POINTS)         # high reflectivity

    # Ground plane: low z, spread across the scene, low intensity.
    ground = np.empty((N_GROUND_POINTS, 4))
    ground[:, 0] = rng.uniform(*SCENE_X, N_GROUND_POINTS)
    ground[:, 1] = rng.uniform(*SCENE_Y, N_GROUND_POINTS)
    ground[:, 2] = GROUND_Z + rng.normal(0.0, 0.05, N_GROUND_POINTS)
    ground[:, 3] = rng.uniform(0.0, 0.2, N_GROUND_POINTS)

    # Background scatter: random clutter through the volume.
    bg = np.empty((N_BACKGROUND_POINTS, 4))
    bg[:, 0] = rng.uniform(*SCENE_X, N_BACKGROUND_POINTS)
    bg[:, 1] = rng.uniform(*SCENE_Y, N_BACKGROUND_POINTS)
    bg[:, 2] = rng.uniform(-2.0, 2.0, N_BACKGROUND_POINTS)
    bg[:, 3] = rng.uniform(0.0, 0.5, N_BACKGROUND_POINTS)

    return np.concatenate([car, ground, bg], axis=0)
