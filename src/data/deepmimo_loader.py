"""DeepMIMO O1 loader + near-field channel reconstruction (Paper 5, synthetic source).

DeepMIMO v4 produces **far-field/planar** channels (its steering is angle-only). Paper 5
needs the **near-field** channel of ``02_SYSTEM_MODEL.md`` §1.2, so we take DeepMIMO's
ray-tracing geometry (per-path gain, AoD azimuth, path-length range) and rebuild

    h = Σ_ℓ αₗ · spherical_steering(θₗ_az, rₗ)

with the *same* spherical steering used to build the codebook (matched-filter
consistency, D-005). See ``docs/decisions.md`` D-008/D-008a for the per-path range proxy
and FoV handling. Elevation is dropped to match the azimuth×range codebook scope.

NOTE: the DeepMIMO-specific extraction (:func:`paths_for_user`) is validated against a
real loaded O1 scenario in :mod:`scripts.generate_synthetic_multimodal`; the pure
reconstruction below is unit-tested with fabricated paths.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from src.codebook.steering import spherical_steering

logger = logging.getLogger(__name__)

FOV_LIMIT_RAD = np.pi / 2 - np.deg2rad(1.0)  # codebook covers |θ| < 90°; 1° guard band


def reconstruct_near_field_channel(azimuths: np.ndarray, ranges: np.ndarray,
                                   gains: np.ndarray, N: int = 1024, f_c: float = 60e9,
                                   spacing: float = 0.5) -> np.ndarray:
    """Rebuild a near-field channel as a superposition of spherical array responses.

    Parameters
    ----------
    azimuths : np.ndarray
        ``(L,)`` per-path azimuths in radians (array-boresight frame).
    ranges : np.ndarray
        ``(L,)`` per-path ranges in metres (> 0, finite).
    gains : np.ndarray
        ``(L,)`` complex per-path gains ``αₗ``.
    N : int
        Number of antenna elements (square UPA, ``N_side = sqrt(N)``).
    f_c : float
        Carrier frequency in Hz.
    spacing : float
        Element spacing in wavelengths.

    Returns
    -------
    np.ndarray
        Complex ``(N,)`` near-field channel ``h``.
    """
    azimuths = np.atleast_1d(np.asarray(azimuths, dtype=np.float64))
    ranges = np.atleast_1d(np.asarray(ranges, dtype=np.float64))
    gains = np.atleast_1d(np.asarray(gains, dtype=np.complex128))
    if not (azimuths.shape == ranges.shape == gains.shape):
        raise ValueError("azimuths, ranges, gains must have the same shape")
    if azimuths.size == 0:
        raise ValueError("no paths provided to reconstruct a channel")

    n_side = int(round(np.sqrt(N)))
    if n_side * n_side != N:
        raise ValueError(f"UPA requires square N; got N={N}")

    h = np.zeros(N, dtype=np.complex128)
    for az, r, g in zip(azimuths, ranges, gains):
        h += g * spherical_steering(az, r, N_side=n_side, spacing=spacing, f_c=f_c)
    return h


def paths_for_user(dataset, u: int, max_paths: int = 25
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract in-FoV (azimuth, range, complex-gain) per-path arrays for one user.

    Uses DeepMIMO v4 per-path fields, verified against a loaded O1 scenario:

    - ``_aod_az_rot`` — AoD azimuth (deg) **in the array-boresight frame** (rotation-
      aware). This is the frame the polar codebook lives in, so it is used directly
      with NO extra offset (raw ``aod_az`` is the global frame — wrong here).
    - ``power`` — per-path power in **dBW** (negative); converted to linear amplitude
      ``sqrt(10**(dBW/10))``.
    - ``phase`` — per-path phase in degrees.
    - ``delay`` — per-path delay in seconds; range proxy ``rₗ = c·delayₗ`` (D-008a).

    Paths outside the array FoV (``|θ| >= 90°``) are dropped and counted (no silent
    drop). Returns at most ``max_paths`` strongest paths.

    Returns
    -------
    (azimuths, ranges, gains) : arrays of shape ``(L_kept,)`` (possibly empty).
    """
    C = 299_792_458.0
    az_deg = np.asarray(dataset["_aod_az_rot"][u], dtype=np.float64)
    power_dbw = np.asarray(dataset["power"][u], dtype=np.float64)
    phase_deg = np.asarray(dataset["phase"][u], dtype=np.float64)
    delay = np.asarray(dataset["delay"][u], dtype=np.float64)

    valid = np.isfinite(power_dbw) & np.isfinite(delay) & np.isfinite(az_deg)
    az = np.deg2rad(az_deg[valid])
    az = (az + np.pi) % (2 * np.pi) - np.pi          # wrap to (-pi, pi]
    rng = C * delay[valid]
    amp = np.sqrt(10.0 ** (power_dbw[valid] / 10.0))  # dBW -> linear amplitude
    gains = amp * np.exp(1j * np.deg2rad(phase_deg[valid]))

    in_fov = np.abs(az) < FOV_LIMIT_RAD
    n_drop = int((~in_fov).sum())
    if n_drop:
        logger.debug("user %d: dropped %d/%d out-of-FoV paths", u, n_drop, az.size)
    az, rng, gains = az[in_fov], rng[in_fov], gains[in_fov]

    if az.size > max_paths:  # keep strongest paths
        order = np.argsort(np.abs(gains))[::-1][:max_paths]
        az, rng, gains = az[order], rng[order], gains[order]
    return az, rng, gains


def range_via_inter_pos(dataset, u: int, path_idx: int) -> float:
    """BS-side bounce distance for one path (alternative range proxy, D-008a).

    Distance from the BS (tx) to the **first interaction point** of the path
    (the apparent source seen by the BS). For a LoS path (no interactions) this is the
    direct BS-user distance. Used only for the D-008a range-proxy comparison.
    """
    tx = np.asarray(dataset["tx_pos"]).reshape(-1)[:3]
    # inter_pos[u] has shape (n_paths, max_inter, 3), NaN-padded.
    pts = np.asarray(dataset["inter_pos"][u], dtype=np.float64)[path_idx]  # (max_inter,3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] == 0:                               # LoS: no bounce
        rx = np.asarray(dataset["rx_pos"])[u, :3]
        return float(np.linalg.norm(rx - tx))
    return float(np.linalg.norm(finite[0] - tx))           # BS -> first interaction
