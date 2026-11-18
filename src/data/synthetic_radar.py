"""Synthetic FMCW radar IF-signal generation (Paper 5, proof-of-concept sensor).

Implements the recipe in ``04_DATASET_SETUP.md`` §6: model the user as a point
reflector at a range bin determined by its distance, add seeded clutter reflectors and
receiver noise. Parameters are documented in ``docs/decisions.md`` D-009/D-010 (proxy
sensor; tunable). All randomness is seeded (no global RNG).

The generated raw IQ has shape ``(n_rx, n_chirps, n_samples)`` and is consumed by
:func:`src.data.radar_processing.radar_preprocessing`.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# --- radar configuration (D-009) --------------------------------------------
# RANGE_RES chosen so 256 range bins span ~307 m (O1 user distances reach ~292 m).
RANGE_RES_M = 1.2                       # range resolution; 256 bins -> 307.2 m
MAX_RANGE_M = RANGE_RES_M * 256         # 307.2 m
VELOCITY_RES_MPS = 0.25                 # Doppler resolution (proxy)
SPACING_LAMBDA = 0.5                    # rx element spacing (half-wavelength)
N_CLUTTER_DEFAULT = 12                  # seeded clutter reflectors per frame
CLUTTER_AMP = 0.25                      # clutter amplitude rel. to target (=1.0)


def simulate_radar_iq(distance: float, velocity: float, azimuth_rad: float,
                      n_rx: int = 4, n_chirps: int = 256, n_samples: int = 256,
                      seed: int = 0, n_clutter: int = N_CLUTTER_DEFAULT,
                      snr_db: float = 20.0) -> np.ndarray:
    """Simulate the raw FMCW radar IQ cube for one frame.

    The user is a point reflector whose **range FFT bin** is
    ``round(distance / RANGE_RES_M)``; Doppler bin follows ``velocity`` and the
    receive phase ramp follows ``azimuth_rad`` across the ``n_rx`` ULA. Seeded clutter
    reflectors (random range/azimuth) and complex-Gaussian noise are added.

    Parameters
    ----------
    distance : float
        User range in metres (> 0). Clamped to ``MAX_RANGE_M`` with a warning.
    velocity : float
        Radial velocity in m/s (sign = approaching/receding).
    azimuth_rad : float
        User azimuth in radians (array-boresight frame).
    n_rx, n_chirps, n_samples : int
        Receive elements, chirps per frame, fast-time samples per chirp.
    seed : int
        RNG seed for clutter + noise (deterministic; D-010).
    n_clutter : int
        Number of seeded clutter reflectors.
    snr_db : float
        Target SNR in dB (sets the noise level relative to unit target amplitude).

    Returns
    -------
    np.ndarray
        Complex ``(n_rx, n_chirps, n_samples)`` IQ cube.
    """
    if distance <= 0:
        raise ValueError(f"distance must be positive, got {distance}")
    rng = np.random.default_rng(seed)

    range_bin = int(round(distance / RANGE_RES_M))
    if range_bin > n_samples - 1:
        logger.warning("radar: distance %.1f m exceeds max range %.1f m; clamping bin",
                       distance, MAX_RANGE_M)
        range_bin = n_samples - 1

    samp = np.arange(n_samples)
    chirp = np.arange(n_chirps)
    rx = np.arange(n_rx)

    def reflector(r_bin: float, d_bin: float, az: float, amp: float) -> np.ndarray:
        f_r = r_bin / n_samples                 # fast-time freq -> range
        f_d = d_bin / n_chirps                  # slow-time freq -> Doppler
        ph_s = np.exp(1j * 2 * np.pi * f_r * samp)            # (n_samples,)
        ph_c = np.exp(1j * 2 * np.pi * f_d * chirp)           # (n_chirps,)
        ph_rx = np.exp(1j * 2 * np.pi * SPACING_LAMBDA * np.sin(az) * rx)  # (n_rx,)
        cube = amp * ph_rx[:, None, None] * ph_c[None, :, None] * ph_s[None, None, :]
        return cube

    doppler_bin = velocity / VELOCITY_RES_MPS
    iq = reflector(range_bin, doppler_bin, azimuth_rad, amp=1.0)

    # Seeded clutter: random range bins, azimuths, small Doppler.
    for _ in range(n_clutter):
        c_rbin = rng.integers(0, n_samples)
        c_az = rng.uniform(-np.pi / 2, np.pi / 2)
        c_dbin = rng.uniform(-n_chirps / 4, n_chirps / 4)
        c_amp = CLUTTER_AMP * rng.uniform(0.3, 1.0)
        iq = iq + reflector(c_rbin, c_dbin, c_az, amp=c_amp)

    # Complex Gaussian noise at the requested SNR (target amplitude = 1.0).
    noise_sigma = 10 ** (-snr_db / 20.0)
    noise = noise_sigma / np.sqrt(2) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape))
    return (iq + noise).astype(np.complex128)
