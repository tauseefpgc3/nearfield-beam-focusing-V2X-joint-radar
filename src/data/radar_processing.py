"""FMCW radar feature extraction: raw IQ -> normalized RD / RA maps (Paper 5).

Implements ``02_SYSTEM_MODEL.md`` §2.1. The range-Doppler map (RD) and range-angle map
(RA) are produced by FFT / DoA beamforming and normalized to ~[0, 1]. See
``docs/decisions.md`` D-009 for shapes and the RAM zero-pad convention.

Signatures match ``05_REPO_STRUCTURE.md``.
"""
from __future__ import annotations

import numpy as np

# Re-exported so callers / tests share one source of the range scale (D-009).
from .synthetic_radar import RANGE_RES_M  # noqa: F401

N_RANGE_BINS = 256
N_DOPPLER_BINS = 128
N_ANGLE_BINS = 64
SPACING_LAMBDA = 0.5


def generate_radar_steering(n_rx: int, n_angle_bins: int) -> np.ndarray:
    """DoA steering matrix for the receive ULA, uniform in sine-angle.

    Parameters
    ----------
    n_rx : int
        Number of receive elements.
    n_angle_bins : int
        Number of azimuth bins to beamform.

    Returns
    -------
    np.ndarray
        Complex ``(n_angle_bins, n_rx)`` steering matrix (each row a unit-modulus
        array response for one azimuth).
    """
    i = np.arange(n_angle_bins)
    sin_theta = (2.0 * i - n_angle_bins + 1.0) / n_angle_bins   # uniform in sine
    rx = np.arange(n_rx)
    phase = 2 * np.pi * SPACING_LAMBDA * np.outer(sin_theta, rx)  # (A, n_rx)
    return np.exp(1j * phase)


def _normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] (constant array -> zeros, no divide-by-zero)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def radar_preprocessing(raw_iq: np.ndarray, n_rx: int = 4, n_chirps: int = 256,
                        n_samples: int = 256) -> dict:
    """Convert a raw IQ cube to normalized RD and RA maps.

    Parameters
    ----------
    raw_iq : np.ndarray
        Complex ``(n_rx, n_chirps, n_samples)`` IQ cube.
    n_rx, n_chirps, n_samples : int
        Cube dimensions (validated against ``raw_iq.shape``).

    Returns
    -------
    dict
        ``{'RD': (256, 128) float32, 'RA': (256, 64) float32}`` normalized to [0, 1].
    """
    raw_iq = np.asarray(raw_iq)
    if raw_iq.shape != (n_rx, n_chirps, n_samples):
        raise ValueError(f"raw_iq shape {raw_iq.shape} != {(n_rx, n_chirps, n_samples)}")

    # Range FFT over fast time -> (n_rx, n_chirps, N_RANGE_BINS).
    range_fft = np.fft.fft(raw_iq, n=N_RANGE_BINS, axis=2)

    # --- RD: Doppler FFT over slow time, magnitude, averaged over rx -----------
    dopp_fft = np.fft.fftshift(np.fft.fft(range_fft, n=n_chirps, axis=1), axes=1)
    rd_full = np.abs(dopp_fft).mean(axis=0)              # (n_chirps, N_RANGE_BINS)
    # Center-crop Doppler to N_DOPPLER_BINS, transpose to (range, doppler).
    start = (n_chirps - N_DOPPLER_BINS) // 2
    rd = rd_full[start:start + N_DOPPLER_BINS, :].T      # (256, 128)
    rd = _normalize01(rd).astype(np.float32)

    # --- RA: DoA beamforming over rx, magnitude, averaged over chirps ----------
    steer = generate_radar_steering(n_rx, N_ANGLE_BINS)  # (64, n_rx)
    # range-compressed per (rx, chirp, range); beamform: sum_rx conj(steer)*x.
    beam = np.einsum("ar,rck->ack", steer.conj(), range_fft)  # (64, n_chirps, 256)
    ra = np.abs(beam).mean(axis=1).T                          # (256, 64)
    ra = _normalize01(ra).astype(np.float32)

    return {"RD": rd, "RA": ra}
