"""DFT (far-field, angle-only) beam codebook (Paper 5 baseline + Gate-0 G0.2).

This is the conventional angle-only codebook the polar codebook is compared against.
It is also the far-field (``j=0``) limit of the polar codebook, so audit E2 checks
that ``polar[:, j=0] == generate_dft_codebook(...)``.

Independent code path (audit E2): the phase is built directly from the element
**index** (a spatial-frequency ramp), not from the Euclidean ``p_n · u`` used in
``steering.planar_steering``. The two must nonetheless agree numerically — that
agreement is the sanity check. Geometry conventions match ``docs/decisions.md`` D-004:
``N_side x N_side`` UPA, half-wavelength-grid spatial frequency ``sin(theta)``, angles
uniform in sine, row-major (y outer / z inner) ordering, centred phase reference.
"""
from __future__ import annotations

import numpy as np


def generate_dft_codebook(N: int, B_theta: int = 64, spacing: float = 0.5
                          ) -> np.ndarray:
    """Generate the far-field DFT (angle-only) codebook.

    Codeword ``i`` is the planar UPA response at azimuth
    ``θ_i = arcsin((2i - B_theta + 1)/B_theta)``, built from the element index:
    ``[f_i]_n = (1/sqrt(N)) exp(+j 2π spacing (m - (N_side-1)/2) sin(θ_i))`` where
    ``m`` is the element's y-index and the value is tiled over the z-index (elevation
    0). This equals ``steering.planar_steering(θ_i) / sqrt(N)`` (audit E2).

    Parameters
    ----------
    N : int
        Total number of antenna elements (square UPA: ``N_side = sqrt(N)``).
    B_theta : int, default 64
        Number of angle (azimuth) bins.
    spacing : float, default 0.5
        Element spacing in wavelengths.

    Returns
    -------
    np.ndarray
        Complex ``(B_theta, N)`` unit-norm DFT codebook.
    """
    N_side = int(round(np.sqrt(N)))
    if N_side * N_side != N:
        raise ValueError(f"UPA requires square N; got N={N} (sqrt={np.sqrt(N):.3f})")

    # Angles uniform in sine (same convention as polar._angles).
    i = np.arange(B_theta)
    sin_theta = (2.0 * i - B_theta + 1.0) / B_theta          # (B_theta,)

    # Centred y-index per element, tiled over the z-index (row-major, y outer).
    m = np.arange(N_side) - (N_side - 1) / 2.0               # (N_side,)
    m_full = np.repeat(m, N_side)                            # (N,), n = m*N_side + l

    # Spatial-frequency phase ramp; outer over (angle, element).
    phase = 2.0 * np.pi * spacing * np.outer(sin_theta, m_full)  # (B_theta, N)
    return np.exp(1j * phase) / np.sqrt(N)
