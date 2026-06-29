"""Polar-domain (angle x range) near-field beam codebook (Paper 5, contribution C1).

Implements ``02_SYSTEM_MODEL.md`` §1.2-1.4: the polar codebook, the Fraunhofer
distance, the optimal-label search, and the beam-focusing-gain ratio. Conventions are
fixed in ``docs/decisions.md``:

- D-001/D-002: ``N_side`` (per-side) geometry sets ``d_F`` and the range sampling,
  never ``N_elements = 16384``.
- D-005: gain is the matched-filter ``|f^H h|^2`` (the plan's ``|h^T f|^2`` shorthand).
- D-006: flat codeword index ``k = i*B_r + j`` so ``(y_theta, y_r) = divmod(k, B_r)``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .steering import planar_steering, spherical_steering, wavelength


def fraunhofer_distance(f_c: float, N_side: int = 128, spacing: float = 0.5) -> float:
    """Fraunhofer (near/far-field boundary) distance ``d_F = 2 D^2 / λ``.

    Uses the **per-side** aperture ``D = (N_side - 1) * spacing * λ`` (docs/decisions
    D-002). At the 128x128 UPA (D-011 resolved) this gives 86.3 / 40.3 / 17.3 m at
    28 / 60 / 140 GHz.

    .. warning::
       ``N_side`` is the *per-side* element count (128 for a 128x128 UPA), **not** the
       16384 total elements. Passing 16384 yields a kilometre-scale ``d_F`` and
       false-passes Gate-0 — the forbidden bug from the session precondition.

    Parameters
    ----------
    f_c : float
        Carrier frequency in Hz.
    N_side : int, default 128
        Per-side UPA element count.
    spacing : float, default 0.5
        Element spacing in wavelengths.

    Returns
    -------
    float
        Fraunhofer distance in metres.
    """
    if N_side <= 1:
        raise ValueError(f"N_side must be > 1, got {N_side}")
    lam = wavelength(f_c)
    aperture = (N_side - 1) * spacing * lam  # per-side aperture D, metres
    return 2.0 * aperture ** 2 / lam


def _polar_ranges(N_side: int, lam: float, B_r: int, beta_delta: float) -> np.ndarray:
    """Non-uniform range samples (Cui & Dai 2022), uniform in 1/r.

    ``r_j = beta_delta * (N_side - 1)^2 * λ / (8 j)`` for ``j >= 1``; ``r_0 = +inf``
    (far-field). ``(N_side - 1)^2`` uses the per-side aperture (docs/decisions D-003).
    """
    ranges = np.empty(B_r, dtype=np.float64)
    ranges[0] = np.inf
    if B_r > 1:
        scale = beta_delta * (N_side - 1) ** 2 * lam / 8.0
        j = np.arange(1, B_r)
        ranges[1:] = scale / j
    return ranges


def _angles(B_theta: int) -> np.ndarray:
    """Angle samples uniform in sine: ``θ_i = arcsin((2i - B_theta + 1)/B_theta)``."""
    i = np.arange(B_theta)
    return np.arcsin((2.0 * i - B_theta + 1.0) / B_theta)


def generate_polar_codebook(N: int, f_c: float, B_theta: int = 64, B_r: int = 8,
                            beta_delta: float = 1.2, antenna_layout: str = "UPA",
                            spacing: float = 0.5
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the polar-domain (angle x range) beam codebook.

    Each codeword is ``f_{i,j} = (1/sqrt(N)) a_NF(θ_i, r_j)`` (unit norm). The ``j=0``
    column is the far-field (planar) DFT codebook, so the polar codebook subsumes DFT.
    Codewords are stacked in flat order ``k = i*B_r + j`` (docs/decisions D-006).

    Parameters
    ----------
    N : int
        Total number of antenna elements (steering-vector dimension), e.g. 16384.
    f_c : float
        Carrier frequency in Hz.
    B_theta : int, default 64
        Number of angle bins.
    B_r : int, default 8
        Number of range bins (``j=0`` is far-field).
    beta_delta : float, default 1.2
        Range-oversampling factor (Cui & Dai 2022).
    antenna_layout : str, default "UPA"
        Array layout. Only ``"UPA"`` (square ``sqrt(N) x sqrt(N)``) is supported.
    spacing : float, default 0.5
        Element spacing in wavelengths.

    Returns
    -------
    codebook : np.ndarray
        Complex ``(B_theta*B_r, N)`` unit-norm codewords.
    angles : np.ndarray
        ``(B_theta,)`` azimuth angles in radians.
    ranges : np.ndarray
        ``(B_r,)`` range samples in metres; ``ranges[0] = inf`` (far-field).
    """
    if antenna_layout != "UPA":
        raise NotImplementedError(
            f"antenna_layout={antenna_layout!r} not supported; only 'UPA'. "
            "No silent fallback (docs/05 coding standards)."
        )
    N_side = int(round(np.sqrt(N)))
    if N_side * N_side != N:
        raise ValueError(f"UPA requires square N; got N={N} (sqrt={np.sqrt(N):.3f})")

    lam = wavelength(f_c)
    angles = _angles(B_theta)
    ranges = _polar_ranges(N_side, lam, B_r, beta_delta)

    codebook = np.empty((B_theta * B_r, N), dtype=np.complex128)
    sqrt_n = np.sqrt(N)
    for i, theta in enumerate(angles):
        for j, r in enumerate(ranges):
            if np.isinf(r):  # far-field bin -> planar steering (DFT codeword)
                a = planar_steering(theta, N_side=N_side, spacing=spacing, f_c=f_c)
            else:
                a = spherical_steering(theta, r, N_side=N_side, spacing=spacing,
                                       f_c=f_c)
            codebook[i * B_r + j] = a / sqrt_n
    return codebook, angles, ranges


def compute_optimal_label(h: np.ndarray, codebook: np.ndarray, B_theta: int, B_r: int
                          ) -> Tuple[Tuple[int, int], float]:
    """Optimal polar label ``(y_theta, y_r)`` and its beamforming gain.

    Gain is the matched-filter ``γ_{i,j} = |f_{i,j}^H h|^2`` (docs/decisions D-005).
    The plan writes ``|h^T f|^2``; the conjugate is the physically correct (and
    label-consistent) inner product. Returns the argmax label and ``max_gain``.

    Parameters
    ----------
    h : np.ndarray
        Complex ``(N,)`` channel vector.
    codebook : np.ndarray
        Complex ``(B_theta*B_r, N)`` codebook (rows in ``k = i*B_r + j`` order).
    B_theta, B_r : int
        Codebook dimensions (used to map the flat index to ``(y_theta, y_r)``).

    Returns
    -------
    (y_theta, y_r) : Tuple[int, int]
        Optimal angle and range indices.
    max_gain : float
        Beamforming gain ``|f^H h|^2`` of the optimal codeword.
    """
    h = np.asarray(h)
    if h.ndim != 1 or h.shape[0] != codebook.shape[1]:
        raise ValueError(
            f"h shape {h.shape} incompatible with codebook {codebook.shape}"
        )
    if codebook.shape[0] != B_theta * B_r:
        raise ValueError(
            f"codebook rows {codebook.shape[0]} != B_theta*B_r = {B_theta * B_r}"
        )
    gains = np.abs(codebook.conj() @ h) ** 2  # |f^H h|^2 per codeword
    k = int(np.argmax(gains))
    y_theta, y_r = divmod(k, B_r)
    return (y_theta, y_r), float(gains[k])


def compute_beam_focusing_gain_ratio(f_pred: np.ndarray, f_opt: np.ndarray,
                                     h: np.ndarray) -> float:
    """Beam-focusing-gain ratio ``|f_pred^H h|^2 / |f_opt^H h|^2`` in [0, 1].

    Computed from the **actual** channel ``h`` (audit B3), not a prediction. The
    near-field loss in dB is ``-10 log10(ratio)``.

    Parameters
    ----------
    f_pred : np.ndarray
        Complex ``(N,)`` predicted/selected codeword.
    f_opt : np.ndarray
        Complex ``(N,)`` optimal (genie) codeword.
    h : np.ndarray
        Complex ``(N,)`` channel vector.

    Returns
    -------
    float
        Gain ratio in [0, 1].
    """
    g_pred = np.abs(np.vdot(f_pred, h)) ** 2  # vdot conjugates the first arg -> f^H h
    g_opt = np.abs(np.vdot(f_opt, h)) ** 2
    if g_opt <= 0:
        raise ValueError("optimal codeword has zero gain; channel may be all-zero")
    return float(g_pred / g_opt)
