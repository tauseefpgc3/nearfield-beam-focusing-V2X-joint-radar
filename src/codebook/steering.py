"""Array steering vectors for the THz V2X UPA (Paper 5).

Implements the near-field (spherical) and far-field (planar) array responses from
``02_SYSTEM_MODEL.md`` §1.2. Geometry conventions are fixed in ``docs/decisions.md``
(D-004): a ``N_side x N_side`` UPA in the y-z plane, boresight +x, half-wavelength
spacing, centred at the origin (array centre = phase reference). The user is at
azimuth ``theta`` / range ``r`` at elevation 0: ``p_u = (r cosθ, r sinθ, 0)``.

Steering vectors are returned **raw** (each entry has unit modulus, so the vector has
L2 norm ``sqrt(N)``). Codebooks normalise by ``1/sqrt(N)`` to obtain unit-norm
codewords (see ``polar.py`` / ``dft.py``).
"""
from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s


def wavelength(f_c: float) -> float:
    """Carrier wavelength ``lambda = c / f_c`` in metres.

    Parameters
    ----------
    f_c : float
        Carrier frequency in Hz (must be > 0).
    """
    if f_c <= 0:
        raise ValueError(f"f_c must be positive, got {f_c}")
    return SPEED_OF_LIGHT / f_c


def upa_element_positions(N_side: int, spacing: float, lambda_: float) -> np.ndarray:
    """Centred UPA element positions in the y-z plane.

    Element ``n = m * N_side + l`` sits at ``(0, y_m, z_l)`` with
    ``y_m = (m - (N_side-1)/2) * d`` and ``d = spacing * lambda`` (metres). Row-major
    in (y, z); this ordering is shared verbatim with ``dft.py`` (audit E2.3).

    Parameters
    ----------
    N_side : int
        Per-side element count (32 for the 32x32 UPA).
    spacing : float
        Element spacing in wavelengths (0.5 for half-wavelength).
    lambda_ : float
        Carrier wavelength in metres.

    Returns
    -------
    np.ndarray
        ``(N_side**2, 3)`` array of (x, y, z) positions in metres.
    """
    if N_side <= 0:
        raise ValueError(f"N_side must be positive, got {N_side}")
    d = spacing * lambda_
    idx = np.arange(N_side)
    coords = (idx - (N_side - 1) / 2.0) * d
    ys, zs = np.meshgrid(coords, coords, indexing="ij")  # (N_side, N_side)
    n = N_side * N_side
    positions = np.zeros((n, 3), dtype=np.float64)
    positions[:, 1] = ys.reshape(-1)
    positions[:, 2] = zs.reshape(-1)
    return positions


def planar_steering(theta: float, N_side: int = 128, spacing: float = 0.5,
                    f_c: float = 28e9) -> np.ndarray:
    """Far-field (planar-wavefront) array response at azimuth ``theta``.

    Far-field limit of :func:`spherical_steering` (``r -> inf``):
    ``[a]_n = exp(+j 2π/λ * (p_n · u))`` with unit direction ``u = (cosθ, sinθ, 0)``.
    With elevation 0 the phase depends only on the y-coordinate (the z-rows are
    degenerate), which is the correct far-field UPA response and equals the DFT
    codeword (audit E2).

    Parameters
    ----------
    theta : float
        Azimuth angle in radians.
    N_side : int
        Per-side UPA element count.
    spacing : float
        Element spacing in wavelengths.
    f_c : float
        Carrier frequency in Hz.

    Returns
    -------
    np.ndarray
        Complex ``(N_side**2,)`` steering vector, each entry of unit modulus.
    """
    lam = wavelength(f_c)
    positions = upa_element_positions(N_side, spacing, lam)
    u = np.array([np.cos(theta), np.sin(theta), 0.0])
    phase = (2.0 * np.pi / lam) * (positions @ u)
    return np.exp(1j * phase)


def spherical_steering(theta: float, r: float, N_side: int = 128, spacing: float = 0.5,
                       f_c: float = 28e9) -> np.ndarray:
    """Near-field (spherical-wavefront) array response at azimuth ``theta``, range ``r``.

    From ``02_SYSTEM_MODEL.md`` §1.2:
    ``[a_NF(θ, r)]_n = exp(-j 2π/λ (||p_n - p_u|| - r))`` with
    ``p_u = (r cosθ, r sinθ, 0)``. ``r`` is the range from the **array centre** (the
    phase reference), so ``||p_u|| = r`` and the ``-r`` term references the centre.

    Parameters
    ----------
    theta : float
        Azimuth angle in radians.
    r : float
        Range from the array centre in metres (must be finite and > 0). For the
        far-field bin (``r -> inf``) use :func:`planar_steering` instead.
    N_side : int
        Per-side UPA element count.
    spacing : float
        Element spacing in wavelengths.
    f_c : float
        Carrier frequency in Hz.

    Returns
    -------
    np.ndarray
        Complex ``(N_side**2,)`` steering vector, each entry of unit modulus.
    """
    if not np.isfinite(r):
        raise ValueError("r must be finite; use planar_steering for the far-field bin")
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    lam = wavelength(f_c)
    positions = upa_element_positions(N_side, spacing, lam)
    p_u = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    dist = np.linalg.norm(positions - p_u, axis=1)  # ||p_n - p_u||
    phase = -(2.0 * np.pi / lam) * (dist - r)
    return np.exp(1j * phase)
