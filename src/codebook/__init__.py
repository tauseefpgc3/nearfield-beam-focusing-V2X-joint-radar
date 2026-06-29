"""Polar / DFT beam codebooks and array steering vectors (Paper 5, contribution C1)."""

from .dft import generate_dft_codebook
from .polar import (
    compute_beam_focusing_gain_ratio,
    compute_optimal_label,
    fraunhofer_distance,
    generate_polar_codebook,
)
from .steering import planar_steering, spherical_steering, wavelength

__all__ = [
    "generate_dft_codebook",
    "generate_polar_codebook",
    "compute_optimal_label",
    "compute_beam_focusing_gain_ratio",
    "fraunhofer_distance",
    "planar_steering",
    "spherical_steering",
    "wavelength",
]
