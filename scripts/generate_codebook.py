"""Generate and save the polar beam codebooks for 28/60/140 GHz.

Writes ``data/codebooks/polar_codebook_{28,60,140}GHz.npz`` containing the codebook,
angles, ranges, d_F, and generation metadata. The codebook is deterministic (pure
physics, no RNG) so it is generated once and loaded everywhere (audit B2).

The stored ``d_F`` is computed by :func:`fraunhofer_distance` — the single source of
truth — so it is identical to any later call (test
``test_saved_codebook_dF_matches_fraunhofer``).

Usage
-----
    python scripts/generate_codebook.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.polar import fraunhofer_distance, generate_polar_codebook

# Fixed array / codebook configuration (configs/base.yaml).
N_ELEMENTS = 16384
N_SIDE = 128
B_THETA = 64
B_R = 8
BETA_DELTA = 1.2
SPACING = 0.5
FREQUENCIES_GHZ = (28, 60, 140)


def main() -> None:
    out_dir = os.path.join(_ROOT, "data", "codebooks")
    os.makedirs(out_dir, exist_ok=True)

    for f_ghz in FREQUENCIES_GHZ:
        f_c = f_ghz * 1e9
        codebook, angles, ranges = generate_polar_codebook(
            N_ELEMENTS, f_c, B_theta=B_THETA, B_r=B_R, beta_delta=BETA_DELTA,
            antenna_layout="UPA", spacing=SPACING,
        )
        d_f = fraunhofer_distance(f_c, N_side=N_SIDE, spacing=SPACING)

        out_path = os.path.join(out_dir, f"polar_codebook_{f_ghz}GHz.npz")
        np.savez(
            out_path,
            codebook=codebook,
            angles=angles,
            ranges=ranges,
            d_F=np.float64(d_f),
            f_c=np.float64(f_c),
            N=np.int64(N_ELEMENTS),
            N_side=np.int64(N_SIDE),
            B_theta=np.int64(B_THETA),
            B_r=np.int64(B_R),
            beta_delta=np.float64(BETA_DELTA),
            spacing=np.float64(SPACING),
        )
        # One line per frequency: shape + d_F (deliverable to auditor).
        print(f"{f_ghz:>3d} GHz | codebook shape {codebook.shape} | "
              f"d_F = {d_f:.4f} m | saved -> {os.path.relpath(out_path, _ROOT)}")


if __name__ == "__main__":
    main()
