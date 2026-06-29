"""PIPELINE SMOKE TEST on PLACEHOLDER geometry -- NOT the real O1 deliverable.

The real O1_60/O1_140 download is blocked by a deepmimo.net 429 rate-limit. To prove the
Batch-2 plumbing (near-field channel reconstruction -> exact label -> seeded radar/LiDAR
-> HDF5 -> verify) is correct and complete, this script runs the *same* per-sample
functions used by ``generate_synthetic_multimodal.py`` on FABRICATED per-path geometry.

Output file is named ``_smoketest_*`` and must NEVER be presented as O1 DeepMIMO data.
Once the download clears, ``python scripts/generate_synthetic_multimodal.py`` produces
the real deliverable with identical code paths.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.generate_synthetic_multimodal import build_sample, load_codebook, write_dataset


def main() -> None:
    f_ghz, f_c = 60, 60e9
    codebook, d_f = load_codebook(f_ghz)
    rng = np.random.default_rng(0)  # placeholder geometry only (deterministic)
    n_users = 300

    # Distances log-uniform in [0.3, 150] m so a meaningful fraction are < d_F (~2.4 m).
    distances = np.exp(rng.uniform(np.log(0.3), np.log(150.0), n_users))
    azimuths_los = rng.uniform(-np.pi / 3, np.pi / 3, n_users)

    samples, traj = [], []
    for u in range(n_users):
        d, th = float(distances[u]), float(azimuths_los[u])
        # LoS path + two weak NLoS paths (different az / longer range).
        az = np.array([th, th + rng.uniform(-0.3, 0.3), th + rng.uniform(-0.5, 0.5)])
        r = np.array([d, d * rng.uniform(1.1, 1.6), d * rng.uniform(1.2, 2.0)])
        g = np.array([1.0 + 0j,
                      (0.3 + 0.1j) * rng.uniform(0.5, 1.0),
                      (0.2 - 0.1j) * rng.uniform(0.5, 1.0)])
        s = build_sample(u, az, r, g, user_xy=(d * np.cos(th), d * np.sin(th)),
                         velocity=0.0, distance=d, f_c=f_c, codebook=codebook)
        samples.append(s)
        traj.append(u // 30)  # 10 placeholder "trajectories"

    out = os.path.join(_ROOT, "data", "processed", "_smoketest_60GHz_full.h5")
    write_dataset(out, samples, np.array(traj), f_c, d_f)
    print(f"\nSMOKE TEST wrote {out} (PLACEHOLDER geometry, NOT O1).")


if __name__ == "__main__":
    main()
