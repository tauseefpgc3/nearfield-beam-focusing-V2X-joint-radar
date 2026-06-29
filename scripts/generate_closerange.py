"""Generate the PRE-REGISTERED close-range synthetic V2X scenario (D-014).

Controlled near-field study: a low-mount intersection / charging-station RSU with vehicles
at close range. Reuses the audited channel/sensor/label machinery from
``generate_synthetic_multimodal`` (D-005 matched-filter label, D-008 near-field multipath
reconstruction, D-010 sensor seeding). The distribution is FIXED by D-014 and must NOT be
tuned to a result.

    python scripts/generate_closerange.py [--freqs 60 140]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.generate_synthetic_multimodal import (CHUNK_USERS, StreamingH5Writer,
                                                   build_sample, load_codebook,
                                                   peak_working_set_mb)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_closerange")

# --- PRE-REGISTERED distribution (D-014) — DO NOT tune to a result --------------
SCENARIO_SEED = 20_240_603            # position + NLoS RNG (distinct from D-010 streams)
N_USERS = 6000
DIST_RANGE_M = (1.0, 20.0)            # user horizontal distance ~ U[1, 20] m
AZ_LIMIT_DEG = 60.0                   # user azimuth ~ U[-60, +60] deg
NLOS_K = (2, 4)                       # number of NLoS scatterers ~ U{2,3,4}
NLOS_R_M = (1.0, 25.0)               # NLoS range ~ U[1, 25] m
NLOS_AZ_DEG = 90.0                    # NLoS azimuth ~ U[-90, +90] deg
NLOS_ATTEN_DB = (10.0, 20.0)         # NLoS 10-20 dB below LoS
FREQS = (60, 140)


def synth_paths(rng):
    """Draw one user's (LoS + seeded NLoS) paths. Returns d, theta, az[], r[], gains[]."""
    d = float(rng.uniform(*DIST_RANGE_M))
    theta = np.deg2rad(rng.uniform(-AZ_LIMIT_DEG, AZ_LIMIT_DEG))
    az = [theta]
    rng_m = [d]
    gains = [1.0 + 0j]                                  # LoS, unit gain
    k = int(rng.integers(NLOS_K[0], NLOS_K[1] + 1))
    for _ in range(k):
        az.append(np.deg2rad(rng.uniform(-NLOS_AZ_DEG, NLOS_AZ_DEG)))
        rng_m.append(float(rng.uniform(*NLOS_R_M)))
        amp = 10.0 ** (-rng.uniform(*NLOS_ATTEN_DB) / 20.0)
        gains.append(amp * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    return d, theta, np.array(az), np.array(rng_m), np.array(gains, dtype=np.complex128)


def trajectory_id(d: float, theta_rad: float) -> int:
    """Synthetic spatial grouping (az 2-deg bin, dist 1-m bin) for leak-free splits."""
    az_bin = int(round(np.degrees(theta_rad)) // 2)
    return az_bin * 100 + int(d // 1)


def generate(f_ghz: int, n_users: int = N_USERS) -> str:
    f_c = f_ghz * 1e9
    codebook, d_f = load_codebook(f_ghz)
    out = os.path.join(_ROOT, "data", "processed", f"closerange_{f_ghz}GHz_full.h5")
    writer = StreamingH5Writer(out, f_c, d_f)
    writer.hf.attrs["scenario"] = "closerange_rsu"
    writer.hf.attrs["seed_scenario"] = SCENARIO_SEED

    chunk: list = []

    def flush(cs: list) -> None:
        if not cs:
            return
        batch = {k: np.stack([s[k] for s in cs]) for k in
                 ("channel", "radar_features", "lidar_features")}
        batch["y_theta"] = np.array([s["y_theta"] for s in cs], np.int32)
        batch["y_r"] = np.array([s["y_r"] for s in cs], np.int32)
        batch["distance"] = np.array([s["distance"] for s in cs], np.float32)
        batch["trajectory_id"] = np.array([s["trajectory_id"] for s in cs], np.int32)
        writer.append(batch)

    written = 0
    for idx in range(n_users):
        rng = np.random.default_rng(SCENARIO_SEED + idx)
        d, theta, az, rng_m, gains = synth_paths(rng)
        user_xy = (d * np.cos(theta), d * np.sin(theta))
        s = build_sample(idx, az, rng_m, gains, user_xy=user_xy, velocity=0.0,
                         distance=d, f_c=f_c, codebook=codebook)
        s["trajectory_id"] = np.int32(trajectory_id(d, theta))
        chunk.append(s)
        if len(chunk) >= CHUNK_USERS:
            flush(chunk)
            written += len(chunk)
            chunk = []
    flush(chunk)
    written += len(chunk)
    writer.close()
    logger.info("closerange %dGHz: %d samples written | peak RSS %.0f MB", f_ghz,
                written, peak_working_set_mb())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=int, nargs="+", default=list(FREQS))
    ap.add_argument("--n-users", type=int, default=N_USERS)
    args = ap.parse_args()
    for f in args.freqs:
        generate(f, n_users=args.n_users)


if __name__ == "__main__":
    main()
