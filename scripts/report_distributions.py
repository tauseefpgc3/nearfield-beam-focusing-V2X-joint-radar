"""Distance + label distribution report for the synthetic datasets (Batch 2 deliverable).

For each processed file prints, per the auditor's request:
  - frac(d < d_F), frac(y_r > 0), frac(y_r >= 2)
  - full y_r histogram (8 bins)
  - mean(y_r) per distance bin
  - the one-line tie between distance and label near-field-ness

This is a diagnostic preview of Gate-0 G0.1/G0.3, NOT the official probes.

    python scripts/report_distributions.py [file1.h5 ...]
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def report(path: str) -> None:
    with h5py.File(path, "r") as hf:
        yr = hf["y_r"][:]
        yt = hf["y_theta"][:]
        dist = hf["distance"][:]
        d_f = float(hf.attrs["d_F"])
        f_c = float(hf.attrs["f_c"])
        adv = hf["polar_adv_db"][:] if "polar_adv_db" in hf else None
        yr_gated = hf["y_r_gated"][:] if "y_r_gated" in hf else None
        gate_db = float(hf.attrs["gain_gate_db"]) if "gain_gate_db" in hf.attrs else None

    n = yr.size
    print(f"\n=== {os.path.relpath(path, _ROOT)}  (f_c={f_c/1e9:.0f} GHz, "
          f"d_F={d_f:.3f} m, N={n}) ===")
    # Committed distance distribution (D-014 proof for the close-range scenario).
    lo, hi = float(dist.min()), float(dist.max())
    edges = np.linspace(lo, hi, 11)
    h_d = np.histogram(dist, bins=edges)[0]
    print(f"  distance min/max = {lo:.2f}/{hi:.2f} m | histogram (10 bins): {h_d.tolist()}")
    frac_nf_dist = float(np.mean(dist < d_f))
    frac_yr_pos = float(np.mean(yr > 0))
    frac_yr_2 = float(np.mean(yr >= 2))
    print(f"  frac(d < d_F)   = {frac_nf_dist:.4f}")
    print(f"  frac(y_r > 0)   = {frac_yr_pos:.4f}")
    print(f"  frac(y_r >= 2)  = {frac_yr_2:.4f}")
    print(f"  y_r histogram   = {np.bincount(yr, minlength=8).tolist()}")
    print(f"  distinct (y_theta,y_r) = {len(set(zip(yt.tolist(), yr.tolist())))}/512")

    # mean(y_r) per distance bin
    edges = [0.0, d_f, 5 * d_f, 10 * d_f, np.inf]
    print("  mean(y_r) per distance bin:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            continue
        hi_s = "inf" if np.isinf(hi) else f"{hi:.1f}"
        print(f"    d in [{lo:7.2f}, {hi_s:>7}) m  n={int(m.sum()):5d}  "
              f"mean(y_r)={yr[m].mean():.3f}  max(y_r)={int(yr[m].max())}")

    # one-line tie
    print(f"  TIE: {frac_nf_dist:.1%} of users are within d_F and "
          f"{frac_yr_pos:.1%} received a near-field (y_r>0) label "
          f"-> {'near-field regime PRESENT' if frac_yr_pos > 0.05 else 'near-field regime ABSENT (labels collapse to far-field bin)'}.")

    # Three-way near-field regime view (anchored on the convention-independent gain).
    if adv is not None:
        f3 = float(np.mean(adv >= 3.0))
        f05 = float(np.mean(adv >= 0.5))
        print("  --- near-field regime, three ways ---")
        print(f"    frac(d < d_F)                       = {frac_nf_dist:.4f}  [strict geometric]")
        print(f"    frac(y_r > 0) raw c.delay           = {frac_yr_pos:.4f}  [raw label rate]")
        print(f"    frac(genie polar adv >= 3 dB)       = {f3:.4f}  [physically meaningful]")
        print(f"    frac(genie polar adv >= 0.5 dB)     = {f05:.4f}  [incl. marginal band]")
        print(f"    marginal band (0.5<=adv<3 dB)       = {f05 - f3:.4f}")
        if yr_gated is not None:
            print(f"    y_r hist  raw   = {np.bincount(yr, minlength=8).tolist()}")
            print(f"    y_r hist  GATED ({gate_db:.1f} dB) = "
                  f"{np.bincount(yr_gated, minlength=8).tolist()}  "
                  f"(frac y_r>0 = {np.mean(yr_gated > 0):.4f})")


def main() -> None:
    files = sys.argv[1:] or sorted(
        glob.glob(os.path.join(_ROOT, "data", "processed", "synthetic_*.h5")))
    if not files:
        print("No synthetic_*.h5 files found.")
        sys.exit(1)
    for f in files:
        report(f)


if __name__ == "__main__":
    main()
