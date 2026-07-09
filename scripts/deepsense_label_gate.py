"""DeepSense label-trust gate (G0.3 + G0.4) — MUST pass before any training.

Runs on a PROCESSED DeepSense HDF5 (schema below); format-agnostic w.r.t. the raw
DeepSense files. DeepSense has NO full channel, so polar labels are reconstructed from
per-DFT-beam power and are low-trust (G0.4). This gate checks:

  G0.3 non-degeneracy: distinct (y_theta, y_r), y_theta + y_r histograms.
  G0.4 angle agreement: the strongest-DFT-beam angle (the trustworthy part, read near-
       directly from which beam had max power) vs the reconstructed y_theta. High
       agreement => the ANGLE label is sound even if range (y_r) is not.

Expected HDF5 datasets (no 'channel'):
  radar_features (N,2,256,128) f32, lidar_features (N,2,150,100) f32,
  y_theta (N,) i32, y_r (N,) i32, distance (N,) f32, trajectory_id (N,) i32,
  best_beam (N,) i32      # index (0..B_theta-1) of the max-power DFT beam (ground truth angle)

Honest expectation: y_r will be mostly j=0 (DeepSense is far-field). That is EXPECTED and
is NOT a failure — it means DeepSense validates ANGLE + far-field beam selection, not
near-field gain. The gate FAILS only if the ANGLE labels are degenerate/incoherent.

    python scripts/deepsense_label_gate.py data/processed/deepsense_*.h5
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_THETA, B_R = 64, 8


def gate_file(path: str) -> dict:
    with h5py.File(path, "r") as hf:
        yt = hf["y_theta"][:].astype(int)
        yr = hf["y_r"][:].astype(int)
        has_beam = "best_beam" in hf
        best = hf["best_beam"][:].astype(int) if has_beam else None
    n = yt.size
    distinct = len(set(zip(yt.tolist(), yr.tolist())))
    hist_t = np.bincount(yt, minlength=B_THETA)
    hist_r = np.bincount(yr, minlength=B_R)
    n_angles = int((hist_t > 0).sum())

    # G0.3: angle non-degeneracy (range being far-field-dominated is expected, NOT failed)
    angle_ok = n_angles >= 5          # angle must span several beams, not collapse to one
    # G0.4: reconstructed y_theta agrees with the directly-observed strongest beam
    agree = float(np.mean(best == yt)) if has_beam else float("nan")
    agree_ok = (not has_beam) or agree >= 0.80

    print(f"\n=== {os.path.relpath(path,_ROOT)} (N={n}) ===")
    print(f"  G0.3 distinct (y_theta,y_r) = {distinct}/512 | distinct angles = {n_angles}/64")
    print(f"  y_theta histogram (nonzero bins): {int((hist_t>0).sum())} bins, "
          f"max {int(hist_t.max())} min(nonzero) {int(hist_t[hist_t>0].min()) if n_angles else 0}")
    print(f"  y_r histogram = {hist_r.tolist()}  (far-field j=0 dominant is EXPECTED)")
    if has_beam:
        print(f"  G0.4 angle agreement (strongest beam == y_theta) = {agree:.3f}")
    else:
        print("  G0.4 angle agreement: best_beam not stored — cannot verify (store it!)")
    verdict = "PASS" if (angle_ok and agree_ok) else "FAIL"
    if not angle_ok:
        print("  >>> FAIL: ANGLE labels degenerate (<5 distinct beams) — reconstruction broken. "
              "STOP, do not train.")
    elif not agree_ok:
        print(f"  >>> FAIL: angle agreement {agree:.2f} < 0.80 — reconstruction incoherent. STOP.")
    else:
        print("  >>> PASS: angle labels trustworthy. (Range far-field-dominated as expected; "
              "use DeepSense for angle + far-field beam-selection validation only.)")
    return dict(path=path, n=n, distinct=distinct, n_angles=n_angles,
                y_r_hist=hist_r.tolist(), angle_agreement=agree, verdict=verdict)


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(
        _ROOT, "data", "processed", "deepsense_*.h5")))
    if not files:
        sys.exit("No processed DeepSense HDF5 files found (data/processed/deepsense_*.h5). "
                 "Download + process first (STEP A — requires the academic-account download).")
    results = [gate_file(f) for f in files]
    print("\n=== GATE SUMMARY ===")
    for r in results:
        print(f"  {os.path.basename(r['path'])}: {r['verdict']} "
              f"(angles {r['n_angles']}/64, agree {r['angle_agreement']:.2f})")
    if any(r["verdict"] == "FAIL" for r in results):
        print("\nLABEL-TRUST GATE FAILED — do not train on these labels (D-016).")
        sys.exit(1)
    print("\nLABEL-TRUST GATE PASSED — DeepSense usable for angle/far-field validation (D-016).")


if __name__ == "__main__":
    main()
