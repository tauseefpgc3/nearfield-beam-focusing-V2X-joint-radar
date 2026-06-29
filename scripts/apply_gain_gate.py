"""Gain-gated near-field relabel (supervisor decision; channels untouched).

For every synthetic sample, compute the genie polar-vs-DFT advantage from the STORED
channel:

    adv_dB = 10*log10( max_k |f_polar_k^H h|^2  /  max_m |f_dft_m^H h|^2 )

A sample keeps its c.delay range bin only if ``adv_dB >= GATE_DB`` (config
``labeling.gain_gate_db``, 3.0 dB); otherwise its range label is set to the far-field
bin (``y_r := 0``). This makes "near-field" a physically meaningful, convention-
independent quantity (D-008a discussion).

This **relabels only** — channels, radar, lidar, distances and the raw ``y_r`` are
left untouched. New datasets are added to each HDF5 file:
  - ``y_r_gated``   (int32)  the gated range label
  - ``y_theta``     unchanged (the gate acts on the range bin, per instruction; the
                    count of samples whose kept y_theta differs from the DFT-optimal
                    angle is reported and is ~0, since sub-gate samples are near-planar)
  - ``polar_adv_db`` (float32) per-sample genie polar-vs-DFT advantage
and attr ``gain_gate_db``.

    python scripts/apply_gain_gate.py [file1.h5 ...]
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_R = 8
CHUNK = 500


def load_gate_db() -> float:
    with open(os.path.join(_ROOT, "configs", "base.yaml")) as f:
        cfg = yaml.safe_load(f)
    return float(cfg["labeling"]["gain_gate_db"])


def codebook_for(f_c: float) -> np.ndarray:
    f_ghz = int(round(f_c / 1e9))
    path = os.path.join(_ROOT, "data", "codebooks", f"polar_codebook_{f_ghz}GHz.npz")
    return np.load(path)["codebook"]


def gate_file(path: str, gate_db: float) -> dict:
    with h5py.File(path, "r+") as hf:
        n = hf["y_theta"].shape[0]
        f_c = float(hf.attrs["f_c"])
        cb = codebook_for(f_c)                 # (512, N)
        cb_c = cb.conj()
        dft_c = cb_c[0::B_R]                   # j=0 (far-field) block, (64, N)

        adv = np.empty(n, dtype=np.float32)
        dft_theta = np.empty(n, dtype=np.int32)
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            h = hf["channel"][s:e]             # (b, N) — only one chunk in RAM
            pol_g = np.abs(cb_c @ h.T) ** 2    # (512, b)
            dft_gains = np.abs(dft_c @ h.T) ** 2  # (64, b)
            adv[s:e] = 10.0 * np.log10(pol_g.max(0) / dft_gains.max(0) + 1e-300)
            dft_theta[s:e] = dft_gains.argmax(0)

        y_r = hf["y_r"][:]
        y_theta = hf["y_theta"][:]
        keep = adv >= gate_db
        y_r_gated = np.where(keep, y_r, 0).astype(np.int32)
        # D-013 resolved: a gated (far-field) sample takes the DFT-optimal angle, not the
        # leftover near-field angle. Above the gate, y_theta is unchanged.
        y_theta_gated = np.where(keep, y_theta, dft_theta).astype(np.int32)
        gated = (~keep) & (y_r > 0)
        theta_fixed = int(np.sum(gated & (y_theta != dft_theta)))  # angles corrected

        for name, data in (("y_r_gated", y_r_gated),
                           ("y_theta_gated", y_theta_gated),
                           ("polar_adv_db", adv)):
            if name in hf:
                del hf[name]
            hf.create_dataset(name, data=data)
        hf.attrs["gain_gate_db"] = gate_db

    return {
        "path": path, "n": n, "f_c": f_c,
        "raw_nf": int(np.sum(y_r > 0)), "gated_nf": int(np.sum(y_r_gated > 0)),
        "hist_before": np.bincount(y_r, minlength=8).tolist(),
        "hist_after": np.bincount(y_r_gated, minlength=8).tolist(),
        "theta_fixed": theta_fixed,
    }


def main() -> None:
    gate_db = load_gate_db()
    files = sys.argv[1:] or sorted(
        glob.glob(os.path.join(_ROOT, "data", "processed", "synthetic_*.h5")))
    print(f"GATE_DB = {gate_db} dB (configs/base.yaml: labeling.gain_gate_db)\n")
    for f in files:
        r = gate_file(f, gate_db)
        print(f"{os.path.relpath(r['path'], _ROOT)}  (f_c={r['f_c']/1e9:.0f} GHz, N={r['n']})")
        print(f"  y_r>0  raw={r['raw_nf']} ({r['raw_nf']/r['n']:.1%})  "
              f"gated={r['gated_nf']} ({r['gated_nf']/r['n']:.1%})")
        print(f"  y_r hist BEFORE: {r['hist_before']}")
        print(f"  y_r hist AFTER : {r['hist_after']}")
        print(f"  (D-013: {r['theta_fixed']} gated samples had y_theta corrected to the "
              f"DFT-optimal angle)\n")


if __name__ == "__main__":
    main()
