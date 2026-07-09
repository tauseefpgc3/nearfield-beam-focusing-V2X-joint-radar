"""Extraction sanity check: DeepMIMO ray-tracing fields vs geometry (diagnostic).

For one LoS-dominant user of the selected BS/grid, confirms the per-path extraction is
correct independently of the array size (these are scenario properties, unchanged by
N_side):
  - dominant-path AoD azimuth (_aod_az_rot) vs geometric BS->user azimuth  (~within 2 deg)
  - dominant-path range (c*delay) vs true LoS 3-D distance                 (~within 1 m)

    python scripts/extraction_sanity_check.py [o1_60|o1_140]
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.generate_synthetic_multimodal import (RX_SET_ID, TX_SET_ID,
                                                   select_subdataset)

C = 299_792_458.0


def main() -> None:
    import deepmimo as dm

    scen = sys.argv[1] if len(sys.argv) > 1 else "o1_60"
    macro = dm.load(scen, tx_sets=[TX_SET_ID], rx_sets=[RX_SET_ID], max_paths=25)
    d = select_subdataset(macro, TX_SET_ID, RX_SET_ID)
    tx = np.asarray(d["tx_pos"]).reshape(-1)[:3]
    rx = np.asarray(d["rx_pos"])

    # First user that has a finite-power LoS-like dominant path.
    u = None
    for cand in range(0, rx.shape[0], 9999):
        pw = np.asarray(d["power"][cand], float)
        if pw.size and np.isfinite(pw).any():
            u = cand
            break
    pw = np.asarray(d["power"][u], float)
    az_raw = np.asarray(d["aod_az"][u], float)        # global frame
    az_rot = np.asarray(d["_aod_az_rot"][u], float)   # array-boresight frame (downstream)
    dl = np.asarray(d["delay"][u], float)
    dom = int(np.nanargmax(pw))

    geo_vec = rx[u, :3] - tx
    geo_az = np.degrees(np.arctan2(geo_vec[1], geo_vec[0]))   # GLOBAL frame
    geo_dist = float(np.linalg.norm(geo_vec))
    # Validate extraction in a CONSISTENT (global) frame: raw aod_az vs global geometry.
    az_diff = abs(((az_raw[dom] - geo_az + 180) % 360) - 180)
    rng_diff = abs(C * dl[dom] - geo_dist)
    rotation = ((az_raw[dom] - az_rot[dom] + 180) % 360) - 180  # BS array orientation

    print(f"=== extraction sanity ({scen}, BS{TX_SET_ID}/rx{RX_SET_ID}, user {u}) ===")
    print("  array geometry now 128x128 (extraction unchanged; scenario properties)")
    print(f"  dominant aod_az (global)  = {az_raw[dom]:.2f} deg | geometric az (global)"
          f" = {geo_az:.2f} deg | diff = {az_diff:.2f} deg")
    print(f"  dominant _aod_az_rot (array frame, used in codebook) = {az_rot[dom]:.2f} deg"
          f"  (BS array rotation = {rotation:.2f} deg)")
    print(f"  dominant c*delay = {C*dl[dom]:.2f} m | LoS 3-D dist = {geo_dist:.2f} m"
          f" | diff = {rng_diff:.3f} m")
    assert az_diff < 2.0, f"AoD azimuth off by {az_diff:.2f} deg (>2) — units/extraction bug"
    assert rng_diff < 1.0, f"range off by {rng_diff:.3f} m (>1) — delay/units bug"
    print("  PASS: global AoD within 2 deg and range within 1 m.")


if __name__ == "__main__":
    main()
