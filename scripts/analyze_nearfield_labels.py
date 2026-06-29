"""Diagnostic: are j>0 labels on FAR-by-LoS users physical, or a c.delay artifact?

For users with LoS distance > d_F that still get a near-field label (y_r>0), print, for
the path that drives the optimal codeword, its effective range under BOTH conventions
(c.delay total-path-length vs inter_pos BS->first-bounce), plus the resulting y_r under
each. Analysis only — changes nothing.

    python scripts/analyze_nearfield_labels.py [60|140] [n_users]
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.polar import compute_optimal_label, fraunhofer_distance
from src.codebook.steering import spherical_steering
from src.data.deepmimo_loader import FOV_LIMIT_RAD, range_via_inter_pos
from scripts.generate_synthetic_multimodal import (B_R, B_THETA, N_ELEMENTS, N_SIDE,
                                                   RX_SET_ID, SPACING, TX_SET_ID,
                                                   N_CAP_USERS, select_subdataset)

C = 299_792_458.0


def extract_paths(d, u):
    """Return (az, r_cdelay, gains, orig_idx) for in-FoV finite paths of user u."""
    az_deg = np.asarray(d["_aod_az_rot"][u], float)
    pdbw = np.asarray(d["power"][u], float)
    ph = np.asarray(d["phase"][u], float)
    dl = np.asarray(d["delay"][u], float)
    valid = np.isfinite(pdbw) & np.isfinite(dl) & np.isfinite(az_deg)
    idx = np.where(valid)[0]
    az = (np.deg2rad(az_deg[idx]) + np.pi) % (2 * np.pi) - np.pi
    fov = np.abs(az) < FOV_LIMIT_RAD
    idx = idx[fov]
    az = az[fov]
    r_cdelay = C * dl[idx]
    gains = np.sqrt(10.0 ** (pdbw[idx] / 10.0)) * np.exp(1j * np.deg2rad(ph[idx]))
    return az, r_cdelay, gains, idx


def channel(az, rng, gains, f_c):
    h = np.zeros(N_ELEMENTS, dtype=np.complex128)
    for a, r, g in zip(az, rng, gains):
        h += g * spherical_steering(a, float(r), N_side=N_SIDE, spacing=SPACING, f_c=f_c)
    return h


def main() -> None:
    import deepmimo as dm

    f_ghz = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    f_c = f_ghz * 1e9
    scen = f"o1_{f_ghz}"

    cbz = np.load(os.path.join(_ROOT, "data", "codebooks",
                               f"polar_codebook_{f_ghz}GHz.npz"))
    codebook, cb_ranges = cbz["codebook"], cbz["ranges"]
    d_f = fraunhofer_distance(f_c, N_side=N_SIDE, spacing=SPACING)

    macro = dm.load(scen, tx_sets=[TX_SET_ID], rx_sets=[RX_SET_ID], max_paths=25)
    d = select_subdataset(macro, TX_SET_ID, RX_SET_ID)
    tx = np.asarray(d["tx_pos"]).reshape(-1)[:3]
    rx = np.asarray(d["rx_pos"])
    n_total = rx.shape[0]
    stride = max(1, n_total // N_CAP_USERS)
    user_idxs = list(range(0, n_total, stride))[:N_CAP_USERS]

    # j=0 (far-field/DFT) block of the polar codebook, for the polar-vs-DFT margin.
    dft_block = codebook[0::B_R]

    def polar_vs_dft_db(h):
        gp = np.max(np.abs(codebook.conj() @ h) ** 2)
        gd = np.max(np.abs(dft_block.conj() @ h) ** 2)
        return 10.0 * np.log10(gp / gd + 1e-300)

    print(f"=== near-field label physics check ({scen}, d_F={d_f:.2f} m, "
          f"cb near ranges={np.round(cb_ranges[1:],2)}) ===")
    print("selecting FAR-by-LoS (d>d_F) users with c.delay-label y_r>0")
    print("dB col = genie polar-vs-DFT advantage (how decisively j>0 beats far-field)\n")
    hdr = (f"{'user':>6} {'LoS(m)':>7} | {'r_cdel*':>8} {'r_inter*':>9} "
           f"{'minCdel':>8} {'minIntr':>8} | {'yrC':>4} {'yrI':>4} | "
           f"{'dB_C':>6} {'dB_I':>6}")
    print(hdr)
    print("-" * len(hdr))

    found = 0
    for u in user_idxs:
        az, r_cdelay, gains, idx = extract_paths(d, u)
        if az.size == 0:
            continue
        los = float(np.linalg.norm(rx[u, :3] - tx))
        if los <= d_f:
            continue
        h_c = channel(az, r_cdelay, gains, f_c)
        (yt_c, yr_c), _ = compute_optimal_label(h_c, codebook, B_THETA, B_R)
        if yr_c == 0:
            continue
        # inter_pos ranges for the SAME paths
        r_inter = np.array([range_via_inter_pos(d, u, int(k)) for k in idx])
        h_i = channel(az, r_inter, gains, f_c)
        (yt_i, yr_i), _ = compute_optimal_label(h_i, codebook, B_THETA, B_R)
        # driving path = path most aligned with the optimal (c.delay) codeword
        f_opt = codebook[yt_c * B_R + yr_c]
        contrib = [abs(np.vdot(f_opt, g * spherical_steering(a, float(r), N_side=N_SIDE,
                   spacing=SPACING, f_c=f_c)))
                   for a, r, g in zip(az, r_cdelay, gains)]
        drive = int(np.argmax(contrib))
        print(f"{u:>6} {los:>7.1f} | {r_cdelay[drive]:>8.1f} {r_inter[drive]:>9.1f} "
              f"{r_cdelay.min():>8.1f} {r_inter.min():>8.1f} | {yr_c:>4} {yr_i:>4} | "
              f"{polar_vs_dft_db(h_c):>6.2f} {polar_vs_dft_db(h_i):>6.2f}")
        found += 1
        if found >= want:
            break

    print("\nr_cdel*/r_inter* = the DRIVING (strongest-aligned) path's range under each "
          "convention.")
    print("dB_C/dB_I = polar-vs-DFT genie advantage of the channel built with c.delay / "
          "inter_pos ranges.")
    print("Physical near-field => the j>0 codeword beats DFT by a real margin (not ~0 dB) "
          "AND a short geometric (inter_pos) bounce exists.")


if __name__ == "__main__":
    main()
