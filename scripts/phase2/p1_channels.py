"""Phase 2a (D-029) step 1a: users, 3-D channels, 256x8 polar codebook, raw genie labels,
and the genie-vs-matched-filter gate (STOP if the mean gap > 1 dB at either frequency).

Draw order per user, rng = default_rng(20261116 + idx):
  d ~ U[1,20], theta ~ U[-60,60] deg, psi ~ U[0, 2pi), K ~ U{2,3,4},
  then per reflection: rho ~ U[1,25], phi ~ U[-90,90] deg, z ~ U[0.5,3.0],
  atten_dB ~ U[10,20], phase ~ U[0, 2pi).
Output data/processed/phase2/channels_{f}GHz.h5 and data/processed/phase2/users.h5.
"""
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "processed", "phase2")
RES = os.path.join(ROOT, "results", "phase2")
os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
C = 299_792_458.0
N, B_TH, B_R, N_USERS, SEED = 128, 256, 8, 6000, 20261116
H_RSU, H_VEH = 2.0, 1.5


def draw(idx):
    r = np.random.default_rng(SEED + idx)
    d = r.uniform(1, 20); th = np.deg2rad(r.uniform(-60, 60)); psi = r.uniform(0, 2 * np.pi)
    K = int(r.integers(2, 5))
    pts = [(d * np.cos(th), d * np.sin(th), H_VEH)]; g = [1.0 + 0j]
    for _ in range(K):
        rho = r.uniform(1, 25); phi = np.deg2rad(r.uniform(-90, 90)); z = r.uniform(0.5, 3.0)
        att = r.uniform(10, 20); ph = r.uniform(0, 2 * np.pi)
        pts.append((rho * np.cos(phi), rho * np.sin(phi), z)); g.append(10 ** (-att / 20) * np.exp(1j * ph))
    return d, th, psi, np.array(pts), np.array(g)


def elem_y(lam):
    return (np.arange(N) - (N - 1) / 2) * lam / 2


def channel(pts, g, lam):
    y = elem_y(lam); q0 = np.array([0, 0, H_RSU]); h = np.zeros(N, complex)
    for p, gg in zip(pts, g):
        rel = p - q0
        dist = np.sqrt(rel[0] ** 2 + (rel[1] - y) ** 2 + rel[2] ** 2)
        h += gg * np.exp(-1j * 2 * np.pi / lam * (dist - np.linalg.norm(rel)))
    return h


def codebook(lam):
    y = elem_y(lam); u = (2 * np.arange(B_TH) - B_TH + 1) / B_TH
    rings = np.r_[np.inf, 1.2 * (N - 1) ** 2 * lam / 8 / np.arange(1, B_R)]
    F = np.empty((B_TH * B_R, N), complex)
    for i, uu in enumerate(u):
        for j, r in enumerate(rings):
            ph = 2 * np.pi / lam * uu * y if np.isinf(r) else -2 * np.pi / lam * (np.sqrt(r * r - 2 * r * uu * y + y * y) - r)
            F[i * B_R + j] = np.exp(1j * ph) / np.sqrt(N)
    return F, u, rings


def main():
    users = [draw(i) for i in range(N_USERS)]
    d = np.array([x[0] for x in users]); th = np.array([x[1] for x in users]); psi = np.array([x[2] for x in users])
    with h5py.File(os.path.join(OUT, "users.h5"), "w") as hf:
        hf["d"] = d; hf["theta"] = th; hf["psi"] = psi
        hf["ant_xyz"] = np.stack([d * np.cos(th), d * np.sin(th), np.full_like(d, H_VEH)], 1)
        hf.attrs.update(seed=SEED, h_rsu=H_RSU, h_veh=H_VEH)
    checks = {}
    for f in (60, 140):
        lam = C / (f * 1e9)
        F, u, rings = codebook(lam)
        H = np.stack([channel(p, g, lam) for (_, _, _, p, g) in users])
        G = np.abs(H @ F.conj().T) ** 2                       # (6000, 2048), float64
        gmax = G.max(1)
        lab = np.argmax(G >= (1 - 1e-12) * gmax[:, None], axis=1)   # smallest index within tolerance
        adv = 10 * np.log10(gmax / G[:, 0::B_R].max(1))
        gap = 10 * np.log10(np.sum(np.abs(H) ** 2, 1) / gmax)
        with h5py.File(os.path.join(OUT, f"channels_{f}GHz.h5"), "w") as hf:
            hf["channel"] = H; hf["gains"] = G; hf["label"] = lab
            hf["y_theta"] = lab // B_R; hf["y_r"] = lab % B_R; hf["adv_db"] = adv; hf["gap_db"] = gap
            hf["u_grid"] = u; hf["rings"] = rings
            hf.attrs.update(f_c=f * 1e9, N=N, B_theta=B_TH, B_r=B_R, d_F=2 * ((N - 1) * lam / 2) ** 2 / lam)
        checks[str(f)] = dict(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)),
                              gap_p95=float(np.percentile(gap, 95)), gap_max=float(gap.max()),
                              frac_adv_ge3=float((adv >= 3).mean()), genie_dft_mean=float(adv.mean()),
                              label_ring_hist=np.bincount(lab % B_R, minlength=B_R).tolist(),
                              n_angle_classes=int(len(np.unique(lab // B_R))),
                              n_near_ties=int(((G >= (1 - 1e-12) * gmax[:, None]).sum(1) > 1).sum()))
        print(f, json.dumps(checks[str(f)]), flush=True)
    ok = all(c["gap_mean"] <= 1.0 for c in checks.values())
    checks["gate_pass"] = bool(ok)
    json.dump(checks, open(os.path.join(RES, "checks_gap.json"), "w"), indent=1)
    print("GAP GATE", "PASS" if ok else "FAIL -> STOP (D-029)")
    sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()
