"""Phase 2a (D-029) step 1b: 77 GHz FMCW radar, 12-element virtual horizontal ULA at
(0, +0.5, 1.2) m, extended vehicle (6 scatterers) + 12 clutter reflectors, SNR 20/10/0 dB.

Geometry/clutter rng = default_rng(20261117 + idx), draw order: 6 x [xi U(-2.25,2.25),
eta U(-1,1), h U(0.3,1.5), amp U(0.3,1), phase U(0,2pi)], then 12 x [range U(1,25),
azimuth U(-90,90) deg, doppler U(-16,16), amp 0.25*U(0.3,1)] (clutter: no random phase, as
the existing model). Noise rng = default_rng(20261117 + 1_000_003*(1+s) + idx), s = SNR index.
Output data/processed/phase2/radar_snr{20,10,0}.h5 (radar_features (6000,2,256,128) float32,
ra_db_min/max, rd_db_min/max) and radar_geometry.h5 (true antenna R, sin(phi) seen from radar).
"""
import os

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "processed", "phase2")
N_S, N_C, N_V = 256, 64, 12
DR = 0.10
RADAR = np.array([0.0, 0.5, 1.2])
ANG = np.deg2rad(np.arange(-75, 75.0001, 1.5))            # 101 bins, uniform in angle
SNRS = (20, 10, 0)
BASE = 20261117
dev = torch.device("cuda")


def reflectors(idx, ant_xy, psi):
    r = np.random.default_rng(BASE + idx)
    R, S, V, A = [], [], [], []
    amps, phs, rel = [], [], []
    c, s_ = np.cos(psi), np.sin(psi)
    for _ in range(6):
        xi = r.uniform(-2.25, 2.25); eta = r.uniform(-1, 1); hh = r.uniform(0.3, 1.5)
        amps.append(r.uniform(0.3, 1.0)); phs.append(r.uniform(0, 2 * np.pi))
        p = np.array([ant_xy[0] + c * xi - s_ * eta, ant_xy[1] + s_ * xi + c * eta, hh]) - RADAR
        rel.append(p)
    amps = np.array(amps); amps /= np.sqrt((amps ** 2).sum())
    for p, a, ph in zip(rel, amps, phs):
        rr = np.linalg.norm(p); R.append(rr); S.append(p[1] / rr); V.append(0.0); A.append(a * np.exp(1j * ph))
    for q in range(12):
        rr = r.uniform(1, 25); az = np.deg2rad(r.uniform(-90, 90)); v = r.uniform(-16, 16)
        if q < 6:                       # D-029 deviation 1: first 6 clutter reflectors static
            v = 0.0
        a = 0.25 * r.uniform(0.3, 1.0)
        R.append(rr); S.append(np.sin(az)); V.append(v); A.append(a + 0j)
    return np.array(R), np.array(S), np.array(V), np.array(A)


def process(cube):
    """cube (B, N_V, N_C, N_S) complex64 on GPU -> (B,2,256,128) features + dB min/max."""
    wr = torch.hann_window(N_S, periodic=False, device=dev); wd = torch.hann_window(N_C, periodic=False, device=dev)
    x = torch.fft.fft(cube * wr, n=N_S, dim=-1)
    x = torch.fft.fftshift(torch.fft.fft(x * wd[:, None], n=N_C, dim=-2), dim=-2)   # (B,V,dop,range)
    rd = (x.abs() ** 2).sum(1).transpose(1, 2)                                        # (B,256,64)
    steer = torch.exp(-1j * np.pi * torch.arange(N_V, device=dev)[None, :] * torch.sin(torch.tensor(ANG, device=dev))[:, None]).to(torch.complex64)
    bf = torch.einsum("am,bmdk->badk", steer, x)                                      # (B,101,dop,range)
    ra = (bf[:, :, N_C // 2].abs() ** 2).transpose(1, 2)    # (B,256,101) zero-Doppler bin (D-029 deviation 1)
    out = torch.zeros(cube.shape[0], 2, 256, 128, device=dev)
    mm = []
    for ch, m in ((0, rd), (1, ra)):
        db = 10 * torch.log10(m + 1e-12)
        lo = db.amin((1, 2), keepdim=True); hi = db.amax((1, 2), keepdim=True)
        out[:, ch, :, :m.shape[2]] = (db - lo) / (hi - lo).clamp_min(1e-9)
        mm.append((lo.flatten(), hi.flatten()))
    return out, mm


def main():
    with h5py.File(os.path.join(OUT, "users.h5"), "r") as hf:
        ant = hf["ant_xyz"][:]; psi = hf["psi"][:]
    n = len(ant)
    refl = [reflectors(i, ant[i, :2], psi[i]) for i in range(n)]
    relA = ant - RADAR
    with h5py.File(os.path.join(OUT, "radar_geometry.h5"), "w") as hf:
        hf["R_true"] = np.linalg.norm(relA, axis=1); hf["sin_true"] = relA[:, 1] / np.linalg.norm(relA, axis=1)
        hf.attrs.update(radar_pos=RADAR, dR=DR, ang_grid_deg=np.rad2deg(ANG))
    ns = torch.arange(N_S, device=dev, dtype=torch.float64); nc = torch.arange(N_C, device=dev, dtype=torch.float64)
    nv = torch.arange(N_V, device=dev, dtype=torch.float64)
    for si, snr in enumerate(SNRS):
        sig2 = (N_S * N_C * N_V) / 10 ** (snr / 10)
        path = os.path.join(OUT, f"radar_snr{snr}.h5")
        with h5py.File(path, "w") as hf:
            F = hf.create_dataset("radar_features", (n, 2, 256, 128), np.float32, chunks=(1, 2, 256, 128))
            mins = {k: np.zeros(n, np.float32) for k in ("rd_db_min", "rd_db_max", "ra_db_min", "ra_db_max")}
            for b0 in range(0, n, 50):
                idx = range(b0, min(b0 + 50, n))
                cubes = []
                for i in idx:
                    R, S, V, A = (torch.tensor(z, device=dev) for z in refl[i])
                    ph = (2 * np.pi * (R[:, None, None, None] / DR / N_S) * ns[None, None, None, :]
                          + 2 * np.pi * (V[:, None, None, None] / N_C) * nc[None, None, :, None]
                          + np.pi * S[:, None, None, None] * nv[None, :, None, None])
                    cube = (A[:, None, None, None] * torch.exp(1j * ph)).sum(0)
                    rng = np.random.default_rng(BASE + 1_000_003 * (1 + si) + i)
                    noise = np.sqrt(sig2 / 2) * (rng.standard_normal((N_V, N_C, N_S)) + 1j * rng.standard_normal((N_V, N_C, N_S)))
                    cubes.append((cube + torch.tensor(noise, device=dev)).to(torch.complex64))
                feat, mm = process(torch.stack(cubes))
                F[b0:b0 + len(idx)] = feat.cpu().numpy()
                for (lo, hi), nm in zip(mm, ("rd", "ra")):
                    mins[f"{nm}_db_min"][b0:b0 + len(idx)] = lo.cpu().numpy(); mins[f"{nm}_db_max"][b0:b0 + len(idx)] = hi.cpu().numpy()
            for k, v in mins.items():
                hf[k] = v
            hf.attrs.update(snr_db=snr, definition="coherent-integration SNR of the vehicle return; sigma^2 per raw sample = 256*64*12/10^(SNR/10)",
                            n_ang=len(ANG), ang_grid_deg=np.rad2deg(ANG), dR=DR, radar_pos=RADAR)
        print("wrote", os.path.basename(path), flush=True)


if __name__ == "__main__":
    main()
