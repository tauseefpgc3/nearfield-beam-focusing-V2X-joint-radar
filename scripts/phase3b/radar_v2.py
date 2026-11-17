"""D-029 deviation 2 / D-032: regenerate the radar with the 6 vehicle scatterers in the car body
(4.5 x 1.8 m around the true body centre = antenna + 1.3 m * heading), same rng and draw order;
everything else exactly as scripts/phase2/p1_radar.py (imported and reused unchanged).
Writes data/processed/phase3b/radar_snr{20,10,0}.h5 and radar_bev{20,10,0}.h5 (D-031 warp)."""
import os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2")); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
import p1_radar as R1
import radar_bev as RB
P2 = os.path.join(ROOT, "data", "processed", "phase2"); OUT = os.path.join(ROOT, "data", "processed", "phase3b")
ANT_BACK = 1.3


def reflectors(idx, ant_xy, psi):
    r = np.random.default_rng(R1.BASE + idx)
    R, S, V, A = [], [], [], []
    amps, phs, rel = [], [], []
    c, s_ = np.cos(psi), np.sin(psi)
    body = ant_xy + ANT_BACK * np.array([c, s_])                 # true body centre
    for _ in range(6):
        xi = r.uniform(-2.25, 2.25); eta = r.uniform(-0.9, 0.9); hh = r.uniform(0.3, 1.5)
        amps.append(r.uniform(0.3, 1.0)); phs.append(r.uniform(0, 2 * np.pi))
        rel.append(np.array([body[0] + c * xi - s_ * eta, body[1] + s_ * xi + c * eta, hh]) - R1.RADAR)
    amps = np.array(amps); amps /= np.sqrt((amps ** 2).sum())
    for p, a, ph in zip(rel, amps, phs):
        rr = np.linalg.norm(p); R.append(rr); S.append(p[1] / rr); V.append(0.0); A.append(a * np.exp(1j * ph))
    for q in range(12):                                          # clutter: unchanged
        rr = r.uniform(1, 25); az = np.deg2rad(r.uniform(-90, 90)); v = r.uniform(-16, 16)
        if q < 6:
            v = 0.0
        a = 0.25 * r.uniform(0.3, 1.0)
        R.append(rr); S.append(np.sin(az)); V.append(v); A.append(a + 0j)
    return np.array(R), np.array(S), np.array(V), np.array(A)


def main():
    os.makedirs(OUT, exist_ok=True)
    # check: clutter part identical to the original generator for a few users
    with h5py.File(os.path.join(P2, "users.h5"), "r") as hf: ant = hf["ant_xyz"][:]; psi = hf["psi"][:]
    for i in (0, 17, 4242):
        a = R1.reflectors(i, ant[i, :2], psi[i]); b = reflectors(i, ant[i, :2], psi[i])
        assert all(np.allclose(x[6:], y[6:]) for x, y in zip(a, b)), "clutter changed"
    R1.reflectors = reflectors; R1.OUT = OUT
    # p1_radar.main reads users.h5 from OUT: link a copy of the frozen users file
    import shutil; shutil.copyfile(os.path.join(P2, "users.h5"), os.path.join(OUT, "users.h5"))
    R1.main()
    G = RB.grid().cuda()
    for snr in R1.SNRS:
        with h5py.File(os.path.join(OUT, f"radar_snr{snr}.h5"), "r") as hf, h5py.File(os.path.join(OUT, f"radar_bev{snr}.h5"), "w") as ho:
            n = hf["radar_features"].shape[0]; D = ho.create_dataset("bev", (n, 1, RB.NX, RB.NY), np.float16, chunks=(1, 1, RB.NX, RB.NY))
            for b in range(0, n, 200):
                ra = torch.from_numpy(hf["radar_features"][b:b + 200, 1:2, :, :101]).cuda()
                D[b:b + len(ra)] = F.grid_sample(ra, G.expand(len(ra), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True).cpu().numpy().astype(np.float16)
        print("bev", snr, flush=True)


if __name__ == "__main__":
    main()
