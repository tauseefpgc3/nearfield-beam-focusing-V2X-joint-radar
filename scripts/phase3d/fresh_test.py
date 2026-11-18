"""D-034 B5: fresh test set, generated with the frozen generators at new indices.

    python scripts/phase3d/fresh_test.py      # -> data/processed/phase3d/fresh/*

Candidate k uses user index 10,000,000 + k in every generator stream (channel/geometry seed
20261116 + idx, LiDAR 20261119 + idx, radar 20261117 + idx and its noise streams); see the
D-034 deviation note for why 10,000,000 and not 1,000,000. A candidate is kept if it falls in a
test block (role 2 of the frozen block table) and lies >= 0.5 m from every training user;
generation stops at 1,200 users. Channels at 60/140 GHz, fixed radar (D-029 dev. 2) at 20 dB
only, standard-vehicle LiDAR.
"""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("phase2", "phase3", "phase3b"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import p1_channels as CH, p1_radar as R1, radar_v2 as RV, radar_bev as RB, lidar_v2 as LV, lidar_var as LR
P2 = os.path.join(ROOT, "data", "processed", "phase2"); OUT = os.path.join(ROOT, "data", "processed", "phase3d", "fresh")
OFF, N_NEW = 10_000_000, 1200
BANDS = [1.0, 5.75, 10.5, 15.25, 20.0]


def roles_table():
    R = np.zeros((4, 15), int)
    for b in range(4):
        R[b, [(b + 5 * k) % 15 for k in range(3)]] = 2; R[b, [(b + 3 + 5 * k) % 15 for k in range(2)]] = 1
    return R


def main():
    os.makedirs(OUT, exist_ok=True)
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: tr = np.sort(h["train"][:]); role0 = h["role"][:]
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: ant0 = h["ant_xyz"][:]; th0 = h["theta"][:]; d0 = h["d"][:]
    RT = roles_table()
    # check the block table reproduces the frozen roles
    sec0 = np.minimum(((np.rad2deg(th0) + 60) // 8).astype(int), 14); band0 = np.minimum(np.searchsorted(BANDS, d0, side="right") - 1, 3)
    assert np.array_equal(RT[band0, sec0], role0), "block table mismatch"
    tree = cKDTree(ant0[tr, :2]); users, k, tried, rej = [], 0, 0, dict(not_test_block=0, near_train=0)
    while len(users) < N_NEW:
        idx = OFF + k; k += 1; tried += 1
        d, th, psi, pts, g = CH.draw(idx)
        sec = min(int((np.rad2deg(th) + 60) // 8), 14); band = min(int(np.searchsorted(BANDS, d, side="right") - 1), 3)
        if RT[band, sec] != 2: rej["not_test_block"] += 1; continue
        if tree.query(pts[0][:2])[0] < 0.5: rej["near_train"] += 1; continue
        users.append((idx, d, th, psi, pts, g))
    n = len(users); ant = np.stack([u[4][0] for u in users]); psi = np.array([u[3] for u in users])
    with h5py.File(os.path.join(OUT, "users.h5"), "w") as hf:
        hf["idx"] = np.array([u[0] for u in users]); hf["d"] = np.array([u[1] for u in users]); hf["theta"] = np.array([u[2] for u in users])
        hf["psi"] = psi; hf["ant_xyz"] = ant; hf.attrs.update(offset=OFF, candidates=tried, **rej)
    print("users", n, "candidates", tried, rej, flush=True)
    for f in (60, 140):
        lam = CH.C / (f * 1e9); Fc, u, rings = CH.codebook(lam)
        H = np.stack([CH.channel(p, g, lam) for (_, _, _, _, p, g) in users]); G = np.abs(H @ Fc.conj().T) ** 2
        gmax = G.max(1); lab = np.argmax(G >= (1 - 1e-12) * gmax[:, None], axis=1); adv = 10 * np.log10(gmax / G[:, 0::8].max(1))
        with h5py.File(os.path.join(OUT, f"channels_{f}GHz.h5"), "w") as hf:
            hf["gains"] = G; hf["label"] = lab; hf["adv_db"] = adv; hf["u_grid"] = u; hf["rings"] = rings
        print(f, "GHz adv>=3:", int((adv >= 3).sum()), flush=True)
    # LiDAR (standard vehicles), stream seeds shifted by OFF
    LV.SEED = LV.SEED + OFF
    std = LR.standard_vehicles(n); Bs, clouds = LR.render_bev(np.arange(n), ant, psi, std)
    with h5py.File(os.path.join(OUT, "lidar_bev.h5"), "w") as hf: hf["bev"] = Bs
    np.savez_compressed(os.path.join(OUT, "lidar_clouds.npz"), *clouds); print("lidar done", flush=True)
    # radar, fixed scatterers, 20 dB only; stream seeds shifted by OFF
    R1.BASE = R1.BASE + OFF; R1.SNRS = (20,); R1.reflectors = RV.reflectors; R1.OUT = OUT
    R1.main()
    G_ = RB.grid().cuda()
    with h5py.File(os.path.join(OUT, "radar_snr20.h5"), "r") as hf, h5py.File(os.path.join(OUT, "radar_bev20.h5"), "w") as ho:
        ra = torch.from_numpy(hf["radar_features"][:, 1:2, :, :101]).cuda(); out = []
        for b in range(0, n, 200):
            out.append(F.grid_sample(ra[b:b + 200], G_.expand(len(ra[b:b + 200]), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True).cpu().numpy())
        ho["bev"] = np.concatenate(out).astype(np.float16)
    print("radar done", flush=True)
    json.dump(dict(users=n, candidates=tried, rejected=rej, offset=OFF), open(os.path.join(ROOT, "results", "phase3d", "fresh_generation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
