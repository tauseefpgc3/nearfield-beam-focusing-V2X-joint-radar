"""D-031 radar BEV: frozen 20 dB range-angle channel (Phase 2a) bilinearly resampled onto the
LiDAR BEV grid (0.2 m, x in [-3.2, 22.4), y in [-20.8, 20.8)) at the known antenna height
1.5 m, seen from the radar (0, 0.5, 1.2). Cells outside the RA grid = 0.
Writes data/processed/phase3/radar_bev20.h5 (6000, 1, 128, 208) float16."""
import os, sys
import h5py, numpy as np, torch
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_v2 import BX0, BY0, CELL, NX, NY, P2, P3
RADAR = np.array([0.0, 0.5, 1.2])


def grid():
    x = BX0 + CELL * (np.arange(NX) + 0.5); y = BY0 + CELL * (np.arange(NY) + 0.5)
    X, Y = np.meshgrid(x, y, indexing="ij")
    rel = np.stack([X - RADAR[0], Y - RADAR[1], np.full_like(X, 1.5 - RADAR[2])], -1); R = np.linalg.norm(rel, axis=-1)
    phi = np.rad2deg(np.arcsin(rel[..., 1] / R))
    a = (phi + 75) / 1.5; k = R / 0.10                       # fractional angle / range bins
    g = np.stack([a / 100 * 2 - 1, k / 255 * 2 - 1], -1)     # grid_sample: (x = width = angle, y = height = range)
    g[(a < 0) | (a > 100) | (k > 255)] = 2.0                # outside -> zero padding
    return torch.tensor(g[None], dtype=torch.float32)


def main():
    G = grid().cuda()
    with h5py.File(os.path.join(P2, "radar_snr20.h5"), "r") as hf, h5py.File(os.path.join(P3, "radar_bev20.h5"), "w") as ho:
        n = hf["radar_features"].shape[0]; D = ho.create_dataset("bev", (n, 1, NX, NY), np.float16, chunks=(1, 1, NX, NY))
        for b in range(0, n, 200):
            ra = torch.from_numpy(hf["radar_features"][b:b + 200, 1:2, :, :101]).cuda()
            D[b:b + len(ra)] = F.grid_sample(ra, G.expand(len(ra), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True).cpu().numpy().astype(np.float16)
    print("done", n)


if __name__ == "__main__":
    main()
