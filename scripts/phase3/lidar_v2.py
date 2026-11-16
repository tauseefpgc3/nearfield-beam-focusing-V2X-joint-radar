"""D-031 realistic LiDAR: Pandar64-class spinning LiDAR at (0, 0, 2.0 m), ray-cast against a
two-box car (body 4.5x1.8x1.0, rear-biased cabin 2.4x1.6x0.5) and the ground plane z = 0,
range noise N(0, 2 cm), 10% return dropout, 600 occluded background points.
World frame: x boresight, y along the array, z up, ground z = 0. Points are cropped to the
existing ROI x in [-20, 40], y in [-20, 20] m.

    python scripts/phase3/lidar_v2.py          # writes data/processed/phase3/lidar_bev.h5
"""
import os

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P2 = os.path.join(ROOT, "data", "processed", "phase2")
P3 = os.path.join(ROOT, "data", "processed", "phase3")
SENSOR = np.array([0.0, 0.0, 2.0])
EL = np.deg2rad(np.linspace(-25.0, 15.0, 64)); AZ = np.deg2rad(np.arange(1800) * 0.2)
R_MIN, R_MAX, SIG_R, P_DROP = 0.3, 200.0, 0.02, 0.10
BODY = (np.array([-2.25, -0.9, 0.0]), np.array([2.25, 0.9, 1.0]))
CABIN = (np.array([-1.6, -0.8, 1.0]), np.array([0.8, 0.8, 1.5]))
ANT_OFF = np.array([-1.3, 0.0])                       # antenna in vehicle frame (xi, eta)
ROI = (-20.0, 40.0, -20.0, 20.0)
# BEV grid for the networks
BX0, BY0, CELL, NX, NY = -3.2, -20.8, 0.2, 128, 208
SEED = 20261119
dev = torch.device("cuda")
_ce, _se = np.cos(EL)[:, None], np.sin(EL)[:, None]
DIRS = torch.tensor(np.stack([_ce * np.cos(AZ)[None], _ce * np.sin(AZ)[None], np.broadcast_to(_se, (64, 1800))], -1).reshape(-1, 3), device=dev)


def users():
    with h5py.File(os.path.join(P2, "users.h5"), "r") as hf:
        return hf["ant_xyz"][:], hf["psi"][:]


def vehicle_centre(ant, psi):
    c, s = np.cos(psi), np.sin(psi)
    return ant[..., :2] - np.stack([c * ANT_OFF[0] - s * ANT_OFF[1], s * ANT_OFF[0] + c * ANT_OFF[1]], -1)


def _slab(o, d, lo, hi):
    """o (B,1,3) d (B,R,3) in the vehicle frame -> entry distance (B,R), inf if no hit."""
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t1 = (torch.tensor(lo, device=dev) - o) * inv; t2 = (torch.tensor(hi, device=dev) - o) * inv
    tmin = torch.minimum(t1, t2).amax(-1); tmax = torch.maximum(t1, t2).amin(-1)
    hit = (tmax >= torch.clamp(tmin, min=0.0))
    t = torch.where(tmin > 0, tmin, tmax)
    return torch.where(hit & (t > 0), t, torch.full_like(t, np.inf))


def render(idx, ant, psi):
    """Point clouds (N,4) [x, y, z, intensity] (world frame) for user indices idx; also the
    number of returns on the vehicle (diagnostic only, not used by any estimator)."""
    cen = vehicle_centre(ant[idx], psi[idx]); B = len(idx)
    c = torch.tensor(np.cos(psi[idx]), device=dev)[:, None]; s = torch.tensor(np.sin(psi[idx]), device=dev)[:, None]
    def to_veh(v):                                     # world displacement -> vehicle frame
        return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
    o_w = torch.tensor(np.c_[SENSOR[:2][None] - cen, np.full(B, SENSOR[2])], device=dev)[:, None, :]   # sensor rel. centre
    o = to_veh(o_w); d = to_veh(DIRS[None].expand(B, -1, -1))
    t_veh = torch.minimum(_slab(o, d, *BODY), _slab(o, d, *CABIN))
    dz = DIRS[:, 2][None].expand(B, -1)
    t_gr = torch.where(dz < -1e-9, -SENSOR[2] / dz, torch.full_like(dz, np.inf))
    t = torch.minimum(t_veh, t_gr); onveh = t_veh <= t_gr
    out, nveh = [], []
    for b, i in enumerate(idx):
        rng = np.random.default_rng(SEED + int(i))
        bg = np.c_[rng.uniform(-20, 40, 600), rng.uniform(-20, 20, 600), rng.uniform(0, 4, 600)]
        # background occlusion: drop if inside the vehicle or the line of sight crosses it
        v = torch.tensor(bg - SENSOR, device=dev); L = v.norm(dim=-1); dv = v / L[:, None]
        cb = torch.tensor(np.cos(psi[i]), device=dev); sb = torch.tensor(np.sin(psi[i]), device=dev)
        ob = o[b:b + 1]
        dvv =torch.stack([cb * dv[:, 0] + sb * dv[:, 1], -sb * dv[:, 0] + cb * dv[:, 1], dv[:, 2]], -1)[None]
        tb = torch.minimum(_slab(ob, dvv, *BODY), _slab(ob, dvv, *CABIN))[0]
        keep_bg = (tb > L).cpu().numpy()
        tt = t[b].cpu().numpy(); ok = np.isfinite(tt) & (tt >= R_MIN) & (tt <= R_MAX)
        dirs = DIRS.cpu().numpy()[ok]; rr = tt[ok]; veh = onveh[b].cpu().numpy()[ok]
        inten = np.where(veh, rng.uniform(0.4, 1.0, ok.sum()), rng.uniform(0.0, 0.2, ok.sum()))
        bgr = np.linalg.norm(bg - SENSOR, axis=1); bgd = (bg - SENSOR) / bgr[:, None]
        bgi = rng.uniform(0.0, 0.5, 600)
        rr = rr + rng.normal(0, SIG_R, len(rr)); bgr = bgr + rng.normal(0, SIG_R, 600)
        drop = rng.random(len(rr)) < P_DROP; dropb = rng.random(600) < P_DROP
        pts = np.r_[SENSOR + dirs * rr[:, None], SENSOR + bgd * bgr[:, None]]
        I = np.r_[inten, bgi]; keep = np.r_[~drop, keep_bg & ~dropb]; isv = np.r_[veh, np.zeros(600, bool)]
        pts = pts[keep]; I = I[keep]; isv = isv[keep]
        m = (pts[:, 0] >= ROI[0]) & (pts[:, 0] < ROI[1]) & (pts[:, 1] >= ROI[2]) & (pts[:, 1] < ROI[3])
        out.append(np.c_[pts[m], I[m]].astype(np.float32)); nveh.append(int(isv[m].sum()))
    return out, nveh


def bev(pc):
    ix = np.floor((pc[:, 0] - BX0) / CELL).astype(int); iy = np.floor((pc[:, 1] - BY0) / CELL).astype(int)
    m = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY); ix, iy, q = ix[m], iy[m], pc[m]
    f = ix * NY + iy; n = np.bincount(f, minlength=NX * NY)
    hmax = np.zeros(NX * NY); np.maximum.at(hmax, f, np.clip(q[:, 2], 0, None))
    isum = np.bincount(f, weights=q[:, 3], minlength=NX * NY)
    out = np.stack([np.log1p(n), hmax / 2.0, np.where(n > 0, isum / np.maximum(n, 1), 0)]).reshape(3, NX, NY)
    return out.astype(np.float16)


def main():
    os.makedirs(P3, exist_ok=True)
    ant, psi = users(); n = len(ant)
    cen = vehicle_centre(ant, psi)
    # pole (0,0) inside the vehicle footprint (frozen users; reported only)
    c, s = np.cos(psi), np.sin(psi); rel = -cen
    xi = c * rel[:, 0] + s * rel[:, 1]; eta = -s * rel[:, 0] + c * rel[:, 1]
    pole = (np.abs(xi) <= 2.25) & (np.abs(eta) <= 0.9)
    with h5py.File(os.path.join(P3, "lidar_bev.h5"), "w") as hf:
        F = hf.create_dataset("bev", (n, 3, NX, NY), np.float16, chunks=(1, 3, NX, NY))
        npts = np.zeros(n, int); nveh = np.zeros(n, int)
        for b0 in range(0, n, 25):
            idx = np.arange(b0, min(b0 + 25, n)); pcs, nv = render(idx, ant, psi)
            F[b0:b0 + len(idx)] = np.stack([bev(p) for p in pcs]); npts[idx] = [len(p) for p in pcs]; nveh[idx] = nv
            if b0 % 1000 == 0:
                print(b0, flush=True)
        hf["n_points"] = npts; hf["n_vehicle_returns"] = nveh; hf["vehicle_centre"] = cen; hf["pole_in_footprint"] = pole
        hf.attrs.update(grid=f"x0={BX0} y0={BY0} cell={CELL} nx={NX} ny={NY}", channels="log1p(count), max height/2m, mean intensity")
    print("points per frame median", np.median(npts), "vehicle returns median / p5 / min", np.median(nveh), np.percentile(nveh, 5), nveh.min(),
          "pole in footprint", int(pole.sum()))


if __name__ == "__main__":
    main()
