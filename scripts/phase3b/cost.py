"""D-032 B5: parameters and per-frame cost (batch 1, median over 200 test frames after warm-up).
Preprocessing: LiDAR point cloud -> BEV (numpy CPU; torch GPU version checked equal);
radar cube -> zero-Doppler RA map -> BEV warp (the p1_radar / radar_bev code, on GPU and CPU).
Inference: network forward + local soft-argmax decode. Geometric estimator and radar picker: CPU.
Simulation (ray casting, cube synthesis) is excluded. Writes results/phase3b/cost.json."""
import json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("verify3", "phase3", "phase2"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
from common import split, users, Net, decode_local, CK
import lidar_v2 as LV, lidar_estimator as GEO, p1_radar as R1, radar_bev as RBV, p3_eval as E
sys.path.insert(0, HERE); import radar_v2 as RV2
N = 200; te = split()["test"][:N]; ant, psi = users()
dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json")))
PRM = dict(GEO.DEFAULT, **dl["best"]["params"]); FB = np.array(dl["fallback_xy"])


def med(fn, items, sync=False, warm=10):
    for it in items[:warm]: fn(it)
    ts = []
    for it in items:
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(it)
        if sync: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def bev_torch(pc, dev):
    q = torch.as_tensor(pc, device=dev)
    ix = torch.floor((q[:, 0] - LV.BX0) / LV.CELL).long(); iy = torch.floor((q[:, 1] - LV.BY0) / LV.CELL).long()
    m = (ix >= 0) & (ix < LV.NX) & (iy >= 0) & (iy < LV.NY); f = ix[m] * LV.NY + iy[m]; q = q[m]
    n = torch.bincount(f, minlength=LV.NX * LV.NY).float()
    h = torch.zeros(LV.NX * LV.NY, device=dev).scatter_reduce(0, f, q[:, 2].clamp_min(0), "amax", include_self=True)
    isum = torch.bincount(f, weights=q[:, 3], minlength=LV.NX * LV.NY)
    return torch.stack([torch.log1p(n), h / 2.0, torch.where(n > 0, isum / n.clamp_min(1), torch.zeros_like(n))]).view(3, LV.NX, LV.NY)


def main():
    res = {}
    clouds = []
    for b in range(0, N, 25): clouds += LV.render(te[b:b + 25], ant, psi)[0]
    # float32 floor at cell boundaries can differ by one ulp between CUDA and numpy, moving a
    # point to the neighbouring cell; record the fraction of cells that agree (timing check only)
    agree = [np.mean(np.isclose(bev_torch(c, "cuda").cpu().numpy(), LV.bev(c).astype(np.float32), atol=1e-2)) for c in clouds[:20]]
    res["gpu_bev_cell_agreement_min"] = float(min(agree)); assert min(agree) > 0.995
    res["points_per_frame_median"] = float(np.median([len(c) for c in clouds]))
    res["lidar_bev_ms"] = dict(cpu_numpy=med(LV.bev, clouds), gpu_torch=med(lambda c: bev_torch(c, "cuda"), clouds, sync=True))
    res["geometric_estimator_cpu_ms"] = med(lambda c: GEO.estimate(c, PRM, FB), clouds)
    # radar: synthesise cubes once (simulation, not timed), then time processing + warp
    ns = torch.arange(R1.N_S, dtype=torch.float64); nc = torch.arange(R1.N_C, dtype=torch.float64); nv = torch.arange(R1.N_V, dtype=torch.float64)
    cubes = []
    for i in te[:60]:
        R, S, V, A = (torch.tensor(z) for z in RV2.reflectors(int(i), ant[i, :2], psi[i]))
        ph = (2 * np.pi * (R[:, None, None, None] / R1.DR / R1.N_S) * ns + 2 * np.pi * (V[:, None, None, None] / R1.N_C) * nc[None, None, :, None]
              + np.pi * S[:, None, None, None] * nv[None, :, None, None])
        cubes.append((A[:, None, None, None] * torch.exp(1j * ph)).sum(0).to(torch.complex64)[None])
    G = RBV.grid()
    for dev in ("cuda", "cpu"):
        R1.dev = torch.device(dev); Gd = G.to(dev); cd = [c.to(dev) for c in cubes]
        def radar_pre(c):
            feat, _ = R1.process(c); return F.grid_sample(feat[:, 1:2, :, :101], Gd, mode="bilinear", padding_mode="zeros", align_corners=True)
        res.setdefault("radar_preprocess_ms", {})[dev] = med(radar_pre, cd, sync=dev == "cuda")
    R1.dev = torch.device("cuda")
    import h5py
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase3b", "radar_snr20.h5"), "r") as h:
        X = h["radar_features"][te[:60]]; lo = h["ra_db_min"][te[:60]]; hi = h["ra_db_max"][te[:60]]
    res["radar_picker_cpu_ms"] = med(lambda k: E.estimate(E.ra_db(X[k:k + 1], lo[k:k + 1], hi[k:k + 1]), "centroid"), list(range(60)))
    # networks
    for s, suf, cin in (("radar", "_r2", 1), ("lidar", "", 3), ("fused", "_r2", 4)):
        ck = torch.load(os.path.join(CK, f"{s}_H_lr0.001_s11{suf}.pt"), map_location="cpu", weights_only=False)
        row = dict(parameters=int(sum(p.numel() for p in Net(cin).parameters())))
        xs = [torch.randn(1, cin, LV.NX, LV.NY) for _ in range(60)]
        for dev in ("cuda", "cpu"):
            m = Net(cin); m.load_state_dict(ck["model"]); m = m.to(dev).eval(); xd = [x.to(dev) for x in xs]
            def inf(x):
                with torch.no_grad(): return decode_local(m(x).cpu().numpy())
            row[f"inference_ms_{dev}"] = med(inf, xd if dev == "cuda" else xd[:40], sync=dev == "cuda", warm=5)
        res[f"{s}_net"] = row
    res["hardware"] = dict(gpu=torch.cuda.get_device_name(0), cpu_threads=torch.get_num_threads())
    json.dump(res, open(os.path.join(ROOT, "results", "phase3b", "cost.json"), "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
