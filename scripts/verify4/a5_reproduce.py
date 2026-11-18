"""D-034 A5: independent reproduction of the headline cells from checkpoints.

Own network definition (written from the D-031 text, loaded strict=True), own decoder, beam
set, loss and shape selection. Imports NO evaluation code from phase3/3b/3c/verify3; the
LiDAR renderer (varied test set) and the frozen geometric estimator are called as black boxes.

    python scripts/verify4/a5_reproduce.py     # -> results/verify4/a5.json
"""
import json, os, sys
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P2 = os.path.join(ROOT, "data", "processed", "phase2"); OUT = os.path.join(ROOT, "results", "verify4")
dev = torch.device("cuda")
X0, Y0, C, NX, NY = -3.2, -20.8, 0.2, 128, 208
xc = X0 + C * (np.arange(NX) + 0.5); yc = Y0 + C * (np.arange(NY) + 0.5)
SEEDS = list(range(11, 221, 11)); MS = (8, 16, 32, 64)


class N(nn.Module):
    def __init__(s, cin):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(True),
                                           nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(True))
        g = np.stack(np.meshgrid(xc / 22.4, yc / 20.8, indexing="ij"))[None]
        s.register_buffer("coords", torch.tensor(g, dtype=torch.float32))
        s.enc = nn.ModuleList([cb(cin + 2, 32), cb(32, 64), cb(64, 128), cb(128, 256)])
        s.dec = nn.ModuleList([cb(384, 128), cb(192, 64), cb(96, 32)]); s.out = nn.Conv2d(32, 1, 1)

    def forward(s, x):
        x = torch.cat([x, s.coords.expand(x.shape[0], -1, -1, -1)], 1); f = []
        for i, e in enumerate(s.enc): x = e(F.max_pool2d(x, 2) if i else x); f.append(x)
        for d, k in zip(s.dec, (2, 1, 0)):
            x = d(torch.cat([F.interpolate(x, size=f[k].shape[-2:], mode="bilinear", align_corners=False), f[k]], 1))
        return s.out(x)[:, 0]


def decode(lg):
    out = np.empty((len(lg), 2))
    for b, m in enumerate(lg):
        i, j = np.unravel_index(np.argmax(m), m.shape); a, bb = max(i - 2, 0), max(j - 2, 0)
        w = m[a:i + 3, bb:j + 3].astype(np.float64); w = np.exp(w - w.max()); w /= w.sum()
        out[b] = (w * xc[a:i + 3, None]).sum(), (w * yc[None, bb:j + 3]).sum()
    return out


def net_xy(path, X):
    ck = torch.load(path, map_location="cpu", weights_only=False); m = N(ck["cin"]); m.load_state_dict(ck["model"], strict=True); m.to(dev).eval()
    with torch.no_grad(): lg = np.concatenate([m(torch.from_numpy(X[b:b + 32]).to(dev).float()).cpu().numpy() for b in range(0, len(X), 32)])
    return decode(lg)


def beams(xy, u, rings, A, R):
    p = np.c_[xy, np.full(len(xy), -0.5)]; r = np.linalg.norm(p, axis=1); su = p[:, 1] / r
    ia = np.argsort(np.abs(su[:, None] - u[None]), 1, kind="mergesort")[:, :A]
    ir = np.argsort(np.abs(1 / r[:, None] - np.where(np.isinf(rings), 0, 1 / rings)[None]), 1, kind="mergesort")[:, :R]
    return (ia[:, :, None] * 8 + ir[:, None, :]).reshape(len(xy), -1)


def loss(G, S): return 10 * np.log10(G.max(1) / np.take_along_axis(G, S, 1).max(1))


def shapes(M): return [(a, M // a) for a in (M // 8, M // 4, M // 2, M) if a >= 1 and M // a <= 8]


def main():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: va = np.sort(h["val"][:]); te = np.sort(h["test"][:])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: ant = h["ant_xyz"][:]; psi = h["psi"][:]
    CH = {}
    for f in (60, 140):
        with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as h: CH[f] = (h["gains"][:], h["u_grid"][:], h["rings"][:])
    def bev(p, idx):
        with h5py.File(p, "r") as h: return h["bev"][idx]
    LB = os.path.join(ROOT, "data", "processed", "phase3", "lidar_bev.h5"); LBV = os.path.join(ROOT, "data", "processed", "phase3b", "lidar_bev_var.h5")
    Xva, Xte, Xtv = bev(LB, va), bev(LB, te), bev(LBV, te)
    ck = lambda sd: os.path.join(ROOT, "checkpoints", "phase3", f"lidar_H_lr0.001_s{sd}.pt")
    PV, PT, PTV = [], [], []
    for sd in SEEDS: PV.append(net_xy(ck(sd), Xva)); PT.append(net_xy(ck(sd), Xte)); PTV.append(net_xy(ck(sd), Xtv)); print("seed", sd, flush=True)
    # geometric estimator (black box) on freshly rendered clouds: standard (val, test) and varied (test)
    sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3")); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3b"))
    import lidar_estimator as GEO, lidar_v2 as LV, lidar_var as LR
    dl = json.load(open(os.path.join(ROOT, "results", "phase3", "design_lidar_estimator.json")))
    prm = dict(GEO.DEFAULT, **dl["best"]["params"]); fb = np.array(dl["fallback_xy"])
    def geo(idx, veh):
        out = []
        for b in range(0, len(idx), 25):
            for c in LR.render(idx[b:b + 25], ant, psi, veh): out.append(GEO.estimate(c, prm, fb)[0])
        return np.array(out)
    std = LR.standard_vehicles(len(ant)); var = LR.varied_vehicles(len(ant))
    gva, gte, gtv = geo(va, std), geo(te, std), geo(te, var); print("geo done", flush=True)
    res = {}
    for f in (60, 140):
        G, u, rings = CH[f]
        Sw = np.c_[np.arange(256)[None].repeat(len(te), 0) * 8, G[te][:, 0::8].argmax(1)[:, None] * 8 + np.arange(1, 8)[None]]
        res[f"sweep263|{f}"] = float(loss(G[te], Sw).mean())
        for M in MS:
            def pick(ev): return min(shapes(M), key=lambda s: loss(G[va], beams(ev, u, rings, *s)).mean())
            ln_s, ln_v = [], []
            for r in range(len(SEEDS)):
                sh = pick(PV[r]); ln_s.append(loss(G[te], beams(PT[r], u, rings, *sh)).mean()); ln_v.append(loss(G[te], beams(PTV[r], u, rings, *sh)).mean())
            sg = pick(gva)
            res[f"lidar_net|standard|{f}|{M}"] = float(np.mean(ln_s)); res[f"lidar_net|varied|{f}|{M}"] = float(np.mean(ln_v))
            res[f"geometric|standard|{f}|{M}"] = float(loss(G[te], beams(gte, u, rings, *sg)).mean())
            res[f"geometric|varied|{f}|{M}"] = float(loss(G[te], beams(gtv, u, rings, *sg)).mean())
    # compare with banked values
    b = json.load(open(os.path.join(ROOT, "results", "phase3b", "results_b.json"))); c1 = json.load(open(os.path.join(ROOT, "results", "phase3c", "c1.json")))
    ref = {}
    for f in (60, 140):
        ref[f"sweep263|{f}"] = b["tables"][f"sweep263|{f}|all"]
        for M in MS:
            ref[f"lidar_net|standard|{f}|{M}"] = b["tables"][f"lidar_net|{f}|{M}|all"]["mean"]
            ref[f"geometric|standard|{f}|{M}"] = b["tables"][f"lidar_geometric|{f}|{M}|all"]["mean"]
            ref[f"lidar_net|varied|{f}|{M}"] = c1["tables"][f"varied|single_vehicle_net|{f}|{M}|all"]["mean"]
            ref[f"geometric|varied|{f}|{M}"] = c1["tables"][f"varied|lidar_geometric|{f}|{M}|all"]["mean"]
    cmp = {k: dict(reproduced=res[k], banked=ref[k], abs_diff=abs(res[k] - ref[k])) for k in res}
    worst = max(v["abs_diff"] for v in cmp.values())
    out = dict(cells=cmp, max_abs_diff_db=worst, verdict="PASS" if worst <= 0.005 else "FAIL")
    json.dump(out, open(os.path.join(OUT, "a5.json"), "w"), indent=1)
    for k, v in cmp.items(): print(f"{k:32s} {v['reproduced']:.4f} {v['banked']:.4f} {v['abs_diff']:.2e}")
    print("max |diff|", worst, out["verdict"])


if __name__ == "__main__":
    main()
