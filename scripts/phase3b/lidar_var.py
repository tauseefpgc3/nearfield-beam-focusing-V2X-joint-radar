"""D-032 parametrised LiDAR renderer: the scripts/phase3/lidar_v2.py ray caster generalised to
per-user vehicle boxes and antenna offsets, an elevation-channel subset, range noise and dropout.
With the standard parameters it reproduces lidar_v2 bit-for-bit (checked in main()).

    python scripts/phase3b/lidar_var.py      # writes data/processed/phase3b/lidar_bev_var.h5
"""
import os, sys
import h5py, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3"))
import lidar_v2 as LV
dev = LV.dev; OUT = os.path.join(ROOT, "data", "processed", "phase3b")
VSEED = 20261121


def standard_vehicles(n):
    """Per-user (body_lo, body_hi, cab_lo, cab_hi, ant_xi) for the frozen template."""
    return dict(body_lo=np.tile(LV.BODY[0], (n, 1)), body_hi=np.tile(LV.BODY[1], (n, 1)),
                cab_lo=np.tile(LV.CABIN[0], (n, 1)), cab_hi=np.tile(LV.CABIN[1], (n, 1)),
                ant_xi=np.full(n, LV.ANT_OFF[0]), L=np.full(n, 4.5), W=np.full(n, 1.8), H=np.full(n, 1.5))


def varied_vehicles(n):
    """D-032 item 3a, rng default_rng(20261121 + idx); draw order L, W, H, antenna fraction."""
    v = {k: np.zeros((n, 3)) for k in ("body_lo", "body_hi", "cab_lo", "cab_hi")}
    v.update(ant_xi=np.zeros(n), L=np.zeros(n), W=np.zeros(n), H=np.zeros(n))
    for i in range(n):
        r = np.random.default_rng(VSEED + i)
        L = r.uniform(3.8, 5.0); W = r.uniform(1.65, 1.95); H = r.uniform(1.35, 1.50); fa = r.uniform(-0.34, -0.20)
        hb = 2.0 / 3.0 * H; cr = -0.356 * L; cl = 0.533 * L
        v["body_lo"][i] = (-L / 2, -W / 2, 0.0); v["body_hi"][i] = (L / 2, W / 2, hb)
        v["cab_lo"][i] = (cr, -(W - 0.2) / 2, hb); v["cab_hi"][i] = (cr + cl, (W - 0.2) / 2, H)
        v["ant_xi"][i] = fa * L; v["L"][i] = L; v["W"][i] = W; v["H"][i] = H
    return v


def centre_of(ant, psi, ant_xi):
    return ant[:, :2] - np.c_[np.cos(psi) * ant_xi, np.sin(psi) * ant_xi]


def _slab(o, d, lo, hi):
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t1 = (lo - o) * inv; t2 = (hi - o) * inv
    tmin = torch.minimum(t1, t2).amax(-1); tmax = torch.maximum(t1, t2).amin(-1)
    hit = tmax >= torch.clamp(tmin, min=0.0); t = torch.where(tmin > 0, tmin, tmax)
    return torch.where(hit & (t > 0), t, torch.full_like(t, np.inf))


def render(idx, ant, psi, veh, ch_step=1, sig_r=LV.SIG_R, p_drop=LV.P_DROP):
    """As lidar_v2.render; veh = per-user dict indexed by user id; ch_step keeps every k-th
    elevation channel (same FOV)."""
    idx = np.asarray(idx); B = len(idx)
    dirs_all = LV.DIRS.view(64, 1800, 3)[::ch_step].reshape(-1, 3)
    cen = centre_of(ant[idx], psi[idx], veh["ant_xi"][idx])
    T = lambda a: torch.tensor(a[idx][:, None, :], device=dev)
    blo, bhi, clo, chi = T(veh["body_lo"]), T(veh["body_hi"]), T(veh["cab_lo"]), T(veh["cab_hi"])
    c = torch.tensor(np.cos(psi[idx]), device=dev)[:, None]; s = torch.tensor(np.sin(psi[idx]), device=dev)[:, None]
    def to_veh(v):
        return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
    o = to_veh(torch.tensor(np.c_[LV.SENSOR[:2][None] - cen, np.full(B, LV.SENSOR[2])], device=dev)[:, None, :])
    d = to_veh(dirs_all[None].expand(B, -1, -1))
    t_veh = torch.minimum(_slab(o, d, blo, bhi), _slab(o, d, clo, chi))
    dz = dirs_all[:, 2][None].expand(B, -1)
    t_gr = torch.where(dz < -1e-9, -LV.SENSOR[2] / dz, torch.full_like(dz, np.inf))
    t = torch.minimum(t_veh, t_gr); onveh = t_veh <= t_gr
    dirs_np = dirs_all.cpu().numpy(); out = []
    for b, i in enumerate(idx):
        rng = np.random.default_rng(LV.SEED + int(i))
        bg = np.c_[rng.uniform(-20, 40, 600), rng.uniform(-20, 20, 600), rng.uniform(0, 4, 600)]
        v = torch.tensor(bg - LV.SENSOR, device=dev); Ln = v.norm(dim=-1); dv = v / Ln[:, None]
        cb = torch.tensor(np.cos(psi[i]), device=dev); sb = torch.tensor(np.sin(psi[i]), device=dev)
        dvv = torch.stack([cb * dv[:, 0] + sb * dv[:, 1], -sb * dv[:, 0] + cb * dv[:, 1], dv[:, 2]], -1)[None]
        ob = o[b:b + 1]
        tb = torch.minimum(_slab(ob, dvv, blo[b:b + 1], bhi[b:b + 1]), _slab(ob, dvv, clo[b:b + 1], chi[b:b + 1]))[0]
        keep_bg = (tb > Ln).cpu().numpy()
        tt = t[b].cpu().numpy(); ok = np.isfinite(tt) & (tt >= LV.R_MIN) & (tt <= LV.R_MAX)
        dd = dirs_np[ok]; rr = tt[ok]; vh = onveh[b].cpu().numpy()[ok]
        inten = np.where(vh, rng.uniform(0.4, 1.0, ok.sum()), rng.uniform(0.0, 0.2, ok.sum()))
        bgr = np.linalg.norm(bg - LV.SENSOR, axis=1); bgd = (bg - LV.SENSOR) / bgr[:, None]
        bgi = rng.uniform(0.0, 0.5, 600)
        rr = rr + rng.normal(0, sig_r, len(rr)); bgr = bgr + rng.normal(0, sig_r, 600)
        drop = rng.random(len(rr)) < p_drop; dropb = rng.random(600) < p_drop
        pts = np.r_[LV.SENSOR + dd * rr[:, None], LV.SENSOR + bgd * bgr[:, None]]
        I = np.r_[inten, bgi]; keep = np.r_[~drop, keep_bg & ~dropb]
        pts = pts[keep]; I = I[keep]
        m = (pts[:, 0] >= LV.ROI[0]) & (pts[:, 0] < LV.ROI[1]) & (pts[:, 1] >= LV.ROI[2]) & (pts[:, 1] < LV.ROI[3])
        out.append(np.c_[pts[m], I[m]].astype(np.float32))
    return out


def render_bev(idx, ant, psi, veh, **kw):
    clouds, bevs = [], []
    for b in range(0, len(idx), 25):
        pcs = render(idx[b:b + 25], ant, psi, veh, **kw); clouds += pcs; bevs += [LV.bev(p) for p in pcs]
    return np.stack(bevs), clouds


def main():
    ant, psi = LV.users(); n = len(ant)
    # bit-for-bit check against the frozen renderer / stored BEV
    chk = np.array([5, 77, 1234, 4321, 5999])
    Bs, _ = render_bev(chk, ant, psi, standard_vehicles(n))
    with h5py.File(os.path.join(LV.P3, "lidar_bev.h5"), "r") as hf: ref = hf["bev"][np.sort(chk)]
    assert np.array_equal(Bs[np.argsort(chk)], ref), "standard render differs from lidar_v2"
    print("standard-vehicle reproduction: exact", flush=True)
    veh = varied_vehicles(n)
    with h5py.File(os.path.join(OUT, "lidar_bev_var.h5"), "w") as hf:
        D = hf.create_dataset("bev", (n, 3, LV.NX, LV.NY), np.float16, chunks=(1, 3, LV.NX, LV.NY))
        for b0 in range(0, n, 25):
            ix = np.arange(b0, min(b0 + 25, n)); D[b0:b0 + len(ix)] = render_bev(ix, ant, psi, veh)[0]
        for k, v in veh.items(): hf[k] = v
    print("varied: L %.2f-%.2f W %.2f-%.2f H %.2f-%.2f ant_xi %.2f-%.2f" % (veh["L"].min(), veh["L"].max(), veh["W"].min(), veh["W"].max(),
          veh["H"].min(), veh["H"].max(), veh["ant_xi"].min(), veh["ant_xi"].max()))


if __name__ == "__main__":
    main()
