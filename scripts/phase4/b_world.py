"""D-036 Experiment B: multi-car scenes with the frozen channel, array, codebook and LiDAR models.

Users: p1_channels.draw order with default_rng(20270101 + i). User car: D-032 varied-vehicle
distribution, default_rng(20270601 + i). Other vehicles: default_rng(20270201 + i) (draw order
documented in scene()). LiDAR: the frozen D-031 ray caster (lidar_v2 / lidar_var) extended to
several vehicles: every ray's first hit over all vehicle boxes and the ground; background
points removed when inside or behind any vehicle; same noise, intensities, dropout and ROI.
"""
import os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("phase2", "phase3", "phase3b"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import p1_channels as P1, lidar_v2 as LV, lidar_var as LR
from p1_lidar_split import roles_table, BANDS
dev = LV.dev
N_USERS = 6000; USEED, VSEED, SSEED, NSEED, PSEED, GSEED = 20270101, 20270601, 20270201, 20270301, 20270401, 20270501
LANE = 3.5; PARK_W = 2.5; DT = 0.1


# ------------------------------------------------------------------ users and split
def draw(i):
    """p1_channels.draw with the new user seed (same distribution and draw order)."""
    r = np.random.default_rng(USEED + i)
    d = r.uniform(1, 20); th = np.deg2rad(r.uniform(-60, 60)); psi = r.uniform(0, 2 * np.pi); K = int(r.integers(2, 5))
    pts = [(d * np.cos(th), d * np.sin(th), P1.H_VEH)]; g = [1.0 + 0j]
    for _ in range(K):
        rho = r.uniform(1, 25); phi = np.deg2rad(r.uniform(-90, 90)); z = r.uniform(0.5, 3.0); att = r.uniform(10, 20); ph = r.uniform(0, 2 * np.pi)
        pts.append((rho * np.cos(phi), rho * np.sin(phi), z)); g.append(10 ** (-att / 20) * np.exp(1j * ph))
    return d, th, psi, np.array(pts), np.array(g)


def users():
    U = [draw(i) for i in range(N_USERS)]
    d = np.array([u[0] for u in U]); th = np.array([u[1] for u in U]); psi = np.array([u[2] for u in U])
    ant = np.c_[d * np.cos(th), d * np.sin(th), np.full(N_USERS, P1.H_VEH)]
    return d, th, psi, ant


def split(d, th, ant):
    """D-029 spatial-block rule: 15 sectors x 4 bands, deterministic rotation, 0.5 m training guard."""
    sec = np.minimum(((np.rad2deg(th) + 60) // 8).astype(int), 14); band = np.minimum(np.searchsorted(BANDS, d, side="right") - 1, 3)
    role = roles_table()[band, sec]; xy = ant[:, :2]; test = np.where(role == 2)[0]
    near = cKDTree(xy[test]).query(xy, distance_upper_bound=0.5)[0] < 0.5
    return dict(train=np.where((role == 0) & ~near)[0], val=np.where(role == 1)[0], test=test, guard_dropped=int(((role == 0) & near).sum()))


def gains(i, lam, F, disp=(0.0, 0.0)):
    _, _, _, pts, g = draw(i); pts = pts.copy(); pts[0, :2] += disp
    return np.abs(F.conj() @ P1.channel(pts, g, lam)) ** 2


# ------------------------------------------------------------------ vehicles
def two_box(L, W, H):
    hb = 2.0 / 3.0 * H; cr = -0.356 * L; cl = 0.533 * L
    return np.array([[-L / 2, -W / 2, 0.0], [L / 2, W / 2, hb], [cr, -(W - 0.2) / 2, hb], [cr + cl, (W - 0.2) / 2, H]])


def one_box(L, W, H):
    b = np.array([[-L / 2, -W / 2, 0.0], [L / 2, W / 2, H]]); return np.r_[b, b]


def user_car(i):
    r = np.random.default_rng(VSEED + i)
    L = r.uniform(3.8, 5.0); W = r.uniform(1.65, 1.95); H = r.uniform(1.35, 1.50); fa = r.uniform(-0.34, -0.20)
    return dict(boxes=two_box(L, W, H), L=L, W=W, H=H, ant_xi=fa * L, kind="user")


def _corners(c, h, L, W):
    u = np.array([np.cos(h), np.sin(h)]); n = np.array([-np.sin(h), np.cos(h)])
    return np.array([c + a * L / 2 * u + b * W / 2 * n for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _overlap(A, B):
    for P in (A, B):
        for k in range(2):
            e = P[k + 1] - P[k]; ax = np.array([-e[1], e[0]])
            pa, pb = A @ ax, B @ ax
            if pa.max() <= pb.min() or pb.max() <= pa.min(): return False
    return True


def _pole_dist(c, h, L, W):
    u = np.array([np.cos(h), np.sin(h)]); n = np.array([-np.sin(h), np.cos(h)]); q = -c
    dx = max(abs(q @ u) - L / 2, 0); dy = max(abs(q @ n) - W / 2, 0); return float(np.hypot(dx, dy))


def scene(i, ant, psi):
    """Other vehicles around user i. Draw order (default_rng(20270201 + i)): user lane; K_o; then per
    vehicle: parked flag; side (parked) or lane (moving); type; L, W, H; speed offset delta;
    up to 200 placement attempts (s, lateral jitter, heading noise)."""
    r = np.random.default_rng(SSEED + i); uc = user_car(i)
    cen_u = ant[:2] - uc["ant_xi"] * np.array([np.cos(psi), np.sin(psi)])
    lu = int(r.integers(0, 2)); K = int(r.integers(0, 6))
    u = np.array([np.cos(psi), np.sin(psi)]); nrm = np.array([-np.sin(psi), np.cos(psi)])
    placed = [_corners(cen_u, psi, uc["L"] + 2.0, uc["W"])]; V = []
    for _ in range(K):
        parked = bool(r.random() < 0.15)
        if parked:
            side = 0 if r.random() < 0.5 else 3; kind = "car" if r.random() < 0.8 else "van"
            eta0 = LANE * (0 - lu) - LANE / 2 - PARK_W / 2 if side == 0 else LANE * (3 - lu) + LANE / 2 + PARK_W / 2; lane = -1 if side == 0 else 4
            hdir = psi if side == 0 else psi + np.pi
        else:
            lane = int(r.integers(0, 4)); x = r.random(); kind = "car" if x < 0.7 else ("van" if x < 0.9 else "truck")
            eta0 = LANE * (lane - lu); hdir = psi if lane <= 1 else psi + np.pi
        if kind == "car": L, W, H = r.uniform(3.8, 5.0), r.uniform(1.65, 1.95), r.uniform(1.35, 1.50)
        elif kind == "van": L, W, H = r.uniform(4.8, 6.0), r.uniform(1.90, 2.10), r.uniform(1.9, 2.6)
        else: L, W, H = r.uniform(8.0, 12.0), r.uniform(2.40, 2.55), r.uniform(3.0, 3.8)
        delta = r.uniform(-3, 3)
        for _a in range(200):
            s = r.uniform(-30, 30); jit = r.uniform(-0.3, 0.3); hn = r.normal(0, np.deg2rad(2.0))
            c = cen_u + s * u + (eta0 + jit) * nrm; h = hdir + hn; Q = _corners(c, h, L + 2.0, W)
            if _pole_dist(c, h, L, W) >= 1.0 and not any(_overlap(Q, P) for P in placed):
                placed.append(Q)
                V.append(dict(c=c, h=h, L=L, W=W, H=H, kind=kind, parked=parked, lane=lane, delta=delta,
                              boxes=two_box(L, W, H) if kind == "car" else one_box(L, W, H))); break
    return dict(user=dict(c=cen_u, h=psi, **uc), others=V, K_drawn=K, lane_user=lu)


def at_time(sc, v, dt):
    """Vehicle list (user first) shifted by velocity * dt (dt < 0 = earlier frame)."""
    out = []
    for j, x in enumerate([sc["user"]] + sc["others"]):
        sp = v if j == 0 else (0.0 if x["parked"] else max(0.0, v + x["delta"]))
        out.append(dict(c=x["c"] + sp * dt * np.array([np.cos(x["h"]), np.sin(x["h"])]), h=x["h"], boxes=x["boxes"]))
    return out


# ------------------------------------------------------------------ LiDAR
def _t_vehicles(o_w, dirs, vs):
    """first-hit distance per vehicle: (V, R)"""
    out = []
    for x in vs:
        c, s = np.cos(x["h"]), np.sin(x["h"])
        to_v = lambda w: torch.stack([c * w[..., 0] + s * w[..., 1], -s * w[..., 0] + c * w[..., 1], w[..., 2]], -1)
        o = to_v(torch.tensor(np.r_[o_w[:2] - x["c"], o_w[2]], device=dev)[None, None]); d = to_v(dirs[None])
        b = torch.tensor(x["boxes"], device=dev)
        out.append(torch.minimum(LR._slab(o, d, b[0], b[1]), LR._slab(o, d, b[2], b[3]))[0])
    return torch.stack(out)


def render(vs, bg_seed, noise_seed=None):
    """-> cloud (N, 4) world xyz + intensity, hit id (N,) (0 user, j other, -1 ground, -2 background),
    user-first-hit ray count, user ray count alone (both before noise and dropout)."""
    D = LV.DIRS; T = _t_vehicles(LV.SENSOR, D, vs)
    dz = D[:, 2]; t_gr = torch.where(dz < -1e-9, -LV.SENSOR[2] / dz, torch.full_like(dz, np.inf))
    tv, iv = T.min(0); t = torch.minimum(tv, t_gr); onveh = tv <= t_gr
    inr = lambda tt: torch.isfinite(tt) & (tt >= LV.R_MIN) & (tt <= LV.R_MAX)
    n_user = int((onveh & (iv == 0) & inr(t)).sum()); tu = torch.minimum(T[0], t_gr); n_alone = int(((T[0] <= t_gr) & inr(tu)).sum())
    rng = np.random.default_rng(bg_seed)
    bg = np.c_[rng.uniform(-20, 40, 600), rng.uniform(-20, 20, 600), rng.uniform(0, 4, 600)]
    if noise_seed is not None: rng = np.random.default_rng(noise_seed)
    v = torch.tensor(bg - LV.SENSOR, device=dev); Ln = v.norm(dim=-1); dv = v / Ln[:, None]
    keep_bg = (_t_vehicles(LV.SENSOR, dv, vs).min(0).values > Ln).cpu().numpy()
    tt = t.cpu().numpy(); ok = np.isfinite(tt) & (tt >= LV.R_MIN) & (tt <= LV.R_MAX)
    dd = D.cpu().numpy()[ok]; rr = tt[ok]; vh = onveh.cpu().numpy()[ok]; hid = np.where(vh, iv.cpu().numpy()[ok], -1)
    inten = np.where(vh, rng.uniform(0.4, 1.0, ok.sum()), rng.uniform(0.0, 0.2, ok.sum()))
    bgr = np.linalg.norm(bg - LV.SENSOR, axis=1); bgd = (bg - LV.SENSOR) / bgr[:, None]; bgi = rng.uniform(0.0, 0.5, 600)
    rr = rr + rng.normal(0, LV.SIG_R, len(rr)); bgr = bgr + rng.normal(0, LV.SIG_R, 600)
    drop = rng.random(len(rr)) < LV.P_DROP; dropb = rng.random(600) < LV.P_DROP
    pts = np.r_[LV.SENSOR + dd * rr[:, None], LV.SENSOR + bgd * bgr[:, None]]; I = np.r_[inten, bgi]
    ids = np.r_[hid, np.full(600, -2)]; keep = np.r_[~drop, keep_bg & ~dropb]; pts, I, ids = pts[keep], I[keep], ids[keep]
    m = (pts[:, 0] >= LV.ROI[0]) & (pts[:, 0] < LV.ROI[1]) & (pts[:, 1] >= LV.ROI[2]) & (pts[:, 1] < LV.ROI[3])
    return np.c_[pts[m], I[m]].astype(np.float32), ids[m].astype(np.int8), n_user, n_alone


def single_check(i, ant, psi):
    """With no other vehicles and the frozen standard car, render() must equal lidar_v2.render."""
    std = dict(c=LV.vehicle_centre(ant[i:i + 1], psi[i:i + 1])[0], h=psi[i], boxes=np.r_[np.array(LV.BODY), np.array(LV.CABIN)])
    pc, _, _, _ = render([std], LV.SEED + i); ref = LV.render(np.array([i]), ant, psi)[0][0]
    return pc, ref


# ------------------------------------------------------------------ radio blockage
def blocked_by(vs, ant):
    """indices j >= 1 of vehicles whose boxes intersect the segment array (0,0,2) -> antenna (x, y, 1.5)."""
    a = np.array([0.0, 0.0, P1.H_RSU]); seg = np.r_[ant[:2], P1.H_VEH] - a; out = []
    for j, x in enumerate(vs[1:], 1):
        c, s = np.cos(x["h"]), np.sin(x["h"]); R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        o = R @ (a - np.r_[x["c"], 0.0]); d = R @ seg; hit = False
        for lo, hi in ((x["boxes"][0], x["boxes"][1]), (x["boxes"][2], x["boxes"][3])):
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo - o) / d; t2 = (hi - o) / d
            t1 = np.where(np.abs(d) < 1e-12, np.where((o >= lo) & (o <= hi), -np.inf, np.inf), t1)
            t2 = np.where(np.abs(d) < 1e-12, np.where((o >= lo) & (o <= hi), np.inf, -np.inf), t2)
            tmin = np.minimum(t1, t2).max(); tmax = np.maximum(t1, t2).min()
            if tmax >= max(tmin, 0.0) and tmin <= 1.0: hit = True
        if hit: out.append(j)
    return out


# ------------------------------------------------------------------ association
def clusters(pc):
    """Connected components of above-ground points (z > 0.3 m) linked at 0.8 m in xy; components with >= 10 points.
    Exact, memory-bounded implementation (added 2026-11-20 after the pair list hit 1.37e9 pairs on a dense test scene):
    points sharing a grid cell of side 0.8/sqrt(2) are always linked, so cells are merged by union-find and two
    nearby cells are joined iff some cross pair is within 0.8 m. Labels are renumbered in order of each component's
    lowest point index, which is scipy.connected_components' numbering, so output equals clusters_pairs()."""
    ng = np.where(pc[:, 2] > 0.3)[0]; lab = np.full(len(pc), -1)
    if len(ng) < 10: return lab
    xy = pc[ng, :2].astype(np.float64); h = 0.8 / np.sqrt(2.0)
    cell = np.floor(xy / h).astype(np.int64); cu, inv = np.unique(cell, axis=0, return_inverse=True); inv = inv.ravel()
    order = np.argsort(inv, kind="stable"); starts = np.searchsorted(inv[order], np.arange(len(cu) + 1))
    pts = lambda c: xy[order[starts[c]:starts[c + 1]]]
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(cu)}
    par = np.arange(len(cu))
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    trees = {}
    tree = lambda c: trees[c] if c in trees else trees.setdefault(c, cKDTree(pts(c)))
    offs = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) > (0, 0)]
    for a, (cx, cy) in enumerate(cu):
        for dx, dy in offs:
            b = key.get((int(cx) + dx, int(cy) + dy))
            if b is None: continue
            ra, rb = find(a), find(b)
            if ra == rb: continue
            pa, pb = pts(a), pts(b)
            if len(pa) > len(pb): d, _ = tree(a).query(pb, k=1, distance_upper_bound=0.8)
            else: d, _ = tree(b).query(pa, k=1, distance_upper_bound=0.8)
            if (d <= 0.8).any(): par[ra] = rb
    comp = np.array([find(c) for c in range(len(cu))])[inv]
    _, first, cl = np.unique(comp, return_index=True, return_inverse=True); cl = cl.ravel()
    rank = np.empty(len(first), np.int64); rank[np.argsort(first, kind="stable")] = np.arange(len(first)); cl = rank[cl]
    cnt = np.bincount(cl); lab[ng] = np.where(cnt[cl] >= 10, cl, -1); return lab


def clusters_pairs(pc):
    """Original pair-list implementation (used for validation before 2026-11-20); kept for the equivalence check."""
    ng = np.where(pc[:, 2] > 0.3)[0]; lab = np.full(len(pc), -1)
    if len(ng) < 10: return lab
    pr = cKDTree(pc[ng, :2]).query_pairs(0.8, output_type="ndarray")
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(ng), len(ng)))
    _, cl = connected_components(A, directed=False); cnt = np.bincount(cl)
    lab[ng] = np.where(cnt[cl] >= 10, cl, -1); return lab


def associate(pc, gps, sigma, rule, lab=None):
    """-> (cloud for the estimator, selected cluster id or -1, fallback flag). R0: unchanged, no GPS."""
    if rule == "R0": return pc, -1, False
    if lab is None: lab = clusters(pc)
    best, bd = -1, np.inf
    for c in np.unique(lab[lab >= 0]):
        dmin = np.hypot(*(pc[lab == c, :2] - gps).T).min()
        if dmin <= 3 * sigma + 0.5 and dmin < bd: best, bd = int(c), dmin
    fb = best < 0
    if rule == "R1": keep = (lab < 0) | (lab == best)
    else: keep = (pc[:, 2] <= 0.3) | (np.hypot(*(pc[:, :2] - gps).T) <= 3 * sigma + 3.0)
    return pc[keep], best, fb
