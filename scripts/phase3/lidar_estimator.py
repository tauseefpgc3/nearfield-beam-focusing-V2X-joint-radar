"""D-031 geometric LiDAR estimator: raw point cloud -> antenna (x, y).

1. ground removal: keep z in [Z_GROUND, Z_TOP] (world; ground z = 0, car top 1.5 m);
2. occupancy grid (CELL_C), 8-connected components; vehicle = the component with the most
   points among those with bounding-box diagonal <= MAX_DIAG and >= N_MIN points;
3. heading (mod 90 deg) by L-shape fitting (Zhang et al. 2017 closeness criterion,
   1 deg steps) on the body-side points (z <= Z_SIDE) if >= N_SIDE of them, else by the
   minimum-area bounding rectangle of all cluster points;
4. length axis = the axis with the larger observed extent;
5. box from the visible faces: on each axis, if the observed extent >= dim - TOL the centre
   is the extent midpoint; otherwise the face nearer the sensor is taken as observed and the
   box extends away from the sensor by the known dimension (4.5 / 1.8 m);
6. front/back from the cabin (spans xi in [-1.6, 0.8] m, rear-biased): count cabin-level
   points (z >= Z_CAB) in xi in (-1.6+DELTA, -0.8-DELTA) under each heading hypothesis
   (that zone is cabin under one hypothesis and open deck under the other); the hypothesis
   with more points wins if ahead by >= N_CAB; antenna = centre - 1.3 m * forward. Else
   AMBIG rule: 'centre' (antenna = box centre) or 'near' (hypothesis nearer the sensor).
Fallback (no valid cluster): FALLBACK position (training-set median antenna position).
"""
import numpy as np
from scipy import ndimage

DEFAULT = dict(Z_GROUND=0.15, Z_TOP=1.7, CELL_C=0.2, MAX_DIAG=6.0, N_MIN=20, Z_SIDE=0.95, N_SIDE=30,
               TOL=0.3, Z_CAB=1.15, N_CAB=5, AMBIG="centre", DELTA=0.1)
L_CAR, W_CAR, ANT = 4.5, 1.8, 1.3
SENSOR_XY = np.zeros(2)


def _closeness(X, th, d0=0.01):
    e1 = np.array([np.cos(th), np.sin(th)]); e2 = np.array([-np.sin(th), np.cos(th)])
    c1 = X @ e1; c2 = X @ e2
    d1 = np.minimum(c1.max() - c1, c1 - c1.min()); d2 = np.minimum(c2.max() - c2, c2 - c2.min())
    return np.sum(1.0 / np.maximum(np.minimum(d1, d2), d0))


def _min_area(X, th):
    e1 = np.array([np.cos(th), np.sin(th)]); e2 = np.array([-np.sin(th), np.cos(th)])
    c1 = X @ e1; c2 = X @ e2
    return -(np.ptp(c1) * np.ptp(c2))


def cluster(pc, p):
    q = pc[(pc[:, 2] >= p["Z_GROUND"]) & (pc[:, 2] <= p["Z_TOP"])]
    if len(q) < p["N_MIN"]:
        return None
    lo = q[:, :2].min(0); ij = np.floor((q[:, :2] - lo) / p["CELL_C"]).astype(int)
    occ = np.zeros(ij.max(0) + 1, bool); occ[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    pl = lab[ij[:, 0], ij[:, 1]]; cnt = np.bincount(pl, minlength=n + 1); cnt[0] = 0
    for c in np.argsort(-cnt):
        if cnt[c] < p["N_MIN"]:
            return None
        m = pl == c; ext = np.ptp(q[m, :2], 0)
        if np.hypot(*ext) <= p["MAX_DIAG"]:
            return q[m]
    return None


def estimate(pc, p=DEFAULT, fallback=np.array([10.0, 0.0])):
    """Returns (antenna_xy (2,), info dict)."""
    q = cluster(pc, p)
    if q is None:
        return fallback.copy(), dict(fallback=True)
    X = q[:, :2]; side = q[:, 2] <= p["Z_SIDE"]
    ths = np.deg2rad(np.arange(0, 90, 1.0))
    if side.sum() >= p["N_SIDE"]:
        th = ths[np.argmax([_closeness(X[side], t) for t in ths])]
    else:
        th = ths[np.argmax([_min_area(X, t) for t in ths])]
    axes = [np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])]
    ext = [np.ptp(X @ a) for a in axes]
    if ext[1] > ext[0]:
        axes = axes[::-1]; ext = ext[::-1]
    centre = np.zeros(2)
    for a, dim in zip(axes, (L_CAR, W_CAR)):
        c = X @ a; s = SENSOR_XY @ a
        if np.ptp(c) >= dim - p["TOL"]:
            m = (c.max() + c.min()) / 2
        elif abs(s - c.min()) <= abs(s - c.max()):
            m = c.min() + dim / 2
        else:
            m = c.max() - dim / 2
        centre += m * a
    fwd = axes[0]; info = dict(fallback=False, heading_mod180=float(np.rad2deg(np.arctan2(fwd[1], fwd[0])) % 180))
    cab = q[:, 2] >= p["Z_CAB"]; xi = (X[cab] - centre) @ fwd; dl = p["DELTA"]
    n_minus = int(((xi > -1.6 + dl) & (xi < -0.8 - dl)).sum())   # rear cabin zone if the cabin is towards -fwd
    n_plus = int(((xi > 0.8 + dl) & (xi < 1.6 - dl)).sum())      # the same zone under the flipped heading
    info.update(n_minus=n_minus, n_plus=n_plus)
    if abs(n_minus - n_plus) >= p["N_CAB"]:
        return centre - ANT * fwd if n_minus > n_plus else centre + ANT * fwd, dict(info, resolved=True)
    info["resolved"] = False
    if p["AMBIG"] == "near":
        h = [centre + ANT * fwd, centre - ANT * fwd]
        return min(h, key=lambda v: np.linalg.norm(v - SENSOR_XY)), info
    return centre, info
