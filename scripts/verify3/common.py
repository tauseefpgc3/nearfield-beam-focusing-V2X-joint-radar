"""Phase 3b audit: independent re-implementation of the network definition, map decoding,
local-sweep construction and loss. Nothing here imports Phase 3 evaluation or training code.
The simulator (scripts/phase3/lidar_v2.py) and the geometric estimator
(scripts/phase3/lidar_estimator.py) are the objects under audit and are only ever called as
black boxes from the audit scripts."""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P2 = os.path.join(ROOT, "data", "processed", "phase2")
P3 = os.path.join(ROOT, "data", "processed", "phase3")
CK = os.path.join(ROOT, "checkpoints", "phase3")
OUT = os.path.join(ROOT, "results", "verify3")
os.makedirs(OUT, exist_ok=True)
SEEDS = (11, 22, 33, 44, 55)
ARRAY = np.array([0.0, 0.0, 2.0]); H_ANT = 1.5
# BEV grid as documented in D-031 (x rows, y columns, 0.2 m)
GX0, GY0, GC, GNX, GNY = -3.2, -20.8, 0.2, 128, 208
XCEN = GX0 + GC * (np.arange(GNX) + 0.5); YCEN = GY0 + GC * (np.arange(GNY) + 0.5)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h:
        return {k: np.sort(h[k][:]) for k in ("train", "val", "test")}


def users():
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h:
        return h["ant_xyz"][:], h["psi"][:]


def channels(f):
    with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as h:
        return dict(G=h["gains"][:], u=h["u_grid"][:], rings=h["rings"][:], adv=h["adv_db"][:])


# ---------------------------------------------------------------- network (own definition)
def _conv2(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(True))


class Net(nn.Module):
    """Must load the Phase 3 checkpoints with strict=True; that is itself a check that the
    checkpoint is exactly the documented architecture (coord channels + 4-level U-Net)."""
    def __init__(self, cin):
        super().__init__()
        gx, gy = np.meshgrid(XCEN / 22.4, YCEN / 20.8, indexing="ij")
        self.register_buffer("coords", torch.tensor(np.stack([gx, gy])[None], dtype=torch.float32))
        w = (32, 64, 128, 256)
        self.enc = nn.ModuleList([_conv2(cin + 2, 32), _conv2(32, 64), _conv2(64, 128), _conv2(128, 256)])
        self.dec = nn.ModuleList([_conv2(256 + 128, 128), _conv2(128 + 64, 64), _conv2(64 + 32, 32)])
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        h = torch.cat([x, self.coords.expand(len(x), -1, -1, -1)], 1); sk = []
        for k, e in enumerate(self.enc):
            h = e(h if k == 0 else F.max_pool2d(h, 2)); sk.append(h)
        for d, s in zip(self.dec, sk[-2::-1]):
            h = d(torch.cat([F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.out(h)[:, 0]


def load_net(sensor, seed, head="H", lr="0.001"):
    ck = torch.load(os.path.join(CK, f"{sensor}_{head}_lr{float(lr):g}_s{seed}.pt"), map_location="cpu", weights_only=False)
    m = Net(ck["cin"]); m.load_state_dict(ck["model"], strict=True); return m.to(DEV).eval(), ck


def decode_local(lg, r=2):
    """Arg-max cell, then softmax-weighted mean of cell centres in the (2r+1)^2 window (numpy)."""
    B = lg.shape[0]; flat = lg.reshape(B, -1); k = flat.argmax(1); ix, iy = k // GNY, k % GNY
    out = np.empty((B, 2))
    for b in range(B):
        x0, x1 = max(ix[b] - r, 0), min(ix[b] + r + 1, GNX); y0, y1 = max(iy[b] - r, 0), min(iy[b] + r + 1, GNY)
        w = lg[b, x0:x1, y0:y1].astype(np.float64); w = np.exp(w - w.max()); w /= w.sum()
        out[b] = [(w.sum(1) * XCEN[x0:x1]).sum(), (w.sum(0) * YCEN[y0:y1]).sum()]
    return out


def logits(model, X, bs=32):
    out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            out.append(model(torch.as_tensor(X[b:b + bs]).to(DEV).float()).cpu().numpy())
    return np.concatenate(out)


def predict(model, X):
    return decode_local(logits(model, X))


def bev_inputs(sensor, idx):
    parts = []
    if sensor in ("radar", "fused"):
        with h5py.File(os.path.join(P3, "radar_bev20.h5"), "r") as h: parts.append(h["bev"][idx])
    if sensor in ("lidar", "fused"):
        with h5py.File(os.path.join(P3, "lidar_bev.h5"), "r") as h: parts.append(h["bev"][idx])
    return np.concatenate(parts, 1)


# ---------------------------------------------------------------- local sweep (own version)
def beam_set(xy, u_grid, rings, A, R):
    """Flat codeword indices (k = 8*angle + ring) of the A x R beams nearest the estimate:
    A angles nearest in u = y/|p - array|, R rings nearest in inverse distance (ring 0 = far field)."""
    rel = np.c_[xy[:, 0], xy[:, 1], np.full(len(xy), H_ANT - ARRAY[2])]
    dist = np.sqrt((rel ** 2).sum(1)); u = rel[:, 1] / dist
    du = np.abs(u[:, None] - u_grid[None, :])
    ang = np.argsort(du, axis=1, kind="mergesort")[:, :A]
    inv_r = np.array([0.0 if not np.isfinite(r) else 1.0 / r for r in rings])
    dr = np.abs(1.0 / dist[:, None] - inv_r[None, :])
    rng_ = np.argsort(dr, axis=1, kind="mergesort")[:, :R]
    return (ang[:, :, None] * len(rings) + rng_[:, None, :]).reshape(len(xy), A * R)


def loss_db(G, S):
    best = np.take_along_axis(G, S, axis=1).max(1)
    return 10.0 * np.log10(G.max(1) / best)


def shape_options(M):
    return [(A, M // A) for A in (M // 8, M // 4, M // 2, M) if A >= 1 and M // A <= 8]


def sweep_with_val_shape(ch, va, te, est_va, est_te, M):
    """Choose (A, R) on validation, return per-user test loss and the shape."""
    best, shp = np.inf, None
    for A, R in shape_options(M):
        v = loss_db(ch["G"][va], beam_set(est_va, ch["u"], ch["rings"], A, R)).mean()
        if v < best: best, shp = v, (A, R)
    return loss_db(ch["G"][te], beam_set(est_te, ch["u"], ch["rings"], *shp)), shp


def boot_mean(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
