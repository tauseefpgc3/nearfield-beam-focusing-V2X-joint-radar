"""Phase 2a pre-tag split check (data-free). Deterministic fine grid over the user region
(theta step 0.05 deg, d step 0.01 m); each grid point gets its line-of-sight nearest
codeword (nearest u_i of 256, ring with nearest 1/r). Checks that every angle class and
every ring occurring in test regions also occurs in the (guarded) training regions.
Usage: python p0_split_check.py [rng|rotate]"""
import sys, numpy as np
C = 299_792_458.0
NS, NB = 15, 4
BANDS = [1.0, 5.75, 10.5, 15.25, 20.0]

def assign(mode):
    roles = np.zeros((NB, NS), int)               # 0 train 1 val 2 test
    if mode == "rng":
        rng = np.random.default_rng(2029)
        for b in range(NB):
            p = rng.permutation(NS); roles[b, p[:3]] = 2; roles[b, p[3:5]] = 1
    else:                                          # deterministic rotation
        for b in range(NB):
            roles[b, [(b + 5 * k) % NS for k in range(3)]] = 2
            roles[b, [(b + 3 + 5 * k) % NS for k in range(2)]] = 1
    return roles

mode = sys.argv[1] if len(sys.argv) > 1 else "rng"; G = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
roles = assign(mode)
print(f"mode={mode}; roles per band (sector -60..60 deg, T=test V=val .=train):")
for b in range(NB): print(f"  band {BANDS[b]:5.2f}-{BANDS[b+1]:5.2f} m: " + "".join(".VT"[r] for r in roles[b]))
print("  sectors never in training:", [s for s in range(NS) if (roles[:, s] != 0).all()],
      "| sectors test in >1 band:", [s for s in range(NS) if (roles[:, s] == 2).sum() > 1])
th = np.deg2rad(np.arange(-60, 60, 0.05) + 0.025); d = np.arange(1.0, 20.0, 0.01) + 0.005
TH, D = np.meshgrid(th, d, indexing="ij"); TH, D = TH.ravel(), D.ravel()
sec = np.minimum(((np.rad2deg(TH) + 60) // 8).astype(int), NS - 1)
band = np.minimum(np.searchsorted(BANDS, D, side="right") - 1, NB - 1)
role = roles[band, sec]
XY = np.stack([D * np.cos(TH), D * np.sin(TH)], 1)
# guard: drop train/val grid points within 0.5 m of any test grid point (coarse test sample)
from scipy.spatial import cKDTree
tree = cKDTree(XY[role == 2][::7])
near = tree.query(XY, distance_upper_bound=G)[0] < G
train = (role == 0) & ~near; test = role == 2
print(f"  grid fractions: train {train.mean():.3f} (after guard) | val {(role==1).mean():.3f} | test {test.mean():.3f} | guard-dropped {(near & (role!=2)).mean():.3f}")
r = np.hypot(D, 0.5); u = D * np.sin(TH) / r
ug = (2 * np.arange(256) - 255) / 256
ai = np.abs(u[:, None] - ug[None]).argmin(1)
ok = True
for f in (60, 140):
    lam = C / (f * 1e9); inv = np.r_[0, np.arange(1, 8) / (1.2 * 127**2 * lam / 8)]
    rj = np.abs(1 / r[:, None] - inv[None]).argmin(1)
    miss_a = sorted(set(ai[test]) - set(ai[train])); miss_r = sorted(set(rj[test]) - set(rj[train]))
    jt = set(zip(ai[test], rj[test])); jr = set(zip(ai[train], rj[train]))
    print(f"  {f} GHz: angle classes test {len(set(ai[test]))} / train {len(set(ai[train]))}, test-only {miss_a} | rings test {sorted(set(rj[test]))} train {sorted(set(rj[train]))}, test-only {miss_r} | joint test-only {len(jt-jr)} of {len(jt)}")
    ok &= not miss_a and not miss_r
print("CHECK", "PASS" if ok else "FAIL")
