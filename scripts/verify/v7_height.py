"""V4 side-check: genie-gain drop of the elevation-0 polar codebook if the D-014 heights
(RSU 2.0 m, vehicle 1.5 m) were applied to the LoS path. 120 users, 60 and 140 GHz."""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); C = 299_792_458.0
def draw(i):
    r = np.random.default_rng(20_240_603 + i); d = r.uniform(1, 20); th = np.deg2rad(r.uniform(-60, 60)); return d, th
for f in (60, 140):
    cb = np.load(os.path.join(ROOT, "data", "codebooks", f"polar_codebook_{f}GHz.npz"))["codebook"]; lam = C / (f * 1e9)
    co = (np.arange(128) - 63.5) * 0.5 * lam; Y, Z = np.meshgrid(co, co, indexing="ij"); Y = Y.ravel(); Z = Z.ravel()
    drop, el = [], []
    for i in range(0, 6000, 50):
        d, th = draw(i); x, y = d * np.cos(th), d * np.sin(th)
        h0 = np.exp(-1j * 2 * np.pi / lam * np.sqrt(x**2 + (Y - y)**2 + Z**2))
        h1 = np.exp(-1j * 2 * np.pi / lam * np.sqrt(x**2 + (Y - y)**2 + (Z + 2.0 - 1.5)**2))  # user 0.5 m below array centre
        g0 = (np.abs(cb.conj() @ h0)**2).max(); g1 = (np.abs(cb.conj() @ h1)**2).max()
        drop.append(10 * np.log10(g0 / g1)); el.append(np.degrees(np.arctan(0.5 / d)))
    drop = np.array(drop); print(f, "GHz LoS-only genie drop: mean %.1f dB, median %.1f, min %.1f, max %.1f | elevation %.1f-%.1f deg" % (drop.mean(), np.median(drop), drop.min(), drop.max(), min(el), max(el)))
