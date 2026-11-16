"""Phase 2a pre-plan check (codebook property only; no dataset, no sensors, no methods).
Genie-vs-matched-filter gap for a pure LoS point on a deterministic (u, r) grid, for a
128-element horizontal ULA with B_theta angle samples uniform in sine x 8 Cui-Dai rings.
For a ULA, |p_n - p_u|^2 = r^2 - 2 r u y_n + y_n^2 depends only on (u = y_u/r, r), so the
2.0/1.5 m heights do not break an (u, r) codebook. Also the expected NLoS energy share."""
import numpy as np
C = 299_792_458.0
def ula(N, lam, u, r):
    y = (np.arange(N) - (N - 1) / 2) * lam / 2
    if np.isinf(r): return np.exp(1j * 2 * np.pi / lam * u * y)
    return np.exp(-1j * 2 * np.pi / lam * (np.sqrt(r**2 - 2 * r * u * y + y**2) - r))
for f in (60, 140):
    lam = C / (f * 1e9); N = 128
    rings = np.r_[np.inf, 1.2 * (N - 1)**2 * lam / 8 / np.arange(1, 8)]
    for B in (128, 256):
        us = (2 * np.arange(B) - B + 1) / B
        CB = np.array([ula(N, lam, u, r) for u in us for r in rings]) / np.sqrt(N)
        gaps = []
        for u in np.linspace(-np.sin(np.pi/3), np.sin(np.pi/3), 397):
            for d in np.linspace(1, 20, 39):
                r = np.hypot(d, 0.5); h = ula(N, lam, u, r)
                gaps.append(10 * np.log10(N / (np.abs(CB.conj() @ h)**2).max()))
        g = np.array(gaps)
        print(f"{f} GHz  B_theta={B}: LoS genie-vs-bound gap mean {g.mean():.2f} dB, median {np.median(g):.2f}, max {g.max():.2f}  | rings {np.round(rings[1:],2)}")
e = (10**-1 - 10**-2) / np.log(10)   # E[10^(-X/10)], X~U(10,20)
print(f"expected NLoS energy share (K~U{{2,3,4}}): {3*e/(1+3*e):.3f} -> {10*np.log10(1+3*e):.2f} dB of ||h||^2 not capturable by one beam (approx.)")
