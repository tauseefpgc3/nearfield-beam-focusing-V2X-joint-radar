"""Side-check: genie polar codeword vs the matched-filter bound ||h||^2, by distance band."""
import os, h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); C = 299_792_458.0
for f in (60, 140):
    z = np.load(os.path.join(ROOT, "results", "verify", f"gains_{f}GHz.npz")); G = z["G64"]; d = z["distance"]
    with h5py.File(os.path.join(ROOT, "data", "processed", f"closerange_{f}GHz_full.h5"), "r") as hf:
        nrm = np.concatenate([np.sum(np.abs(hf["channel"][s:s+500])**2, 1) for s in range(0, 6000, 500)])
    gap = 10 * np.log10(nrm / G.max(1)); lam = C / (f * 1e9); r1 = 1.2 * 127**2 * lam / 8
    print(f, "GHz ring r_1 = %.2f m | genie-vs-MRT gap: mean %.1f dB; d<r_1: %.1f dB; r_1<=d: %.1f dB" % (r1, gap.mean(), gap[d < r1].mean(), gap[d >= r1].mean()))
