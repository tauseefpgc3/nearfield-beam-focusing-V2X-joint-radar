"""Fig 4 (HEADLINE) — near-field beam-focusing dB-loss CDF, K=4 shortlist, n=20 (D-018).

Source: results/indomain_n20/close_n20.json :: results.<f>.<pred>.gain_ratio_k4_nf
        (per near-field test sample, ratio = realized/genie; pooled over all 20 seeds).
PRIMARY curves = soft-avg (C3) multimodal — the D-018-recommended predictor; SECONDARY
(dashed) = radar-alone single-sensor operating point. dB-loss = -10*log10(ratio). All caption
stats (median, %<=1 dB, n) are computed live from the JSON. Genie/DeepSense untouched.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, save_fig, setup_mpl

CJ = os.path.join(RESULTS, "indomain_n20", "close_n20.json")
FREQS = [("60", "60 GHz", "#1f77b4"), ("140", "140 GHz", "#d62728")]
# (predictor key, label, linestyle, linewidth, primary?)
PREDS = [("softavg", "soft-avg", "-", 1.6, True), ("radar", "radar", "--", 1.1, False)]


def dbloss(aud, fghz, pred):
    r = np.asarray(aud.arr(CJ, f"results.{fghz}.{pred}.gain_ratio_k4_nf"), dtype=float)
    return -10.0 * np.log10(np.clip(r, 1e-6, 1.0))


def main():
    plt = setup_mpl()
    aud = Audit("fig4_cdf")
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for fghz, flab, color in FREQS:
        for pred, plab, ls, lw, primary in PREDS:
            d = np.sort(dbloss(aud, fghz, pred))
            cdf = np.arange(1, len(d) + 1) / len(d)
            med = float(np.median(d)); w1 = float((d <= 1.0).mean()) * 100
            ax.plot(d, cdf, color=color, ls=ls, lw=lw,
                    alpha=1.0 if primary else 0.75,
                    label=f"{flab} {plab} ($n$={len(d)}, med {med:.2f} dB, {w1:.0f}% $\\leq$1 dB)")
            aud.note(f"{flab} {plab}: n={len(d)}, median={med:.3f} dB, frac<=1dB={w1:.1f}%")
    ax.axvline(1.0, color="0.4", ls=":", lw=0.9)
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Near-field beam-focusing loss vs genie (dB), $K$=4 shortlist")
    ax.set_ylabel("Empirical CDF (near-field-subset users)")
    ax.set_title("Near-field beam focusing (20 seeds)", fontsize=8, pad=16)
    ax.text(0.5, 1.015, "solid = soft-avg (recommended); dashed = radar-alone (single-sensor)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, style="italic")
    ax.legend(loc="lower right", fontsize=6.0)
    save_fig(fig, "fig4_dbloss_cdf")

    import os as _os
    from _paperkit import FIG_DIR
    with open(_os.path.join(FIG_DIR, "fig4_dbloss_cdf.caption.txt"), "w", encoding="utf-8") as f:
        f.write("Fig 4. Empirical CDF of the near-field beam-focusing loss vs genie (K=4 "
                "shortlist), near-field subset, pooled over 20 seeds (D-018). Solid = the "
                "recommended soft-avg (C3) multimodal predictor; dashed = the radar-alone "
                "single-sensor operating point. All statistics are computed from "
                "results/indomain_n20/close_n20.json.\n")
    aud.write()


if __name__ == "__main__":
    main()
