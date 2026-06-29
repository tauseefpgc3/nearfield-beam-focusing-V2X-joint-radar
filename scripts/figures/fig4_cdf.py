"""Fig 4 (HEADLINE) — radar near-field beam-focusing dB-loss CDF, K=4 shortlist.

Source: results/indomain/radar_closerange_{60,140}GHz_seed*_metrics.json :: gain_ratio_k4_nf
        (per near-field test sample, ratio = realized_gain / genie_gain in (0,1]).
dB-loss = -10*log10(ratio); pooled over all 5 seeds. The CDF reads "fraction of
near-field users beam-focused to within X dB of the genie via a K=4 RF sweep."
JSON-only, no recompute of the ratios.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, SEEDS, metrics_path, save_fig, setup_mpl

FREQS = [("60", "60 GHz", "#1f77b4"), ("140", "140 GHz", "#d62728")]


def pooled_dbloss(aud: Audit, fghz: str) -> np.ndarray:
    ratios = []
    for s in SEEDS:
        p = metrics_path("radar", f"closerange_{fghz}GHz", s)
        ratios.extend(aud.arr(p, "gain_ratio_k4_nf"))
    r = np.asarray(ratios, dtype=float)
    r = np.clip(r, 1e-6, 1.0)               # ratio in (0,1]; guard log of 0
    return -10.0 * np.log10(r)              # dB-loss >= 0


def main():
    plt = setup_mpl()
    aud = Audit("fig4_cdf")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))   # IEEE single-column
    ns = {}                                    # near-field-subset pooled count per freq
    for fghz, label, color in FREQS:
        d = np.sort(pooled_dbloss(aud, fghz))
        ns[fghz] = len(d)
        cdf = np.arange(1, len(d) + 1) / len(d)
        med = float(np.median(d))
        within1 = float((d <= 1.0).mean()) * 100
        ax.plot(d, cdf, color=color, lw=1.3,
                label=f"{label} ($n$={len(d)}, median {med:.2f} dB, "
                      f"{within1:.0f}% $\\leq$1 dB)")
        aud.note(f"{label}: pooled n={len(d)}, median={med:.3f} dB, "
                 f"frac<=1dB={within1:.1f}%")
    ax.axvline(1.0, color="0.4", ls=":", lw=0.9)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Near-field beam-focusing loss vs genie (dB), $K$=4 shortlist")
    ax.set_ylabel("Empirical CDF (near-field-subset users)")
    # Title + subtitle make explicit this is the NEAR-FIELD SUBSET only (3 dB gate), K=4 —
    # NOT all test users — so the CDF cannot be misread as an all-users result. (Restyled to
    # IEEE sizes: 8 pt title, 6.5 pt italic subtitle. Kept, not removed.)
    ax.set_title("Radar-driven near-field beam focusing", fontsize=8, pad=16)
    ax.text(0.5, 1.015, f"near-field subset only (3 dB gate), $K$=4 — "
            f"$n$={ns['60']} (60 GHz) / {ns['140']} (140 GHz)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, style="italic")
    ax.legend(loc="lower right")
    save_fig(fig, "fig4_dbloss_cdf")
    # Sidecar caption for the manuscript (explicit subset framing).
    import os as _os
    from _paperkit import FIG_DIR
    with open(_os.path.join(FIG_DIR, "fig4_dbloss_cdf.caption.txt"), "w",
              encoding="utf-8") as f:
        f.write("Fig 4. Empirical CDF of the radar-driven near-field beam-focusing loss "
                "relative to the genie, using a K=4 RF verification shortlist. Computed over "
                "the NEAR-FIELD SUBSET ONLY (users passing the 3 dB near-field gate, D-013): "
                f"n={ns['60']} at 60 GHz and n={ns['140']} at 140 GHz, pooled across 5 seeds. "
                "Far-field users are excluded, so this is not an all-users distribution.\n")
    aud.note(f"near-field SUBSET only (3 dB gate), K=4; n={ns['60']} (60 GHz)/{ns['140']} "
             "(140 GHz); far-field users excluded (not all-users).")
    aud.write()


if __name__ == "__main__":
    main()
