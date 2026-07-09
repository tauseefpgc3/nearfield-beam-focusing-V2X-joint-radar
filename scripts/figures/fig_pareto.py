"""Fig (Pareto) — beam-focusing loss vs RF-measurement overhead.

Source: results/gain_overhead_n20.json. Log-x scatter/line of near-field K-shortlist dB-loss
vs number of RF measurements, for soft-avg (primary) and radar-alone (secondary), plus the
hierarchical sweep (71 meas) and the exhaustive genie (512 meas). Shows our predictor reaching
near-genie loss ~an order of magnitude cheaper than the hierarchical sweep. ieee_style.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, save_fig, setup_mpl

GO = os.path.join(RESULTS, "gain_overhead_n20.json")
FREQS = [("60", "60 GHz", "#1f77b4"), ("140", "140 GHz", "#d62728")]
KS = (1, 2, 4, 8)


def main():
    if not os.path.exists(GO):
        print("  [fig_pareto] gain_overhead_n20.json absent -> skipped"); return
    plt = setup_mpl()
    aud = Audit("fig_pareto")
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    nm = aud.val(GO, "hier_measurements")
    for fghz, flab, color in FREQS:
        # ours: soft-avg (solid, primary) and radar (dashed, secondary)
        for pred, ls, lw, mk, plab in (("softavg", "-", 1.6, "o", "soft-avg"),
                                       ("radar", "--", 1.1, "s", "radar")):
            xs = list(KS)
            ys = [aud.val(GO, f"results.{fghz}.{pred}.K{k}.mean") for k in KS]
            ax.plot(xs, ys, ls=ls, lw=lw, marker=mk, ms=3.5, color=color, alpha=1.0 if pred == "softavg" else 0.7,
                    label=f"{flab} {plab}")
        # hierarchical baseline point
        hm = aud.val(GO, f"results.{fghz}.hierarchical.mean")
        ax.scatter([nm], [hm], marker="*", s=70, color=color, edgecolor="black",
                   linewidth=0.5, zorder=5)
    ax.scatter([512], [0.0], marker="D", s=28, color="0.3", zorder=5)
    ax.annotate("exhaustive\n(genie)", (512, 0.0), textcoords="offset points", xytext=(-4, 6),
                ha="right", fontsize=6, color="0.3")
    ax.annotate(f"hierarchical\n({nm} meas)", (nm, aud.val(GO, "results.60.hierarchical.mean")),
                textcoords="offset points", xytext=(6, 2), fontsize=6, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("RF measurements (log scale)")
    ax.set_ylabel("Near-field beam-focusing loss (dB)")
    ax.set_title("Gain vs overhead (20 seeds)", fontsize=8)
    ax.set_ylim(bottom=-0.3)
    ax.legend(loc="upper right", fontsize=6.0)
    save_fig(fig, "fig_pareto")
    aud.write()
    print("  [fig_pareto] written")


if __name__ == "__main__":
    main()
