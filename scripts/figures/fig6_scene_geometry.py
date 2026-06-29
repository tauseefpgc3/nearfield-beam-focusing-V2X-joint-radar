"""Fig 6 — scene-geometry mechanism: CNN extracts beam info beyond raw-feature similarity,
and that margin collapses from V2I to V2V.

Source: results/deepsense/genie_decomp.json
For scenario 31 (V2I/RSU) and 36 (V2V): random chance, full-feature 1-NN, and the trained
CNN top-1 beam accuracy (5 seeds, mean±95%CI). The CNN−1-NN gap (annotated) is large for
V2I (stable RSU scene geometry the CNN can exploit) and ~0 for V2V (mobile, no stable
geometry) — explaining why real-data V2V radar accuracy is lower. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, save_fig, setup_mpl

GD = os.path.join(RESULTS, "deepsense", "genie_decomp.json")
SCEN = [("31", "V2I (RSU)"), ("36", "V2V")]


def main():
    plt = setup_mpl()
    aud = Audit("fig6_scene_geometry")
    rand = aud.val(GD, "random") * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.9))   # IEEE single-column
    width = 0.36
    colors = {"knn1": "#ff7f0e", "cnn": "#1f77b4"}
    for gi, (scen, link) in enumerate(SCEN):
        kn = aud.val(GD, f"scenarios.{scen}.knn1.mean") * 100
        kn_e = (aud.val(GD, f"scenarios.{scen}.knn1.hi") -
                aud.val(GD, f"scenarios.{scen}.knn1.mean")) * 100
        cn = aud.val(GD, f"scenarios.{scen}.cnn.mean") * 100
        cn_e = (aud.val(GD, f"scenarios.{scen}.cnn.hi") -
                aud.val(GD, f"scenarios.{scen}.cnn.mean")) * 100
        gap = aud.val(GD, f"scenarios.{scen}.cnn_minus_knn1.mean") * 100
        # The annotated gap is the banked CNN−1-NN mean (cnn_minus_knn1.mean), NOT prose.
        aud.note(f"sc{scen} annotated gap = {gap:+.1f} pp (from cnn_minus_knn1.mean; "
                 f"CANONICAL — supersedes earlier ~{'9' if scen=='31' else '1'} pp prose)")
        x0 = gi
        ax.bar(x0 - width / 2, kn, width, yerr=kn_e, capsize=2, color=colors["knn1"],
               edgecolor="black", lw=0.5, label="1-NN (raw feature)" if gi == 0 else None)
        ax.bar(x0 + width / 2, cn, width, yerr=cn_e, capsize=2, color=colors["cnn"],
               edgecolor="black", lw=0.5, label="CNN (trained)" if gi == 0 else None)
        top = max(kn + kn_e, cn + cn_e)
        ax.annotate(f"gap {gap:+.1f} pp", xy=(x0, top + 1.5), ha="center", fontsize=7,
                    fontweight="bold")
    ax.axhline(rand, color="0.4", ls=":", lw=0.9, label=f"random (1/64 = {rand:.1f}%)")
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([f"Sc {s}\n{l}" for s, l in SCEN])
    ax.set_ylabel("Top-1 beam accuracy (%)")
    ax.set_ylim(0, 32)
    ax.set_title("Scene-geometry exploitation: V2I vs V2V (real radar)")
    ax.legend(loc="upper right")
    save_fig(fig, "fig6_scene_geometry")
    aud.write()


if __name__ == "__main__":
    main()
