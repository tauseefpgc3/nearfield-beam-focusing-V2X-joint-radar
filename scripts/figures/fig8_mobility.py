"""Fig 8 — future-beam accuracy vs prediction horizon (DeepSense radar, D-020).

Source: results/mobility/future_beam.json. Top-1 beam accuracy vs horizon (Δ frames / seconds)
for Sc31 (V2I) and Sc36 (V2V), 5-seed mean ± 95% CI. Skips gracefully (no file written)
if the mobility results are not yet banked. ieee_style; vector PDF + 400 dpi PNG.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, save_fig, setup_mpl

FBJ = os.path.join(RESULTS, "mobility", "future_beam.json")
SERIES = [("31", "V2I (Sc31)", "#1f77b4", "o"), ("36", "V2V (Sc36)", "#d62728", "s")]


def main():
    if not os.path.exists(FBJ):
        print("  [fig8] future_beam.json absent -> skipped (mobility not yet banked)")
        return
    plt = setup_mpl()
    aud = Audit("fig8_mobility")
    deltas = aud.arr(FBJ, "deltas")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for scen, lab, color, mk in SERIES:
        xs, ys, err = [], [], []
        for d in deltas:
            hz = aud.val(FBJ, f"results.{scen}.{d}.horizon_s")
            m = aud.val(FBJ, f"results.{scen}.{d}.top1.mean")
            hi = aud.val(FBJ, f"results.{scen}.{d}.top1.hi")
            xs.append(hz); ys.append(m * 100); err.append((hi - m) * 100)
        ax.errorbar(xs, ys, yerr=err, marker=mk, color=color, lw=1.3, markersize=4,
                    capsize=2, label=lab)
    rand = 100.0 / 64
    ax.axhline(rand, color="0.4", ls=":", lw=0.9, label=f"random (1/64 = {rand:.1f}%)")
    ax.set_xlabel("Prediction horizon (s ahead; 0.1 s/frame)")
    ax.set_ylabel("Top-1 beam accuracy (%)")
    ax.set_title("Future-beam prediction under mobility")
    ax.set_ylim(bottom=0)
    ax.legend()
    save_fig(fig, "fig8_mobility")
    aud.write()
    print("  [fig8] written")


if __name__ == "__main__":
    main()
