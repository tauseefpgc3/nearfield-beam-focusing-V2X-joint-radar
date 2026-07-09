"""Fig 5 — modality-degradation robustness (close-range 60 GHz, joint top-1).

Source: results/degradation_closerange60.json (mean/lo/hi over 5 seeds, per condition).
Conditions: clean, lidar dropped, ±5° misalignment, radar dropped, both dropped.
The both-out bar collapses to ~0 (no running-stat leakage; GroupNorm), confirming the
model genuinely uses the sensors rather than memorizing position priors. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, save_fig, setup_mpl

# (json-key, display label, color)
CONDS = [
    ("clean", "Clean\n(both)", "#2ca02c"),
    ("lidar_out", "LiDAR\ndropped", "#1f77b4"),
    ("misalign5deg", "±5°\nmisalign", "#9467bd"),
    ("radar_out", "Radar\ndropped", "#ff7f0e"),
    ("both_out", "Both\ndropped", "#d62728"),
]


def main():
    plt = setup_mpl()
    aud = Audit("fig5_degradation")
    _n20 = os.path.join(RESULTS, "degradation_closerange60_n20.json")
    p = _n20 if os.path.exists(_n20) else os.path.join(RESULTS, "degradation_closerange60.json")
    labels, means, los, his = [], [], [], []
    for key, lab, _ in CONDS:
        m = aud.val(p, f"{key}.mean")
        labels.append(lab); means.append(m)
        los.append(m - aud.val(p, f"{key}.lo")); his.append(aud.val(p, f"{key}.hi") - m)
    colors = [c for _, _, c in CONDS]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))   # IEEE single-column
    x = range(len(CONDS))
    ax.bar(x, means, yerr=[los, his], capsize=2, color=colors, edgecolor="black",
           lw=0.5, width=0.7)
    # Place each value label clear ABOVE its upper error-bar cap (m + hi), not at the bar top.
    top = max(m + h for m, h in zip(means, his))
    for xi, m, h in zip(x, means, his):
        ax.text(xi, m + h + 0.022 * top, f"{m:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel(r"Joint top-1 accuracy ($\theta$, $r$)")
    ax.set_ylim(0, top * 1.18)
    nseed = int(aud.val(p, "clean.n"))
    ax.set_title(f"Sensor-degradation robustness ({nseed} seeds, 95% CI)")
    save_fig(fig, "fig5_degradation")
    aud.write()


if __name__ == "__main__":
    main()
