"""Fig 2 — Gate-0 near-field characterization (close-range PRIMARY vs O1 ANCHOR).

Source: results/gate0/gate0.json
(a) frac(genie polar-vs-DFT advantage ≥ 3 dB) per dataset/freq — the physically meaningful
    near-field occurrence. PRIMARY close-range (27%/14%) vs O1 anchor (8%/7%).
(b) mean genie advantage (dB) vs distance bin, close-range 60/140 GHz — the near-field gain
    is large (+9/+4 dB) at <4 m and decays to ~0 by d_F; the 3 dB gate line is the label
    threshold (D-013). JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, save_fig, setup_mpl

GATE = os.path.join(RESULTS, "gate0", "gate0.json")


def main():
    plt = setup_mpl()
    aud = Audit("fig2_gate0")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 2.7))   # IEEE double-column

    # (a) near-field occurrence: frac(adv >= 3 dB) per dataset row
    g01 = aud.arr(GATE, "G0_1")            # logged as length; individual cells read below
    labels, fracs, colors = [], [], []
    palette = {"PRIMARY": "#1f77b4", "ANCHOR": "#bbbbbb"}
    for i, row in enumerate(g01):
        role = aud.val(GATE, f"G0_1.{i}.role")
        f_ghz = aud.val(GATE, f"G0_1.{i}.f_ghz")
        frac = aud.val(GATE, f"G0_1.{i}.frac_adv")
        tag = "PRIMARY" if role.startswith("PRIMARY") else "ANCHOR"
        labels.append(f"{'Close' if tag=='PRIMARY' else 'O1'}\n{f_ghz} GHz")
        fracs.append(frac * 100); colors.append(palette[tag])
    x = range(len(labels))
    axa.bar(x, fracs, color=colors, edgecolor="black", lw=0.5, width=0.7)
    for xi, fr in zip(x, fracs):
        axa.text(xi, fr + 0.6, f"{fr:.0f}%", ha="center", va="bottom", fontsize=6.5)
    axa.set_xticks(list(x)); axa.set_xticklabels(labels)
    axa.set_ylabel("Users with genie advantage $\\geq$ 3 dB (%)")
    axa.set_ylim(0, max(fracs) * 1.25)
    axa.set_title("(a) Near-field occurrence")
    from matplotlib.patches import Patch
    axa.legend(handles=[Patch(facecolor=palette["PRIMARY"], label="PRIMARY (close-range)"),
                        Patch(facecolor=palette["ANCHOR"], label="ANCHOR (O1, far-field)")],
               loc="upper right")

    # (b) genie advantage vs distance bin (close-range)
    g02 = aud.arr(GATE, "G0_2")
    styles = {60: ("#1f77b4", "o", "60 GHz"), 140: ("#d62728", "s", "140 GHz")}
    for i, rec in enumerate(g02):
        f_ghz = aud.val(GATE, f"G0_2.{i}.f_ghz")
        bins = rec["bins"]
        centers, means = [], []
        for j, b in enumerate(bins):
            lo = aud.val(GATE, f"G0_2.{i}.bins.{j}.lo")
            hi = aud.val(GATE, f"G0_2.{i}.bins.{j}.hi")
            mean_db = aud.val(GATE, f"G0_2.{i}.bins.{j}.mean_adv_db")
            centers.append(lo if hi is None else 0.5 * (lo + hi)); means.append(mean_db)
        c, mk, lab = styles[f_ghz]
        axb.plot(centers, means, marker=mk, color=c, lw=1.3, markersize=4, label=lab)
    axb.axhline(3.0, color="0.4", ls="--", lw=0.9, label="3 dB gate (D-013)")
    axb.set_xlabel("Distance from RSU (m)")
    axb.set_ylabel("Mean genie advantage, polar vs DFT (dB)")
    axb.set_title("(b) Near-field gain vs distance")
    axb.legend()
    fig.tight_layout()
    save_fig(fig, "fig2_gate0")
    # caption sidecar — framing updated per D-018/D-019 (diagram unchanged)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "paper", "figures", "fig2_gate0.caption.txt"),
            "w", encoding="utf-8") as fh:
        fh.write("Fig 2. Near-field occurrence (Gate-0). Panel (a): fraction of users whose "
                 "genie polar-vs-DFT advantage exceeds 3 dB (close-range PRIMARY vs O1 "
                 "far-field ANCHOR). Panel (b): mean genie advantage vs distance. This "
                 "characterises the near-field regime the predictor targets. The recommended "
                 "operating point is the soft-avg (C3) multimodal predictor; radar-alone is "
                 "the single-sensor fallback (see Tables 1-2, n=20). Genie quantities are "
                 "seed-independent.\n")
    aud.write()


if __name__ == "__main__":
    main()
