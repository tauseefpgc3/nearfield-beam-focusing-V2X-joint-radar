"""Table 1 — in-domain near-field results (close-range synthetic, 5 seeds, mean±95%CI).

Source: results/indomain/{radar,lidar,fusion}_closerange_{60,140}GHz_seed*_metrics.json
Columns: top-1 angle, near-field top-1 range, near-field K=4 coverage,
         near-field K=8 beam-focusing dB-loss (the physical headline metric).
mean±CI over seeds {11,22,33,44,55} via src/eval/statistics.mean_ci. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, SEEDS, metrics_path, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci

FAMILIES = [("radar", "Radar"), ("lidar", "LiDAR"), ("fusion", "Fusion")]
FREQS = ["60", "140"]
# (dotted cell, label, percent?)
COLS = [
    ("top1_theta", "Angle top-1", True),
    ("range_nf", "NF range top-1", True),
    ("coverage.4.nf", "NF cov@4", True),
    ("dbloss.8.nf", "NF dB-loss@8", False),
]


def ci(aud, fam, fghz, cell):
    vals = [aud.val(metrics_path(fam, f"closerange_{fghz}GHz", s), cell) for s in SEEDS]
    return mean_ci(vals)


def cell_str(c, pct):
    if pct:
        return f"{c['mean']*100:.1f}±{(c['hi']-c['mean'])*100:.1f}"
    return f"{c['mean']:.2f}±{c['hi']-c['mean']:.2f}"


def main():
    aud = Audit("table1_indomain")
    rows = []
    for fghz in FREQS:
        for fam, famlab in FAMILIES:
            cells = [cell_str(ci(aud, fam, fghz, c), pct) for c, _, pct in COLS]
            rows.append((f"{fghz} GHz", famlab, cells))

    hdr = ["Freq", "Modality"] + [l for _, l, _ in COLS]
    md = ["| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for fghz, fam, cells in rows:
        md.append("| " + " | ".join([fghz, fam] + cells) + " |")
    md_s = ("# Table 1 — In-domain near-field results (close-range, 5 seeds, mean±95%CI)\n\n"
            + "\n".join(md) + "\n\nPercentages where marked; NF dB-loss@8 in dB "
            "(lower is better). NF = near-field subset (3 dB gate).\n")

    tex = [r"\begin{tabular}{ll" + "r" * len(COLS) + "}", r"\toprule",
           " & ".join(hdr) + r" \\", r"\midrule"]
    prev = None
    for fghz, fam, cells in rows:
        lead = fghz if fghz != prev else ""
        prev = fghz
        tex.append(" & ".join([lead, fam] + cells) + r" \\")
        if fam == "Fusion":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table1_indomain", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
