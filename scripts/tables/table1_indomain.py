"""Table 1 — in-domain near-field results. Prefers the n=20 predictors (D-018) when banked,
else the original n=5 modality breakdown.

n=20 source: results/indomain_n20/close_n20.json (radar-alone + soft-avg C3 recommended
predictor, both freqs; radar/fusion are the checkpoints trained to 20 seeds — LiDAR-alone is
NOT retrained at 20 seeds and stays an n=5 reference, noted in the footnote).
n=5 fallback: results/indomain/{radar,lidar,fusion}_closerange_*_metrics.json.
mean±95%CI via src/eval/statistics.mean_ci. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, SEEDS, metrics_path, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci

N20 = os.path.join(RESULTS, "indomain_n20", "close_n20.json")
FREQS = ["60", "140"]


def cell(c, pct):
    return (f"{c['mean']*100:.1f}±{(c['hi']-c['mean'])*100:.1f}" if pct
            else f"{c['mean']:.2f}±{c['hi']-c['mean']:.2f}")


def build_n20(aud):
    preds = [("softavg", "Soft-avg (recommended)"), ("radar", "Radar-alone")]
    cols = [("top1_theta", "Angle top-1", True), ("range_nf", "NF range top-1", True),
            ("cov4_nf", "NF cov@4", True), ("db8_nf", "NF dB-loss@8", False)]
    rows = []
    for fghz in FREQS:
        for pk, plab in preds:
            cells = [cell(mean_ci(aud.arr(N20, f"results.{fghz}.{pk}.{c}")), pct)
                     for c, _, pct in cols]
            rows.append((f"{fghz} GHz", plab, cells))
    hdr = ["Freq", "Predictor"] + [l for _, l, _ in cols]
    foot = ("\n\n20-seed mean±95%CI (D-018). NF = near-field subset (3 dB gate). "
            "Soft-avg (C3) is the recommended multimodal predictor; radar-alone is the "
            "single-sensor operating point. LiDAR-alone is **not** retrained in the n=20 "
            "extension (it remains an n=5 reference — see the n=5 Table 1 variant).\n")
    return hdr, rows, "In-domain near-field results (n=20)", foot


def build_n5(aud):
    fams = [("radar", "Radar"), ("lidar", "LiDAR"), ("fusion", "Fusion")]
    cols = [("top1_theta", "Angle top-1", True), ("range_nf", "NF range top-1", True),
            ("coverage.4.nf", "NF cov@4", True), ("dbloss.8.nf", "NF dB-loss@8", False)]
    rows = []
    for fghz in FREQS:
        for fam, flab in fams:
            cells = [cell(mean_ci([aud.val(metrics_path(fam, f"closerange_{fghz}GHz", s), c)
                                   for s in SEEDS]), pct) for c, _, pct in cols]
            rows.append((f"{fghz} GHz", flab, cells))
    hdr = ["Freq", "Modality"] + [l for _, l, _ in cols]
    return hdr, rows, "In-domain near-field results (5 seeds)", \
        "\n\nPercentages where marked; NF dB-loss@8 in dB. NF = near-field subset (3 dB gate).\n"


def main():
    aud = Audit("table1_indomain")
    hdr, rows, title, foot = build_n20(aud) if os.path.exists(N20) else build_n5(aud)
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for a, b, cells in rows:
        md.append("| " + " | ".join([a, b] + cells) + " |")
    md_s = f"# Table 1 — {title}\n\n" + "\n".join(md) + foot

    # two-line headers keep the natural width small; \resizebox then guarantees a
    # \columnwidth fit so the last "NF dB-loss@8" column can never clip (0 overfull).
    tex_hdr = [hdr[0], hdr[1]] + [r"\shortstack{" + l.replace(" ", r"\\", 1) + "}"
                                  for l in hdr[2:]]
    tex = [r"\resizebox{\columnwidth}{!}{\setlength{\tabcolsep}{4pt}\begin{tabular}{ll"
           + "r" * (len(hdr) - 2) + "}", r"\toprule",
           " & ".join(tex_hdr) + r" \\", r"\midrule"]
    prev = None
    for a, b, cells in rows:
        lead = a if a != prev else ""; prev = a
        tex.append(" & ".join([lead, b] + cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}"]
    write_table("table1_indomain", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
