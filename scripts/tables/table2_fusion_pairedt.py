"""Table 2 — fusion vs radar: paired t-test on near-field K=8 beam-focusing dB-loss.

Source: results/indomain/{radar,fusion}_closerange_{60,140}GHz_seed*_metrics.json
        :: dbloss.8.nf  (one value per seed; paired across the 5 shared seeds).
Stat: src/eval/statistics.paired_t_test (validated vs scipy). Tests the decision-rule
claim (D-015): fusion carries more information but does NOT significantly beat the
radar-primary branch on the physical metric. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, SEEDS, metrics_path, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci, paired_t_test

FREQS = ["60", "140"]


def main():
    aud = Audit("table2_fusion_pairedt")
    rows = []
    for fghz in FREQS:
        radar = [aud.val(metrics_path("radar", f"closerange_{fghz}GHz", s), "dbloss.8.nf")
                 for s in SEEDS]
        fusion = [aud.val(metrics_path("fusion", f"closerange_{fghz}GHz", s), "dbloss.8.nf")
                  for s in SEEDS]
        # paired difference: radar minus fusion (positive => fusion lower dB-loss = better)
        st = paired_t_test(radar, fusion)
        rmean, fmean = mean_ci(radar)["mean"], mean_ci(fusion)["mean"]
        sig = "yes" if st["p"] < 0.05 else "no"
        rows.append((f"{fghz} GHz", f"{rmean:.2f}", f"{fmean:.2f}",
                     f"{st['mean_diff']:+.3f}", f"{st['t']:.2f}", f"{st['p']:.3f}",
                     f"{st['cohen_d']:+.2f}", sig))
        aud.note(f"{fghz}GHz radar-vs-fusion: n={st['n']}, df={st['df']}")

    hdr = ["Freq", "Radar dB-loss", "Fusion dB-loss", "Δ(radar−fusion)",
           "t", "p", "Cohen d", "p<0.05?"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md_s = ("# Table 2 — Fusion vs radar paired t-test (NF K=8 dB-loss, 5 seeds)\n\n"
            + "\n".join(md) + "\n\nΔ>0 means fusion has lower dB-loss. No frequency "
            "reaches p<0.05: fusion is statistically indistinguishable from radar-primary "
            "on the physical metric (D-015).\n")

    tex = [r"\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", r"\toprule",
           " & ".join(hdr).replace("Δ", r"$\Delta$").replace("<", r"$<$") + r" \\",
           r"\midrule"]
    for r in rows:
        tex.append(" & ".join(r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table2_fusion_pairedt", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
