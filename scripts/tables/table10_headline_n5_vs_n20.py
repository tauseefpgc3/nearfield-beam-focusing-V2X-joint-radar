"""Table 10 — headline radar/fusion NF K=8 dB-loss: n=5 vs n=20, side by side (D-018).

Shows whether the headline numbers (radar-alone 0.85/1.60; fusion D-015 0.70/1.70) and the
significance move when the study is powered up from 5 to 20 seeds. Sources:
results/fusion_rules/taskA_stats.json (n=5) and taskA_stats_n20.json (n=20). If the n=20
file is absent (training not yet complete) the table renders a 'pending' placeholder so
run_all stays green. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

N5 = os.path.join(RESULTS, "fusion_rules", "taskA_stats.json")
N20 = os.path.join(RESULTS, "fusion_rules", "taskA_stats_n20.json")
FREQS = ["60", "140"]


def main():
    aud = Audit("table10_headline_n5_vs_n20")
    if not os.path.exists(N20):
        md = ("# Table 10 — headline n=5 vs n=20 (PENDING)\n\n"
              "_The n=20 sweep (D-018) has not finished; `taskA_stats_n20.json` is not yet "
              "banked. Re-run after training completes: "
              "`python scripts/tables/table10_headline_n5_vs_n20.py`._\n")
        write_table("table10_headline_n5_vs_n20",
                    "% Table 10 pending: n=20 not yet banked (D-018).\n", md)
        aud.note("n=20 not banked yet -> placeholder")
        aud.write()
        print("  [table10] n=20 pending -> placeholder written")
        return

    rows = []
    for f in FREQS:
        for rule, lab in [("radar_ang+bestR", "Fusion D-015"), ("soft_avg", "Fusion soft-avg")]:
            k = f"{f}GHz__radar_vs_{rule}"
            ra5 = aud.val(N5, f"comparisons.{k}.radar_mean")
            r5 = aud.val(N5, f"comparisons.{k}.rule_mean")
            p5 = aud.val(N5, f"comparisons.{k}.paired_t_p")
            ra20 = aud.val(N20, f"comparisons.{k}.radar_mean")
            r20 = aud.val(N20, f"comparisons.{k}.rule_mean")
            p20 = aud.val(N20, f"comparisons.{k}.paired_t_p")
            w20 = aud.val(N20, f"comparisons.{k}.wilcoxon_p")
            rows.append((f"{f} GHz", lab, f"{ra5:.2f} / {r5:.2f}", f"{p5:.3f}",
                         f"{ra20:.2f} / {r20:.2f}", f"{p20:.3f}", f"{w20:.3f}"))
    n5 = aud.val(N5, "n_seeds"); n20 = aud.val(N20, "n_seeds")

    hdr = ["Freq", "Comparison", f"radar/rule (n={n5})", f"t-p (n={n5})",
           f"radar/rule (n={n20})", f"t-p (n={n20})", f"W-p (n={n20})"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md_s = (f"# Table 10 — Headline NF K=8 dB-loss: n={n5} vs n={n20} (D-018)\n\n"
            + "\n".join(md) + "\n\nradar/rule = radar-alone / fusion-rule mean dB-loss "
            "(lower better). Powering up from 5 to 20 seeds shows whether the headline "
            "numbers and the fusion significance move; see Table 9 (n=20) for the full "
            "battery (Wilcoxon, Cohen's d CI, TOST, Bonferroni).\n")

    tex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           " & ".join(hdr).replace("=", "$=$") + r" \\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join(r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table10_headline_n5_vs_n20", "\n".join(tex) + "\n", md_s)
    aud.write()
    print(f"  [table10] n={n5} vs n={n20} written")


if __name__ == "__main__":
    main()
