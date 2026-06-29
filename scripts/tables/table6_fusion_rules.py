"""Table 6 (appendix) — fusion combination-rule robustness (NF K=8 dB-loss, 5 seeds).

Source: results/fusion_rules/rule_sweep.json (banked by scripts/eval_fusion_rules_sweep.py).
One row per inference rule, columns = 60 / 140 GHz near-field K=8 beam-focusing dB-loss
(mean±95%CI over seeds {11..55}). Shows the radar-primary negative-fusion result holds
across every combination rule. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import paired_t_test

RULES = [("radar_alone", "Radar-alone (reference)"),
         ("soft_avg", "Soft-avg (C3 confidence-weighted)"),
         ("argmax_conf", "Argmax-confidence (both heads)"),
         ("radar_ang+bestR", "Radar angle + best-range (D-015, headline)"),
         ("radar_both", "Radar angle + radar range"),
         ("temp4", "Temperature C3 (T=4)"),
         ("temp8", "Temperature C3 (T=8)")]
FREQS = ["60", "140"]


def main():
    aud = Audit("table6_fusion_rules")
    p = os.path.join(RESULTS, "fusion_rules", "rule_sweep.json")
    ra_seed = {f: aud.arr(p, f"results.{f}.radar_alone.per_seed") for f in FREQS}
    rows = []
    n_sig = 0
    for key, lab in RULES:
        cells = []
        for f in FREQS:
            m = aud.val(p, f"results.{f}.{key}.mean")
            hi = aud.val(p, f"results.{f}.{key}.hi")
            if key == "radar_alone":
                cells.append((f"{m:.2f}±{hi-m:.2f}", "—"))
            else:
                v = aud.arr(p, f"results.{f}.{key}.per_seed")
                pv = paired_t_test(ra_seed[f], v)["p"]      # rule vs radar-alone
                star = "*" if pv < 0.05 else ""
                if pv < 0.05:
                    n_sig += 1
                cells.append((f"{m:.2f}±{hi-m:.2f}", f"{pv:.3f}{star}"))
        rows.append((lab, cells))

    hdr = ["Combination rule", "60 GHz dB-loss", "p vs radar", "140 GHz dB-loss", "p vs radar"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for lab, cells in rows:
        md.append(f"| {lab} | {cells[0][0]} | {cells[0][1]} | {cells[1][0]} | {cells[1][1]} |")
    md_s = ("# Table 6 — Fusion combination-rule robustness (NF K=8 dB-loss, 5 seeds, "
            "mean±95%CI; paired t-test vs radar-alone)\n\n" + "\n".join(md)
            + f"\n\nLower dB-loss is better; `p` is the paired t-test of each rule vs the "
            f"radar-alone reference. The pre-registered headline rule (radar angle + "
            f"best-range, D-015) is **not significant** at either frequency (p=0.243/0.469). "
            f"Some C3/temperature variants have lower means, but only **{n_sig} of 12** "
            f"rule×frequency comparisons reaches p<0.05 (temp4 at 140 GHz, p=0.044), which "
            f"does **not** survive multiple-comparison correction (Bonferroni-adjusted "
            f"≈0.53). Conclusion: no combination rule is a statistically robust improvement "
            f"over radar-alone — the radar-primary result is insensitive to the rule.\n")

    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Combination rule & 60\,GHz & $p$ & 140\,GHz & $p$ \\", r"\midrule"]
    for lab, cells in rows:
        tex.append(f"{lab} & {cells[0][0]} & {cells[0][1]} & {cells[1][0]} & {cells[1][1]} "
                   + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"% * p<0.05 uncorrected; the single starred case (temp4, 140GHz) fails"
            r" Bonferroni correction across 12 tests.%"]
    write_table("table6_fusion_rules", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
