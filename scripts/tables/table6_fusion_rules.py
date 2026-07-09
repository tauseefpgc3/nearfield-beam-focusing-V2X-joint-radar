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
         ("radar_ang+bestR", "Radar angle + best-range (D-015, single-sensor rule)"),
         ("radar_both", "Radar angle + radar range"),
         ("temp4", "Temperature C3 (T=4)"),
         ("temp8", "Temperature C3 (T=8)")]
FREQS = ["60", "140"]


def main():
    aud = Audit("table6_fusion_rules")
    # prefer the n=20 sweep (D-018) if banked, else n=5
    _n20 = os.path.join(RESULTS, "fusion_rules", "rule_sweep_n20.json")
    p = _n20 if os.path.exists(_n20) else os.path.join(RESULTS, "fusion_rules", "rule_sweep.json")
    nseed = len(aud.arr(p, "seeds"))
    ra_seed = {f: aud.arr(p, f"results.{f}.radar_alone.per_seed") for f in FREQS}
    rows = []
    n_sig = 0; d015_p = {}; sig_rules = []; min_sig_p = 1.0; bonf_p = {}
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
                if key == "radar_ang+bestR":
                    d015_p[f] = pv
                if pv < 0.05:
                    n_sig += 1; sig_rules.append(f"{lab} @{f}GHz"); min_sig_p = min(min_sig_p, pv)
                    bonf_p[f"{lab} @{f}GHz"] = pv
                cells.append((f"{m:.2f}±{hi-m:.2f}", f"{pv:.3f}{'*' if pv < 0.05 else ''}"))
        rows.append((lab, cells))
    m_tests = (len(RULES) - 1) * len(FREQS)     # non-radar rules x freqs

    hdr = ["Combination rule", "60 GHz dB-loss", "p vs radar", "140 GHz dB-loss", "p vs radar"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for lab, cells in rows:
        md.append(f"| {lab} | {cells[0][0]} | {cells[0][1]} | {cells[1][0]} | {cells[1][1]} |")
    # data-driven caption (all p-values from the JSON via paired_t_test above)
    d015_txt = "/".join(f"{d015_p[f]:.3f}" for f in FREQS)
    if n_sig:
        bonf = f"the lowest ({min_sig_p:.3f}) gives Bonferroni-adjusted ≈{min(min_sig_p*m_tests,1):.2f}"
        sig_txt = (f"**{n_sig} of {m_tests}** rule×frequency comparisons reach p<0.05 "
                   f"({', '.join(sig_rules)}); {bonf}")
    else:
        sig_txt = f"**none** of the {m_tests} rule×frequency comparisons reaches p<0.05"
    md_s = (f"# Table 6 — Fusion combination-rule robustness (NF K=8 dB-loss, {nseed} seeds, "
            "mean±95%CI; paired t-test vs radar-alone)\n\n" + "\n".join(md)
            + f"\n\nLower dB-loss is better; `p` is the paired t-test of each rule vs the "
            f"radar-alone reference. The pre-registered single-sensor rule (D-015, radar angle + "
            f"best-range) has p={d015_txt} (60/140 GHz). {sig_txt}. See Table 9/Table 11 "
            f"Panel B for the full battery and the clean modality-vs-ensembling control.\n")

    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Combination rule & 60\,GHz & $p$ & 140\,GHz & $p$ \\", r"\midrule"]
    for lab, cells in rows:
        tex.append(f"{lab} & {cells[0][0]} & {cells[0][1]} & {cells[1][0]} & {cells[1][1]} "
                   + r"\\")
    survive = [r for r, pv in bonf_p.items() if pv * m_tests < 0.05]
    surv_txt = (", ".join(survive) + f" survive Bonferroni (x{m_tests})") if survive \
        else f"none survive Bonferroni (x{m_tests})"
    tex += [r"\bottomrule", r"\end{tabular}",
            f"% * p<0.05 uncorrected; {surv_txt}.%"]
    write_table("table6_fusion_rules", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
