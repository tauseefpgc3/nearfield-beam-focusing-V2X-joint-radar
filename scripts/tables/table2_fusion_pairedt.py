"""Table 2 — fusion vs radar on NF K=8 dB-loss. Prefers n=20 (D-018) when banked, else n=5.

n=20 source: results/fusion_rules/taskA_stats_n20.json — soft-avg (C3, recommended) and
D-015 radar-primary, each vs radar-alone, both freqs: paired t, Wilcoxon, Cohen's d (95% CI),
TOST ±0.5 dB. n=5 fallback: the original radar-vs-fusion paired t over the *_metrics.json.
JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, SEEDS, metrics_path, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci, paired_t_test

N20 = os.path.join(RESULTS, "fusion_rules", "taskA_stats_n20.json")
FREQS = ["60", "140"]


def main():
    aud = Audit("table2_fusion_pairedt")
    if os.path.exists(N20):
        n = aud.val(N20, "n_seeds")
        rules = [("soft_avg", "Soft-avg (C3)"), ("radar_ang+bestR", "D-015 radar-primary")]
        hdr = ["Freq", "Comparison", "radar / rule", "paired-t p", "Wilcoxon p",
               "Cohen d [95% CI]", "TOST±0.5", "sig?"]
        md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for f in FREQS:
            for rk, rlab in rules:
                k = f"{f}GHz__radar_vs_{rk}"
                rm = aud.val(N20, f"comparisons.{k}.radar_mean")
                vm = aud.val(N20, f"comparisons.{k}.rule_mean")
                tp = aud.val(N20, f"comparisons.{k}.paired_t_p")
                wp = aud.val(N20, f"comparisons.{k}.wilcoxon_p")
                d = aud.val(N20, f"comparisons.{k}.cohen_d")
                dl = aud.val(N20, f"comparisons.{k}.cohen_d_ci95.0")
                dh = aud.val(N20, f"comparisons.{k}.cohen_d_ci95.1")
                eq = aud.val(N20, f"comparisons.{k}.tost_equivalent")
                sig = "yes" if tp < 0.05 else "no"
                md.append(f"| {f} GHz | vs {rlab} | {rm:.2f} / {vm:.2f} | {tp:.3f} | {wp:.3f} "
                          f"| {d:+.2f} [{dl:+.2f}, {dh:+.2f}] | {'equiv' if eq else '—'} | {sig} |")
        anysig = aud.val(N20, "bonferroni.any_significant")
        md_s = (f"# Table 2 — Fusion vs radar-alone (NF K=8 dB-loss, n={n} seeds, D-018)\n\n"
                + "\n".join(md) + f"\n\nLower dB-loss is better. At n={n}, soft-avg (C3) is "
                "significantly below radar-alone at both frequencies (and survives Bonferroni "
                f"— any-rule-significant={anysig}); D-015 radar-primary is TOST-equivalent to "
                "radar-alone at 60 GHz. See the D-019 clean control (Table 11 Panel B): the "
                "soft-avg gain is largely generic ensembling, not the LiDAR modality.\n")
        tex_hdr = " & ".join(hdr).replace("±", r"$\pm$")
    else:
        n = len(SEEDS)
        hdr = ["Freq", "Radar dB-loss", "Fusion dB-loss", "Δ(radar−fusion)", "t", "p", "Cohen d"]
        md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for fghz in FREQS:
            radar = [aud.val(metrics_path("radar", f"closerange_{fghz}GHz", s), "dbloss.8.nf") for s in SEEDS]
            fusion = [aud.val(metrics_path("fusion", f"closerange_{fghz}GHz", s), "dbloss.8.nf") for s in SEEDS]
            st = paired_t_test(radar, fusion)
            md.append(f"| {fghz} GHz | {mean_ci(radar)['mean']:.2f} | {mean_ci(fusion)['mean']:.2f} "
                      f"| {st['mean_diff']:+.3f} | {st['t']:.2f} | {st['p']:.3f} | {st['cohen_d']:+.2f} |")
        md_s = (f"# Table 2 — Fusion vs radar paired t-test (NF K=8 dB-loss, {n} seeds)\n\n"
                + "\n".join(md) + "\n\nNo frequency reaches p<0.05 at n=5 (underpowered).\n")
        tex_hdr = " & ".join(hdr).replace("Δ", r"$\Delta$").replace("−", "$-$")

    tex = [r"\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", r"\toprule", tex_hdr + r" \\", r"\midrule"]
    for line in md[2:]:
        tex.append(" & ".join(x.strip() for x in line.strip("|").split("|")).replace("±", r"$\pm$") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table2_fusion_pairedt", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
