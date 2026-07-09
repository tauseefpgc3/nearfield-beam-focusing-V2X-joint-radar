"""Table 9 (appendix) — radar-vs-fusion statistics, per-seed losses + full test battery.

Source: results/fusion_rules/taskA_stats.json (banked by scripts/eval_taskA_stats.py).
Panel A: per-seed near-field K=8 dB-loss (radar-alone, radar-primary D-015, soft-avg C3),
60 & 140 GHz. Panel B: paired t-test, Wilcoxon, Cohen's d_z (95% CI), TOST (+/-0.5 dB).
Honest scope note: n = the available seeds (5), NOT the requested 20. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

# Prefer the adequately-powered n=20 stats if it has been banked; else the n=5 file.
_N20 = os.path.join(RESULTS, "fusion_rules", "taskA_stats_n20.json")
TSJ = _N20 if os.path.exists(_N20) else os.path.join(RESULTS, "fusion_rules", "taskA_stats.json")
RULES = [("radar_alone", "Radar-alone"), ("radar_ang+bestR", "Fusion (D-015)"),
         ("soft_avg", "Fusion (soft-avg C3)")]


def main():
    aud = Audit("table9_fusion_stats")
    n = aud.val(TSJ, "n_seeds")
    seeds = aud.arr(TSJ, "seeds")

    # Panel A — per-seed losses
    perseed = {f: {r: aud.arr(TSJ, f"per_seed.{f}.{r}") for r, _ in RULES} for f in ("60", "140")}
    hdrA = ["Seed", "60: radar", "60: D-015", "60: soft-avg", "140: radar", "140: D-015", "140: soft-avg"]
    mdA = ["| " + " | ".join(hdrA) + " |", "|" + "|".join(["---"] * len(hdrA)) + "|"]
    for i, s in enumerate(seeds):
        r = [f"{perseed['60']['radar_alone'][i]:.2f}", f"{perseed['60']['radar_ang+bestR'][i]:.2f}",
             f"{perseed['60']['soft_avg'][i]:.2f}", f"{perseed['140']['radar_alone'][i]:.2f}",
             f"{perseed['140']['radar_ang+bestR'][i]:.2f}", f"{perseed['140']['soft_avg'][i]:.2f}"]
        mdA.append(f"| {s} | " + " | ".join(r) + " |")

    # Panel B — aggregate stats
    hdrB = ["Comparison (freq)", "radar / rule", "paired-t p", "Wilcoxon p",
            "Cohen d [95% CI]", "TOST±0.5 p", "equiv?"]
    mdB = ["| " + " | ".join(hdrB) + " |", "|" + "|".join(["---"] * len(hdrB)) + "|"]
    for f in ("60", "140"):
        for rule, lab in [("radar_ang+bestR", "D-015"), ("soft_avg", "soft-avg")]:
            k = f"{f}GHz__radar_vs_{rule}"
            rm = aud.val(TSJ, f"comparisons.{k}.radar_mean")
            vm = aud.val(TSJ, f"comparisons.{k}.rule_mean")
            tp = aud.val(TSJ, f"comparisons.{k}.paired_t_p")
            wp = aud.val(TSJ, f"comparisons.{k}.wilcoxon_p")
            d = aud.val(TSJ, f"comparisons.{k}.cohen_d")
            dl = aud.val(TSJ, f"comparisons.{k}.cohen_d_ci95.0")
            dh = aud.val(TSJ, f"comparisons.{k}.cohen_d_ci95.1")
            to = aud.val(TSJ, f"comparisons.{k}.tost_p")
            eq = aud.val(TSJ, f"comparisons.{k}.tost_equivalent")
            mdB.append(f"| radar vs {lab} ({f} GHz) | {rm:.2f} / {vm:.2f} | {tp:.3f} | "
                       f"{wp:.3f} | {d:+.2f} [{dl:+.2f}, {dh:+.2f}] | {to:.3f} | "
                       f"{'yes' if eq else 'no'} |")

    bf_any = aud.val(TSJ, "bonferroni.any_significant")
    bf_m = aud.val(TSJ, "bonferroni.m")
    bf_sig = aud.arr(TSJ, "bonferroni.significant") if bf_any else []
    # Data-driven verdict (so the caption reflects the ACTUAL n, not a hardcoded story).
    def _p(rule, f):
        return aud.val(TSJ, f"comparisons.{f}GHz__radar_vs_{rule}.paired_t_p")
    def _eq(rule, f):
        return aud.val(TSJ, f"comparisons.{f}GHz__radar_vs_{rule}.tost_equivalent")
    sa = {f: _p("soft_avg", f) for f in ("60", "140")}
    d015_eq = {f: _eq("radar_ang+bestR", f) for f in ("60", "140")}
    sa_sig = [f"{f} GHz (p={sa[f]:.3f})" for f in ("60", "140") if sa[f] < 0.05]
    eq_freqs = [f"{f} GHz" for f in ("60", "140") if d015_eq[f]]
    sa_verdict = ("**significantly lower loss than radar-alone at " + ", ".join(sa_sig) + "**"
                  if sa_sig else "not significantly different from radar-alone")
    d015_verdict = ("TOST-**equivalent** to radar-alone (within ±0.5 dB) at " +
                    ", ".join(eq_freqs) if eq_freqs else
                    "neither significantly different nor TOST-equivalent (inconclusive)")
    bonf_verdict = (f"**survives Bonferroni** ({bf_m} comparisons): " + ", ".join(bf_sig)
                    if bf_any else f"no rule survives Bonferroni ({bf_m} comparisons)")
    md = (f"# Table 9 — Radar vs fusion, per-seed losses + statistics (near-field K=8 "
          f"dB-loss, n={n} seeds; D-018)\n\n**Panel A — per-seed loss (dB):**\n\n"
          + "\n".join(mdA)
          + "\n\n**Panel B — aggregate tests (radar-alone vs rule):**\n\n" + "\n".join(mdB)
          + f"\n\n**Verdict (n={n}):** soft-avg (C3) fusion has {sa_verdict}; it "
          f"{bonf_verdict}. Radar-primary (D-015) is {d015_verdict}. Per D-018 the reported "
          f"outcome is whatever this powered study yields; if soft-avg is significant after "
          f"correction the paper reframes to 'a soft ensemble gives a small significant gain; "
          f"radar-primary is an equivalence-tested single-sensor operating point.'\n")

    # LaTeX: Panel B only (compact); per-seed panel A as a second tabular
    texA = [r"\begin{tabular}{r rrr rrr}", r"\toprule",
            r"Seed & \multicolumn{3}{c}{60\,GHz} & \multicolumn{3}{c}{140\,GHz} \\",
            r"& radar & D-015 & soft-avg & radar & D-015 & soft-avg \\", r"\midrule"]
    for i, s in enumerate(seeds):
        texA.append(f"{s} & {perseed['60']['radar_alone'][i]:.2f} & "
                    f"{perseed['60']['radar_ang+bestR'][i]:.2f} & {perseed['60']['soft_avg'][i]:.2f} & "
                    f"{perseed['140']['radar_alone'][i]:.2f} & "
                    f"{perseed['140']['radar_ang+bestR'][i]:.2f} & {perseed['140']['soft_avg'][i]:.2f} \\\\")
    texA += [r"\bottomrule", r"\end{tabular}"]
    write_table("table9_fusion_stats", "\n".join(texA) + "\n", md)
    aud.write()


if __name__ == "__main__":
    main()
