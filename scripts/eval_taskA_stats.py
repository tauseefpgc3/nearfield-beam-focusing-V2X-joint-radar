"""Task A statistics — radar vs fusion at the AVAILABLE seeds (no new training).

Reads the banked per-seed near-field K=8 dB-loss from results/fusion_rules/rule_sweep.json
(5 seeds: 11,22,33,44,55) and computes, for radar-alone vs the two headline combination
rules (radar-primary 'radar_ang+bestR' and soft-avg C3), at 60 and 140 GHz:
  paired t-test, Wilcoxon signed-rank, Cohen's d_z with bootstrap 95% CI, and a TOST
  equivalence test with margin +/-0.5 dB.
Also applies Bonferroni correction over the full Appendix-A rule sweep (all fusion rules x
both frequencies vs radar-alone). Writes results/fusion_rules/taskA_stats.json.

NOTE (honest scope): this is n=5 seeds, NOT the reviewer-requested 20 — the extended sweep
was not run. TOST at n=5 has low power (wide CIs), so 'not equivalent' here means
'inconclusive', not 'different'. Every number is labelled with n.

    python scripts/eval_taskA_stats.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.statistics import (bonferroni, cohen_d_ci, mean_ci, paired_t_test,
                                 tost_paired, wilcoxon_signed_rank)

FUSION_DIR = os.path.join(_ROOT, "results", "fusion_rules")
FREQS = ["60", "140"]
HEADLINE = ["radar_ang+bestR", "soft_avg"]          # the two rules the reviewer named
ALL_FUSION = ["soft_avg", "argmax_conf", "radar_ang+bestR", "radar_both", "temp4", "temp8"]
TOST_MARGIN = 0.5


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-sweep", default="rule_sweep.json",
                    help="input filename under results/fusion_rules/")
    ap.add_argument("--out", default="taskA_stats.json",
                    help="output filename under results/fusion_rules/")
    args = ap.parse_args()
    rs = json.load(open(os.path.join(FUSION_DIR, args.rule_sweep)))
    seeds = rs["seeds"]
    n = len(seeds)
    out = {"n_seeds": n, "seeds": seeds, "tost_margin_db": TOST_MARGIN,
           "metric": "near-field K=8 beam-focusing dB-loss vs genie",
           "note": f"n={n} seeds (available), NOT the requested 20; low power",
           "per_seed": {}, "comparisons": {}, "bonferroni": {}}

    # per-seed table (radar-alone + the two headline rules), per freq
    for f in FREQS:
        out["per_seed"][f] = {"radar_alone": rs["results"][f]["radar_alone"]["per_seed"]}
        for r in HEADLINE:
            out["per_seed"][f][r] = rs["results"][f][r]["per_seed"]

    # full stat battery: radar-alone vs each headline rule, both freqs
    for f in FREQS:
        ra = rs["results"][f]["radar_alone"]["per_seed"]
        for r in HEADLINE:
            v = rs["results"][f][r]["per_seed"]
            pt = paired_t_test(ra, v)            # radar - rule (>0 => rule lower/better)
            wc = wilcoxon_signed_rank(ra, v)
            dd = cohen_d_ci(ra, v, seed=0)
            to = tost_paired(ra, v, margin=TOST_MARGIN)
            out["comparisons"][f"{f}GHz__radar_vs_{r}"] = dict(
                freq=f, rule=r,
                radar_mean=mean_ci(ra)["mean"], rule_mean=mean_ci(v)["mean"],
                mean_diff_radar_minus_rule=pt["mean_diff"],
                paired_t_p=pt["p"], paired_t=pt["t"],
                wilcoxon_p=wc["p"], wilcoxon_stat=wc["statistic"],
                cohen_d=dd["d"], cohen_d_ci95=[dd["lo"], dd["hi"]],
                tost_margin=TOST_MARGIN, tost_p=to["p_tost"],
                tost_equivalent=to["equivalent"], tost_ci=list(to["ci"]))

    # Bonferroni over the full rule sweep (all fusion rules x 2 freqs vs radar-alone)
    fam = []
    for f in FREQS:
        ra = rs["results"][f]["radar_alone"]["per_seed"]
        for r in ALL_FUSION:
            fam.append((f"{f}GHz:{r}", paired_t_test(ra, rs["results"][f][r]["per_seed"])["p"]))
    bf = bonferroni([p for _, p in fam])
    out["bonferroni"] = dict(family=[k for k, _ in fam], m=bf["m"],
                             alpha_corrected=bf["alpha_corrected"],
                             raw_p=[p for _, p in fam], corrected_p=bf["corrected"],
                             any_significant=bool(any(bf["reject"])),
                             significant=[k for (k, _), rej in zip(fam, bf["reject"]) if rej])

    json.dump(out, open(os.path.join(FUSION_DIR, args.out), "w"), indent=2)
    print(f"Wrote results/fusion_rules/{args.out} (n={n} seeds)")
    for key, c in out["comparisons"].items():
        print(f"  {key}: radar {c['radar_mean']:.2f} vs {c['rule']} {c['rule_mean']:.2f} | "
              f"t-p={c['paired_t_p']:.3f} W-p={c['wilcoxon_p']:.3f} "
              f"d={c['cohen_d']:+.2f}[{c['cohen_d_ci95'][0]:+.2f},{c['cohen_d_ci95'][1]:+.2f}] "
              f"TOST±0.5 p={c['tost_p']:.3f} equiv={c['tost_equivalent']}")
    print(f"  Bonferroni (m={bf['m']}, alpha_corr={bf['alpha_corrected']:.4f}): "
          f"any significant = {out['bonferroni']['any_significant']}")


if __name__ == "__main__":
    main()
