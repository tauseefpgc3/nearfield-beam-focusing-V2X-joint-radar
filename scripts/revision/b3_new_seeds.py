"""B3 (pre-registered D-023): primary-endpoint tests on the 15 new seeds alone vs n=20 vs n=5.

Reads the banked per-seed NF K=8 losses in results/fusion_rules/rule_sweep_n20.json and
applies exactly the battery of scripts/eval_taskA_stats.py. CPU only, no inference.

    python scripts/revision/b3_new_seeds.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, dump                                   # noqa: E402
from src.eval.statistics import (bonferroni, cohen_d_ci, mean_ci, paired_t_test,  # noqa: E402
                                 tost_paired, wilcoxon_signed_rank)

FREQS = ["60", "140"]
HEADLINE = ["radar_ang+bestR", "soft_avg"]
ALL_FUSION = ["soft_avg", "argmax_conf", "radar_ang+bestR", "radar_both", "temp4", "temp8"]
SETS = {"new15": [66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220],
        "all20": [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220],
        "orig5": [11, 22, 33, 44, 55]}


def battery(rs, seeds):
    pos = [rs["seeds"].index(s) for s in seeds]
    get = lambda f, r: [rs["results"][f][r]["per_seed"][i] for i in pos]   # noqa: E731
    comps = {}
    for f in FREQS:
        ra = get(f, "radar_alone")
        for r in HEADLINE:
            v = get(f, r)
            pt, wc = paired_t_test(ra, v), wilcoxon_signed_rank(ra, v)
            dd, to = cohen_d_ci(ra, v, seed=0), tost_paired(ra, v, margin=0.5)
            comps[f"{f}GHz__radar_vs_{r}"] = dict(
                radar_mean=mean_ci(ra)["mean"], rule_mean=mean_ci(v)["mean"],
                mean_diff_radar_minus_rule=pt["mean_diff"], diff_ci95=list(pt["ci95"]),
                paired_t=pt["t"], paired_t_p=pt["p"], wilcoxon_stat=wc["statistic"], wilcoxon_p=wc["p"],
                cohen_d=dd["d"], cohen_d_ci95=[dd["lo"], dd["hi"]],
                tost_p=to["p_tost"], tost_equivalent=to["equivalent"], tost_ci90=list(to["ci"]),
                n_rule_better=int(np.sum(np.array(ra) > np.array(v))), n=len(seeds))
    fam = [(f"{f}GHz:{r}", paired_t_test(get(f, "radar_alone"), get(f, r))["p"])
           for f in FREQS for r in ALL_FUSION]
    bf = bonferroni([p for _, p in fam])
    return dict(comparisons=comps, bonferroni=dict(
        m=bf["m"], alpha_corrected=bf["alpha_corrected"],
        raw_p=dict(zip([k for k, _ in fam], [p for _, p in fam])),
        corrected_p=dict(zip([k for k, _ in fam], bf["corrected"])),
        significant=[k for (k, _), rej in zip(fam, bf["reject"]) if rej]))


def main():
    rs = json.load(open(os.path.join(ROOT, "results", "fusion_rules", "rule_sweep_n20.json")))
    out = {"source": "results/fusion_rules/rule_sweep_n20.json (banked per-seed NF K=8 dB-loss)",
           "prereg": "docs/decisions.md D-023, release commit 99b5cda",
           "note": "orig5 fusion ckpts: radar branch frozen; new15: radar branch trained (A5)",
           "results": {k: battery(rs, v) for k, v in SETS.items()}}
    # consistency: our all20 recomputation vs the banked taskA_stats_n20.json
    bank = json.load(open(os.path.join(ROOT, "results", "fusion_rules", "taskA_stats_n20.json")))
    diffs = {}
    for key, c in bank["comparisons"].items():
        mine = out["results"]["all20"]["comparisons"][key]
        diffs[key] = max(abs(mine["paired_t_p"] - c["paired_t_p"]), abs(mine["wilcoxon_p"] - c["wilcoxon_p"]),
                         abs(mine["cohen_d"] - c["cohen_d"]), abs(mine["tost_p"] - c["tost_p"]))
    out["all20_vs_banked_taskA_stats_n20_maxabs"] = diffs
    dump("B3_new_seeds.json", out)
    for sk, r in out["results"].items():
        print(f"== {sk}")
        for k, c in r["comparisons"].items():
            print(f"  {k}: radar {c['radar_mean']:.3f} rule {c['rule_mean']:.3f} diff {c['mean_diff_radar_minus_rule']:+.3f} "
                  f"t-p {c['paired_t_p']:.4f} W-p {c['wilcoxon_p']:.4f} d {c['cohen_d']:+.2f} "
                  f"[{c['cohen_d_ci95'][0]:+.2f},{c['cohen_d_ci95'][1]:+.2f}] TOST p {c['tost_p']:.4f} "
                  f"eq={c['tost_equivalent']} better {c['n_rule_better']}/{c['n']}")
        print(f"  Bonferroni m={r['bonferroni']['m']} significant: {r['bonferroni']['significant']}")
    print("all20 vs banked:", {k: f"{v:.2e}" for k, v in diffs.items()})


if __name__ == "__main__":
    main()
