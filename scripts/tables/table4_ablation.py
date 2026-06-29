"""Table 4 — E10 range-codebook resolution ablation (B_r ∈ {4,8,16}, 3 seeds).

Source: results/ablation/br{4,8,16}_seed{11,22,33}.json
Comparable physical metric = near-field K=4 / K=8 beam-focusing dB-loss vs each B_r's own
genie (lower=better). top-1 range is reported but NOT comparable across B_r (fewer bins =
easier), and populated_bins shows codebook utilization. mean±95%CI over 3 seeds. JSON-only.

NOTE: the COMMON-near-field-set absolute-gain re-evaluation (ablate_br_common.py) was
console-only and is NOT banked as JSON; that cross-check is logged in MISSING_FROM_JSON.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci

B_RS = [4, 8, 16]
SEEDS = [11, 22, 33]


def path(b_r, seed):
    return os.path.join(RESULTS, "ablation", f"br{b_r}_seed{seed}.json")


def ci(aud, b_r, cell):
    return mean_ci([aud.val(path(b_r, s), cell) for s in SEEDS])


def main():
    aud = Audit("table4_ablation")
    rows = []
    for b_r in B_RS:
        k4 = ci(aud, b_r, "K4_nf_db")
        k8 = ci(aud, b_r, "K8_nf_db")
        tr = ci(aud, b_r, "top1_range_nf")
        pop = ci(aud, b_r, "populated_bins")
        rows.append((b_r, f"{k4['mean']:.2f}±{k4['hi']-k4['mean']:.2f}",
                     f"{k8['mean']:.2f}±{k8['hi']-k8['mean']:.2f}",
                     f"{tr['mean']*100:.1f}±{(tr['hi']-tr['mean'])*100:.1f}",
                     f"{pop['mean']:.1f}/{b_r}"))
    # IMPORTANT (E10 metric trap, supervisor-flagged): these dB-loss columns are EACH
    # measured on that B_r's OWN gated near-field set against its OWN genie, so they are
    # NOT comparable across B_r (a coarser B_r=4 has an easier self-genie → looks better).
    # The valid cross-B_r ranking is the common-near-field-set absolute realized gain
    # (results/ablation/common_nf_absgain.json), where B_r=8 wins. Read it here.
    common = os.path.join(RESULTS, "ablation", "common_nf_absgain.json")
    KEPT = aud.val(common, "best_b_r")             # JSON-derived (common-set cross-check)
    abs_gain = {b: (aud.val(common, f"br{b}.K8_abs_db.mean"),
                    aud.val(common, f"br{b}.K8_abs_db.hi") -
                    aud.val(common, f"br{b}.K8_abs_db.mean")) for b in B_RS}
    aud.note(f"per-gate dB-loss NOT comparable across B_r; common-set absolute gain (banked) "
             f"selects B_r={KEPT} (D-016).")

    hdr = ["B_r", "NF dB-loss@4*", "NF dB-loss@8*", "NF range top-1 (%)*",
           "Populated bins", "Abs. gain@8 (dB)†"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for b, k4, k8, tr, pop in rows:
        mark = " (kept)" if b == KEPT else ""
        ag = abs_gain[b]
        md.append(f"| {b}{mark} | {k4} | {k8} | {tr} | {pop} | {ag[0]:+.2f}±{ag[1]:.2f} |")
    md_s = ("# Table 4 — Range-codebook resolution ablation (E10, 3 seeds, mean±95%CI)\n\n"
            + "\n".join(md) + "\n\n\\*Per-gate dB-loss and top-1 range are **not comparable "
            "across B_r**: each is measured on that B_r's own gated near-field set against "
            "its own (coarser) genie, so a smaller B_r is flattered.\n\n"
            "†The comparable metric: **absolute** realized beam-focusing gain at K=8 on a "
            "**common** near-field set (identical users), referenced to the finest B_r=16 "
            f"genie. On this metric **B_r={KEPT} wins** ({abs_gain[KEPT][0]:+.2f} dB) and is "
            "kept (D-016) — B_r=4 loses by under-resolving range, B_r=16 by sparse, "
            "noisy bins.\n")

    tex = [r"\begin{tabular}{rrrrrr}", r"\toprule",
           " & ".join(hdr).replace("B_r", r"$B_r$").replace("*", r"$^\ddagger$")
           .replace("†", r"$^\dagger$") + r" \\", r"\midrule"]
    for b, k4, k8, tr, pop in rows:
        star = r"\,(kept)" if b == KEPT else ""
        ag = abs_gain[b]
        tex.append(f"{b}{star} & {k4} & {k8} & {tr} & {pop} & "
                   f"{ag[0]:+.2f}$\\pm${ag[1]:.2f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"% ddagger: per-gate metrics NOT comparable across B_r."
            r" dagger: common-set absolute gain selects B_r=8 (D-016).%"]
    write_table("table4_ablation", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
