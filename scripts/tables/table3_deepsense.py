"""Table 3 — DeepSense6G real-data validation: top-1 beam (angle) accuracy.

Source: results/indomain/{radar,lidar,fusion_warmstart}_deepsense_scenario{31,36}_seed*.json
        :: test.theta  (one per seed; mean±95%CI over 5 seeds).
Scenario 31 = V2I (RSU), scenario 36 = V2V. In-domain top-1 rows come from the per-seed
training JSONs (test.theta); the zero-shot transfer rows come from
results/deepsense/transfer.json (banked by scripts/deepsense_decomp.py). JSON-only.
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci

# (scenario, role, family, family-label)
ROWS = [
    (31, "V2I (RSU)", "radar", "Radar"),
    (31, "V2I (RSU)", "lidar", "LiDAR"),
    (31, "V2I (RSU)", "fusion_warmstart", "Fusion"),
    (36, "V2V", "radar", "Radar"),
]


def seed_files(family, scen):
    return sorted(glob.glob(os.path.join(
        RESULTS, "indomain", f"{family}_deepsense_scenario{scen}_seed*.json")))


def main():
    aud = Audit("table3_deepsense")
    out = []
    for scen, role, fam, famlab in ROWS:
        files = seed_files(fam, scen)
        vals, n_test = [], "—"
        for f in files:
            vals.append(aud.val(f, "test.theta"))
            try:                                  # fusion test block omits n; theta is the point
                n_test = aud.val(f, "test.n")
            except KeyError:
                pass
        c = mean_ci(vals)
        out.append((f"Sc {scen}", role, famlab, len(files), str(n_test),
                    f"{c['mean']*100:.2f}±{(c['hi']-c['mean'])*100:.2f}"))

    # Zero-shot transfer rows (banked in results/deepsense/transfer.json)
    tp = os.path.join(RESULTS, "deepsense", "transfer.json")
    rand = aud.val(tp, "random") * 100
    transfer_rows = []
    for key in ("sc31_to_sc36", "synthetic_to_sc31"):
        desc = aud.val(tp, f"combos.{key}.desc")
        n = aud.val(tp, f"combos.{key}.n")
        m = aud.val(tp, f"combos.{key}.top1.mean")
        e = aud.val(tp, f"combos.{key}.top1.hi") - m
        transfer_rows.append((desc, str(n), f"{m*100:.2f}±{e*100:.2f}"))

    hdr = ["Scenario", "Link", "Modality", "Seeds", "n(test)", "Top-1 beam (%)"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for s, role, fam, ns, nt, acc in out:
        md.append(f"| {s} | {role} | {fam} | {ns} | {nt} | {acc} |")
    md += ["", f"**Zero-shot transfer** (random chance = {rand:.2f}%):",
           "", "| Transfer | n | Top-1 beam (%) |", "|---|---|---|"]
    for desc, n, acc in transfer_rows:
        md.append(f"| {desc} | {n} | {acc} |")
    md_s = ("# Table 3 — DeepSense6G real-data validation (top-1 beam accuracy, mean±95%CI)\n\n"
            + "\n".join(md) + "\n\nIn-domain rows: model trained and tested within scenario "
            "(trajectory-grouped split). Zero-shot rows collapse to ≈chance, showing the "
            "learned features are scene-specific (no free real-world transfer).\n")

    tex = [r"\begin{tabular}{lllrrr}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule"]
    for s, role, fam, ns, nt, acc in out:
        tex.append(f"{s} & {role} & {fam} & {ns} & {nt} & {acc} " + r"\\")
    tex += [r"\midrule",
            r"\multicolumn{6}{l}{\emph{Zero-shot transfer (random = %.2f\%%)}} \\" % rand]
    for desc, n, acc in transfer_rows:
        tex.append(r"\multicolumn{3}{l}{%s} & & %s & %s \\" % (desc, n, acc))
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table3_deepsense", "\n".join(tex) + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
