"""Table 8 — beam-focusing gain vs RF-measurement overhead. Prefers n=20 when banked.

n=20 source: results/gain_overhead_n20.json (hierarchical 71-measurement sweep + our soft-avg
and radar predictors at K in {1,2,4,8}, 20-seed mean±95%CI). Exhaustive polar sweep = 512
measurements, 0 dB (genie). n=5 fallback: hierarchical_sweep.json + radar metrics. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, SEEDS, metrics_path, write_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci

GO20 = os.path.join(RESULTS, "gain_overhead_n20.json")
FREQS = ["60", "140"]


def main():
    aud = Audit("table8_gain_overhead")
    rows = []
    if os.path.exists(GO20):
        nm = aud.val(GO20, "hier_measurements"); nseed = aud.val(GO20, "n_seeds")
        rows.append(("Exhaustive polar sweep (genie)", "512", {f: "0.00" for f in FREQS}))
        rows.append((f"Hierarchical sweep (angle→range)", str(nm),
                     {f: f"{aud.val(GO20, f'results.{f}.hierarchical.mean'):+.2f}".lstrip('+')
                      + f"±{aud.val(GO20, f'results.{f}.hierarchical.hi')-aud.val(GO20, f'results.{f}.hierarchical.mean'):.2f}"
                      for f in FREQS}))
        for pred, plab in (("softavg", "Ours soft-avg (recommended)"), ("radar", "Ours radar-alone")):
            for k in (1, 2, 4, 8):
                cells = {}
                for f in FREQS:
                    m = aud.val(GO20, f"results.{f}.{pred}.K{k}.mean")
                    hi = aud.val(GO20, f"results.{f}.{pred}.K{k}.hi")
                    cells[f] = f"{m:.2f}±{hi-m:.2f}"
                rows.append((f"{plab}, K={k}", str(k), cells))
        title = f"Beam-focusing gain vs RF-measurement overhead (n={nseed})"
        note = ("\n\nLower dB-loss is better. The recommended soft-avg predictor reaches "
                f"near-genie focusing with a K=8 shortlist — ~{round(nm/8)}× fewer RF "
                f"measurements than the {nm}-measurement hierarchical sweep, at lower loss.\n")
    else:
        hp = os.path.join(RESULTS, "baselines", "hierarchical_sweep.json")
        nm = aud.val(hp, "n_measurements")
        rows.append(("Exhaustive polar sweep (genie)", "512", {f: "0.00" for f in FREQS}))
        rows.append(("Hierarchical sweep", str(nm),
                     {f: f"{aud.val(hp, f'results.{f}.mean'):.2f}" for f in FREQS}))
        for k in (1, 2, 4, 8):
            cells = {f: f"{mean_ci([aud.val(metrics_path('radar', f'closerange_{f}GHz', s), f'dbloss.{k}.nf') for s in SEEDS])['mean']:.2f}"
                     for f in FREQS}
            rows.append((f"Ours (radar), K={k}", str(k), cells))
        title = "Beam-focusing gain vs RF-measurement overhead (5 seeds)"
        note = "\n\nLower is better.\n"

    hdr = ["Method", "# RF meas.", "60 GHz dB-loss", "140 GHz dB-loss"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for name, meas, cells in rows:
        md.append(f"| {name} | {meas} | {cells['60']} | {cells['140']} |")
    md_s = f"# Table 8 — {title}, mean±95%CI\n\n" + "\n".join(md) + note

    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Method & \# RF meas. & 60\,GHz & 140\,GHz \\", r"\midrule"]
    for name, meas, cells in rows:
        tex.append(f"{name} & {meas} & {cells['60']} & {cells['140']} ".replace("→", r"$\to$") + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table8_gain_overhead", "\n".join(tex).replace("→", r"$\to$") + "\n", md_s)
    aud.write()


if __name__ == "__main__":
    main()
