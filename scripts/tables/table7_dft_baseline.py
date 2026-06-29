"""Table 7 (appendix) — sensor-driven polar vs DFT-only codebook (NF K=8 dB-loss, 5 seeds).

Source: results/dft_baseline/dft_sweep.json (banked by scripts/eval_dft_baseline.py).
Our radar model evaluated with the full polar (angle x range) shortlist vs restricted to the
64 far-field DFT codewords (j=0 slice), on the same near-field test subset. Quantifies what a
sensor-driven angle-only DFT codebook leaves on the table in the near field. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

FREQS = ["60", "140"]
ROWS = [("polar", "Polar (angle×range, ours)"),
        ("dft", "DFT-only (angle, j=0 slice)"),
        ("dft_minus_polar", "Penalty (DFT − polar)")]


def main():
    aud = Audit("table7_dft_baseline")
    p = os.path.join(RESULTS, "dft_baseline", "dft_sweep.json")
    rows = []
    for key, lab in ROWS:
        cells = []
        for f in FREQS:
            m = aud.val(p, f"results.{f}.{key}.mean")
            hi = aud.val(p, f"results.{f}.{key}.hi")
            sign = "+" if key == "dft_minus_polar" else ""
            cells.append(f"{sign}{m:.2f}±{hi-m:.2f}")
        rows.append((lab, cells))

    hdr = ["Codebook (sensor-driven)", "60 GHz", "140 GHz"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for lab, cells in rows:
        md.append("| " + " | ".join([lab] + cells) + " |")
    md_s = ("# Table 7 — Sensor-driven polar vs DFT-only codebook "
            "(near-field K=8 dB-loss vs polar genie, 5 seeds, mean±95%CI)\n\n"
            + "\n".join(md) + "\n\nLower is better. Restricting the same model to the "
            "angle-only DFT slice (range=∞) cannot focus in range and pays the listed "
            "near-field penalty; the polar codebook recovers it.\n")

    tex = [r"\begin{tabular}{lrr}", r"\toprule",
           r"Codebook (sensor-driven) & 60\,GHz & 140\,GHz \\", r"\midrule"]
    for lab, cells in rows:
        tex.append(f"{lab} & {cells[0]} & {cells[1]} " + r"\\".replace("×", r"$\times$"))
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table7_dft_baseline",
                "\n".join(tex).replace("×", r"$\times$").replace("∞", r"$\infty$").replace("−", "$-$") + "\n",
                md_s)
    aud.write()


if __name__ == "__main__":
    main()
