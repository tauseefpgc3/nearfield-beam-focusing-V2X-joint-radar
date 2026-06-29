"""Consistency STOP-report: which paper numbers are NOT yet banked as results/*.json.

Governing rule: every figure/table number must come from a JSON cell; if a needed number
is not in a JSON, STOP and report it (do not fill from memory). This script writes
paper/audit/MISSING_FROM_JSON.md enumerating the gaps, the figure/table they block, and
the exact committed (eval-only, no-retrain) command that would bank them. It does NOT
invent any values.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import AUD_DIR, RESULTS, have

GAPS = [
    ("Fig 2 (Gate-0 near-field occurrence + genie-by-distance)",
     "results/gate0/gate0.json",
     "Numbers exist only in results/gate0/report.md (markdown), not a structured JSON: "
     "frac(d<d_F), frac(y_r>0), frac(genie adv≥3 dB) per dataset/freq, and the "
     "mean/median genie advantage per distance bin.",
     "add a --emit-json flag to scripts/gate0_verification.py and re-run "
     "(deterministic, reads processed h5, NO training)."),
    ("Fig 6 (scene-geometry: CNN-over-genie gap, V2I vs V2V)",
     "results/deepsense/genie_decomp.json",
     "The radar genie decomposition (RA-argmax, RD-range-only, full-feature 1-NN, CNN "
     "top-1) for scenario 31 and 36 was printed to console only.",
     "re-run the genie-decomposition eval on the existing DeepSense checkpoints with a "
     "--save-json flag (eval-only, NO training)."),
    ("Table 3 extra rows (genie decomposition + zero-shot transfer)",
     "results/deepsense/transfer.json",
     "Zero-shot 31→36 and synthetic→real top-1, plus the genie-decomposition rows, are "
     "console-only. (The radar/lidar/fusion in-domain top-1 rows ARE banked and tabulated.)",
     "re-run the zero-shot eval on existing checkpoints with a --save-json flag "
     "(eval-only, NO training)."),
    ("Table 4 cross-check (E10 common-near-field-set absolute gain)",
     "results/ablation/common_nf_absgain.json",
     "ablate_br_common.py printed the common-set absolute realized-gain numbers to console "
     "only. (The per-gate K=4/K=8 dB-loss that Table 4 uses IS banked.)",
     "add a json.dump to scripts/ablate_br_common.py and re-run (eval-only, NO training)."),
]


def main():
    os.makedirs(AUD_DIR, exist_ok=True)
    out = os.path.join(AUD_DIR, "MISSING_FROM_JSON.md")
    lines = ["# MISSING FROM JSON — figures/tables blocked by un-banked numbers", "",
             "Per the governing rule, these numbers are **not** in any `results/*.json` "
             "and were therefore **not** filled from memory. Each is recoverable by "
             "re-running a committed **eval-only** step (no retraining, no new data) that "
             "writes the JSON; the figure/table script then reads it.", ""]
    blocking = []
    for title, target, what, how in GAPS:
        status = "ALREADY BANKED ✓" if have(os.path.join(RESULTS, *target.split("/")[1:])) \
            else "MISSING ✗"
        if "MISSING" in status:
            blocking.append(title)
        lines += [f"## {title}", f"- **status:** {status}",
                  f"- **needs JSON:** `{target}`", f"- **missing cells:** {what}",
                  f"- **to bank (eval-only):** {how}", ""]
    lines += ["---", "",
              f"**{len(blocking)} of {len(GAPS)} still blocking:** "
              + (", ".join(blocking) if blocking else "none — all banked."), "",
              "All other figures/tables (Fig 3, Fig 4 headline, Fig 5; Tables 1, 2, 4, 5, "
              "and the in-domain rows of Table 3) are fully JSON-sourced and were generated."]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  missing-audit -> paper/audit/MISSING_FROM_JSON.md "
          f"({len(blocking)} blocking gaps)")
    for b in blocking:
        print(f"    [BLOCKED] {b}")


if __name__ == "__main__":
    main()
