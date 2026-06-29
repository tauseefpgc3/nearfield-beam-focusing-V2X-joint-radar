# MISSING FROM JSON — figures/tables blocked by un-banked numbers

Per the governing rule, these numbers are **not** in any `results/*.json` and were therefore **not** filled from memory. Each is recoverable by re-running a committed **eval-only** step (no retraining, no new data) that writes the JSON; the figure/table script then reads it.

## Fig 2 (Gate-0 near-field occurrence + genie-by-distance)
- **status:** ALREADY BANKED ✓
- **needs JSON:** `results/gate0/gate0.json`
- **missing cells:** Numbers exist only in results/gate0/report.md (markdown), not a structured JSON: frac(d<d_F), frac(y_r>0), frac(genie adv≥3 dB) per dataset/freq, and the mean/median genie advantage per distance bin.
- **to bank (eval-only):** add a --emit-json flag to scripts/gate0_verification.py and re-run (deterministic, reads processed h5, NO training).

## Fig 6 (scene-geometry: CNN-over-genie gap, V2I vs V2V)
- **status:** ALREADY BANKED ✓
- **needs JSON:** `results/deepsense/genie_decomp.json`
- **missing cells:** The radar genie decomposition (RA-argmax, RD-range-only, full-feature 1-NN, CNN top-1) for scenario 31 and 36 was printed to console only.
- **to bank (eval-only):** re-run the genie-decomposition eval on the existing DeepSense checkpoints with a --save-json flag (eval-only, NO training).

## Table 3 extra rows (genie decomposition + zero-shot transfer)
- **status:** ALREADY BANKED ✓
- **needs JSON:** `results/deepsense/transfer.json`
- **missing cells:** Zero-shot 31→36 and synthetic→real top-1, plus the genie-decomposition rows, are console-only. (The radar/lidar/fusion in-domain top-1 rows ARE banked and tabulated.)
- **to bank (eval-only):** re-run the zero-shot eval on existing checkpoints with a --save-json flag (eval-only, NO training).

## Table 4 cross-check (E10 common-near-field-set absolute gain)
- **status:** ALREADY BANKED ✓
- **needs JSON:** `results/ablation/common_nf_absgain.json`
- **missing cells:** ablate_br_common.py printed the common-set absolute realized-gain numbers to console only. (The per-gate K=4/K=8 dB-loss that Table 4 uses IS banked.)
- **to bank (eval-only):** add a json.dump to scripts/ablate_br_common.py and re-run (eval-only, NO training).

---

**0 of 4 still blocking:** none — all banked.

All other figures/tables (Fig 3, Fig 4 headline, Fig 5; Tables 1, 2, 4, 5, and the in-domain rows of Table 3) are fully JSON-sourced and were generated.
