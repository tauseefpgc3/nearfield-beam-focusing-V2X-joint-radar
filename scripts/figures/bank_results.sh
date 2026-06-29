#!/usr/bin/env bash
# Re-bank the eval-only JSONs that the figures/tables read (NO training, GPU required).
# Run this once before run_all.sh if results/gate0/gate0.json, results/deepsense/*.json or
# results/ablation/common_nf_absgain.json are missing. All steps are deterministic evals on
# committed checkpoints/data — they do not retrain and do not regenerate datasets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8

echo "=== Gate-0 structured JSON (Fig 2) ==="
$PY scripts/gate0_verification.py
echo "=== DeepSense genie-decomp + zero-shot (Fig 6, Table 3 extras) ==="
$PY scripts/deepsense_decomp.py
echo "=== E10 common-NF-set absolute gain (Table 4 cross-check) ==="
$PY scripts/ablate_br_common.py
echo "=== Fusion combination-rule sweep (Table 6) ==="
$PY scripts/eval_fusion_rules_sweep.py
echo "=== DFT-only baseline (Table 7) ==="
$PY scripts/eval_dft_baseline.py
echo "Banking done. Now run scripts/figures/run_all.sh"
