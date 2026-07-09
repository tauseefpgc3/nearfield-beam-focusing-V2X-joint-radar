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
echo "=== Hierarchical gain-vs-overhead baseline (Table 8) ==="
$PY scripts/eval_hierarchical_baseline.py
N20="11 22 33 44 55 66 77 88 99 110 121 132 143 154 165 176 187 198 209 220"
echo "=== Task A n=20 powered sweep (D-018; only if all 20-seed checkpoints exist) ==="
if $PY -c "import os,sys;S=[int(x) for x in '$N20'.split()];sys.exit(0 if all(os.path.exists(f'checkpoints/closerange_{f}GHz_{m}/seed{s}.pt') for f in ('60','140') for s in S for m in ('radar','fusion_warmstart')) else 1)"; then
  $PY scripts/eval_fusion_rules_sweep.py --seeds $N20 --out rule_sweep_n20.json
  $PY scripts/eval_taskA_stats.py --rule-sweep rule_sweep_n20.json --out taskA_stats_n20.json
  echo "=== Modality-vs-ensembling confound check (Table 11) ==="
  $PY scripts/eval_ensemble_confound.py
  echo "=== n=20 in-domain metrics (Fig 4 CDF, Fig 3, Tables 1/2) ==="
  $PY scripts/eval_indomain_n20.py
  echo "=== degradation n=20 (Fig 5) ==="
  $PY scripts/degradation_eval.py --seeds $N20 --out degradation_closerange60_n20.json
  echo "=== gain-vs-overhead n=20 (Table 8) ==="
  $PY scripts/eval_gain_overhead_n20.py
else
  echo "  [skip] n=20 checkpoints incomplete; run scripts/sweep_taskA.py first"
fi
echo "Banking done. Now run scripts/figures/run_all.sh"
