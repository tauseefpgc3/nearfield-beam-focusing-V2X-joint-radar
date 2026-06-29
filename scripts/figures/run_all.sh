#!/usr/bin/env bash
# Regenerate every Paper-5 figure and table from banked results/*.json (no GPU, no train).
# Each script writes a paper/audit/<name>.audit.txt trail mapping every number to a JSON cell.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

echo "=== Figures ==="
$PY scripts/figures/fig2_gate0.py
$PY scripts/figures/fig3_topk.py
$PY scripts/figures/fig4_cdf.py
$PY scripts/figures/fig5_degradation.py
$PY scripts/figures/fig6_scene_geometry.py
$PY scripts/figures/fig7_polar_focus.py

echo "=== Architecture skeleton (TikZ, Fig 1) ==="
if command -v pdflatex >/dev/null 2>&1; then
  ( cd "$ROOT/paper/figures" && pdflatex -interaction=nonstopmode -halt-on-error \
      fig1_architecture.tex >/dev/null 2>&1 \
      && rm -f fig1_architecture.aux fig1_architecture.log \
      && echo "  figure -> paper/figures/fig1_architecture.pdf" \
      || echo "  [warn] fig1_architecture.tex failed to compile (see paper/figures)" )
else
  echo "  [skip] pdflatex not found; fig1_architecture.tex delivered as editable skeleton"
fi

echo "=== Tables ==="
$PY scripts/tables/table1_indomain.py
$PY scripts/tables/table2_fusion_pairedt.py
$PY scripts/tables/table3_deepsense.py
$PY scripts/tables/table4_ablation.py
$PY scripts/tables/table5_o1anchor.py
$PY scripts/tables/table6_fusion_rules.py
$PY scripts/tables/table7_dft_baseline.py

echo "=== Canonical numbers (single source of truth) ==="
$PY scripts/canonical_numbers.py

echo "=== Missing-from-JSON consistency report ==="
$PY scripts/figures/missing_audit.py

echo "Done. Outputs in paper/figures, paper/tables; audit trails in paper/audit."
