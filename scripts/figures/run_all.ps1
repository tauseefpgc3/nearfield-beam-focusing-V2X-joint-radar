# Regenerate every Paper-5 figure and table from banked results/*.json (no GPU, no train).
# Windows runner mirroring run_all.sh. Each script writes a paper/audit trail.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "=== Figures ==="
& $py scripts/figures/fig2_gate0.py
& $py scripts/figures/fig3_topk.py
& $py scripts/figures/fig4_cdf.py
& $py scripts/figures/fig5_degradation.py
& $py scripts/figures/fig6_scene_geometry.py
& $py scripts/figures/fig7_polar_focus.py

Write-Host "=== Architecture skeleton (TikZ, Fig 1) ==="
if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
  Push-Location "$root\paper\figures"
  & pdflatex -interaction=nonstopmode -halt-on-error fig1_architecture.tex *> $null
  if (Test-Path fig1_architecture.pdf) {
    Remove-Item fig1_architecture.aux, fig1_architecture.log -ErrorAction SilentlyContinue
    Write-Host "  figure -> paper/figures/fig1_architecture.pdf"
  } else { Write-Host "  [warn] fig1_architecture.tex failed to compile" }
  Pop-Location
} else {
  Write-Host "  [skip] pdflatex not found; fig1_architecture.tex delivered as editable skeleton"
}

Write-Host "=== Tables ==="
& $py scripts/tables/table1_indomain.py
& $py scripts/tables/table2_fusion_pairedt.py
& $py scripts/tables/table3_deepsense.py
& $py scripts/tables/table4_ablation.py
& $py scripts/tables/table5_o1anchor.py
& $py scripts/tables/table6_fusion_rules.py
& $py scripts/tables/table7_dft_baseline.py

Write-Host "=== Canonical numbers (single source of truth) ==="
& $py scripts/canonical_numbers.py

Write-Host "=== Missing-from-JSON consistency report ==="
& $py scripts/figures/missing_audit.py

Write-Host "Done. Outputs in paper/figures, paper/tables; audit trails in paper/audit."
