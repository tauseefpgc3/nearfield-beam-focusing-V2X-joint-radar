# Re-bank the eval-only JSONs that the figures/tables read (NO training, GPU required).
# Deterministic evals on committed checkpoints/data; no retraining, no dataset regen.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=== Gate-0 structured JSON (Fig 2) ==="
& $py scripts/gate0_verification.py
Write-Host "=== DeepSense genie-decomp + zero-shot (Fig 6, Table 3 extras) ==="
& $py scripts/deepsense_decomp.py
Write-Host "=== E10 common-NF-set absolute gain (Table 4 cross-check) ==="
& $py scripts/ablate_br_common.py
Write-Host "=== Fusion combination-rule sweep (Table 6) ==="
& $py scripts/eval_fusion_rules_sweep.py
Write-Host "=== DFT-only baseline (Table 7) ==="
& $py scripts/eval_dft_baseline.py
Write-Host "=== Hierarchical gain-vs-overhead baseline (Table 8) ==="
& $py scripts/eval_hierarchical_baseline.py
$N20 = 11,22,33,44,55,66,77,88,99,110,121,132,143,154,165,176,187,198,209,220
$have = $true
foreach ($f in "60","140") { foreach ($s in $N20) { foreach ($m in "radar","fusion_warmstart") {
  if (-not (Test-Path "checkpoints/closerange_${f}GHz_${m}/seed${s}.pt")) { $have = $false } } } }
Write-Host "=== Task A n=20 powered sweep (D-018; only if all 20-seed checkpoints exist) ==="
if ($have) {
  & $py scripts/eval_fusion_rules_sweep.py --seeds $N20 --out rule_sweep_n20.json
  & $py scripts/eval_taskA_stats.py --rule-sweep rule_sweep_n20.json --out taskA_stats_n20.json
  Write-Host "=== Modality-vs-ensembling confound check (Table 11) ==="
  & $py scripts/eval_ensemble_confound.py
} else { Write-Host "  [skip] n=20 checkpoints incomplete; run scripts/sweep_taskA.py first" }
Write-Host "Banking done. Now run scripts/figures/run_all.ps1"
