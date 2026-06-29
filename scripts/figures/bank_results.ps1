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
Write-Host "Banking done. Now run scripts/figures/run_all.ps1"
