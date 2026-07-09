#!/usr/bin/env bash
# Combined overnight chain: Stage 1 (D-019 clean ensemble) -> Stage 2 (D-020 mobility).
# Fire-and-forget, resumable (skips existing checkpoints/results). One master log.
# HARD-FAIL gates: Stage-1 leakage!=0.000 (eval exit 2/3) and Stage-2 anchor fail (exit 4)
# stop the chain loudly and do NOT proceed. GPU required (train scripts sys.exit if absent).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
PY="${PYTHON:-python}"
mkdir -p results/sweep_logs

log(){ echo "[$(date '+%F %T')] $*"; }
hardfail(){ log "########################################################"; \
            log "## CHAIN HARD-FAIL: $*"; \
            log "## Chain STOPPED — nothing further trained/evaluated."; \
            log "## Fix the cause and re-run scripts/overnight_chain.sh (resumable)."; \
            log "########################################################"; exit 1; }

# train one radar model to a specific checkpoint; skip if it exists; per-run log; summarise
train_run(){ # $1 data  $2 split_seed  $3 tag  $4 ckpt_name  $5 init_seed(optional)  $6 runlog
  local data="$1" seed="$2" tag="$3" name="$4" init="$5" rl="$6" ck="checkpoints/$7/$4.pt"
  if [ -f "$ck" ]; then log "  [skip] $tag $name (ckpt exists)"; return 0; fi
  log "  [train] $tag $name (split$seed init${init:-$seed})"
  local extra=""; [ -n "$init" ] && extra="--init-seed $init"
  $PY scripts/train_single.py --modality radar --data "$data" --seed "$seed" \
      --tag "$tag" --ckpt-name "$name" --epochs 50 $extra > "$rl" 2>&1 \
      || hardfail "training failed: $tag $name (see $rl)"
  grep -h "GPU-compute confirmed" "$rl" | head -1 | sed 's/^/    /'
  grep -hE "^=== FINAL|theta " "$rl" | tail -1 | sed 's/^/    /'
}

log "================ OVERNIGHT CHAIN START ================"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || hardfail "CUDA not available (no CPU fallback allowed)"
$PY -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# ------------------------- STAGE 1: D-019 -------------------------
log "===== STAGE 1: D-019 clean same-split radar ensemble (18 trainings) ====="
for f in 60 140; do
  data="data/processed/closerange_${f}GHz_full.h5"
  for s in 11 22 33; do
    for k in 201 202 203; do
      train_run "$data" "$s" "d019" "split${s}_init${k}" "$k" \
                "results/sweep_logs/d019_${f}_s${s}_i${k}.log" "closerange_${f}GHz_d019"
    done
  done
done
log "STAGE 1 EVAL: clean-ensemble battery + leakage assertion (must be 0.000)"
$PY scripts/eval_d019_clean.py || hardfail "D-019 eval — leakage!=0.000 or missing checkpoint"
log "STAGE 1 COMPLETE (all leakage=0.000)."

# ------------------------- STAGE 2: MOBILITY -------------------------
log "===== STAGE 2: D-020 mobility future-beam ====="
log "Building mobility datasets"
$PY scripts/build_mobility_pairs.py || hardfail "mobility dataset build"
log "Training Delta=0 anchor (10 trainings)"
for scen in 31 36; do
  data="data/processed/mobility/mobility_sc${scen}_delta0_full.h5"
  for s in 11 22 33 44 55; do
    train_run "$data" "$s" "radar" "seed${s}" "" \
              "results/sweep_logs/mob_sc${scen}_d0_s${s}.log" "mobility_sc${scen}_delta0_radar"
  done
done
log "ANCHOR CHECK: Delta=0 must reproduce banked top-1 within 5-seed 95% CI"
$PY scripts/eval_mobility.py --deltas 0 --anchor-check \
  || hardfail "mobility Delta=0 anchor outside banked CI — pipeline suspect, horizons NOT trained"
log "ANCHOR PASSED. Training Delta in {1,2,3} (30 trainings)"
for scen in 31 36; do
  for d in 1 2 3; do
    data="data/processed/mobility/mobility_sc${scen}_delta${d}_full.h5"
    for s in 11 22 33 44 55; do
      train_run "$data" "$s" "radar" "seed${s}" "" \
                "results/sweep_logs/mob_sc${scen}_d${d}_s${s}.log" "mobility_sc${scen}_delta${d}_radar"
    done
  done
done
log "STAGE 2 EVAL: full (scenario x Delta) + bank future_beam.json"
$PY scripts/eval_mobility.py --deltas 0 1 2 3 || hardfail "mobility full eval"
log "STAGE 2 COMPLETE."

# ------------------------- FINALISE -------------------------
log "Regenerating all figures/tables (run_all)"
bash scripts/figures/run_all.sh 2>&1 | tail -3
log "================ OVERNIGHT CHAIN COMPLETE — BOTH STAGES CLEAN ================"
