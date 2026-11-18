#!/usr/bin/env bash
# Phase 2a: one frequency; for each seed, radar-alone then the three-branch model
# (warm-started from that seed's radar model). Resumable: skips existing checkpoints.
F=$1; cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8
mkdir -p results/phase2/logs
for S in 11 22 33 44 55; do
  for M in radar fusion; do
    CK="checkpoints/phase2/${M}_${F}GHz_seed${S}.pt"
    [ -f "$CK" ] && { echo "[$(date '+%F %T')] skip $M $F s$S"; continue; }
    echo "[$(date '+%F %T')] train $M $F s$S"
    python scripts/phase2/p2_train.py --model $M --freq $F --seed $S > results/phase2/logs/${M}_${F}_s${S}.log 2>&1 || { echo "FAILED $M $F s$S"; exit 1; }
    tail -1 results/phase2/logs/${M}_${F}_s${S}.log
  done
done
echo "[$(date '+%F %T')] CHAIN $F DONE"
