#!/usr/bin/env bash
# D-033 C2: 20 seeds per scenario, sequential, resumable. usage: chain_ds.sh "8 31 36"
cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8; mkdir -p results/phase3c/logs
for S in $1; do for SEED in $(seq 11 11 220); do T="ds${S}_s${SEED}"
  [ -f "checkpoints/phase3c/$T.pt" ] && continue
  python scripts/phase3c/train_ds.py --scenario $S --seed $SEED > results/phase3c/logs/$T.log 2>&1 || { echo "FAILED $T"; exit 1; }
  echo "$T $(tail -1 results/phase3c/logs/$T.log)"
done; done
