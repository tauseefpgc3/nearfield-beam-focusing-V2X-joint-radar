#!/usr/bin/env bash
# D-036 A: 3 scenarios x 5 folds x 5 seeds, resumable.
cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8; mkdir -p results/phase4/logs/a
for S in 32 33 34; do for K in 0 1 2 3 4; do for SEED in 11 22 33 44 55; do T="f${K}_ds${S}_s${SEED}"
  [ -f "checkpoints/phase4/cv/f${K}/ds${S}_s${SEED}.pt" ] && continue
  python scripts/phase4/a_train.py --fold $K --scenario $S --seed $SEED > results/phase4/logs/a/$T.log 2>&1 || { echo "FAILED $T"; exit 1; }
  echo "$T $(tail -1 results/phase4/logs/a/$T.log)"
done; done; done
echo A_TRAIN_DONE
