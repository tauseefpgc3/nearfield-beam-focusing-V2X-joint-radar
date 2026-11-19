#!/usr/bin/env bash
# D-036 B: prepare inputs, then R0/R1/R2 x 5 seeds, resumable.
cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8; mkdir -p results/phase4/logs/b
[ -f data/processed/phase4/multicar/val_R2_s3.h5 ] || python scripts/phase4/b_prepare.py > results/phase4/logs/b_prepare.log 2>&1 || { echo "FAILED prepare"; exit 1; }
for R in R0 R1 R2; do for SEED in 11 22 33 44 55; do T="lidar_H_lr0.001_s${SEED}_${R}"
  [ -f "checkpoints/phase4/multicar/$T.pt" ] && continue
  python scripts/phase4/b_train.py --rule $R --seed $SEED > results/phase4/logs/b/$T.log 2>&1 || { echo "FAILED $T"; exit 1; }
  echo "$T $(tail -1 results/phase4/logs/b/$T.log)"
done; done
echo B_TRAIN_DONE
