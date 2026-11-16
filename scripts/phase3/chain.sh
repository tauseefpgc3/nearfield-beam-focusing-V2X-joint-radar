#!/usr/bin/env bash
# D-031: train a list of configs sequentially for one sensor. Resumable.
# usage: chain.sh SENSOR "HEAD:LR:SEED HEAD:LR:SEED ..."
S=$1; cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8
for c in $2; do IFS=: read H LR SEED <<< "$c"; T="${S}_${H}_lr${LR}_s${SEED}"
  [ -f "checkpoints/phase3/$T.pt" ] && continue
  python scripts/phase3/train_net.py --sensor $S --head $H --lr $LR --seed $SEED > results/phase3/logs/$T.log 2>&1 || { echo "FAILED $T"; exit 1; }
  echo "$T $(tail -1 results/phase3/logs/$T.log)"
done
