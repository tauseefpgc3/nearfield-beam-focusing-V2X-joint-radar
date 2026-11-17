#!/usr/bin/env bash
# D-032: train H / lr 1e-3 (frozen at revision-design-v2) for a list of seeds. Resumable.
# usage: chain3b.sh SENSOR SUFFIX "SEED SEED ..." [extra train_net args]
S=$1; SUF=$2; SEEDS=$3; shift 3; cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8
for SEED in $SEEDS; do T="${S}_H_lr0.001_s${SEED}${SUF}"
  [ -f "checkpoints/phase3/$T.pt" ] && continue
  python scripts/phase3/train_net.py --sensor $S --head H --lr 0.001 --seed $SEED --suffix "$SUF" "$@" > results/phase3b/logs/$T.log 2>&1 || { echo "FAILED $T"; exit 1; }
  echo "$T $(tail -1 results/phase3b/logs/$T.log)"
done
