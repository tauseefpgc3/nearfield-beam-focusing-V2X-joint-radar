#!/usr/bin/env bash
# D-034 B6: selection grid (head x lr, seed 11, per frequency), choose on validation loss at M=8
# (final epoch), then seeds 11..55 of the chosen setting. Resumable.
cd "$(dirname "$0")/../.."; export PYTHONIOENCODING=utf-8; mkdir -p results/phase3d/logs
run() { T="cls$1_$2_lr$3_s$4"; [ -f "checkpoints/phase3d/$T.pt" ] && return
  python scripts/phase3d/train_cls.py --freq $1 --head $2 --lr $3 --seed $4 > results/phase3d/logs/$T.log 2>&1 || { echo "FAILED $T"; exit 1; }
  echo "$T $(tail -1 results/phase3d/logs/$T.log)"; }
for F in 60 140; do for H in G D; do for LR in 0.001 0.0003; do run $F $H $LR 11; done; done; done
python - <<'PY' > results/phase3d/cls_selection.json
import json, glob, os, re
out = {}
for f in (60, 140):
    c = {}
    for p in glob.glob(f"results/phase3d/logs/cls{f}_*_s11.log"):
        h = json.loads(open(p).read().strip().splitlines()[-1]); m = re.match(rf"cls{f}_(\w)_lr([\d.e-]+)_s11", os.path.basename(p)[:-4])
        c[f"{m.group(1)}|{m.group(2)}"] = dict(val_loss_M8=h["val_loss_M8"], val_top1=h["val_top1"], train_loss=h["train_loss"], prior=h["prior"])
    best = min(c, key=lambda k: c[k]["val_loss_M8"]); out[str(f)] = dict(candidates=c, chosen=best)
print(json.dumps(out, indent=1))
PY
cat results/phase3d/cls_selection.json
for F in 60 140; do
  CH=$(python -c "import json;print(json.load(open('results/phase3d/cls_selection.json'))['$F']['chosen'])"); H=${CH%|*}; LR=${CH#*|}
  for S in 11 22 33 44 55; do run $F $H $LR $S; done
done
echo DONE
