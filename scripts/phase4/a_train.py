"""D-036 Experiment A: the frozen Phase 3c trainer (train_ds.main, unchanged) on fold k's inputs.

    python scripts/phase4/a_train.py --fold 0 --scenario 32 --seed 11
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); import train_ds as T
k = sys.argv[sys.argv.index("--fold") + 1]; del sys.argv[sys.argv.index("--fold"):sys.argv.index("--fold") + 2]
T.DATA = os.path.join(ROOT, "data", "processed", "phase4", "cv", f"f{k}"); T.CK = os.path.join(ROOT, "checkpoints", "phase4", "cv", f"f{k}")
T.main()
