"""A3: recompute the D-029 block split from the user geometry alone and compare with split.h5;
check disjointness, the 0.5 m train-test guard, and which user sets each Phase 3 design step read."""
import json, os, re
import h5py, numpy as np
from scipy.spatial import cKDTree
from common import ROOT, P2, OUT, split, users

ant, psi = users(); xy = ant[:, :2]
d = np.hypot(xy[:, 0], xy[:, 1]); th = np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0]))
sector = np.clip(np.floor((th + 60) / 8).astype(int), 0, 14)
band = np.clip(np.searchsorted([1.0, 5.75, 10.5, 15.25, 20.0], d, side="right") - 1, 0, 3)
role = np.zeros(len(d), int)                        # 0 train, 1 val, 2 test (D-029 rotation)
for b in range(4):
    for k in range(3): role[(band == b) & (sector == (b + 5 * k) % 15)] = 2
    for k in range(2): role[(band == b) & (sector == (b + 3 + 5 * k) % 15)] = 1
test = np.where(role == 2)[0]
guard = cKDTree(xy[test]).query(xy, k=1)[0] < 0.5
mine = dict(train=np.where((role == 0) & ~guard)[0], val=np.where(role == 1)[0], test=test)
S = split()
res = {k: dict(n=len(S[k]), identical_to_recomputed=bool(np.array_equal(S[k], mine[k]))) for k in S}
res["disjoint"] = all(len(np.intersect1d(S[a], S[b])) == 0 for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
res["min_test_to_train_m"] = float(cKDTree(xy[S["train"]]).query(xy[S["test"]])[0].min())
res["min_test_to_val_m"] = float(cKDTree(xy[S["val"]]).query(xy[S["test"]])[0].min())
res["min_val_to_train_m"] = float(cKDTree(xy[S["train"]]).query(xy[S["val"]])[0].min())
blocks = lambda ix: set(zip(band[ix].tolist(), sector[ix].tolist()))
res["test_blocks_shared_with_train_or_val"] = len(blocks(S["test"]) & (blocks(S["train"]) | blocks(S["val"])))
res["n_test_blocks"] = len(blocks(S["test"]))
# static scan of every Phase 3 design-stage script for which split keys it reads
scan = {}
for fn in ("lidar_v2.py", "radar_bev.py", "lidar_estimator.py", "tune_lidar_estimator.py", "train_net.py", "select_design.py", "eval_test.py"):
    s = open(os.path.join(ROOT, "scripts", "phase3", fn), encoding="utf8").read()
    scan[fn] = sorted(set(re.findall(r'hf\["(train|val|test)"\]', s)))
res["split_keys_read_by_script"] = scan
# diagnostic renders done in the Phase 3a session before the design tag (users 0,1,2,3,10,50)
res["render_smoke_users_0_1_2_3_10_50"] = {int(i): next((k for k in S if i in S[k]), "guard-dropped train-block user (in no split)") for i in (0, 1, 2, 3, 10, 50)}
res["n_guard_dropped"] = int(len(xy) - sum(len(v) for v in S.values()))
json.dump(res, open(os.path.join(OUT, "a3_split.json"), "w"), indent=1); print(json.dumps(res, indent=1))
