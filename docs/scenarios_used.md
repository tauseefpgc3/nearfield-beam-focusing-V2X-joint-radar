# DeepSense6G scenarios used

**Status: Scenario 31 downloaded + processed + gate-passed (2026-06-11).** Scenario 36
downloaded but still in split zips (not yet extracted/processed). DeepSense6G is gated
behind academic registration + license acceptance at <https://www.deepsense6g.net/>
(human download task; no programmatic API).

| role | scenario ID | description | download date | frames | format |
|---|---|---|---|---|---|
| V2V multi-modal | **Scenario 31** (`scenario31_new/scenario31`) | unit1 RSU (60 GHz pwr + radar + LiDAR) + unit2 UE GPS; range 10.8–84.8 m; 52 sequences | 2026-06-11 | **7012** | dev format (single 60 GHz pwr, `.npy` radar (4,256,250), `.ply` LiDAR, `unit1_beam` 1–64) |
| V2V (4-array, 360°) | **Scenario 36** (`Scenario36/`) | mobile V2V; 4 radars (4,256,128) + 4 mmWave arrays (256-beam `overall-beam`); range 3.5–358 m; 124 seqs | 2026-06-12 | 8267 (stride-3) | radar **axis verified** (rx,chirps=256,**samples=128**=range last axis); **option A** radar1→pwr1 64-beam; **best-beam is 0-indexed** (gate caught a 0-vs-1 off-by-one) |

Processed: `data/processed/deepsense_scenario31_full.h5` (radar/lidar features, y_theta=best
beam, y_r=0 far-field, distance, trajectory_id=seq_index, best_beam; **no channel**).
Label-trust gate (`scripts/deepsense_label_gate.py`): G0.3 64/64 angles, **G0.4 agreement
1.000 → PASS** (D-016).

**Expected per-scenario layout** (DeepSense6G): `unitN/` folders with mmWave per-beam
**power** (NOT channel), FMCW radar, LiDAR point cloud, camera, and a CSV of synchronized
sample rows incl. the **best-beam index** (the trustworthy angle ground truth) and GPS
positions. The processor (`process_deepsense_scenario.py`, to be finalized against the
real file layout) must store: `radar_features (2,256,128)`, `lidar_features (2,150,100)`,
`y_theta`, `y_r` (reconstructed, low-trust), `distance`, `trajectory_id`, `best_beam` —
and **no `channel`** (DeepSense has none; that is the whole point of G0.4).
