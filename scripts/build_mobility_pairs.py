"""D-020 Stage 2: build future-beam (X_t -> y_{t+Delta}) datasets from DeepSense radar.

For each scenario in {31,36} and Delta in {0,1,2,3}, writes
data/processed/mobility/mobility_sc{scen}_delta{Delta}_full.h5 with the SAME keys
train_single expects (radar_features, y_theta_gated, y_r_gated, trajectory_id). Pairs are
formed WITHIN each trajectory only (frames are temporally ordered by row index): source
frame t -> target label at t+Delta, both in the same trajectory. trajectory_id is carried
from the SOURCE frame so the banked trajectory-grouped split protocol is preserved.

Guarantees / prints: no cross-trajectory pairs (asserted); Delta=0 reproduces the original
(labels unchanged); pair counts; frame interval (DeepSense ~10 Hz -> 0.1 s) and measured
mean per-frame range change (physical horizon mapping). Resumable (skips existing h5).

    python scripts/build_mobility_pairs.py
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = [31, 36]
DELTAS = [0, 1, 2, 3]
FRAME_DT_S = 0.1        # DeepSense6G nominal ~10 Hz (raw timestamps not retained; documented)
OUT_DIR = os.path.join(_ROOT, "data", "processed", "mobility")


def build(scen, delta):
    src_h5 = os.path.join(_ROOT, "data", "processed", f"deepsense_scenario{scen}_full.h5")
    out_h5 = os.path.join(OUT_DIR, f"mobility_sc{scen}_delta{delta}_full.h5")
    if os.path.exists(out_h5):
        print(f"  [skip] {os.path.basename(out_h5)} exists"); return
    with h5py.File(src_h5, "r") as hf:
        traj = hf["trajectory_id"][:]
        yth = hf["y_theta_gated"][:]
        dist = hf["distance"][:]
        n0 = traj.shape[0]
        # source/target index pairs within each trajectory (temporal order = ascending index)
        src_idx, tgt_idx = [], []
        for tt in np.unique(traj):
            gi = np.sort(np.where(traj == tt)[0])          # frames of this trajectory, in order
            if len(gi) > delta:
                src_idx.extend(gi[:len(gi) - delta].tolist())
                tgt_idx.extend(gi[delta:].tolist())
        src_idx = np.asarray(src_idx); tgt_idx = np.asarray(tgt_idx)
        # ASSERT: never pair across trajectory boundaries
        assert np.all(traj[src_idx] == traj[tgt_idx]), "cross-trajectory pair detected!"
        if delta == 0:
            assert np.array_equal(src_idx, tgt_idx), "delta=0 must be identity"
        # Globally sort pairs by source index (keep src/tgt aligned) so the streamed h5py
        # read below uses strictly increasing indices (h5py fancy-index requirement).
        order = np.argsort(src_idx, kind="stable")
        src_idx = src_idx[order]; tgt_idx = tgt_idx[order]
        os.makedirs(OUT_DIR, exist_ok=True)
        m = len(src_idx)
        disp = float(np.mean(np.abs(dist[tgt_idx] - dist[src_idx]))) if delta > 0 else 0.0
        with h5py.File(out_h5, "w") as of:
            rf = hf["radar_features"]
            dset = of.create_dataset("radar_features", (m,) + rf.shape[1:], dtype=rf.dtype)
            for s in range(0, m, 512):                      # streamed copy (memory-safe)
                e = min(s + 512, m)
                dset[s:e] = rf[src_idx[s:e]]                # increasing indices -> h5py ok
            of.create_dataset("y_theta_gated", data=yth[tgt_idx].astype(np.int64))
            of.create_dataset("y_r_gated", data=np.zeros(m, np.int64))   # far-field
            of.create_dataset("trajectory_id", data=traj[src_idx].astype(np.int32))
            of.attrs.update(scenario=f"scenario{scen}", delta=delta,
                            frame_dt_s=FRAME_DT_S, horizon_s=delta * FRAME_DT_S,
                            mean_disp_m=disp, source=f"deepsense_scenario{scen}",
                            note="future-beam X_t -> y_{t+delta}, within-trajectory (D-020)")
    print(f"  sc{scen} delta={delta}: {m}/{n0} pairs | horizon {delta*FRAME_DT_S:.1f}s | "
          f"mean disp {disp:.2f} m -> {os.path.basename(out_h5)}")


def main():
    print(f"Building mobility datasets (frame dt = {FRAME_DT_S}s nominal):")
    for scen in SCEN:
        for delta in DELTAS:
            build(scen, delta)
    print("Done.")


if __name__ == "__main__":
    main()
