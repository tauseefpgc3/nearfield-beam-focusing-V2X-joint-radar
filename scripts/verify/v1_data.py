"""Independent verification, step 1: data-level findings V2, V3, V4, V6, V12 (genie-DFT).

Re-derived from the generator source (scripts/generate_closerange.py,
src/data/synthetic_radar.py, src/data/radar_processing.py, src/codebook/*) and the banked
HDF5. Does not import scripts/revision. Output: results/verify/v1_data.json
"""
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.splits import trajectory_grouped_split  # the split the training used

OUT = os.path.join(ROOT, "results", "verify")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220]
C = 299_792_458.0
res = {}


# ---- my own re-implementation of the geometry draw (generate_closerange.py:43-56) ----
def draw(idx):
    rng = np.random.default_rng(20_240_603 + idx)
    d = float(rng.uniform(1.0, 20.0))
    th = np.deg2rad(rng.uniform(-60.0, 60.0))
    az, rr, g = [th], [d], [1.0 + 0j]
    for _ in range(int(rng.integers(2, 5))):
        az.append(np.deg2rad(rng.uniform(-90, 90)))
        rr.append(float(rng.uniform(1, 25)))
        amp = 10 ** (-rng.uniform(10, 20) / 20)
        g.append(amp * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    return d, th, np.array(az), np.array(rr), np.array(g)


def my_channel(az, rr, g, f, z_user=0.0, z_array=0.0):
    """Independent spherical-wave sum over a 128x128 UPA in the y-z plane."""
    lam = C / f
    dd = 0.5 * lam
    co = (np.arange(128) - 63.5) * dd
    Y, Z = np.meshgrid(co, co, indexing="ij")
    Y = Y.ravel(); Z = Z.ravel() + z_array
    h = np.zeros(128 * 128, complex)
    for a, r, gg in zip(az, rr, g):
        pu = np.array([r * np.cos(a), r * np.sin(a), z_user])
        dist = np.sqrt(pu[0] ** 2 + (Y - pu[1]) ** 2 + (Z - pu[2]) ** 2)
        ref = np.linalg.norm(pu - np.array([0, 0, z_array]))
        h += gg * np.exp(-1j * 2 * np.pi / lam * (dist - ref))
    return h


geo = [draw(i) for i in range(6000)]
D = np.array([x[0] for x in geo]); TH = np.array([x[1] for x in geo])
XY = np.stack([D * np.cos(TH), D * np.sin(TH)], 1)

for f in (60, 140):
    r = {}
    z = np.load(os.path.join(OUT, f"gains_{f}GHz.npz"))
    G = z["G32"].astype(np.float64)
    yt, yr, ytg, yrg = z["y_theta"], z["y_r"], z["y_theta_gated"], z["y_r_gated"]
    dist, traj, adv = z["distance"], z["trajectory_id"], z["polar_adv_db"]
    r["distance_regen_maxabs_err_m"] = float(np.abs(D - dist).max())

    # ---------------- V3: stored channel == in-house spherical model? ------------
    with h5py.File(os.path.join(ROOT, "data", "processed", f"closerange_{f}GHz_full.h5"), "r") as hf:
        chk = {}
        for i in (0, 1, 2, 777, 5999):
            hs = hf["channel"][i]
            hm = my_channel(*geo[i][2:], f * 1e9)
            chk[i] = dict(n_paths=len(geo[i][2]), rel_err=float(np.linalg.norm(hs - hm) / np.linalg.norm(hs)))
        r["V3_channel_rebuild_rel_err"] = chk
        RA = hf["radar_features"][:, 1, :, :64]          # (6000,256,64) read only RA

    # ---------------- V2: radar peak vs ground truth ------------------------------
    flat = RA.reshape(6000, -1).argmax(1)
    rb, ab = np.divmod(flat, 64)
    true_rbin = np.rint(D / 1.2).astype(int)
    sin_grid = (2 * np.arange(64) - 63) / 64
    near_ang = np.abs(np.sin(TH)[:, None] - sin_grid[None]).argmin(1)
    r["V2"] = dict(
        radar_peak_rangebin_eq_round_d_over_1p2=float((rb == true_rbin).mean()),
        radar_peak_angle_eq_y_theta_gated=float((ab == ytg).mean()),
        radar_peak_angle_eq_y_theta_raw=float((ab == yt).mean()),
        radar_peak_angle_eq_nearest_grid_to_LoS_az=float((ab == near_ang).mean()),
        radar_peak_angle_within1_of_y_theta_gated=float((np.abs(ab - ytg) <= 1).mean()),
        y_theta_gated_eq_nearest_grid_to_LoS=float((ytg == near_ang).mean()),
        radar_peak_angle_eq_y_theta_gated_on_yrg_gt0=float((ab[yrg > 0] == ytg[yrg > 0]).mean()),
    )
    # range: map radar distance to a polar ring the way a peak picker would
    lam = C / (f * 1e9)
    scale = 1.2 * 127 ** 2 * lam / 8
    inv_r = np.r_[0.0, np.arange(1, 8) / scale]
    dhat = np.maximum(rb, 1) * 1.2
    jhat = np.abs(inv_r[None] - 1 / dhat[:, None]).argmin(1)
    jtrue = np.abs(inv_r[None] - 1 / D[:, None]).argmin(1)
    r["V2"].update(
        ring_from_peak_eq_y_r_raw=float((jhat == yr).mean()),
        ring_from_true_d_eq_y_r_raw=float((jtrue == yr).mean()),
        ring_from_peak_eq_y_r_gated_on_yrg_gt0=float((jhat[yrg > 0] == yrg[yrg > 0]).mean()),
        joint_peak_eq_gated_label_on_yrg_gt0=float(((jhat == yrg) & (ab == ytg))[yrg > 0].mean()),
        frac_y_r_gated_gt0=float((yrg > 0).mean()),
    )

    # ---------------- genie-DFT loss over all users (V12 part) ----------------------
    genie = G.max(1); dftg = G[:, 0::8].max(1)
    gdft = 10 * np.log10(genie / dftg)
    r["polar_adv_db_matches_my_genie_dft"] = float(np.abs(gdft - adv).max())
    r["genie_label_eq_raw_label"] = float((G.argmax(1) == yt * 8 + yr).mean())

    # ---------------- V6: splits ----------------------------------------------------
    tests, trains = {}, {}
    per_seed = []
    for s in SEEDS:
        tr, va, te = trajectory_grouped_split(traj, seed=s)
        tests[s], trains[s] = te, tr
        # nearest training user in the horizontal plane
        dmin = np.sqrt(((XY[te][:, None, :] - XY[tr][None]) ** 2).sum(-1)).min(1)
        nf = yrg[te] > 0
        per_seed.append(dict(seed=s, n_test=len(te), n_test_nf=int(nf.sum()),
                             genie_dft_all=float(gdft[te].mean()),
                             genie_dft_nf=float(gdft[te][nf].mean()),
                             frac_nn_train_within_0p5m=float((dmin < 0.5).mean()),
                             frac_nn_train_within_0p25m=float((dmin < 0.25).mean()),
                             median_nn_train_m=float(np.median(dmin))))
    ov = [len(np.intersect1d(tests[a], tests[b])) / len(tests[a])
          for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]
    nf_pool = np.concatenate([tests[s][yrg[tests[s]] > 0] for s in SEEDS])
    all_pool = np.concatenate([tests[s] for s in SEEDS])
    r["V6"] = dict(
        pairwise_test_overlap_mean=float(np.mean(ov)), min=float(np.min(ov)), max=float(np.max(ov)),
        frac_nn_within_0p5m_mean=float(np.mean([p["frac_nn_train_within_0p5m"] for p in per_seed])),
        nf_pooled_n=int(len(nf_pool)), nf_unique=int(len(np.unique(nf_pool))),
        all_pooled_n=int(len(all_pool)), all_unique=int(len(np.unique(all_pool))),
        n_trajectories=int(len(np.unique(traj))),
        users_per_traj_mean=float(6000 / len(np.unique(traj))),
    )
    r["genie_dft_all_20seed_mean"] = float(np.mean([p["genie_dft_all"] for p in per_seed]))
    r["genie_dft_nf_20seed_mean"] = float(np.mean([p["genie_dft_nf"] for p in per_seed]))
    r["genie_dft_all_6000"] = float(gdft.mean())
    r["per_seed"] = per_seed
    res[str(f)] = r
    print(f, json.dumps({k: v for k, v in r.items() if k != "per_seed"}, indent=1))

# ---------------- V4: heights (what the pre-registered 2.0 m / 1.5 m would change) ----
# geometry check: steering places array centre and user at z = 0 (steering.py), so the
# vertical offset is zero. Quantify the effect of the D-014 offset on labels for 300 users.
cb60 = np.load(os.path.join(ROOT, "data", "codebooks", "polar_codebook_60GHz.npz"))["codebook"]
chg, loss = [], []
for i in range(0, 6000, 20):
    h0 = my_channel(*geo[i][2:], 60e9)
    # vehicle at 1.5 m, RSU at 2.0 m; scatterers kept at z=0 (worst case for LoS only)
    az, rr, g = geo[i][2:]
    lam = C / 60e9
    co = (np.arange(128) - 63.5) * 0.5 * lam
    Y, Z = np.meshgrid(co, co, indexing="ij"); Y = Y.ravel(); Z = Z.ravel() + 2.0
    h1 = np.zeros_like(h0)
    for k, (a, rgn, gg) in enumerate(zip(az, rr, g)):
        zu = 1.5 if k == 0 else 0.0
        pu = np.array([rgn * np.cos(a), rgn * np.sin(a), zu])
        dist = np.sqrt(pu[0] ** 2 + (Y - pu[1]) ** 2 + (Z - pu[2]) ** 2)
        h1 += gg * np.exp(-1j * 2 * np.pi / lam * dist)
    g0 = np.abs(cb60.conj() @ h0) ** 2; g1 = np.abs(cb60.conj() @ h1) ** 2
    chg.append(g0.argmax() != g1.argmax())
    loss.append(10 * np.log10(g1.max() / g1[g0.argmax()]))
res["V4_height_effect_60GHz_300users"] = dict(
    frac_label_changes=float(np.mean(chg)), mean_loss_of_flat_label_db=float(np.mean(loss)),
    median_loss_db=float(np.median(loss)))
print(res["V4_height_effect_60GHz_300users"])
json.dump(res, open(os.path.join(OUT, "v1_data.json"), "w"), indent=1)
