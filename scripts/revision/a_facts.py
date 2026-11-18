"""TVT revision Phase 1, Part A facts (eval-only, no training, no data regeneration).

Banks one JSON per item under results/revision/:
  A1_splits.json   dataset sizes, what a "trajectory" is, per-seed split sizes, pairwise
                   test-set overlap across the 20 seeds, Fig-4 pooling vs unique users,
                   spatial proximity of test users to training users.
  A3_array.json    codebook range grid / d_F / label facts per frequency.
  A4_sensors.json  radar range-bin occupancy, radar-vs-codebook angle grid, RA-peak facts.
  A5_model.json    tensor shapes entering cross-attention, attention-weight degeneracy,
                   radar-branch drift vs the standalone radar checkpoint.

    python scripts/revision/a_facts.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import h5py
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.polar import _angles                       # noqa: E402
from src.data.radar_processing import generate_radar_steering  # noqa: E402
from src.data.splits import trajectory_grouped_split          # noqa: E402
from src.data.synthetic_radar import RANGE_RES_M              # noqa: E402
from src.models.radar_encoder import RadarPolarModel          # noqa: E402
from src.models.sensing_polar import SensingPolarPredictor    # noqa: E402

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165,
         176, 187, 198, 209, 220]
FREQS = ["60", "140"]
OUT = os.path.join(_ROOT, "results", "revision")
B_R = 8


def h5(f):
    return os.path.join(_ROOT, "data", "processed", f"closerange_{f}GHz_full.h5")


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"wrote results/revision/{name}")


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                median=float(np.median(x)))


def a1():
    out = {"seeds": SEEDS, "per_freq": {}}
    cache = {}
    for f in FREQS:
        with h5py.File(h5(f), "r") as hf:
            traj = hf["trajectory_id"][:]
            dist = hf["distance"][:].astype(float)
            yr = hf["y_r_gated"][:]
            yr_raw = hf["y_r"][:]
            adv = hf["polar_adv_db"][:].astype(float)
            dF = float(hf.attrs["d_F"])
        cache[f] = (traj, dist)
        uniq, cnt = np.unique(traj, return_counts=True)
        splits = {s: trajectory_grouped_split(traj, seed=s) for s in SEEDS}
        nf = yr > 0
        per_seed = {str(s): dict(train=int(len(tr)), val=int(len(va)), test=int(len(te)),
                                 test_nf=int(nf[te].sum()),
                                 test_r_lt_dF=int((dist[te] < dF).sum()))
                    for s, (tr, va, te) in splits.items()}
        # pairwise test overlap
        T = {s: set(splits[s][2].tolist()) for s in SEEDS}
        Tnf = {s: set(np.asarray(splits[s][2])[nf[splits[s][2]]].tolist()) for s in SEEDS}
        frac, jac, frac_nf = [], [], []
        for a, b in itertools.combinations(SEEDS, 2):
            inter = len(T[a] & T[b])
            frac.append(inter / len(T[a]))
            jac.append(inter / len(T[a] | T[b]))
            frac_nf.append(len(Tnf[a] & Tnf[b]) / max(1, len(Tnf[a])))
        times_in_test = np.zeros(len(traj), int)
        for s in SEEDS:
            times_in_test[splits[s][2]] += 1
        pooled_nf = sum(len(Tnf[s]) for s in SEEDS)
        uniq_nf = len(set().union(*Tnf.values()))
        new = SEEDS[5:]
        out["per_freq"][f] = dict(
            n_total=int(len(traj)), d_F_m=dF,
            n_nf_gated=int(nf.sum()), frac_nf_gated=float(nf.mean()),
            n_raw_yr_gt0=int((yr_raw > 0).sum()),
            n_r_lt_dF=int((dist < dF).sum()), frac_r_lt_dF=float((dist < dF).mean()),
            n_adv_ge_3dB=int((adv >= 3.0).sum()),
            n_trajectory_ids=int(len(uniq)),
            samples_per_trajectory=dict(min=int(cnt.min()), median=float(np.median(cnt)),
                                        max=int(cnt.max()), mean=float(cnt.mean())),
            frac_traj_singletons=float((cnt == 1).mean()),
            per_seed_split=per_seed,
            pairwise_test_overlap_frac_of_A=stats(frac),
            pairwise_test_jaccard=stats(jac),
            pairwise_nf_test_overlap_frac=stats(frac_nf),
            n_pairs=len(frac),
            times_in_test_hist={str(k): int(v) for k, v in
                                zip(*np.unique(times_in_test, return_counts=True))},
            union_test_all20=int((times_in_test > 0).sum()),
            fig4_pooled_nf_test_n=int(pooled_nf),
            fig4_unique_nf_users=int(uniq_nf),
            new15_pooled_nf=int(sum(len(Tnf[s]) for s in new)),
            new15_unique_nf=int(len(set().union(*[Tnf[s] for s in new]))),
        )
    # same positions at both frequencies?
    t60, d60 = cache["60"]; t140, d140 = cache["140"]
    out["traj_ids_identical_across_freqs"] = bool(np.array_equal(t60, t140))
    out["distances_identical_across_freqs"] = bool(np.allclose(d60, d140))
    # spatial proximity (same geometry at both freqs) — needs azimuth: recover from the
    # pre-registered generator RNG (no regeneration of channels/sensors; positions only).
    from scripts.generate_closerange import SCENARIO_SEED, synth_paths
    xy = np.zeros((6000, 2)); az = np.zeros(6000); dd = np.zeros(6000)
    for idx in range(6000):
        d, theta, *_ = synth_paths(np.random.default_rng(SCENARIO_SEED + idx))
        xy[idx] = (d * np.cos(theta), d * np.sin(theta)); az[idx] = np.degrees(theta); dd[idx] = d
    out["positions_recomputed_match_h5_distance_maxabs_m"] = float(np.abs(dd - d60).max())
    nn_m, nn_ang, nn_rng, within05, within1 = [], [], [], [], []
    for s in SEEDS:
        tr, _, te = trajectory_grouped_split(t60, seed=s)
        D = np.linalg.norm(xy[te][:, None, :] - xy[tr][None, :, :], axis=2)
        j = D.argmin(1)
        nn = D[np.arange(len(te)), j]
        nn_m.append(np.median(nn)); within05.append((nn < 0.5).mean()); within1.append((nn < 1.0).mean())
        nn_ang.append(np.median(np.abs(az[te] - az[tr][j])))
        nn_rng.append(np.median(np.abs(dd[te] - dd[tr][j])))
    # reference: nearest neighbour within the whole set (excluding self)
    Dall = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2); np.fill_diagonal(Dall, np.inf)
    out["spatial_proximity_test_to_train"] = dict(
        note="per seed: for each test user, Euclidean (x,y) distance to nearest TRAIN user; "
             "values are means over 20 seeds of the per-seed statistic",
        median_nn_dist_m=float(np.mean(nn_m)), frac_within_0p5m=float(np.mean(within05)),
        frac_within_1m=float(np.mean(within1)),
        median_abs_az_diff_deg_to_nn=float(np.mean(nn_ang)),
        median_abs_range_diff_m_to_nn=float(np.mean(nn_rng)),
        reference_median_nn_dist_any_user_m=float(np.median(Dall.min(1))))
    out["trajectory_definition"] = ("trajectory_id = (round(az_deg)//2)*100 + floor(d_m): a "
                                    "2-deg x 1-m spatial cell; users are static, one frame each")
    dump("A1_splits.json", out)
    return out


def a3():
    out = {}
    for f in FREQS:
        z = np.load(os.path.join(_ROOT, "data", "codebooks", f"polar_codebook_{f}GHz.npz"))
        with h5py.File(h5(f), "r") as hf:
            yr = hf["y_r_gated"][:]; yr_raw = hf["y_r"][:]; dist = hf["distance"][:]
            yt = hf["y_theta"][:]; ytg = hf["y_theta_gated"][:]; adv = hf["polar_adv_db"][:]
        far = yr == 0
        out[f] = dict(ranges_m=[None if np.isinf(r) else float(r) for r in z["ranges"]],
                      beta_delta=float(z["beta_delta"]), d_F_m=float(z["d_F"]),
                      angles_deg=[float(a) for a in np.degrees(z["angles"])][:3] + ["..."] +
                      [float(a) for a in np.degrees(z["angles"])][-3:],
                      y_r_gated_hist=np.bincount(yr, minlength=B_R).tolist(),
                      y_r_raw_hist=np.bincount(yr_raw, minlength=B_R).tolist(),
                      nonNF_label="y_r_gated=0 (far-field DFT column), y_theta_gated=DFT-optimal angle",
                      n_gated_far=int(far.sum()),
                      n_gated_far_but_raw_yr_gt0=int((far & (yr_raw > 0)).sum()),
                      n_angle_changed_by_gate=int((yt != ytg).sum()),
                      dist_of_nf_users_m=stats(dist[~far]) if (~far).any() else None,
                      dist_of_far_users_m=stats(dist[far]),
                      adv_db=stats(adv))
    dump("A3_array.json", out)


def a4():
    out = {}
    sin_radar = np.angle(generate_radar_steering(2, 64)[:, 1]) / (2 * np.pi * 0.5)
    sin_cb = np.sin(_angles(64))
    out["angle_grid_max_abs_sin_diff"] = float(np.abs(sin_radar - sin_cb).max())
    out["angle_grid_identical"] = bool(np.allclose(sin_radar, sin_cb, atol=1e-9))
    for f in FREQS:
        with h5py.File(h5(f), "r") as hf:
            dist = hf["distance"][:].astype(float)
            ytg = hf["y_theta_gated"][:]
            ra_peak_r, ra_peak_a, rd_peak_d = [], [], []
            for s in range(0, 6000, 500):
                x = hf["radar_features"][s:s + 500]                # (b,2,256,128)
                ra = x[:, 1, :, :64]
                flat = ra.reshape(len(ra), -1).argmax(1)
                ra_peak_r.append(flat // 64); ra_peak_a.append(flat % 64)
                rd = x[:, 0]
                rd_peak_d.append(rd.reshape(len(rd), -1).argmax(1) % 128)
        ra_peak_r = np.concatenate(ra_peak_r); ra_peak_a = np.concatenate(ra_peak_a)
        rd_peak_d = np.concatenate(rd_peak_d)
        rbin = np.rint(dist / RANGE_RES_M).astype(int)
        u, c = np.unique(rbin, return_counts=True)
        out[f] = dict(range_res_m=RANGE_RES_M, n_range_bins_total=256,
                      gt_range_bins_occupied=u.tolist(), gt_range_bin_counts=c.tolist(),
                      n_occupied_bins=int(len(u)),
                      frac_RA_peak_rangebin_eq_round_d_over_1p2=float((ra_peak_r == rbin).mean()),
                      frac_RA_peak_anglebin_eq_y_theta_gated=float((ra_peak_a == ytg).mean()),
                      frac_RA_peak_anglebin_within1_of_y_theta_gated=float((np.abs(ra_peak_a.astype(int) - ytg) <= 1).mean()),
                      RD_peak_doppler_bin_values=np.unique(rd_peak_d).tolist()[:10],
                      velocity_used="0.0 for every close-range user (generate_closerange.py:91)")
    dump("A4_sensors.json", out)


def a5():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = SensingPolarPredictor().to(dev).eval()
    shapes = {}
    hooks = [m.radar_encoder.register_forward_hook(lambda mod, i, o: shapes.__setitem__("z_radar", list(o.shape))),
             m.lidar_encoder.register_forward_hook(lambda mod, i, o: shapes.__setitem__("z_lidar", list(o.shape))),
             m.fusion.attn.register_forward_hook(
                 lambda mod, i, o: shapes.update(attn_query=list(i[0].shape), attn_key=list(i[1].shape),
                                                 attn_value=list(i[2].shape),
                                                 attn_weights=list(o[1].shape) if o[1] is not None else None))]
    ck = os.path.join(_ROOT, "checkpoints", "closerange_60GHz_fusion_warmstart", "seed11.pt")
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
    with h5py.File(h5("60"), "r") as hf:
        xr = torch.from_numpy(hf["radar_features"][:8]).to(dev)
        xl = torch.from_numpy(hf["lidar_features"][:8]).to(dev)
    with torch.no_grad():
        m(xr, xl)
        z_r = m.radar_encoder(xr); z_l = m.lidar_encoder(xl)
        q = z_r.unsqueeze(1); kv = z_l.unsqueeze(1)
        a1, w1 = m.fusion.attn(q, kv, kv)
        a2, _ = m.fusion.attn(torch.randn_like(q) * 10, kv, kv)     # different query
        # attention output == out_proj(v_proj(z_l)) exactly?
        W = m.fusion.attn.in_proj_weight; bvec = m.fusion.attn.in_proj_bias; d = z_l.shape[1]
        v = z_l @ W[2 * d:].T + bvec[2 * d:]
        lin = m.fusion.attn.out_proj(v)
    for h in hooks:
        h.remove()
    out = dict(input_radar=[8, 2, 256, 128], input_lidar=[8, 2, 150, 100], **shapes,
               tokens_per_modality=1,
               attn_weights_all_equal_1=bool(torch.allclose(w1, torch.ones_like(w1))),
               attn_output_independent_of_query_maxabs=float((a1 - a2).abs().max()),
               attn_output_equals_linear_of_lidar_maxabs=float((a1.squeeze(1) - lin).abs().max()),
               losses="sum over 3 branches of CE(theta)+1.0*CE(range); no loss on ensemble "
                      "(scripts/train_fusion.py:72-75,191)")
    # radar branch of the fusion model vs the standalone radar model it was warm-started from
    drift = {}
    for f in FREQS:
        rel = []
        frozen = {}
        for s in SEEDS:
            ckf = torch.load(os.path.join(_ROOT, "checkpoints", f"closerange_{f}GHz_fusion_warmstart",
                                          f"seed{s}.pt"), map_location="cpu", weights_only=False)
            frozen[str(s)] = bool(ckf["args"].get("freeze_radar"))
            fs = ckf["model"]
            rs = torch.load(os.path.join(_ROOT, "checkpoints", f"closerange_{f}GHz_radar",
                                         f"seed{s}.pt"), map_location="cpu", weights_only=False)["model"]
            num = den = 0.0
            for k, v in rs.items():
                fk = ("radar_encoder." + k[len("encoder."):]) if k.startswith("encoder.") else \
                     ("head_radar." + k[len("head."):])
                if v.dtype.is_floating_point:
                    num += float((fs[fk].float() - v.float()).pow(2).sum()); den += float(v.float().pow(2).sum())
            rel.append(np.sqrt(num / den))
        drift[f] = dict(rel_l2_change_radar_branch_vs_standalone=stats(rel),
                        rel_l2_per_seed=dict(zip(map(str, SEEDS), map(float, rel))),
                        freeze_radar_flag_in_ckpt_args=frozen)
    out["radar_branch_vs_standalone"] = drift
    out["radar_alone_definition"] = ("separately trained RadarPolarModel checkpoint "
                                     "(checkpoints/closerange_{f}GHz_radar/seed{s}.pt), used as "
                                     "the warm-start of the fusion radar branch, which is then "
                                     "trained further (not frozen in sweep_taskA.py)")
    dump("A5_model.json", out)


if __name__ == "__main__":
    a1(); a3(); a4(); a5()
