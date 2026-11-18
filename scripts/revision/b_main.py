"""TVT revision Phase 1 — B1, B2, B5, B6, B7 (pre-registered D-021, D-022, D-025, D-026, D-027).

Eval-only on banked checkpoints/data. One inference pass per (freq, seed) feeds all five
diagnostics. Writes results/revision/B{1,2,5,6,7}_*.json.

    python scripts/revision/b_main.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (B_R, B_THETA, FREQS, KS, NCW, ROOT, SEEDS, SETS, agg, dump,  # noqa: E402
                     infer_fusion, infer_radar, joint_rank, load_freq, load_model,
                     losses_from_rank, masks)
from scripts.eval_fusion_rules import combine                                  # noqa: E402
from src.data.splits import trajectory_grouped_split                            # noqa: E402

I_GRID = np.repeat(np.arange(B_THETA), B_R)          # flat k -> angle index
J_GRID = np.tile(np.arange(B_R), B_THETA)            # flat k -> range index
FLAT = np.arange(NCW)
DFT_COLS = np.arange(0, NCW, B_R)


def nested():
    return defaultdict(nested)


def softavg(pth, prg):
    cth, crg = pth.max(-1).values, prg.max(-1).values
    return combine("soft_avg", pth, prg, cth, crg)


def ece(p, y, nb=15):
    conf, pred = p.max(1).values.numpy(), p.argmax(1).numpy()
    corr = (pred == y).astype(float)
    edges = np.linspace(0, 1, nb + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e), float(corr.mean()), float(conf.mean())


def peak_rank(ra, ranges):
    """B5a. ra (n,256,64). Returns (rank (n,8), a, j_hat)."""
    n = ra.shape[0]
    flat = ra.reshape(n, -1).argmax(1)
    rb, a = flat // 64, flat % 64
    d_hat = np.maximum(rb, 1) * 1.2
    inv = np.where(np.isinf(ranges), 0.0, 1.0 / ranges)              # 1/r_j, 1/r_0 = 0
    j_hat = np.abs(inv[None, :] - 1.0 / d_hat[:, None]).argmin(1)
    di = np.abs(I_GRID[None, :] - a[:, None]); dj = np.abs(J_GRID[None, :] - j_hat[:, None])
    key = (di + dj).astype(np.int64) * 10**9 + dj * 10**6 + di * 10**3 + FLAT[None, :]
    rank = np.argsort(key, 1)[:, :8]
    return rank, a, j_hat


@torch.no_grad()
def knn_rank(Xtr, ytr_flat, Xte, device, k_unique=8):
    """B5b. Nearest neighbours' joint codewords, de-duplicated, first k_unique."""
    tr = torch.from_numpy(Xtr).to(device)
    tr_sq = (tr * tr).sum(1)
    ranks = np.empty((len(Xte), k_unique), np.int64)
    for s in range(0, len(Xte), 128):
        q = torch.from_numpy(Xte[s:s + 128]).to(device)
        d2 = (q * q).sum(1, keepdim=True) + tr_sq[None, :] - 2.0 * (q @ tr.T)
        order = d2.argsort(1)[:, :512].cpu().numpy()
        for r, row in enumerate(order):
            labs = ytr_flat[row]
            _, first = np.unique(labs, return_index=True)
            u = labs[np.sort(first)]
            if len(u) < k_unique:                                    # extend (rare)
                full = d2[r].argsort().cpu().numpy(); labs = ytr_flat[full]
                _, first = np.unique(labs, return_index=True); u = labs[np.sort(first)]
            ranks[s + r] = u[:k_unique]
    del tr
    torch.cuda.empty_cache()
    return ranks


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    dev = torch.device("cuda")
    t0 = time.time()
    # L[f][pred][set][K] -> per-seed means; P[...] -> pooled per-sample arrays
    L, P = nested(), nested()
    sizes = nested()
    uniq = {f: {st: set() for st in SETS} for f in FREQS}
    B2 = nested(); B6 = nested(); B7 = nested()

    def add(f, pred, M, lossd):
        for st in SETS:
            for k, v in lossd.items():
                L[f][pred][st].setdefault(k, []).append(float(v[M[st]].mean()) if M[st].any() else float("nan"))
                P[f][pred][st].setdefault(k, []).extend(v[M[st]].tolist())

    for f in FREQS:
        D = load_freq(f, dev)
        print(f"[{f} GHz] loaded in {time.time()-t0:.0f}s", flush=True)
        G = D["G"]
        for s in SEEDS:
            tr, _, te = trajectory_grouped_split(D["traj"], seed=s)
            te = np.sort(te); tr = np.sort(tr)
            M = masks(D, te); Gt = G[te]; yt, yr = D["yt"][te], D["yr"][te]
            for st in SETS:
                sizes[f][st].setdefault("per_seed", []).append(int(M[st].sum()))
                uniq[f][st].update(te[M[st]].tolist())
            xr, xl = D["xr"][te], D["xl"][te]
            rm = load_model("radar", f, s, dev); fm = load_model("fusion", f, s, dev)
            rpt, rpr = infer_radar(rm, xr, dev)
            pth, prg = infer_fusion(fm, xr, xl, dev)
            sat, sar = softavg(pth, prg)
            genie = Gt.max(1)
            # ---------------- B1 ----------------
            rank_sa = joint_rank(sat, sar); rank_r = joint_rank(rpt, rpr)
            rank_dft = DFT_COLS[rpt.topk(8, 1).indices.numpy()]
            ls_sa = losses_from_rank(Gt, rank_sa); ls_r = losses_from_rank(Gt, rank_r)
            add(f, "softavg", M, ls_sa); add(f, "radar", M, ls_r)
            add(f, "dft_restricted", M, losses_from_rank(Gt, rank_dft))
            gdft = 10 * np.log10(genie / Gt[:, DFT_COLS].max(1))
            add(f, "genie_dft", M, {0: gdft})
            # ---------------- B2 ----------------
            prior = np.bincount(D["yr"][tr], minlength=B_R) / len(tr)
            B2[f]["train_range_prior"].setdefault("per_seed", []).append(prior.tolist())
            for name, (pt, pr, rank) in {"softavg": (sat, sar, rank_sa),
                                         "radar": (rpt, rpr, rank_r)}.items():
                for k in (4, 8):
                    na = np.array([len(set(I_GRID[r[:k]])) for r in rank])
                    nr = np.array([len(set(J_GRID[r[:k]])) for r in rank])
                    for st in SETS:
                        B2[f][name][st].setdefault(f"distinct_angles_K{k}", []).append(float(na[M[st]].mean()))
                        B2[f][name][st].setdefault(f"distinct_ranges_K{k}", []).append(float(nr[M[st]].mean()))
                single = np.array([len(set(I_GRID[r])) == 1 and len(set(J_GRID[r])) == 8 for r in rank])
                for st in SETS:
                    B2[f][name][st].setdefault("frac_K8_single_angle_all8_ranges", []).append(float(single[M[st]].mean()))
                ih = pt.argmax(1).numpy()
                rank_a = ih[:, None] * B_R + np.arange(B_R)[None, :]
                add(f, f"{name}__angle_top1_sweep8", M, {8: losses_from_rank(Gt, rank_a, ks=(8,))[8]})
                pri = torch.from_numpy(prior).float().unsqueeze(0).expand(len(te), -1)
                add(f, f"{name}__range_prior", M, losses_from_rank(Gt, joint_rank(pt, pri)))
            # ---------------- B5 ----------------
            prank, a_pk, jh = peak_rank(xr[:, 1, :, :64], D["ranges"])
            add(f, "peak_pick", M, losses_from_rank(Gt, prank))
            add(f, "peak_angle_sweep8", M, {8: losses_from_rank(Gt, a_pk[:, None] * B_R + np.arange(B_R)[None, :], ks=(8,))[8]})
            for st in SETS:
                L[f]["peak_pick_diag"][st].setdefault("angle_top1_vs_label", []).append(float((a_pk == yt)[M[st]].mean()))
                L[f]["peak_pick_diag"][st].setdefault("range_idx_eq_label", []).append(float((jh == yr)[M[st]].mean()))
            ytr_flat = D["yt"][tr] * B_R + D["yr"][tr]
            Xr_tr = D["xr"][tr].reshape(len(tr), -1).astype(np.float32)
            Xr_te = xr.reshape(len(te), -1).astype(np.float32)
            add(f, "knn1_radar", M, losses_from_rank(Gt, knn_rank(Xr_tr, ytr_flat, Xr_te, dev)))
            Xl_tr = D["xl"][tr].reshape(len(tr), -1).astype(np.float32)
            Xl_te = xl.reshape(len(te), -1).astype(np.float32)
            nr_, nl_ = np.linalg.norm(Xr_tr, axis=1).mean(), np.linalg.norm(Xl_tr, axis=1).mean()
            Xc_tr = np.concatenate([Xr_tr / nr_, Xl_tr / nl_], 1); Xc_te = np.concatenate([Xr_te / nr_, Xl_te / nl_], 1)
            del Xr_tr, Xl_tr
            add(f, "knn1_radar_lidar", M, losses_from_rank(Gt, knn_rank(Xc_tr, ytr_flat, Xc_te, dev)))
            del Xc_tr, Xc_te
            # ---------------- B6 ----------------
            labs = yt * B_R + yr
            cnt = np.bincount(labs, minlength=NCW)
            B6[f]["most_common_label_freq"].setdefault("per_seed", []).append(float(cnt.max() / len(labs)))
            B6[f]["most_common_label"].setdefault("per_seed", []).append(int(cnt.argmax()))
            for mode, kw in {"feature_dropout": dict(rd=True, ld=True),
                             "zero_inputs": dict(zero_inputs=True)}.items():
                bt, br = infer_fusion(fm, xr, xl, dev, **kw)
                st_, sr_ = softavg(bt, br)
                pj = st_.argmax(1).numpy() * B_R + sr_.argmax(1).numpy()
                u, c = np.unique(pj, return_counts=True)
                rec = B6[f][mode]
                rec.setdefault("joint_acc", []).append(float((pj == labs).mean()))
                rec.setdefault("angle_acc", []).append(float((st_.argmax(1).numpy() == yt).mean()))
                rec.setdefault("range_acc", []).append(float((sr_.argmax(1).numpy() == yr).mean()))
                rec.setdefault("n_distinct_pred", []).append(int(len(u)))
                rec.setdefault("top_pred_class", []).append(int(u[c.argmax()]))
                rec.setdefault("top_pred_share", []).append(float(c.max() / len(pj)))
                rec.setdefault("top_pred_class_test_freq", []).append(float(cnt[u[c.argmax()]] / len(labs)))
                rec.setdefault("top_pred_is_most_common_label", []).append(bool(u[c.argmax()] == cnt.argmax()))
                ent_t = float(-(st_ * st_.clamp_min(1e-12).log()).sum(1).mean())
                ent_r = float(-(sr_ * sr_.clamp_min(1e-12).log()).sum(1).mean())
                rec.setdefault("entropy_theta_nats", []).append(ent_t)
                rec.setdefault("entropy_range_nats", []).append(ent_r)
            # ---------------- B7 ----------------
            conds = {"clean": (pth, prg),
                     "radar_dropped": infer_fusion(fm, xr, xl, dev, rd=True),
                     "lidar_dropped": infer_fusion(fm, xr, xl, dev, ld=True)}
            for cond, (qt, qr) in conds.items():
                et, er = softavg(qt, qr)
                preds = {"branch_radar": (qt[0], qr[0]), "branch_lidar": (qt[1], qr[1]),
                         "branch_fused": (qt[2], qr[2]), "ensemble_softavg": (et, er)}
                if cond == "clean":
                    preds["standalone_radar"] = (rpt, rpr)
                for pn, (a_, b_) in preds.items():
                    for head, p_, y_ in (("angle", a_, yt), ("range", b_, yr)):
                        e_, acc_, cf_ = ece(p_, y_)
                        r_ = B7[f][cond][pn][head]
                        r_.setdefault("ece", []).append(e_); r_.setdefault("acc", []).append(acc_)
                        r_.setdefault("mean_conf", []).append(cf_)
            del rm, fm
            torch.cuda.empty_cache()
            print(f"  {f}GHz seed{s}: NF K8 radar={L[f]['radar']['nf_ge3dB'][8][-1]:.3f} "
                  f"softavg={L[f]['softavg']['nf_ge3dB'][8][-1]:.3f} "
                  f"peak={L[f]['peak_pick']['nf_ge3dB'][8][-1]:.3f} "
                  f"knn={L[f]['knn1_radar']['nf_ge3dB'][8][-1]:.3f} | {time.time()-t0:.0f}s", flush=True)
        del D

    # ---------------- tripwire (D-021) ----------------
    rs = json.load(open(os.path.join(ROOT, "results", "fusion_rules", "rule_sweep_n20.json")))
    trip = {}
    for f in FREQS:
        for mine, banked in (("radar", "radar_alone"), ("softavg", "soft_avg")):
            mv = np.array(L[f][mine]["nf_ge3dB"][8])
            bk = dict(zip(rs["seeds"], rs["results"][f][banked]["per_seed"]))
            d = np.abs(mv - np.array([bk[s] for s in SEEDS[:len(mv)]]))
            trip[f"{f}_{mine}"] = float(d.max())
    trip_ok = all(v <= 0.01 for v in trip.values())
    print(f"TRIPWIRE (max |diff| vs rule_sweep_n20, dB): {trip} -> {'PASS' if trip_ok else 'FAIL'}")

    def pack(preds):
        out = {}
        for f in FREQS:
            out[f] = {}
            for pr in preds:
                out[f][pr] = {st: {str(k): agg(L[f][pr][st][k], P[f][pr][st].get(k))
                                   for k in L[f][pr][st]} for st in SETS}
        return out

    size_block = {f: {st: dict(per_seed_mean=float(np.mean(sizes[f][st]["per_seed"])),
                               per_seed_min=int(np.min(sizes[f][st]["per_seed"])),
                               per_seed_max=int(np.max(sizes[f][st]["per_seed"])),
                               unique_users_over_20_seeds=len(uniq[f][st]))
                      for st in SETS} for f in FREQS}
    common = dict(seeds=SEEDS, sets={"all": "(i) all close-range test users",
                                     "r_lt_dF": "(ii) distance < d_F",
                                     "nf_ge3dB": "(iii) y_r_gated>0 (>=3 dB genie advantage; paper subset)",
                                     "complement": "(iv) complement of (iii)"},
                  set_sizes=size_block, tripwire_vs_rule_sweep_n20=dict(max_abs_diff_db=trip, passed=trip_ok),
                  prereg="docs/decisions.md D-021..D-027, release commit 99b5cda")
    dump("B1_subsets.json", dict(common, metric="dB loss vs polar genie; K=0 key = genie-DFT loss",
                                 results=pack(["softavg", "radar", "dft_restricted", "genie_dft"])))

    def summ(d):
        return {k: (agg(v) if isinstance(v, list) and v and not isinstance(v[0], list) else v)
                for k, v in d.items()}
    b2 = {f: {"train_range_prior_mean": np.mean(B2[f]["train_range_prior"]["per_seed"], 0).tolist(),
              **{pn: {st: summ(B2[f][pn][st]) for st in SETS} for pn in ("softavg", "radar")}}
          for f in FREQS}
    dump("B2_range_head.json", dict(common, shortlist_composition=b2,
                                    losses=pack(["softavg", "radar", "softavg__angle_top1_sweep8",
                                                 "radar__angle_top1_sweep8", "softavg__range_prior",
                                                 "radar__range_prior"])))
    b5 = pack(["peak_pick", "peak_angle_sweep8", "knn1_radar", "knn1_radar_lidar", "softavg", "radar"])
    for f in FREQS:
        b5[f]["peak_pick_diag"] = {st: {k: agg(v) for k, v in L[f]["peak_pick_diag"][st].items()} for st in SETS}
    dump("B5_learning_free.json", dict(common, results=b5))
    b6 = {}
    for f in FREQS:
        b6[f] = {"most_common_label_freq": agg(B6[f]["most_common_label_freq"]["per_seed"]),
                 "most_common_label_per_seed": B6[f]["most_common_label"]["per_seed"],
                 "chance_1_over_512": 1 / 512}
        for mode in ("feature_dropout", "zero_inputs"):
            r = B6[f][mode]
            b6[f][mode] = {k: (agg(v) if not isinstance(v[0], (bool, np.bool_)) and k not in ("top_pred_class", "n_distinct_pred")
                               else v) for k, v in r.items()}
            b6[f][mode]["n_distinct_pred_per_seed"] = r["n_distinct_pred"]
            b6[f][mode]["top_pred_class_per_seed"] = r["top_pred_class"]
    dump("B6_both_dropped.json", dict(seeds=SEEDS, prereg=common["prereg"], results=b6,
                                      log_64=float(np.log(64)), log_8=float(np.log(8))))
    b7 = {f: {c: {pn: {h: {m: agg(v) for m, v in B7[f][c][pn][h].items()} for h in B7[f][c][pn]}
                  for pn in B7[f][c]} for c in B7[f]} for f in FREQS}
    dump("B7_calibration.json", dict(seeds=SEEDS, prereg=common["prereg"], ece_bins=15, results=b7))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
