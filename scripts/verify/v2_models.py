"""Independent verification, step 2: model-level findings V1, V5, V8, V9, V10, V12, V13 inputs.

Uses only the original model/split code (src/) and banked checkpoints + HDF5. Loss
definition re-derived from scripts/eval_headline.py / eval_fusion_rules.py:
loss_K = 10 log10( max_k G / max_{k in top-K} G ), G = |conj(cb) @ h|^2 (complex64), with
p_theta, p_r = softmax(float(logits)) under fp16 autocast. Gains come from
results/verify/gains_{f}GHz.npz (v0_gains.py). Output results/verify/v2_models.json
"""
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.splits import trajectory_grouped_split
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import SensingPolarPredictor

OUT = os.path.join(ROOT, "results", "verify")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220]
KS = (1, 2, 4, 8)
C = 299_792_458.0
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = False


def loss_of(G, sl):
    """G (n,512) float64, sl (n,K) flat codeword indices -> per-user dB loss."""
    return 10 * np.log10(G.max(1) / np.take_along_axis(G, sl, 1).max(1))


def topk_joint(pt, pr, K=8):
    joint = (pt[:, :, None] * pr[:, None, :]).reshape(len(pt), 512)
    return torch.from_numpy(joint).topk(K, 1).indices.numpy()


# ---------------- learning-free peak picker variants (radar RA map only) ---------------
def peak(RA):
    flat = RA.reshape(len(RA), -1).argmax(1)
    return np.divmod(flat, 64)                    # range bin, angle bin


def ring_of(dhat, f):
    lam = C / (f * 1e9)
    inv_r = np.r_[0.0, np.arange(1, 8) / (1.2 * 127 ** 2 * lam / 8)]
    return np.abs(inv_r[None] - 1 / dhat[:, None]).argmin(1)


II, JJ = np.divmod(np.arange(512), 8)


def pp_rank(a, j, key):
    """Rank all 512 codewords per user by the given key; returns (n,8) top-8 indices."""
    out = np.zeros((len(a), 8), int)
    for u in range(len(a)):
        di, dj = np.abs(II - a[u]), np.abs(JJ - j[u])
        if key == "L1_ringtie":       # D-025 construction
            order = np.lexsort((np.arange(512), di, dj, di + dj))
        elif key == "L1_angletie":
            order = np.lexsort((np.arange(512), dj, di, di + dj))
        elif key == "ring_first":     # j_hat fixed; fan out in range first: (a, j±..), then angle
            order = np.lexsort((np.arange(512), dj, di))
        elif key == "angle_first":    # j_hat fixed; fan out in angle first
            order = np.lexsort((np.arange(512), di, dj))
        elif key == "box2x4":         # Chebyshev-ish: angle |di|<=? rank by max(2di, dj)
            order = np.lexsort((np.arange(512), dj, di, np.maximum(2 * di, dj)))
        elif key == "dft_only":       # far-field: j=0 only, angles around a
            order = np.lexsort((np.arange(512), di, JJ != 0))
        out[u] = order[:8]
    return out


def pp_soft(RA, rb, j, gamma=4.0):
    """Soft picker: p_theta from the RA angle cut at the peak range, p_r ~ exp(-|j-jhat|)."""
    cut = RA[np.arange(len(RA)), rb, :] ** gamma + 1e-9
    pt = cut / cut.sum(1, keepdims=True)
    pr = np.exp(-np.abs(np.arange(8)[None] - j[:, None]).astype(float))
    pr /= pr.sum(1, keepdims=True)
    return topk_joint(pt, pr)


res = {}
hand = {}
for f in (60, 140):
    z = np.load(os.path.join(OUT, f"gains_{f}GHz.npz"))
    G = z["G32"].astype(np.float64); yrg = z["y_r_gated"]; ytg = z["y_theta_gated"]
    traj = z["trajectory_id"]
    with h5py.File(os.path.join(ROOT, "data", "processed", f"closerange_{f}GHz_full.h5"), "r") as hf:
        XR = hf["radar_features"][:]
        XL = hf["lidar_features"][:]
    RA_all = XR[:, 1, :, :64]
    rb_all, a_all = peak(RA_all)
    j_all = ring_of(np.maximum(rb_all, 1) * 1.2, f)
    variants = ["L1_ringtie", "L1_angletie", "ring_first", "angle_first", "box2x4", "dft_only"]
    PP = {v: pp_rank(a_all, j_all, v) for v in variants}
    PP["soft_g4"] = pp_soft(RA_all, rb_all, j_all, 4.0)
    PP["soft_g1"] = pp_soft(RA_all, rb_all, j_all, 1.0)
    PP["peak_angle_all8"] = a_all[:, None] * 8 + np.arange(8)[None]

    per = []
    for s in SEEDS:
        _, _, te = trajectory_grouped_split(traj, seed=s)
        te = np.sort(te)
        nf = yrg[te] > 0
        Gt = G[te]
        rec = dict(seed=s, n=len(te), n_nf=int(nf.sum()))

        # radar-alone
        rck = torch.load(os.path.join(ROOT, "checkpoints", f"closerange_{f}GHz_radar", f"seed{s}.pt"),
                         map_location="cpu", weights_only=False)
        rm = RadarPolarModel().to(dev); rm.load_state_dict(rck["model"]); rm.eval()
        fck = torch.load(os.path.join(ROOT, "checkpoints", f"closerange_{f}GHz_fusion_warmstart",
                                      f"seed{s}.pt"), map_location="cpu", weights_only=False)
        fm = SensingPolarPredictor().to(dev); fm.load_state_dict(fck["model"]); fm.eval()

        # --- V5: frozen vs unfrozen from weights + saved args
        fsd, rsd = fck["model"], rck["model"]
        dmax = 0.0
        for k, v in rsd.items():
            fk = ("radar_encoder." + k[len("encoder."):]) if k.startswith("encoder.") else \
                 ("head_radar." + k[len("head."):])
            dmax = max(dmax, float((fsd[fk].float() - v.float()).abs().max()))
        rec["V5"] = dict(max_abs_diff_radar_branch_vs_standalone=dmax,
                         args_freeze_radar=fck["args"].get("freeze_radar"),
                         args_warmstart=fck["args"].get("warmstart_radar"),
                         fusion_epoch=fck.get("epoch"), radar_epoch=rck.get("epoch"),
                         radar_init_seed=rck["args"].get("init_seed"),
                         radar_args_seed=rck["args"].get("seed"), fusion_args_seed=fck["args"].get("seed"))

        P = {}
        with torch.no_grad():
            xr = torch.from_numpy(XR[te]).to(dev); xl = torch.from_numpy(XL[te]).to(dev)
            outs, outr, outd, outz = [], [], [], []
            for b in range(0, len(te), 128):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = rm(xr[b:b + 128]); fo = fm(xr[b:b + 128], xl[b:b + 128])
                    fd = fm(xr[b:b + 128], xl[b:b + 128], radar_dropout=True, lidar_dropout=True)
                    fz = fm(torch.zeros_like(xr[b:b + 128]), torch.zeros_like(xl[b:b + 128]))
                outr.append((o["theta_logits"].float().softmax(1), o["range_logits"].float().softmax(1)))
                for lst, oo in ((outs, fo), (outd, fd), (outz, fz)):
                    pth = torch.stack([F.softmax(oo[f"{br}_theta_logits"].float(), 1) for br in ("radar", "lidar", "fused")])
                    prg = torch.stack([F.softmax(oo[f"{br}_range_logits"].float(), 1) for br in ("radar", "lidar", "fused")])
                    lst.append((pth, prg))
            cat = lambda L, i: torch.cat([x[i] for x in L], -2).cpu().numpy()
            P["radar_alone"] = (cat(outr, 0), cat(outr, 1))

            def soft(L):
                pth = torch.cat([x[0] for x in L], 1); prg = torch.cat([x[1] for x in L], 1)
                wt = pth.max(-1).values; wr = prg.max(-1).values
                wt = wt / wt.sum(0, keepdim=True).clamp_min(1e-12); wr = wr / wr.sum(0, keepdim=True).clamp_min(1e-12)
                pt = (wt[..., None] * pth).sum(0); pr = (wr[..., None] * prg).sum(0)
                ar = torch.arange(pth.shape[1], device=pth.device)
                prim = (pth[0], prg[prg.max(-1).values.argmax(0), ar])
                return (pt.cpu().numpy(), pr.cpu().numpy()), tuple(x.cpu().numpy() for x in prim)
            P["soft_avg"], P["radar_primary"] = soft(outs)
            Pd, _ = soft(outd); Pz, _ = soft(outz)

        # --- losses
        L = {}
        for name, (pt, pr) in P.items():
            sl = topk_joint(pt, pr)
            L[name] = {k: loss_of(Gt, sl[:, :k]) for k in KS}
            # V8: top-1 angle + all 8 ranges
            ih = pt.argmax(1)
            L[name]["ang1_all8"] = loss_of(Gt, ih[:, None] * 8 + np.arange(8)[None])
            L[name]["n_distinct_angles_K8"] = np.array([len(set(II[r])) for r in sl])
            L[name]["one_angle_8_ranges_K8"] = np.array([len(set(II[r])) == 1 for r in sl]).astype(float)
            L[name]["top1_theta"] = (ih == ytg[te]).astype(float)
        for v, sl in PP.items():
            sl = sl[te]
            L["pp_" + v] = {k: loss_of(Gt, sl[:, :k]) for k in KS if k <= sl.shape[1]}
        gd = 10 * np.log10(Gt.max(1) / Gt[:, 0::8].max(1))
        L["genie_dft"] = {k: gd for k in KS}
        rec["loss"] = {name: {str(k): dict(all=float(v.mean()), nf=float(v[nf].mean()))
                              for k, v in d.items()} for name, d in L.items()}

        # --- V10: both sensors dropped
        for tag, (pt, pr) in (("feat_zeroed", Pd), ("input_zeroed", Pz)):
            joint = (pt[:, :, None] * pr[:, None, :]).reshape(len(te), 512)
            pred = joint.argmax(1)
            true = ytg[te] * 8 + yrg[te]
            vals, cnt = np.unique(pred, return_counts=True)
            tv, tc = np.unique(true, return_counts=True)
            rec[f"V10_{tag}"] = dict(
                acc=float((pred == true).mean()), n_distinct_pred=int(len(vals)),
                top_pred=int(vals[cnt.argmax()]), top_pred_share=float(cnt.max() / len(te)),
                freq_of_top_pred_in_test=float((true == vals[cnt.argmax()]).mean()),
                most_common_true=int(tv[tc.argmax()]), most_common_true_freq=float(tc.max() / len(te)),
                entropy_theta=float(-(pt * np.log(pt + 1e-12)).sum(1).mean()),
                entropy_r=float(-(pr * np.log(pr + 1e-12)).sum(1).mean()))

        # --- hand-worked examples: seed 11, three near-field users
        if s == 11:
            ex = []
            nf_idx = np.where(nf)[0]
            for q in (nf_idx[0], nf_idx[len(nf_idx) // 2], nf_idx[-1]):
                u = int(te[q])
                pt, pr = P["soft_avg"]; sl_soft = topk_joint(pt[q:q + 1], pr[q:q + 1])[0]
                sl_pp = PP["L1_ringtie"][u]
                g = G[u]
                ex.append(dict(
                    user=u, distance_m=float(z["distance"][u]), label=(int(ytg[u]), int(yrg[u])),
                    polar_adv_db=float(z["polar_adv_db"][u]),
                    radar_peak_rangebin=int(rb_all[u]), radar_peak_anglebin=int(a_all[u]),
                    dhat_m=float(max(rb_all[u], 1) * 1.2), jhat=int(j_all[u]),
                    genie_k=int(g.argmax()), genie_ij=[int(x) for x in divmod(int(g.argmax()), 8)],
                    genie_gain=float(g.max()),
                    pp_shortlist=[[int(x) for x in divmod(int(k), 8)] + [float(g[k])] for k in sl_pp],
                    pp_loss_K8=float(10 * np.log10(g.max() / g[sl_pp].max())),
                    soft_shortlist=[[int(x) for x in divmod(int(k), 8)] + [float(g[k])] for k in sl_soft],
                    soft_loss_K8=float(10 * np.log10(g.max() / g[sl_soft].max())),
                    soft_top1_angle=int(pt[q].argmax()),
                    soft_ang1_all8=[[int(pt[q].argmax()), j, float(g[pt[q].argmax() * 8 + j])] for j in range(8)],
                    soft_ang1_all8_loss=float(10 * np.log10(g.max() / g[pt[q].argmax() * 8:pt[q].argmax() * 8 + 8].max())),
                ))
            hand[str(f)] = ex
        per.append(rec)
        print(f, s, "radar", round(rec["loss"]["radar_alone"]["8"]["all"], 3), round(rec["loss"]["radar_alone"]["8"]["nf"], 3),
              "soft", round(rec["loss"]["soft_avg"]["8"]["all"], 3), round(rec["loss"]["soft_avg"]["8"]["nf"], 3),
              "pp", round(rec["loss"]["pp_L1_ringtie"]["8"]["all"], 3), round(rec["loss"]["pp_L1_ringtie"]["8"]["nf"], 3),
              "V5", rec["V5"]["args_freeze_radar"], f'{rec["V5"]["max_abs_diff_radar_branch_vs_standalone"]:.2e}', flush=True)
        del xr, xl
        torch.cuda.empty_cache()

    # aggregate 20-seed means
    agg = {}
    for name in per[0]["loss"]:
        agg[name] = {k: {sub: float(np.mean([p["loss"][name][k][sub] for p in per])) for sub in ("all", "nf")}
                     for k in per[0]["loss"][name]}
    res[str(f)] = dict(per_seed=per, mean=agg)
    del XR, XL

# ---- V9: does the cross-attention output depend on the query? -----------------------
fm = SensingPolarPredictor().to(dev).eval()
fm.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "closerange_60GHz_fusion_warmstart", "seed11.pt"),
                              map_location="cpu", weights_only=False)["model"])
with torch.no_grad():
    zl = torch.randn(1, 128, device=dev).repeat(64, 1)
    zr = torch.randn(64, 128, device=dev) * 3
    ao, w = fm.fusion.attn(zr.unsqueeze(1), zl.unsqueeze(1), zl.unsqueeze(1))
    res["V9"] = dict(attn_weight_min=float(w.min()), attn_weight_max=float(w.max()),
                     attn_out_spread_across_64_queries=float((ao - ao[0:1]).abs().max()),
                     attn_out_scale=float(ao.abs().mean()))
print(res["V9"])
json.dump(res, open(os.path.join(OUT, "v2_models.json"), "w"), indent=1)
json.dump(hand, open(os.path.join(OUT, "v2_hand_examples.json"), "w"), indent=1)
