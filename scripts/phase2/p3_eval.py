"""Phase 2a (D-029) gate evaluation. Methods (a) radar-alone CNN, (b) three-branch soft-avg,
(c) geometric picker (8 variants, selected on validation at 20 dB), (d) two-stage sweep.
Loss vs genie at K in {1,2,4,8}; sets: all test users (primary), adv >= 3 dB (secondary),
horizontal d >= 1.6 m (sensitivity); SNR 20 (primary), 10, 0 (stress). Paired per-user
bootstrap (10,000, rng 0) of (a)-(c) and (b)-(c). Output results/phase2/gate_results.json
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
from scipy import stats
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

P = os.path.join(ROOT, "data", "processed", "phase2")
CK = os.path.join(ROOT, "checkpoints", "phase2")
SEEDS = [11, 22, 33, 44, 55]; SNRS = (20, 10, 0); KS = (1, 2, 4, 8)
B_TH, B_R = 256, 8
ANG = np.deg2rad(np.arange(-75, 75.0001, 1.5)); DR = 0.10
RADAR = np.array([0.0, 0.5, 1.2]); H_RSU, H_VEH = 2.0, 1.5
dev = torch.device("cuda")


def loss_of(G, sl):
    return 10 * np.log10(G.max(1) / np.take_along_axis(G, sl, 1).max(1))


def topk_joint(pt, pr, K=8):
    j = (pt.astype(np.float64)[:, :, None] * pr.astype(np.float64)[:, None, :]).reshape(len(pt), -1)
    return np.argsort(-j, axis=1, kind="stable")[:, :K]          # ties -> lower flat index


# ---------------- (c) geometric picker ---------------------------------------------------
def ra_db(feat, lo, hi):
    return feat[:, 1, :, :101] * (hi - lo)[:, None, None] + lo[:, None, None]


def parab(ym, y0, yp):
    den = ym - 2 * y0 + yp
    return np.where(np.abs(den) > 1e-12, 0.5 * (ym - yp) / den, 0.0)


def estimate(db, kind):
    """db (n,256,101) RA in dB -> (R_hat, sin_hat) in the radar frame."""
    n = len(db); m = db.copy(); m[:, :3] = -np.inf
    k, a = np.divmod(m.reshape(n, -1).argmax(1), 101)
    if kind == "peak":
        R = np.empty(n); S = np.empty(n)
        for u in range(n):
            dk = parab(db[u, k[u] - 1, a[u]], db[u, k[u], a[u]], db[u, k[u] + 1, a[u]]) if 3 <= k[u] < 255 else 0.0
            da = parab(db[u, k[u], a[u] - 1], db[u, k[u], a[u]], db[u, k[u], a[u] + 1]) if 0 < a[u] < 100 else 0.0
            R[u] = (k[u] + dk) * DR; S[u] = np.sin(ANG[a[u]] + da * np.deg2rad(1.5))
        return R, S
    R = np.empty(n); S = np.empty(n)                     # intensity-weighted centroid
    for u in range(n):
        k0, a0 = k[u], a[u]; pk = db[u, k0, a0]
        ks = slice(max(k0 - 15, 3), min(k0 + 16, 256)); as_ = slice(max(a0 - 8, 0), min(a0 + 9, 101))
        W = db[u, ks, as_]; ok = W >= pk - 10
        # 4-connected component containing the peak
        comp = np.zeros_like(ok); st = [(k0 - ks.start, a0 - as_.start)]
        while st:
            i, j = st.pop()
            if 0 <= i < ok.shape[0] and 0 <= j < ok.shape[1] and ok[i, j] and not comp[i, j]:
                comp[i, j] = True; st += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        ii, jj = np.nonzero(comp)
        w = 10 ** ((W[ii, jj] - pk) / 10); Rc = (ks.start + ii) * DR; ph = ANG[as_.start + jj]
        x = np.sum(w * Rc * np.cos(ph)) / w.sum(); y = np.sum(w * Rc * np.sin(ph)) / w.sum()
        R[u] = np.hypot(x, y); S[u] = y / max(R[u], 1e-9)
    return R, S


def to_codeword(R, S, ugrid, rings):
    py = R * S; dz = H_VEH - RADAR[2]
    px = np.sqrt(np.maximum(R ** 2 - py ** 2 - dz ** 2, 0))
    yc = py + RADAR[1]; zc = H_VEH - H_RSU
    r = np.sqrt(px ** 2 + yc ** 2 + zc ** 2); u = yc / np.maximum(r, 1e-9)
    order_a = np.argsort(np.abs(u[:, None] - ugrid[None]), axis=1, kind="stable")[:, :2]
    inv = np.where(np.isinf(rings), 0.0, 1 / rings)
    order_r = np.argsort(np.abs(inv[None] - 1 / np.maximum(r, 1e-9)[:, None]), axis=1, kind="stable")
    return order_a, order_r


def picker_lists(order_a, order_r, tk):
    if tk == "c1":
        return order_a[:, :1] * B_R + order_r[:, :8]
    seq = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    return np.stack([order_a[:, ai] * B_R + order_r[:, rj] for ai, rj in seq], 1)


# ---------------- learned models ----------------------------------------------------------
def infer(model_kind, freq, seed, xr, xl):
    if model_kind == "radar":
        m = RadarPolarModel(d_z=128, B_theta=B_TH, B_r=B_R)
    else:
        m = SensingPolarPredictor(d_z=128, B_theta=B_TH, B_r=B_R)
    m.load_state_dict(torch.load(os.path.join(CK, f"{model_kind}_{freq}GHz_seed{seed}.pt"), map_location="cpu", weights_only=False)["model"])
    m = m.to(dev).eval(); PT, PR = [], []
    with torch.no_grad():
        for b in range(0, len(xr), 128):
            r = torch.from_numpy(xr[b:b + 128]).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if model_kind == "radar":
                    o = m(r); pt = o["theta_logits"].float().softmax(1); pr = o["range_logits"].float().softmax(1)
                else:
                    o = m(r, torch.from_numpy(xl[b:b + 128]).to(dev))
                    pth = torch.stack([F.softmax(o[f"{x}_theta_logits"].float(), 1) for x in BRANCHES])
                    prg = torch.stack([F.softmax(o[f"{x}_range_logits"].float(), 1) for x in BRANCHES])
                    wt = pth.max(-1).values; wt = wt / wt.sum(0, keepdim=True).clamp_min(1e-12)
                    wr = prg.max(-1).values; wr = wr / wr.sum(0, keepdim=True).clamp_min(1e-12)
                    pt = (wt[..., None] * pth).sum(0); pr = (wr[..., None] * prg).sum(0)
            PT.append(pt.cpu().numpy()); PR.append(pr.cpu().numpy())
    return np.concatenate(PT), np.concatenate(PR)


def ci_t(x):
    x = np.asarray(x); m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), per_seed=[float(v) for v in x])


def boot(delta, rng_seed=0, B=10000):
    rng = np.random.default_rng(rng_seed); n = len(delta)
    bs = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "learned better" if hi < 0 else ("picker better" if lo > 0 else "no detectable difference")
    return dict(mean=float(delta.mean()), lo=float(lo), hi=float(hi), verdict=verdict)


def load_radar(snr, idx):
    with h5py.File(os.path.join(P, f"radar_snr{snr}.h5"), "r") as hf:
        return hf["radar_features"][idx], hf["ra_db_min"][idx], hf["ra_db_max"][idx]


def main():
    with h5py.File(os.path.join(P, "split.h5"), "r") as hf:
        tr = np.sort(hf["train"][:]); va = np.sort(hf["val"][:]); te = np.sort(hf["test"][:]); snr_idx = hf["snr_idx"][:]
    with h5py.File(os.path.join(P, "users.h5"), "r") as hf:
        d = hf["d"][:]
    with h5py.File(os.path.join(P, "radar_geometry.h5"), "r") as hf:
        Rt = hf["R_true"][:]; St = hf["sin_true"][:]
    with h5py.File(os.path.join(P, "lidar.h5"), "r") as hf:
        XL = hf["lidar_features"][te]
    # radar estimates (frequency independent)
    est = {}
    for snr in SNRS:
        for nm, ix in (("te", te), ("va", va)):
            X, lo, hi = load_radar(snr, ix); db = ra_db(X, lo, hi)
            for kind in ("peak", "centroid"):
                est[(snr, nm, kind)] = estimate(db, kind)
    # training-set bias: train users at their assigned SNR
    bias = {}
    for kind in ("peak", "centroid"):
        Rh = np.empty(len(tr)); Sh = np.empty(len(tr))
        for s, snr in enumerate(SNRS):
            pos = np.where(snr_idx[tr] == s)[0]
            X, lo, hi = load_radar(snr, tr[pos]); Rh[pos], Sh[pos] = estimate(ra_db(X, lo, hi), kind)
        bias[kind] = (float(np.median(Rt[tr] - Rh)), float(np.median(St[tr] - Sh)))
    res = dict(bias=bias, n=dict(train=len(tr), val=len(va), test=len(te)), freqs={})
    for freq in (60, 140):
        with h5py.File(os.path.join(P, f"channels_{freq}GHz.h5"), "r") as hf:
            G = hf["gains"][:]; adv = hf["adv_db"][:]; ug = hf["u_grid"][:]; rings = hf["rings"][:]
        sets = {"all": np.ones(len(te), bool), "adv_ge3": adv[te] >= 3, "d_ge_1p6": d[te] >= 1.6}
        Gte, Gva = G[te], G[va]
        # ---- picker variants
        variants = [(e, b, t) for e in ("peak", "centroid") for b in ("nobias", "bias") for t in ("c1", "c2")]
        def plist(snr, nm, v):
            e, b, t = v; R, S = est[(snr, nm, e)]
            if b == "bias":
                R = R + bias[e][0]; S = np.clip(S + bias[e][1], -1, 1)
            oa, orr = to_codeword(R, S, ug, rings); return picker_lists(oa, orr, t)
        val_score = {"|".join(v): float(np.mean([loss_of(Gva, plist(20, "va", v)[:, :k]).mean() for k in KS])) for v in variants}
        chosen = min(val_score, key=val_score.get)
        # ---- two-stage sweep (d)
        ist = Gte[:, 0::B_R].argmax(1)
        sweep_sets = np.concatenate([np.arange(B_TH)[None].repeat(len(te), 0) * B_R, ist[:, None] * B_R + np.arange(1, B_R)[None]], 1)
        L_sweep = loss_of(Gte, sweep_sets)
        fr = dict(picker_val_score_20dB=val_score, picker_chosen=chosen,
                  sweep_263={s: float(L_sweep[m].mean()) for s, m in sets.items()},
                  genie_dft_test={s: float(adv[te][m].mean()) for s, m in sets.items()},
                  set_sizes={s: int(m.sum()) for s, m in sets.items()}, snr={})
        for snr in SNRS:
            X, _, _ = load_radar(snr, te)
            Lper = {}
            for v in variants:
                sl = plist(snr, "te", v); Lper["picker|" + "|".join(v)] = {k: loss_of(Gte, sl[:, :k]) for k in KS}
            Lpk = Lper["picker|" + chosen]
            learned = {}
            for mk, name in (("radar", "a_radar"), ("fusion", "b_softavg")):
                per = []
                for seed in SEEDS:
                    pt, pr = infer(mk, freq, seed, X, XL); sl = topk_joint(pt, pr)
                    per.append({k: loss_of(Gte, sl[:, :k]) for k in KS})
                learned[name] = per
            out = {}
            for sname, m in sets.items():
                cell = {}
                for k in KS:
                    c = {"c_picker": float(Lpk[k][m].mean())}
                    for name, per in learned.items():
                        c[name] = ci_t([p[k][m].mean() for p in per])
                        mean_u = np.mean([p[k] for p in per], 0)
                        c[name + "_vs_c"] = boot(mean_u[m] - Lpk[k][m])
                        c[name + "_vs_c"]["n_seeds_better_than_picker"] = int(sum(p[k][m].mean() < Lpk[k][m].mean() for p in per))
                    c["picker_all_variants"] = {v: float(L[k][m].mean()) for v, L in Lper.items()}
                    cell[str(k)] = c
                out[sname] = cell
            fr["snr"][str(snr)] = out
            print(freq, snr, "all K8: a %.3f b %.3f c %.3f | %s" % (out["all"]["8"]["a_radar"]["mean"], out["all"]["8"]["b_softavg"]["mean"],
                  out["all"]["8"]["c_picker"], out["all"]["8"]["b_softavg_vs_c"]["verdict"]), flush=True)
        res["freqs"][str(freq)] = fr
    # gate verdict: all users, 20 dB, K=4 and 8, both freqs, (b) vs (c)
    cells = {f"{f}GHz_K{k}": res["freqs"][str(f)]["snr"]["20"]["all"][str(k)]["b_softavg_vs_c"]["verdict"] for f in (60, 140) for k in (4, 8)}
    v = set(cells.values())
    res["gate"] = dict(cells=cells, verdict="learned beats the geometric baseline" if v == {"learned better"} else
                       ("geometric baseline beats learned" if v == {"picker better"} else "mixed"))
    print(json.dumps(res["gate"], indent=1))
    json.dump(res, open(os.path.join(ROOT, "results", "phase2", "gate_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
