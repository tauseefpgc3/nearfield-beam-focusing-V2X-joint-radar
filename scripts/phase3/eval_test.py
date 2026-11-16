"""D-031 test evaluation — run ONCE, after tag revision-design-v2.
Every method seeds the same local sweep (shape chosen per method/freq/M/seed on validation).
Writes results/phase3/test_results.json."""
import json, os, sys
import h5py, numpy as np, torch
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2"))
import p3_eval as E
import lidar_v2 as LV, lidar_estimator as LE
from p5_e23_sweep import sweep_loss, shapes, err_stats
from train_net import UNet, load_inputs, predict
P2 = LV.P2; R3 = os.path.join(ROOT, "results", "phase3"); CK = os.path.join(ROOT, "checkpoints", "phase3")
SEEDS = (11, 22, 33, 44, 55); MS = (8, 16, 32, 64); FREQS = (60, 140)


def boot(delta, B=10000):
    rng = np.random.default_rng(0); n = len(delta)
    bs = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(B)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(delta.mean()), lo=float(lo), hi=float(hi), verdict="network better" if hi < 0 else ("network worse" if lo > 0 else "no detectable difference"))


def ci_t(x):
    x = np.asarray(x); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), per_seed=[float(v) for v in x])


def radar_picker(ix):
    bias = json.load(open(os.path.join(ROOT, "results", "phase2", "gate_results.json")))["bias"]["centroid"]
    X, lo, hi = E.load_radar(20, ix); R, S = E.estimate(E.ra_db(X, lo, hi), "centroid")
    R = R + bias[0]; S = np.clip(S + bias[1], -1, 1); py = R * S; px = np.sqrt(np.maximum(R ** 2 - py ** 2 - 0.3 ** 2, 0))
    return np.c_[px, py + E.RADAR[1]]


def lidar_geo(ix, ant, psi, params, fb):
    est, fbk = [], []
    for b in range(0, len(ix), 25):
        pcs, _ = LV.render(ix[b:b + 25], ant, psi)
        for p in pcs:
            e, info = LE.estimate(p, params, fb); est.append(e); fbk.append(info["fallback"])
    return np.array(est), float(np.mean(fbk))


def main():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as hf: va = np.sort(hf["val"][:]); te = np.sort(hf["test"][:])
    ant, psi = LV.users(); xy = ant[:, :2]
    dl = json.load(open(os.path.join(R3, "design_lidar_estimator.json"))); dn = json.load(open(os.path.join(R3, "design_networks.json")))
    params = dict(LE.DEFAULT, **dl["best"]["params"]); fb = np.array(dl["fallback_xy"])
    est = {"radar_picker": [(radar_picker(va), radar_picker(te))]}
    lv, fv = lidar_geo(va, ant, psi, params, fb); lt, ft = lidar_geo(te, ant, psi, params, fb)
    est["lidar_geometric"] = [(lv, lt)]
    dev = torch.device("cuda")
    for s in ("radar", "lidar", "fused"):
        h, lr = dn[s]["chosen"].split("|"); runs = []
        Xva, Xte = load_inputs(s, va), load_inputs(s, te)
        for seed in SEEDS:
            ck = torch.load(os.path.join(CK, f"{s}_{h}_lr{float(lr):g}_s{seed}.pt"), map_location="cpu", weights_only=False)
            m = UNet(ck["cin"]).to(dev); m.load_state_dict(ck["model"])
            runs.append((predict(m, Xva, h, dev), predict(m, Xte, h, dev)))
        est[f"{s}_net"] = runs
    res = dict(lidar_fallback_rate=dict(val=fv, test=ft), design=dict(lidar=params, networks={s: dn[s]["chosen"] for s in dn}),
               position_error={}, loss={}, val_loss={}, bootstrap={}, reference={})
    for name, runs in est.items():
        per = [err_stats(t, xy[te]) for _, t in runs]
        pooled = err_stats(np.concatenate([t for _, t in runs]), np.tile(xy[te], (len(runs), 1)))
        res["position_error"][name] = dict(pooled=pooled, per_seed_median_range={k: [min(p[k]["median_abs"] for p in per), max(p[k]["median_abs"] for p in per)] for k in ("range", "sideways")})
    per_user = {}
    for f in FREQS:
        with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as hf:
            G = hf["gains"][:]; ug = hf["u_grid"][:]; rings = hf["rings"][:]; adv = hf["adv_db"][:]
        sets = {"all": np.ones(len(te), bool), "adv_ge3": adv[te] >= 3}
        sw = np.concatenate([np.arange(256)[None].repeat(len(te), 0) * 8, G[te][:, 0::8].argmax(1)[:, None] * 8 + np.arange(1, 8)[None]], 1)
        Ls = E.loss_of(G[te], sw); res["reference"][f"sweep263_{f}"] = {k: float(Ls[m].mean()) for k, m in sets.items()}
        for name, runs in est.items():
            for M in MS:
                LT, LVv, shp = [], [], []
                for v, t in runs:
                    vs = {(A, R): sweep_loss(G[va], v, ug, rings, A, R).mean() for A, R in shapes(M)}
                    A, R = min(vs, key=vs.get); shp.append(f"{A}x{R}"); LVv.append(float(vs[(A, R)]))
                    LT.append(sweep_loss(G[te], t, ug, rings, A, R))
                per_user[(name, f, M)] = np.mean(LT, 0); res["val_loss"][f"{name}|{f}|{M}"] = float(np.mean(LVv))
                cell = dict(shapes=shp)
                for k, m in sets.items():
                    vals = [float(L[m].mean()) for L in LT]
                    cell[k] = ci_t(vals) if len(vals) > 1 else dict(mean=vals[0])
                    cell[k]["frac_within_1dB"] = float(np.mean([(L[m] <= 1).mean() for L in LT]))
                res["loss"][f"{name}|{f}|{M}"] = cell
        for M in MS:
            single = ["radar_picker", "lidar_geometric", "radar_net", "lidar_net"]
            best = min(single, key=lambda n: res["val_loss"][f"{n}|{f}|{M}"])
            for k, m in sets.items():
                b = {"radar_net_vs_radar_picker": boot(per_user[("radar_net", f, M)][m] - per_user[("radar_picker", f, M)][m]),
                     "lidar_net_vs_lidar_geometric": boot(per_user[("lidar_net", f, M)][m] - per_user[("lidar_geometric", f, M)][m]),
                     "fused_net_vs_best_single": dict(boot(per_user[("fused_net", f, M)][m] - per_user[(best, f, M)][m]), best_single=best)}
                res["bootstrap"][f"{f}|{M}|{k}"] = b
    res["reference"]["phase2b_sigma_curve"] = {f: {s: {M: v["test_mean"] for M, v in d.items()} for s, d in fr.items() if s.startswith("sigma")}
                                              for f, fr in json.load(open(os.path.join(ROOT, "results", "phase2", "feasibility", "e23_sweep.json")))["freqs"].items()}
    json.dump(res, open(os.path.join(R3, "test_results.json"), "w"), indent=1)
    for f in FREQS:
        for k in ("all", "adv_ge3"):
            print(f, k, "  ".join(f"{n}: " + "/".join(f"{res['loss'][f'{n}|{f}|{M}'][k]['mean']:.2f}" for M in MS) for n in est), flush=True)
    print(json.dumps(res["bootstrap"], indent=0)[:4000]); print(json.dumps(res["position_error"], indent=0)[:3000]); print(res["lidar_fallback_rate"])


if __name__ == "__main__":
    main()
