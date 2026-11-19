"""D-036 Experiment B evaluation (multi-car simulation).

    python scripts/phase4/b_eval.py        # -> results/phase4/b_results.json

1. Validation (v = 0, frame t0, sigma in {1, 3}): every model x rule R0/R1/R2; per seed the sweep
   shape is chosen on the pooled-sigma validation estimates (frozen rule, new validation gains);
   rule choice per model = lowest mean validation loss at M = 8, 60 GHz (mean over sigma and seeds).
2. Test, chosen rule only: frames t0 and t0 - 0.1 s at v = 0, 10, 20 m/s, sigma 1 and 3; two-frame
   motion compensation (D-035 E2); tau = frame age U[0, 100 ms] (default_rng(2036)) + b4 processing
   + median association time. Loss vs genie on the channel with the user LoS moved by v * tau.
   263-beam sweep chosen on the t0 channel, used 2.35 ms later. CIs: nets t over seeds; geometric
   and sweep per-user bootstrap (10,000, seed 0).
"""
import json, os, sys, time
import h5py, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
for p in ("phase3b", "verify4", "phase3e"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import b_world as W
import eval_b as B
import a5_reproduce as A5
D = os.path.join(ROOT, "data", "processed", "phase4", "multicar"); OUTF = os.path.join(ROOT, "results", "phase4", "b_results.json")
FR, MS, RULES, SIGS, SPEEDS = (60, 140), (8, 16, 32, 64), ("R0", "R1", "R2"), (1.0, 3.0), (0, 10, 20)
T_SWEEP = 263 * (1 / 120e3 + 4.6875e-6 / 8)
CK3 = os.path.join(ROOT, "checkpoints", "phase3"); CKM = os.path.join(ROOT, "checkpoints", "phase4", "multicar")
MODELS = dict(single_vehicle_net=[os.path.join(CK3, f"lidar_H_lr0.001_s{sd}.pt") for sd in A5.SEEDS],
              varied_trained_net=[os.path.join(CK3, f"lidar_H_lr0.001_s{sd}_var.pt") for sd in A5.SEEDS],
              multicar_net=None, lidar_geometric=None)
MC_SEEDS = (11, 22, 33, 44, 55)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
gps_err = lambda i: np.random.default_rng(W.GSEED + i).normal(0, 1, 2)
CB = {f: W.P1.codebook(W.P1.C / (f * 1e9))[0] for f in FR}; U = {f: B.CH[f]["u"] for f in FR}; RG = {f: B.CH[f]["rings"] for f in FR}


def ckpts(model, rule):
    if model == "multicar_net": return [os.path.join(CKM, f"lidar_H_lr0.001_s{sd}_{rule}.pt") for sd in MC_SEEDS]
    return MODELS[model]


def est(model, rule, X, clouds):
    if model == "lidar_geometric": return B.geo(clouds)[0][None]
    return np.array([A5.net_xy(p, X) for p in ckpts(model, rule)])


def gain_mat(idx, f, disp):
    lam = W.P1.C / (f * 1e9); return np.stack([W.gains(int(i), lam, CB[f], disp[j]) for j, i in enumerate(idx)])


def pick_shape(G, e, f, M):
    sc = {sh: B.loss_db(G, B.beam_set(e, U[f], RG[f], *sh)).mean() for sh in B.shape_options(M)}
    k = min(sc, key=sc.get); return k, float(sc[k])


def frame(i, vs, bg_seed, noise_seed, gps_true, sig_list, rules):
    """render + association for each sigma and rule -> {(sig, rule): (bev, cloud, fallback, selected)}, ids, n_user, n_alone, assoc times."""
    pc, ids, nu, na = W.render(vs, bg_seed, noise_seed); t = time.perf_counter(); lab = W.clusters(pc); tc = time.perf_counter() - t; out = {}; ta = {}
    for s in sig_list:
        g = gps_true + s * gps_err(int(i))
        for r in rules:
            t = time.perf_counter(); c, sel, fb = W.associate(pc, g, s, r, lab); ta[r] = tc + time.perf_counter() - t
            out[(s, r)] = (W.LV.bev(c), c, fb, sel, g)
    return out, pc, ids, lab, nu, na, ta


def main():
    t0 = time.time(); Mt = np.load(os.path.join(D, "meta.npz")); ant, psi = Mt["ant"], Mt["psi"]; va, te = Mt["val"], Mt["test"]
    if os.environ.get("B_SMOKE"): va, te = va[:24], te[:30]   # smoke test only (never used for results)
    proc = json.load(open(os.path.join(ROOT, "results", "phase3d", "b4_timing.json")))["processing_ms"]
    R = dict(n_val=int(len(va)), n_test=int(len(te)), n_train=int(len(Mt["train"])), guard_dropped=int(Mt["guard_dropped"]))
    # ---------------- self-check: multi-vehicle renderer == frozen renderer for a single standard car
    a0, p0 = W.LV.users(); R["renderer_single_car_max_abs_diff"] = float(max(np.abs(np.subtract(*W.single_check(i, a0, p0))).max() for i in (0, 5, 123)))
    assert R["renderer_single_car_max_abs_diff"] == 0.0
    # ---------------- validation
    G0va = {f: gain_mat(va, f, np.zeros((len(va), 2))) for f in FR}
    vclouds = {k: [] for k in [(s, r) for s in SIGS for r in RULES]}
    for i in va:
        sc = W.scene(int(i), ant[i], psi[i]); o, *_ = frame(i, W.at_time(sc, 0.0, 0.0), W.NSEED + int(i), None, ant[i, :2], SIGS, RULES)
        for k, v in o.items(): vclouds[k].append(v)
    log("val rendered")
    VAL, CHOSEN, SHAPES = {}, {}, {}
    for m in MODELS:
        best = None
        for r in RULES:
            E = []
            for s in SIGS:
                X = np.stack([v[0] for v in vclouds[(s, r)]]); e = est(m, r, X, [v[1] for v in vclouds[(s, r)]])
                fb = np.array([v[2] for v in vclouds[(s, r)]]); g = np.stack([v[4] for v in vclouds[(s, r)]]); e[:, fb] = g[fb]; E.append(e)
            E = np.concatenate(E, 1); Gp = {f: np.r_[G0va[f], G0va[f]] for f in FR}
            sc = [pick_shape(Gp[60], E[k], 60, 8)[1] for k in range(len(E))]
            VAL[f"{m}|{r}"] = dict(val_loss_M8_60=float(np.mean(sc)), per_seed=sc)
            if best is None or np.mean(sc) < best[0]: best = (float(np.mean(sc)), r, E)
        CHOSEN[m] = best[1]; E = best[2]
        SHAPES[m] = {f"{f}|{M}": [pick_shape(np.r_[G0va[f], G0va[f]], E[k], f, M)[0] for k in range(len(E))] for f in FR for M in MS}
        log("val", m, {r: VAL[f"{m}|{r}"]["val_loss_M8_60"] for r in RULES}, "->", CHOSEN[m])
    R.update(validation=VAL, chosen_rule=CHOSEN, shapes={m: {k: [list(x) for x in v] for k, v in s.items()} for m, s in SHAPES.items()})
    # learning check, multi-car nets
    R["multicar_learning_check"] = {f"{r}|{sd}": (lambda h: dict(final=h["train_loss"], prior=h["prior"], learned=bool(h["train_loss"] < 0.95 * h["prior"])))(
        torch.load(os.path.join(CKM, f"lidar_H_lr0.001_s{sd}_{r}.pt"), map_location="cpu", weights_only=False)["history"][-1]) for r in RULES for sd in MC_SEEDS}
    # ---------------- test scenes, occlusion, blockage, association
    used = sorted(set(CHOSEN.values())); n = len(te); H = np.c_[np.cos(psi[te]), np.sin(psi[te])]
    scenes = [W.scene(int(i), ant[i], psi[i]) for i in te]
    Ko = np.array([len(s["others"]) for s in scenes]); kinds = [[o["kind"] + ("_parked" if o["parked"] else "") for o in s["others"]] for s in scenes]
    blk = [W.blocked_by(W.at_time(s, 0.0, 0.0), ant[i]) for s, i in zip(scenes, te)]; blocked = np.array([len(b) > 0 for b in blk])
    INP = {}; occ = np.zeros(n); succ = {s: np.zeros(n, bool) for s in SIGS}; nocand = {s: np.zeros(n, bool) for s in SIGS}; TA = {r: [] for r in used}
    for v in SPEEDS:
        for j, i in enumerate(te):
            if v == SPEEDS[0]:
                o, pc, ids, lab, nu, na, ta = frame(i, W.at_time(scenes[j], v, 0.0), W.NSEED + int(i), None, ant[i, :2], SIGS, used)
                occ[j] = nu / max(na, 1)
                for s in SIGS:
                    _, sel, fb = W.associate(pc, ant[i, :2] + s * gps_err(int(i)), s, "R1", lab)
                    nocand[s][j] = fb; nu_c = (ids == 0).sum(); succ[s][j] = bool(sel >= 0 and nu_c > 0 and ((ids == 0) & (lab == sel)).sum() >= 0.5 * nu_c)
                for r in used:
                    if r != "R0": TA[r].append(ta[r])
                for k_, val_ in o.items(): INP.setdefault(("t0",) + k_, []).append(val_)
            k = SPEEDS.index(v); ap = ant[i, :2] - v * W.DT * H[j]
            o, *_ = frame(i, W.at_time(scenes[j], v, -W.DT), W.NSEED + int(i), W.PSEED + 1000 * k + int(i), ap, SIGS, used)
            for k_, val_ in o.items(): INP.setdefault((v,) + k_, []).append(val_)
        log("test frames", v)
    ta_med = {r: (float(np.median(TA[r])) if TA.get(r) else 0.0) for r in used}
    occ_cls = np.where(occ >= 0.5, "visible", np.where(occ > 0, "partly_occluded", "fully_occluded"))
    R["scenes"] = dict(K_o_counts={str(k): int((Ko == k).sum()) for k in range(6)}, K_drawn_total=int(sum(s["K_drawn"] for s in scenes)), K_placed_total=int(Ko.sum()),
                       vehicle_types={t: int(sum(x.count(t) for x in kinds)) for t in ("car", "van", "truck", "car_parked", "van_parked")},
                       occlusion_counts={c: int((occ_cls == c).sum()) for c in ("visible", "partly_occluded", "fully_occluded")}, occlusion_ratio_median=float(np.median(occ)))
    by_type = {}
    for t in ("car", "van", "truck", "car_parked", "van_parked"):
        nveh = sum(x.count(t) for x in kinds); nb = sum(sum(1 for jj in b if kinds[j][jj - 1] == t) for j, b in enumerate(blk))
        by_type[t] = dict(scenes_blocked_by_type=float(np.mean([any(kinds[j][jj - 1] == t for jj in b) for j, b in enumerate(blk)])), vehicles=nveh,
                          per_vehicle_blocking_rate=(nb / nveh if nveh else None))
    R["radio_blockage"] = dict(note="the channel model does not include this blockage; results for blocked users are optimistic for every method, including the sweep",
                               blocked_fraction=float(blocked.mean()), by_K_o={str(k): (float(blocked[Ko == k].mean()) if (Ko == k).any() else None) for k in range(6)}, by_type=by_type)
    R["association"] = {str(int(s)): dict(success_rate=float(succ[s].mean()), no_candidate_rate=float(nocand[s].mean()),
                                          by_occlusion={c: float(succ[s][occ_cls == c].mean()) if (occ_cls == c).any() else None for c in ("visible", "partly_occluded", "fully_occluded")})
                        for s in SIGS}
    R["association_time_ms_median"] = {r: v * 1e3 for r, v in ta_med.items()}; log("scenes", R["scenes"], R["radio_blockage"]["blocked_fraction"], R["association"])
    # ---------------- estimates
    tau_proc = {m: (proc["lidar_geometric"] if m == "lidar_geometric" else proc["lidar_net"]) / 1e3 + ta_med[CHOSEN[m]] for m in MODELS}
    age = np.random.default_rng(2036).uniform(0.0, 0.1, n); tau = {m: age + tau_proc[m] for m in MODELS}
    R["tau_ms"] = {m: dict(mean=float(tau[m].mean() * 1e3), p95=float(np.percentile(tau[m], 95) * 1e3)) for m in MODELS}
    EST = {}
    for m in MODELS:
        r = CHOSEN[m]
        for s in SIGS:
            for fr in ("t0",) + SPEEDS:
                L = INP[(fr, s, r)]; e = est(m, r, np.stack([x[0] for x in L]), [x[1] for x in L])
                fb = np.array([x[2] for x in L]); g = np.stack([x[4] for x in L]); e[:, fb] = g[fb]; EST[(m, s, fr)] = e
        log("estimates", m)
    del INP
    # ---------------- losses
    subs = dict(all=np.ones(n, bool), **{f"K_o={k}": Ko == k for k in range(6)}, K_o_ge1=Ko >= 1, **{c: occ_cls == c for c in ("visible", "partly_occluded", "fully_occluded")},
                radio_blocked=blocked, radio_unblocked=~blocked)
    R["subset_sizes"] = {k: int(v.sum()) for k, v in subs.items()}
    T, SW, PER, PE = {}, {}, {}, {}
    for f in FR:
        G0 = gain_mat(te, f, np.zeros((n, 2))); Sw = np.c_[np.arange(256)[None].repeat(n, 0) * 8, G0[:, 0::8].argmax(1)[:, None] * 8 + np.arange(1, 8)[None]]
        for v in SPEEDS:
            Lsw = B.loss_db(gain_mat(te, f, v * T_SWEEP * H), Sw)
            for sb, msk in subs.items():
                if msk.any(): SW[f"{f}|{v}|{sb}"] = B.boot(Lsw[msk])
            Gm = {}
            for m in MODELS:
                key = round(float(tau_proc[m]), 9)
                if key not in Gm: Gm[key] = gain_mat(te, f, v * tau[m][:, None] * H)
                Gt = Gm[key]
                for s in SIGS:
                    e0 = EST[(m, s, "t0")]; ep = EST[(m, s, v)]; comp = e0 + (e0 - ep) / W.DT * tau[m][None, :, None]
                    if f == FR[0]:
                        e = np.hypot(*(comp - (ant[te, :2] + v * tau[m][:, None] * H)[None]).reshape(-1, 2).T); PE[f"{m}|{int(s)}|{v}"] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)))
                    for M in MS:
                        Ls = [B.loss_db(Gt, B.beam_set(comp[k], U[f], RG[f], *SHAPES[m][f"{f}|{M}"][k])) for k in range(len(comp))]
                        PER[(m, s, f, M, v)] = np.mean(Ls, 0)
                        for sb, msk in subs.items():
                            if not msk.any(): continue
                            T[f"{m}|{int(s)}|{f}|{M}|{v}|{sb}"] = (B.ci_t([L[msk].mean() for L in Ls]) | dict(n_runs=len(Ls))) if len(Ls) > 1 else (B.boot(Ls[0][msk]) | dict(n_runs=1))
        log("losses", f)
    R.update(tables=T, sweep263=SW, position_error=PE)
    # ---------------- reads
    pk = "multicar_net|3|60|32|10|all"; sw = SW["60|10|all"]
    R["primary"] = dict(condition="multicar_net (chosen rule), 60 GHz, M = 32, v = 10 m/s, compensated, sigma = 3 m, all test users",
                        net=T[pk], sweep=sw, matches_sweep_in_traffic=bool(T[pk]["mean"] <= sw["hi"]))
    R["frozen_minus_multicar"] = {m: B.boot(PER[(m, 3.0, 60, 32, 10)] - PER[("multicar_net", 3.0, 60, 32, 10)]) for m in MODELS if m != "multicar_net"}
    def two_sample(a, b, Bn=10000):
        rng = np.random.default_rng(0); bs = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(Bn)])
        lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(mean=float(a.mean() - b.mean()), lo=float(lo), hi=float(hi))
    R["traffic_effect_Ko_ge1_minus_Ko0"] = {m: two_sample(PER[(m, 3.0, 60, 32, 10)][Ko >= 1], PER[(m, 3.0, 60, 32, 10)][Ko == 0]) for m in MODELS}
    SM = {}
    for m in MODELS:
        for s in SIGS:
            for f in FR:
                for v in SPEEDS:
                    hi = SW[f"{f}|{v}|all"]["hi"]; ok = [M for M in MS if T[f"{m}|{int(s)}|{f}|{M}|{v}|all"]["mean"] <= hi]
                    SM[f"{m}|{int(s)}|{f}|{v}"] = ok[0] if ok else None
    R["smallest_M_within_sweep_ci"] = SM; R["runtime_s"] = time.time() - t0
    json.dump(R, open(OUTF, "w"), indent=1); log("done", R["primary"])


if __name__ == "__main__":
    main()
