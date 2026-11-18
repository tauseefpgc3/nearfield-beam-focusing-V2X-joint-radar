"""D-035 E1/E2: sensor age and vehicle speed (evaluation only; frozen models and predictions).

E1 stale estimate: the true antenna moves v * tau along the heading after the LiDAR frame
(tau = frame age U[0, 100 ms] from default_rng(2035) + measured processing); beams chosen from
the static estimate (preds_d.npz) with the frozen validation shape, scored on the moved channel.
E2 compensation: an earlier frame at t0 - 0.1 s is rendered (static clutter, fresh noise),
the same estimator gives p_prev; v_hat = (p_t0 - p_prev) / 0.1; beams from p_t0 + v_hat * tau.

    python scripts/phase3e/motion.py     # -> results/phase3e/motion.json
"""
import json, os, sys, time
import h5py, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("phase3b", "verify4", "phase2", "phase3"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
import eval_b as B
import a5_reproduce as A5
import p1_channels as P1
import lidar_v2 as LV, lidar_var as LR
dev = LV.dev
OUTD = os.path.join(ROOT, "results", "phase3e"); os.makedirs(OUTD, exist_ok=True)
SPEEDS = (0, 2, 5, 10, 15, 20); FR, MS = (60, 140), (8, 16, 32, 64); DT = 0.1
T_SWEEP = 263 * (1 / 120e3 + 4.6875e-6 / 8)          # 263 beams x (symbol + normal CP) at 120 kHz
SEEDS = A5.SEEDS
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
te, va, ant, psi = B.te, B.va, B.ant, B.psi
H = np.c_[np.cos(psi), np.sin(psi)]


def render_prev(idx, antx, veh, noise_base):
    """lidar_var.render with static clutter from the original stream (first 1,800 draws) and
    noise / intensity / dropout from default_rng(noise_base + i); noise_base None = original stream."""
    idx = np.asarray(idx); Bn = len(idx)
    dirs_all = LV.DIRS.view(64, 1800, 3).reshape(-1, 3)
    cen = LR.centre_of(antx[idx], psi[idx], veh["ant_xi"][idx])
    T = lambda a: torch.tensor(a[idx][:, None, :], device=dev)
    blo, bhi, clo, chi = T(veh["body_lo"]), T(veh["body_hi"]), T(veh["cab_lo"]), T(veh["cab_hi"])
    c = torch.tensor(np.cos(psi[idx]), device=dev)[:, None]; s = torch.tensor(np.sin(psi[idx]), device=dev)[:, None]
    to_veh = lambda v: torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
    o = to_veh(torch.tensor(np.c_[LV.SENSOR[:2][None] - cen, np.full(Bn, LV.SENSOR[2])], device=dev)[:, None, :])
    d = to_veh(dirs_all[None].expand(Bn, -1, -1))
    t_veh = torch.minimum(LR._slab(o, d, blo, bhi), LR._slab(o, d, clo, chi))
    dz = dirs_all[:, 2][None].expand(Bn, -1)
    t_gr = torch.where(dz < -1e-9, -LV.SENSOR[2] / dz, torch.full_like(dz, np.inf))
    t = torch.minimum(t_veh, t_gr); onveh = t_veh <= t_gr; dirs_np = dirs_all.cpu().numpy(); out = []
    for b, i in enumerate(idx):
        rng = np.random.default_rng(LV.SEED + int(i))
        bg = np.c_[rng.uniform(-20, 40, 600), rng.uniform(-20, 20, 600), rng.uniform(0, 4, 600)]
        if noise_base is not None: rng = np.random.default_rng(noise_base + int(i))
        v = torch.tensor(bg - LV.SENSOR, device=dev); Ln = v.norm(dim=-1); dv = v / Ln[:, None]
        cb = torch.tensor(np.cos(psi[i]), device=dev); sb = torch.tensor(np.sin(psi[i]), device=dev)
        dvv = torch.stack([cb * dv[:, 0] + sb * dv[:, 1], -sb * dv[:, 0] + cb * dv[:, 1], dv[:, 2]], -1)[None]
        ob = o[b:b + 1]
        tb = torch.minimum(LR._slab(ob, dvv, blo[b:b + 1], bhi[b:b + 1]), LR._slab(ob, dvv, clo[b:b + 1], chi[b:b + 1]))[0]
        keep_bg = (tb > Ln).cpu().numpy()
        tt = t[b].cpu().numpy(); ok = np.isfinite(tt) & (tt >= LV.R_MIN) & (tt <= LV.R_MAX)
        dd = dirs_np[ok]; rr = tt[ok]; vh = onveh[b].cpu().numpy()[ok]
        inten = np.where(vh, rng.uniform(0.4, 1.0, ok.sum()), rng.uniform(0.0, 0.2, ok.sum()))
        bgr = np.linalg.norm(bg - LV.SENSOR, axis=1); bgd = (bg - LV.SENSOR) / bgr[:, None]
        bgi = rng.uniform(0.0, 0.5, 600)
        rr = rr + rng.normal(0, LV.SIG_R, len(rr)); bgr = bgr + rng.normal(0, LV.SIG_R, 600)
        drop = rng.random(len(rr)) < LV.P_DROP; dropb = rng.random(600) < LV.P_DROP
        pts = np.r_[LV.SENSOR + dd * rr[:, None], LV.SENSOR + bgd * bgr[:, None]]
        I = np.r_[inten, bgi]; keep = np.r_[~drop, keep_bg & ~dropb]; pts = pts[keep]; I = I[keep]
        m = (pts[:, 0] >= LV.ROI[0]) & (pts[:, 0] < LV.ROI[1]) & (pts[:, 1] >= LV.ROI[2]) & (pts[:, 1] < LV.ROI[3])
        out.append(np.c_[pts[m], I[m]].astype(np.float32))
    return out


def frames(idx, antx, veh, noise_base):
    clouds, bevs = [], []
    for b in range(0, len(idx), 25):
        pcs = render_prev(idx[b:b + 25], antx, veh, noise_base); clouds += pcs; bevs += [LV.bev(p) for p in pcs]
    return np.stack(bevs), clouds


def estimates(X, clouds):
    ck = lambda sd: os.path.join(ROOT, "checkpoints", "phase3", f"lidar_H_lr0.001_s{sd}.pt")
    return dict(lidar_net=np.array([A5.net_xy(ck(sd), X) for sd in SEEDS]), lidar_geometric=B.geo(clouds)[0][None])


def moved_gains(f, disp):
    """Gains (n_test, 2048) with the LoS point of each test user shifted by disp (n_test, 2)."""
    lam = P1.C / (f * 1e9); F = CB[f]; Hm = np.empty((len(te), P1.N), complex)
    for j, i in enumerate(te):
        _, _, _, pts, g = P1.draw(int(i)); pts = pts.copy(); pts[0, :2] += disp[j]; Hm[j] = P1.channel(pts, g, lam)
    return np.abs(Hm @ F.conj().T) ** 2


def ci_seed(Ls, m): x = np.array([L[m].mean() for L in Ls]); return B.ci_t(x) | dict(n_runs=len(x))


def ci_user(L, m): return B.boot(L[m]) | dict(n_runs=1)


def pe(est, true):
    e = np.hypot(*(est - true[None]).reshape(-1, 2).T); return dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)))


def main():
    t0 = time.time(); Z = np.load(os.path.join(ROOT, "results", "phase3d", "preds_d.npz"))
    RD = json.load(open(os.path.join(ROOT, "results", "phase3d", "results_d.json")))
    proc = json.load(open(os.path.join(ROOT, "results", "phase3d", "b4_timing.json")))["processing_ms"]
    METH = ("lidar_net", "lidar_geometric")
    age = np.random.default_rng(2035).uniform(0.0, 0.1, len(te))
    tau = {m: age + proc[m] / 1e3 for m in METH}
    R = dict(tau_ms={m: dict(mean=float(tau[m].mean() * 1e3), p95=float(np.percentile(tau[m], 95) * 1e3), processing_ms=proc[m]) for m in METH},
             sweep_time_ms=T_SWEEP * 1e3, speeds=SPEEDS)
    global CB; CB = {f: P1.codebook(P1.C / (f * 1e9))[0] for f in FR}
    vehs = dict(test=LR.standard_vehicles(len(ant)), varied=LR.varied_vehicles(len(ant)))
    # ---- self-checks: renderer copy reproduces the stored BEVs; net inference reproduces preds_d
    chk = {}
    for sname, path in (("test", B.LB), ("varied", B.LBV)):
        Xs, _ = frames(te[:25], ant, vehs[sname], None)
        with h5py.File(path, "r") as h: ref = h["bev"][te[:25]]
        chk[f"render_{sname}_max_abs_diff"] = float(np.abs(Xs.astype(np.float32) - ref.astype(np.float32)).max())
    with h5py.File(B.LB, "r") as h: Xt = h["bev"][te]
    chk["net_seed11_vs_preds_d_max_abs_m"] = float(np.abs(A5.net_xy(os.path.join(ROOT, "checkpoints", "phase3", "lidar_H_lr0.001_s11.pt"), Xt) - Z["lidar_net_test"][0]).max())
    R["self_checks"] = chk; log("self-checks", chk)
    assert chk["render_test_max_abs_diff"] == 0 and chk["render_varied_max_abs_diff"] == 0 and chk["net_seed11_vs_preds_d_max_abs_m"] < 1e-4, chk
    # ---- static estimates and frozen shapes
    est0 = {s: {m: Z[f"{m}_{s}"] for m in METH} for s in ("test", "varied")}
    shp = {(m, f, M): [B.choose_shape(f, Z[f"{m}_val"][r], M)[0] for r in range(len(Z[f"{m}_val"]))] for m in METH for f in FR for M in MS}
    # ---- E2 earlier frames (per set, speed)
    prev, verr = {}, {}
    for sname in ("test", "varied"):
        for k, v in enumerate(SPEEDS):
            antp = ant.copy(); antp[te, :2] -= v * DT * H[te]
            X, cl = frames(te, antp, vehs[sname], 40_000_000 + 10_000 * k)
            prev[(sname, v)] = estimates(X, cl); del X, cl; log("prev frame", sname, v)
    # ---- channels at the moved positions (shared by both sets: same users, same channel)
    G = {}
    for f in FR:
        for v in SPEEDS:
            for m in METH: G[(f, v, m)] = moved_gains(f, v * tau[m][:, None] * H[te])
            G[(f, v, "sweep")] = moved_gains(f, v * T_SWEEP * H[te])
        log("gains", f)
    adv = {f: B.CH[f]["adv"][te] for f in FR}; xyt = ant[te, :2]
    T, PER, PE, VE, NEAR = {}, {}, {}, {}, {}
    for v in SPEEDS:
        NEAR[str(v)] = {m: int((np.hypot(*(xyt + v * tau[m][:, None] * H[te]).T) < 1.0).sum()) for m in METH}
    for sname in ("test", "varied"):
        for m in METH:
            for v in SPEEDS:
                true = xyt + v * tau[m][:, None] * H[te]
                e0 = est0[sname][m]; vh = (e0 - prev[(sname, v)][m]) / DT
                ests = dict(stale=e0, comp=e0 + vh * tau[m][None, :, None])
                VE[f"{sname}|{m}|{v}"] = pe(vh, v * H[te])
                for mode, ev in ests.items():
                    PE[f"{sname}|{m}|{mode}|{v}"] = pe(ev, true)
                    for f in FR:
                        Gm = G[(f, v, m)]
                        for M in MS:
                            Ls = [B.loss_db(Gm, B.beam_set(ev[r], B.CH[f]["u"], B.CH[f]["rings"], *shp[(m, f, M)][r])) for r in range(len(ev))]
                            PER[(sname, m, mode, f, M, v)] = np.mean(Ls, 0)
                            for sub, msk in (("all", np.ones(len(te), bool)), ("adv_ge3", adv[f] >= 3)):
                                T[f"{sname}|{m}|{mode}|{f}|{M}|{v}|{sub}"] = ci_seed(Ls, msk) if len(Ls) > 1 else ci_user(Ls[0], msk)
            log("losses", sname, m)
    # reference sweep: chosen on the channel at t0, used T_SWEEP later
    SW = {}
    for f in FR:
        G0 = B.CH[f]["G"][te]; Sw = np.c_[np.arange(256)[None].repeat(len(te), 0) * 8, G0[:, 0::8].argmax(1)[:, None] * 8 + np.arange(1, 8)[None]]
        for v in SPEEDS:
            L = B.loss_db(G[(f, v, "sweep")], Sw)
            for sub, msk in (("all", np.ones(len(te), bool)), ("adv_ge3", adv[f] >= 3)): SW[f"{f}|{v}|{sub}"] = float(L[msk].mean())
    # v = 0 bug check against results_d
    diffs = [abs(T[f"{s}|{m}|stale|{f}|{M}|0|{sub}"]["mean"] - RD["tables"][f"{s}|{m}|{f}|{M}|{sub}"]["mean"])
             for s in ("test", "varied") for m in METH for f in FR for M in MS for sub in ("all", "adv_ge3")]
    diffs.append(max(abs(SW[f"{f}|0|{sub}"] - RD["tables"][f"test|sweep263|{f}|263|{sub}"]["mean"]) for f in FR for sub in ("all", "adv_ge3")))
    R["v0_check_max_abs_diff_db"] = float(max(diffs)); log("v0 check", R["v0_check_max_abs_diff_db"])
    # primary read: increase over v = 0 (stale) and compensated minus stale, M = 8, all users
    INC, CMP = {}, {}
    for sname in ("test", "varied"):
        for m in METH:
            for f in FR:
                for v in SPEEDS[1:]:
                    INC[f"{sname}|{m}|{f}|{v}"] = B.boot(PER[(sname, m, "stale", f, 8, v)] - PER[(sname, m, "stale", f, 8, 0)])
                for v in SPEEDS:
                    CMP[f"{sname}|{m}|{f}|{v}"] = B.boot(PER[(sname, m, "comp", f, 8, v)] - PER[(sname, m, "stale", f, 8, v)])
    # break-even speed vs the reference sweep
    BE = {}
    for sname in ("test", "varied"):
        for m in METH:
            for mode in ("stale", "comp"):
                for f in FR:
                    for M in MS:
                        for sub in ("all", "adv_ge3"):
                            vs = [v for v in SPEEDS if T[f"{sname}|{m}|{mode}|{f}|{M}|{v}|{sub}"]["mean"] > SW[f"{f}|{v}|{sub}"]]
                            BE[f"{sname}|{m}|{mode}|{f}|{M}|{sub}"] = vs[0] if vs else None
    R.update(tables=T, sweep263=SW, increase_vs_static_M8=INC, comp_minus_stale_M8=CMP, break_even_speed=BE,
             position_error=PE, speed_error=VE, users_moved_within_1m_of_array=NEAR,
             subset_sizes={str(f): int((adv[f] >= 3).sum()) for f in FR}, runtime_s=time.time() - t0)
    json.dump(R, open(os.path.join(OUTD, "motion.json"), "w"), indent=1); log("done")


if __name__ == "__main__":
    main()
