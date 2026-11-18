"""D-034 A1 (varied-vehicle training labels) and A2 (channel/genie standard vs varied).
Fresh checks: the stored varied geometry and checkpoints are read directly; the renderer is
called as a black box only to look at where the car returns fall.

    python scripts/verify4/a1_a2_varied.py      # -> results/verify4/a1_a2.json
"""
import json, os, sys, inspect
import h5py, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3")); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3b"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2"))
OUT = os.path.join(ROOT, "results", "verify4"); os.makedirs(OUT, exist_ok=True)
P2 = os.path.join(ROOT, "data", "processed", "phase2")


def main():
    R = {}
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: ant = h["ant_xyz"][:]; psi = h["psi"][:]
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: tr = np.sort(h["train"][:]); te = np.sort(h["test"][:])
    with h5py.File(os.path.join(ROOT, "data", "processed", "phase3b", "lidar_bev_var.h5"), "r") as h:
        V = {k: h[k][:] for k in ("ant_xi", "L", "W", "H", "body_lo", "body_hi")}
    hd = np.c_[np.cos(psi), np.sin(psi)]
    # ---- check 1: what the _var checkpoints were trained on
    ck = []
    for sd in range(11, 221, 11):
        c = torch.load(os.path.join(ROOT, "checkpoints", "phase3", f"lidar_H_lr0.001_s{sd}_var.pt"), map_location="cpu", weights_only=False)
        ck.append(dict(seed=sd, lidar_bev=c["args"]["lidar_bev"], head=c["args"]["head"], sensor=c["args"]["sensor"]))
    import train_net
    src = inspect.getsource(train_net.main)
    R["check1_training_target"] = dict(
        label_line=[l.strip() for l in src.splitlines() if "xy = torch.tensor(hf[\"ant_xyz\"]" in l or "Ytr = xy[tr]" in l],
        all_var_checkpoints_use_varied_bev=all(c["lidar_bev"] and c["lidar_bev"].replace("\\", "/").endswith("phase3b/lidar_bev_var.h5") for c in ck),
        checkpoint_bev_paths=sorted({c["lidar_bev"] for c in ck}), n_checkpoints=len(ck))
    # ---- check 2/3: own antenna vs car centre vs template antenna (independent arithmetic)
    xi = V["ant_xi"]; cen = ant[:, :2] - xi[:, None] * hd                      # varied car centre (renderer's rule, recomputed)
    import lidar_var as LR
    cen_r = LR.centre_of(ant, psi, xi)
    own = cen + xi[:, None] * hd; tmpl = cen - 1.3 * hd
    R["check2_own_antenna_equals_label_max_err_m"] = float(np.abs(own - ant[:, :2]).max())
    R["check2_centre_matches_renderer_max_err_m"] = float(np.abs(cen - cen_r).max())
    dt = np.linalg.norm(tmpl - ant[:, :2], axis=1)
    R["check3_template_vs_own_antenna_m"] = dict(matches_abs_xi_plus_1p3=float(np.abs(dt - np.abs(xi + 1.3)).max()),
                                                 train_median=float(np.median(dt[tr])), train_max=float(dt[tr].max()),
                                                 ant_xi_range=[float(xi.min()), float(xi.max())])
    # ---- check 4: car returns surround the car's own centre (render 5 training users)
    five = tr[:5]; clouds = LR.render(five, ant, psi, V | {"cab_lo": None} if False else _veh(V), ch_step=1)
    rows = []
    for k, i in enumerate(five):
        pc = clouds[k]; rel = pc[:, :2] - cen[i]; c, s = hd[i]
        lx = c * rel[:, 0] + s * rel[:, 1]; ly = -s * rel[:, 0] + c * rel[:, 1]
        on = (np.abs(lx) <= V["L"][i] / 2 + 0.05) & (np.abs(ly) <= V["W"][i] / 2 + 0.05) & (pc[:, 2] > 0.05)
        relt = pc[:, :2] - (ant[i, :2] + 1.3 * hd[i]); ltx = c * relt[:, 0] + s * relt[:, 1]; lty = -s * relt[:, 0] + c * relt[:, 1]
        on_t = (np.abs(ltx) <= 2.25 + 0.05) & (np.abs(lty) <= 0.9 + 0.05) & (pc[:, 2] > 0.05)
        q = pc[on]
        rows.append(dict(user=int(i), heading_deg=float(np.rad2deg(psi[i]) % 360), L=float(V["L"][i]), W=float(V["W"][i]), H=float(V["H"][i]),
                         ant_xi_m=float(xi[i]), antenna_label_xy=ant[i, :2].round(4).tolist(), own_antenna_xy=own[i].round(4).tolist(),
                         car_centre_xy=cen[i].round(4).tolist(), template_antenna_xy=tmpl[i].round(4).tolist(),
                         template_minus_own_m=float(dt[i]), returns_in_own_box=int(on.sum()), returns_in_template_box=int(on_t.sum()),
                         own_box_returns_centroid_minus_centre_m=(q[:, :2].mean(0) - cen[i]).round(3).tolist() if len(q) else None))
    R["check4_five_training_users"] = rows
    R["A1_verdict"] = "PASS" if (R["check1_training_target"]["all_var_checkpoints_use_varied_bev"] and R["check2_own_antenna_equals_label_max_err_m"] < 1e-9
                                 and all(r["returns_in_own_box"] > 0 for r in rows)) else "FAIL"
    # ---- A2: channel inputs and genie
    import p1_channels as CH
    sig = inspect.signature(CH.channel); R["A2_channel_function_args"] = list(sig.parameters)
    R["A2_draw_uses_vehicle_geometry"] = any(w in inspect.getsource(CH.draw) for w in ("ant_xi", "BODY", "body", "CABIN"))
    R["A2_note"] = ("Both test sets index the same channels_{f}GHz.h5 gains by user id; the varied set changes only the LiDAR BEV. "
                    "Channel = LoS to the antenna point + 2-4 point reflectors; no car body (blockage/scattering) in the channel.")
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: d = h["d"][:]
    sw = {}
    for f in (60, 140):
        with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as h: G = h["gains"][te]; adv = h["adv_db"][te]
        best_ang = G[:, 0::8].argmax(1)
        S = np.c_[np.arange(256)[None].repeat(len(te), 0) * 8, best_ang[:, None] * 8 + np.arange(1, 8)[None]]
        L = 10 * np.log10(G.max(1) / np.take_along_axis(G, S, 1).max(1)); d2 = np.hypot(*ant[te, :2].T)
        sw[str(f)] = {k: dict(mean_db=float(L[m].mean()), n=int(m.sum())) for k, m in (("all", np.ones(len(te), bool)), ("adv_ge3", adv >= 3), ("d_ge_2m", d2 >= 2))}
    R["A2_sweep263_varied_test"] = sw
    R["A2_verdict"] = "IDENTICAL (antenna-anchored; the varied set re-renders only the LiDAR)" if not R["A2_draw_uses_vehicle_geometry"] else "DIFFER"
    json.dump(R, open(os.path.join(OUT, "a1_a2.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "check4_five_training_users"}, indent=1))
    for r in rows: print(r)


def _veh(V):
    """Rebuild the full per-user varied dict (cabin boxes) exactly as lidar_var does."""
    import lidar_var as LR
    full = LR.varied_vehicles(len(V["ant_xi"]))
    assert np.array_equal(full["ant_xi"], V["ant_xi"]) and np.array_equal(full["L"], V["L"]), "stored varied geometry != generator"
    return full


if __name__ == "__main__":
    main()
