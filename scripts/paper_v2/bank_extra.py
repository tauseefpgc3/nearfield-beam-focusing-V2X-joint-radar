"""D-033 C3: bank the derived inputs of the paper assets as JSON (read by build_assets.py).

    python scripts/paper_v2/bank_extra.py [setup] [nearfield] [cdf]

setup     -> results/phase3c/setup_params.json   (constants read from the frozen generator code + split)
nearfield -> results/phase3c/nearfield_char.json  (share of users with adv >= 3 dB vs distance; beam-map summary)
             results/phase3c/nearfield_beam_map.json (normalised gain maps for the ULA illustration)
cdf       -> results/phase3c/position_error_cdf.json (horizontal position-error quantiles per method)
"""
import json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("phase2", "phase3", "phase3b"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
RES = os.path.join(ROOT, "results", "phase3c"); P2 = os.path.join(ROOT, "data", "processed", "phase2")
C = 299_792_458.0


def dump(name, obj):
    json.dump(obj, open(os.path.join(RES, name), "w"), indent=1); print("wrote", name)


def setup():
    import p1_channels as CHN, p1_radar as RD, lidar_v2 as LV
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: n = {k: int(h[k].shape[0]) for k in ("train", "val", "test")}
    fr = {}
    for f in (60, 140):
        lam = C / (f * 1e9); D = (CHN.N - 1) * lam / 2
        rings = [float(x) for x in np.r_[np.inf, 1.2 * (CHN.N - 1) ** 2 * lam / 8 / np.arange(1, CHN.B_R)]]
        fr[str(f)] = dict(wavelength_mm=lam * 1e3, aperture_m=D, fraunhofer_m=2 * D ** 2 / lam, ring_radii_m=rings[1:])
    out = dict(array=dict(elements=CHN.N, spacing="half wavelength", height_m=CHN.H_RSU, axis="y"),
               codebook=dict(angles=CHN.B_TH, rings=CHN.B_R, size=CHN.B_TH * CHN.B_R, sweep_reference_beams=CHN.B_TH + CHN.B_R - 1),
               users=dict(total=CHN.N_USERS, antenna_height_m=CHN.H_VEH, distance_min_m=1.0, distance_max_m=20.0, angle_max_deg=60.0,
                          reflections_min=2, reflections_max=4, split=n, guard_dropped=CHN.N_USERS - sum(n.values())),
               frequencies=fr,
               lidar=dict(channels=int(len(LV.EL)), elevation_min_deg=-25.0, elevation_max_deg=15.0, azimuth_step_deg=0.2,
                          range_noise_m=LV.SIG_R, dropout=LV.P_DROP, sensor_height_m=float(LV.SENSOR[2]), bev_cell_m=LV.CELL,
                          bev_cells_x=LV.NX, bev_cells_y=LV.NY, body_length_m=4.5, body_width_m=1.8, antenna_offset_m=1.3),
               radar=dict(carrier_ghz=77, virtual_elements=RD.N_V, chirps=RD.N_C, samples=RD.N_S, range_bin_m=RD.DR,
                          angle_bins=int(len(RD.ANG)), vehicle_scatterers=6, clutter=12, snr_db=list(RD.SNRS)),
               training=dict(optimizer="AdamW", lr=1e-3, weight_decay=1e-4, epochs=40, batch=16, schedule="cosine", seeds=20))
    dump("setup_params.json", out)


def nearfield():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: kept = np.sort(np.r_[h["train"][:], h["val"][:], h["test"][:]])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: d = h["d"][:][kept]
    edges = np.arange(1, 21, 1.0); out = dict(distance_bin_edges_m=edges.tolist(), users=int(len(kept)))
    for f in (60, 140):
        with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as h: adv = h["adv_db"][:][kept]
        k = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
        out[str(f)] = dict(share_adv_ge3=[float((adv[k == b] >= 3).mean()) for b in range(len(edges) - 1)],
                           mean_adv_db=[float(adv[k == b].mean()) for b in range(len(edges) - 1)],
                           users_per_bin=[int((k == b).sum()) for b in range(len(edges) - 1)],
                           share_adv_ge3_all=float((adv >= 3).mean()), mean_adv_db_all=float(adv.mean()))
    dump("nearfield_char.json", out)
    # beam illustration: 60 GHz, 128-element ULA at (0, 0, 2), plane z = 1.5 m; focus point at 3 m, 20 deg
    import p1_channels as CHN
    f = 60; lam = C / (f * 1e9); F, u, rings = CHN.codebook(lam); y = CHN.elem_y(lam)
    r0, th0 = 3.0, np.deg2rad(20.0); i = int(np.argmin(np.abs(u - np.sin(th0)))); j = int(np.argmin(np.abs(1 / rings - 1 / r0)))
    xs = np.linspace(0.25, 8.0, 311); ys = np.linspace(-1.0, 4.0, 201)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); q0 = np.array([0, 0, CHN.H_RSU])
    dist = np.sqrt(X[..., None] ** 2 + (Y[..., None] - y) ** 2 + (1.5 - q0[2]) ** 2)
    a = np.exp(-1j * 2 * np.pi / lam * dist) / np.sqrt(len(y))                    # steering to each point
    raw = {nm: np.abs(a @ F[k].conj()) ** 2 for nm, k in (("far_field", i * 8), ("focused", i * 8 + j))}
    ref = max(v.max() for v in raw.values())                                    # common normalisation
    maps = {nm: 10 * np.log10(np.maximum(v / ref, 1e-6)) for nm, v in raw.items()}
    # -3 dB depth of the focused beam along the steering ray
    rr = np.linspace(0.5, 30, 3000); pts = np.c_[rr * np.cos(th0), rr * np.sin(th0)]
    dd = np.sqrt(pts[:, :1] ** 2 + (pts[:, 1:] - y) ** 2 + 0.25); ar = np.exp(-1j * 2 * np.pi / lam * dd) / np.sqrt(len(y))
    gr = np.abs(ar @ F[i * 8 + j].conj()) ** 2; gr /= gr.max(); above = rr[gr >= 0.5]
    gff = np.abs(ar @ F[i * 8].conj()) ** 2
    pk = int(np.argmin(np.abs(rr - r0)))
    summ = dict(frequency_ghz=f, focus_range_m=r0, focus_angle_deg=20.0, codeword_angle_index=i, codeword_ring_index=j,
                ring_radius_m=float(rings[j]), depth_3db_min_m=float(above.min()), depth_3db_max_m=float(above.max()),
                gain_at_focus_far_field_over_focused_db=float(10 * np.log10(gff[pk] / (np.abs(ar[pk] @ F[i * 8 + j].conj()) ** 2))))
    o = json.load(open(os.path.join(RES, "nearfield_char.json"))); o["beam_illustration"] = summ; dump("nearfield_char.json", o)
    dump("nearfield_beam_map.json", dict(x_m=xs.round(4).tolist(), y_m=ys.round(4).tolist(),
                                         **{k: np.round(v, 2).tolist() for k, v in maps.items()}, **summ))


def cdf():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: te = np.sort(h["test"][:])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: xy = h["ant_xyz"][:, :2][te]
    pc = np.load(os.path.join(ROOT, "results", "phase3b", "preds_clean.npz")); c1 = np.load(os.path.join(RES, "preds_c1.npz"))
    q = np.r_[np.arange(0, 100, 1.0), 99.5, 100.0]; out = dict(quantiles=q.tolist(), standard={}, varied={})
    def qs(pt): e = np.hypot(*(pt.reshape(-1, 2) - np.tile(xy, (len(pt), 1))).T); return np.percentile(e, q).tolist()
    for nm in ("radar_picker", "lidar_geometric", "radar_net", "lidar_net", "fused_net"): out["standard"][nm] = qs(pc[f"{nm}_test"])
    out["standard"]["varied_trained_net"] = qs(c1["varied_net_test_standard"])
    out["varied"]["lidar_geometric"] = qs(c1["geometric_test_varied"][None]); out["varied"]["single_vehicle_net"] = qs(c1["single_net_test_varied"])
    out["varied"]["varied_trained_net"] = qs(c1["varied_net_test_varied"])
    dump("position_error_cdf.json", out)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    for a in (sys.argv[1:] or ["setup", "nearfield", "cdf"]): globals()[a]()
