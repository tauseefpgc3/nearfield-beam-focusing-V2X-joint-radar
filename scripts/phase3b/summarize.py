"""Print the D-032 tables from results/phase3b/results_b.json (+ cost.json) as markdown;
also writes results/phase3b/tables.md."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(os.path.join(ROOT, "results", "phase3b", "results_b.json")))
NM = dict(radar_picker="Radar picker", lidar_geometric="LiDAR geometric", radar_net="Radar net", lidar_net="LiDAR net", fused_net="Fused net")
MS = (8, 16, 32, 64); out = []
fmt = lambda c: f"{c['mean']:.2f}" + (f" [{c['lo']:.2f}, {c['hi']:.2f}]" if "lo" in c else "")
P = R["primary"]; out.append(f"## B1 primary endpoint: {P['verdict']}\n")
out.append("| f, M | bootstrap Δ (dB) [95% CI] | seed-level Δ mean [CI] | t-test p (Holm) | Wilcoxon p |\n|---|---|---|---|---|")
for k, v in P["cells"].items():
    b = v["bootstrap"]; s = v["seed_delta"]
    out.append(f"| {k} | {b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] | {s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] | {v['p_t']:.2g} ({v['p_t_holm']:.2g}) | {v['p_wilcoxon']:.2g} |")
for sub in ("all", "near_field", "adv_ge3", "all|d_ge_2m", "near_field|d_ge_2m", "adv_ge3|d_ge_2m"):
    for f in (60, 140):
        n = R["subset_sizes"][str(f)][sub]
        out.append(f"\n### Loss vs genie (dB), {f} GHz, subset {sub} (n = {n}); 263-beam sweep {R['tables'][f'sweep263|{f}|{sub}']:.2f}\n")
        out.append("| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8/16) |\n|---|---|---|---|---|---|")
        for m in NM:
            c = [R["tables"][f"{m}|{f}|{M}|{sub}"] for M in MS]
            out.append(f"| {NM[m]} | " + " | ".join(fmt(x) for x in c) + f" | {c[0]['within_1dB']:.2f} / {c[1]['within_1dB']:.2f} |")
out.append("\n## Position error (m), median / p90\n\n| Method | range (all) | sideways (all) | range (d≥2 m) | sideways (d≥2 m) |\n|---|---|---|---|---|")
for m in NM:
    e = R["position_error"][m]
    out.append(f"| {NM[m]} | " + " | ".join(f"{e[s][a]['median']:.3f} / {e[s][a]['p90']:.2f}" for s in ("all", "d_ge_2m") for a in ("range", "sideways")) + " |")
out.append("\n## Bootstraps (Δ dB [CI], verdict), all users\n\n| f, M | radar net − picker | LiDAR net − geometric | fused − best single |\n|---|---|---|---|")
for f in (60, 140):
    for M in MS:
        b = R["bootstrap"][f"{f}|{M}|all"]
        cell = lambda x: f"{x['mean']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] {x['verdict'].replace('network ', '')}"
        out.append(f"| {f}, {M} | {cell(b['radar_net_vs_radar_picker'])} | {cell(b['lidar_net_vs_lidar_geometric'])} | {cell(b['fused_net_vs_best_single'])} ({b['fused_net_vs_best_single']['best_single']}) |")
for sub in ("adv_ge3", "all|d_ge_2m"):
    out.append(f"\n### Bootstrap verdicts, subset {sub}\n\n| f, M | radar net − picker | LiDAR net − geometric | fused − best single |\n|---|---|---|---|")
    for f in (60, 140):
        for M in MS:
            b = R["bootstrap"][f"{f}|{M}|{sub}"]
            out.append(f"| {f}, {M} | " + " | ".join(f"{b[k]['mean']:+.2f} {b[k]['verdict'].replace('network ', '')}" for k in ("radar_net_vs_radar_picker", "lidar_net_vs_lidar_geometric", "fused_net_vs_best_single")) + " |")
out.append("\n## B3 value of range focusing: Δ_focus = far-field-only − full (dB), all users\n\n| Method | 60: M=8 | 16 | 32 | 64 | 140: M=8 | 16 | 32 | 64 |\n|---|---|---|---|---|---|---|---|---|")
for m in NM:
    out.append(f"| {NM[m]} | " + " | ".join(f"{R['far_field_only'][f'{m}|{f}|{M}|all']['delta_focus']['mean']:+.2f}" for f in (60, 140) for M in MS) + " |")
out.append(f"| genie far-field-only loss (= mean adv) | {R['far_field_only']['genie_ff_only|60|all']:.2f} | | | | {R['far_field_only']['genie_ff_only|140|all']:.2f} | | | |")
out.append("\n### B3, adv ≥ 3 dB subset\n\n| Method | 60: M=8 | 16 | 32 | 64 | 140: M=8 | 16 | 32 | 64 |\n|---|---|---|---|---|---|---|---|---|")
for m in NM:
    out.append(f"| {NM[m]} | " + " | ".join(f"{R['far_field_only'][f'{m}|{f}|{M}|adv_ge3']['delta_focus']['mean']:+.2f}" for f in (60, 140) for M in MS) + " |")
def stress_rows(title, D):
    out.append(f"\n## {title}\n\n| Setting | Method | 60: M=8 | 16 | 32 | 64 | 140: M=8 | 16 | 32 | 64 | err med (m) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for st, row in D.items():
        for m, c in row.items():
            out.append(f"| {st} | {m} | " + " | ".join(f"{c[f'{f}|{M}']['mean']:.2f}" for f in (60, 140) for M in MS) + f" | {c['err_median_m']:.3f} |")
stress_rows("B4 stress tests (frozen models; loss vs genie, dB, all users)", R["stress"])
stress_rows("3a varied vehicles (varied test set)", {"varied": R["varied_vehicles"] and {k: v for k, v in R["varied_vehicles"].items() if not k.startswith("reference")}})
stress_rows("3b antenna-offset error δ (true offset 1.3+δ m; estimator assumes 1.3 m)", R["antenna_offset_error"])
cp = os.path.join(ROOT, "results", "phase3b", "cost.json")
if os.path.exists(cp):
    C = json.load(open(cp))
    out.append("\n## B5 cost (per frame, batch 1, median ms)\n\n| Method | parameters | preprocessing GPU | preprocessing CPU | inference GPU | inference CPU |\n|---|---|---|---|---|---|")
    lb, rp = C["lidar_bev_ms"], C["radar_preprocess_ms"]
    out.append(f"| Radar net | {C['radar_net']['parameters']:,} | {rp['cuda']:.1f} | {rp['cpu']:.1f} | {C['radar_net']['inference_ms_cuda']:.1f} | {C['radar_net']['inference_ms_cpu']:.0f} |")
    out.append(f"| LiDAR net | {C['lidar_net']['parameters']:,} | {lb['gpu_torch']:.1f} | {lb['cpu_numpy']:.1f} | {C['lidar_net']['inference_ms_cuda']:.1f} | {C['lidar_net']['inference_ms_cpu']:.0f} |")
    out.append(f"| Fused net | {C['fused_net']['parameters']:,} | {lb['gpu_torch'] + rp['cuda']:.1f} | {lb['cpu_numpy'] + rp['cpu']:.1f} | {C['fused_net']['inference_ms_cuda']:.1f} | {C['fused_net']['inference_ms_cpu']:.0f} |")
    out.append(f"| LiDAR geometric | 0 (7 tuned thresholds) | – | – | – | {C['geometric_estimator_cpu_ms']:.1f} |")
    out.append(f"| Radar picker | 0 | {rp['cuda']:.1f} (RA map) | {rp['cpu']:.1f} (RA map) | – | {C['radar_picker_cpu_ms']:.1f} |")
txt = "\n".join(out); open(os.path.join(ROOT, "results", "phase3b", "tables.md"), "w", encoding="utf8").write(txt); print(txt)
