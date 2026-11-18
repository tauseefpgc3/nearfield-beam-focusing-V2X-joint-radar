"""Print the C1 tables from results/phase3c/c1.json; also writes results/phase3c/c1_tables.md."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(os.path.join(ROOT, "results", "phase3c", "c1.json"))); out = []
NM = dict(lidar_geometric="Geometric", single_vehicle_net="LiDAR net, single vehicle", varied_trained_net="LiDAR net, varied-trained")
H = R["headline"]; out.append(f"## Headline (varied test, all users): {H['verdict']}\n\n| f, M | bootstrap Δ [CI] | seed Δ [CI] | Holm p | Wilcoxon p |\n|---|---|---|---|---|")
for k, v in H["cells"].items():
    b, s = v["bootstrap"], v["seed_delta"]; out.append(f"| {k} | {b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] | {s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] | {v['p_t_holm']:.1e} | {v['p_wilcoxon']:.1e} |")
f2 = lambda c: f"{c['mean']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"
for ts in ("varied", "standard"):
    for sub in ("all", "adv_ge3", "d_ge_2m"):
        for f in (60, 140):
            n = R["subset_sizes"][str(f)][sub]
            out.append(f"\n### {ts} test set, {f} GHz, {sub} (n = {n}): loss vs genie, dB [95% CI]\n\n| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |\n|---|---|---|---|---|---|")
            for m, nm in NM.items():
                c = [R["tables"][f"{ts}|{m}|{f}|{M}|{sub}"] for M in (8, 16, 32, 64)]
                out.append(f"| {nm} | " + " | ".join(f2(x) for x in c) + f" | {c[0]['within_1dB']:.2f} |")
    out.append(f"\n### {ts} test set: horizontal position error (cm), median / p90\n\n| Method | all | adv ≥ 3 dB (60 GHz) | ≥ 2 m |\n|---|---|---|---|")
    for m, nm in NM.items():
        P = R["position_error"][f"{ts}|{m}"]
        out.append(f"| {nm} | " + " | ".join(f"{P[s]['euclid']['median']*100:.1f} / {P[s]['euclid']['p90']*100:.1f}" for s in ("all", "adv_ge3", "d_ge_2m")) + " |")
out.append("\n## Secondary: varied-trained − single-vehicle (bootstrap Δ dB [CI], verdict), all users\n\n| test set | f, M | Δ | verdict |\n|---|---|---|---|")
for ts in ("varied", "standard"):
    for f in (60, 140):
        for M in (8, 16, 32, 64):
            c = R["secondary"][f"{ts}|{f}|{M}|all"]["varied_vs_single"]; out.append(f"| {ts} | {f}, {M} | {c['mean']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] | {c['verdict']} |")
t = "\n".join(out); open(os.path.join(ROOT, "results", "phase3c", "c1_tables.md"), "w", encoding="utf8").write(t); print(t)
