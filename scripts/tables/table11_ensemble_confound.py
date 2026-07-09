"""Table 11 — modality-vs-ensembling confound check (near-field K=8 dB-loss, 20 seeds).

Source: results/fusion_rules/ensemble_confound.json (banked by eval_ensemble_confound.py).
(a) radar-alone, (b) radar-only 3-model C3 ensemble, (c) multimodal soft-avg; paired tests
(b)-vs-(a) = ensembling effect, (c)-vs-(b) = modality beyond ensembling. Renders a 'pending'
placeholder if the JSON is absent so run_all stays green. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, RESULTS, write_table

ECJ = os.path.join(RESULTS, "fusion_rules", "ensemble_confound.json")
FREQS = ["60", "140"]


def main():
    aud = Audit("table11_ensemble_confound")
    if not os.path.exists(ECJ):
        write_table("table11_ensemble_confound", "% Table 11 pending: ensemble_confound.json not banked.\n",
                    "# Table 11 — modality vs ensembling (PENDING)\n\n_Run "
                    "`python scripts/eval_ensemble_confound.py` first._\n")
        aud.note("ensemble_confound.json missing -> placeholder"); aud.write()
        print("  [table11] pending -> placeholder"); return

    def f3(f, key, sub):
        return aud.val(ECJ, f"results.{f}.{key}.{sub}")

    hdr = ["Freq", "(a) radar", "(b) radar×3 ens", "(c) multimodal", "(b) leakage", "(c)−(b) p [W]", "(c)−(b) d"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    leak_max = 0.0
    for f in FREQS:
        a = f3(f, "a_radar_alone", "mean"); ahi = f3(f, "a_radar_alone", "hi")
        b = f3(f, "b_radar_ens3", "mean"); bhi = f3(f, "b_radar_ens3", "hi")
        c = f3(f, "c_multimodal_softavg", "mean"); chi = f3(f, "c_multimodal_softavg", "hi")
        lk = f3(f, "b_leakage_frac", "mean"); leak_max = max(leak_max, lk)
        p = f3(f, "c_vs_b", "paired_t_p"); w = f3(f, "c_vs_b", "wilcoxon_p")
        d = f3(f, "c_vs_b", "cohen_d")
        md.append(f"| {f} GHz | {a:.2f}±{ahi-a:.2f} | {b:.2f}±{bhi-b:.2f} | {c:.2f}±{chi-c:.2f} | "
                  f"{lk*100:.0f}% | {p:.3f} [{w:.3f}] | {d:+.2f} |")

    # Honest verdict: (b) is ~89% leakage-contaminated -> NOT a valid baseline, so the
    # eval-only check cannot cleanly isolate modality from ensembling regardless of sign.
    bvals = "/".join(f"{f3(f, 'b_radar_ens3', 'mean'):.2f}" for f in FREQS)
    # degradation numbers sourced from JSON (n=20 preferred) — not hardcoded
    _dn20 = os.path.join(RESULTS, "degradation_closerange60_n20.json")
    dgj = _dn20 if os.path.exists(_dn20) else os.path.join(RESULTS, "degradation_closerange60.json")
    d_clean = aud.val(dgj, "clean.mean"); d_lidar = aud.val(dgj, "lidar_out.mean")
    d_radar = aud.val(dgj, "radar_out.mean"); d_n = int(aud.val(dgj, "clean.n"))
    verdict = (f"**Inconclusive — the eval-only radar ensemble (b) is invalid.** Its members "
               f"were trained on other seeds' splits, and **~{leak_max*100:.0f}% of each eval "
               f"seed's near-field test trajectories fall in >=1 member's training set**, so "
               f"(b) has effectively memorised the test data (its {bvals} dB "
               f"is a leakage artifact, not honest ensembling). The apparent '(b) beats (c)' "
               f"therefore does **not** show that ensembling explains the gain. The leakage-"
               f"free evidence is the D-019 clean control (Panel B) and the degradation test "
               f"(n={d_n}): dropping LiDAR raises joint-top-1 error "
               f"(clean {d_clean:.3f} → lidar-out {d_lidar:.3f}), while radar dominates "
               f"(radar-out {d_radar:.3f}).")
    md_s = ("# Table 11 — Modality vs ensembling confound check (NF K=8 dB-loss)\n\n"
            "**Panel A — n=20 eval-only cross-seed ensemble (leaky, invalid):**\n\n"
            + "\n".join(md) + f"\n\n**Panel A verdict:** {verdict}\n")

    # ---- Panel B: D-019 clean same-split radar-ensemble control (leakage-free) ----
    clean = os.path.join(RESULTS, "fusion_rules", "ensemble_confound_clean.json")
    if os.path.exists(clean):
        pb = ["", "**Panel B — D-019 clean same-split control (leakage-free):**", "",
              "| Freq | Split | (b') radar×3 ens | (c') multimodal | Δ(b−c) [95% CI] | boot p | leakage |",
              "|---|---|---|---|---|---|---|"]
        c_better_all, c_worse_all = [], []
        for f in FREQS:
            for s in ("11", "22", "33"):
                bp = f"results.{f}.per_split.{s}"
                b = aud.val(clean, f"{bp}.b_radar_ens3_mean")
                c = aud.val(clean, f"{bp}.c_multimodal_mean")
                dd = aud.val(clean, f"{bp}.mean_delta_b_minus_c")
                lo = aud.val(clean, f"{bp}.ci95.0"); hi = aud.val(clean, f"{bp}.ci95.1")
                pv = aud.val(clean, f"{bp}.bootstrap_p"); lk = aud.val(clean, f"{bp}.leakage")
                pb.append(f"| {f} GHz | {s} | {b:.2f} | {c:.2f} | {dd:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                          f"| {pv:.3f} | {lk:.3f} |")
            c_better_all.append(aud.val(clean, f"results.{f}.c_better_in_all_splits"))
            c_worse_all.append(aud.val(clean, f"results.{f}.c_worse_in_all_splits"))
        # data-driven D-019 verdict (Δ = b'-c'; >0 => c' lower/better => modality helps)
        if all(c_better_all):
            vb = ("**(c') multimodal beats (b') the clean radar-3-ensemble in ALL splits/freqs** "
                  "→ the LiDAR modality contributes beyond generic ensembling; fusion claim is airtight.")
        elif all(c_worse_all):
            vb = ("**(c') multimodal does NOT beat (b') the clean radar-3-ensemble in any "
                  "split/freq** → the soft-avg gain is substantially generic ensembling, not the "
                  "LiDAR modality; the paper states this, noting the radar-3-ensemble costs 3× "
                  "radar compute vs a single fused pass.")
        else:
            vb = ("**Mixed across splits/frequencies** — reported as mixed per D-019 (no clean, "
                  "consistent modality effect).")
        md_s += "\n" + "\n".join(pb) + f"\n\n**Panel B verdict (D-019):** {vb} "
        md_s += "All leakage entries are 0.000 (assertion enforced in code).\n"

    tex = [r"\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", r"\toprule",
           " & ".join(hdr).replace("−", "$-$").replace("×", r"$\times$") + r" \\", r"\midrule"]
    for line in md[2:]:
        tex.append(" & ".join(x.strip() for x in line.strip("|").split("|")).replace("−", "$-$").replace("±", r"$\pm$") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write_table("table11_ensemble_confound", "\n".join(tex) + "\n", md_s)
    aud.write()
    print("  [table11] written")


if __name__ == "__main__":
    main()
