"""D-034 A4: position-error definitions. Lists the definition each report uses and re-expresses
the Phase 2b required-accuracy sweep (E2) in the horizontal Euclidean error. The injected
errors are regenerated exactly (default_rng(2032), shape (6 sigmas, 6000 users, 2)); the run
must first reproduce the banked per-axis medians.

    python scripts/verify4/a4_error_definition.py   # -> results/verify4/a4.json
"""
import json, os
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P2 = os.path.join(ROOT, "data", "processed", "phase2")
SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

DEFINITIONS = {
    "results/phase2/feasibility/e23_sweep.json (Phase 2b E2/E3)": "per-axis: |range| and |sideways| components (median_abs, p90_abs, rms); sigma = per-axis std",
    "results/phase3b/results_b.json position_error": "per-axis: |range| and |sideways| medians / p90",
    "results/phase3b/results_b.json stress err_median_m": "Euclidean horizontal median",
    "results/phase3c/c1.json position_error": "Euclidean horizontal (euclid) plus range and sideways components",
    "results/phase3c/position_error_cdf.json": "Euclidean horizontal quantiles",
    "paper_assets_v2 canonical_macros_v2.tex PosErr*RangeCm/SideCm": "per-axis medians (to be replaced)",
    "paper_assets_v2 canonical_macros_v2.tex CPosErr*OnVariedCm": "Euclidean median",
    "paper_assets_v2 fig_required_accuracy.pdf": "per-axis: mean of range and sideways medians (curve and marks; axis labelled per-axis)",
    "paper_assets_v2 tab_varied.tex / tab_stress.tex / fig_position_cdf.pdf": "Euclidean horizontal",
}


def main():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as h: te = np.sort(h["test"][:])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: n = len(h["d"])
    xy = None
    with h5py.File(os.path.join(P2, "users.h5"), "r") as h: xy = h["ant_xyz"][:, :2]
    noise = np.random.default_rng(2032).standard_normal((len(SIGMAS), n, 2))
    banked = json.load(open(os.path.join(ROOT, "results", "phase2", "feasibility", "e23_sweep.json")))["errors"]
    rad = xy[te] / np.linalg.norm(xy[te], axis=1, keepdims=True); perp = np.c_[-rad[:, 1], rad[:, 0]]
    out, worst = {}, 0.0
    for s, sg in enumerate(SIGMAS):
        e = sg * noise[s, te]; k = f"sigma{sg:g}"
        rng_m = float(np.median(np.abs((e * rad).sum(1)))); side_m = float(np.median(np.abs((e * perp).sum(1))))
        worst = max(worst, abs(rng_m - banked[k]["range"]["median_abs"]), abs(side_m - banked[k]["sideways"]["median_abs"]))
        eu = np.hypot(*e.T)
        out[k] = dict(sigma_per_axis_m=sg, per_axis_mean_of_medians_m=(rng_m + side_m) / 2,
                      euclid_median_m=float(np.median(eu)), euclid_p90_m=float(np.percentile(eu, 90)))
    rep = dict(definitions=DEFINITIONS, reproduction_max_abs_diff_m=worst, reproduced=worst < 1e-12, e2_euclidean=out,
               rule="Paper assets: horizontal Euclidean error |p_hat - p| (median, p90) everywhere; per-axis only as labelled supplements.")
    json.dump(rep, open(os.path.join(ROOT, "results", "verify4", "a4.json"), "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
