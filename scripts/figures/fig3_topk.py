"""Fig 3 — radar top-K shortlist: near-field coverage and dB-loss vs K (60/140 GHz).

Source: results/indomain/radar_closerange_{60,140}GHz_seed*_metrics.json
        coverage.<K>.nf  and  dbloss.<K>.nf  for K in {1,2,4,8}.
Markers = mean over 5 seeds; error bars = 95% CI (src/eval/statistics.mean_ci).
Shows that a small RF verification sweep (K) recovers the near-field genie:
coverage rises toward 1, dB-loss falls toward 0. JSON-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paperkit import Audit, SEEDS, metrics_path, save_fig, setup_mpl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.statistics import mean_ci

KS = [1, 2, 4, 8]
FREQS = [("60", "60 GHz", "#1f77b4", "o"), ("140", "140 GHz", "#d62728", "s")]


import os as _os
from _paperkit import RESULTS
_N20 = _os.path.join(RESULTS, "indomain_n20", "close_n20.json")
_USE_N20 = _os.path.exists(_N20)
_KEY = {"coverage": "cov", "dbloss": "db"}   # close_n20 field names


def series(aud: Audit, fghz: str, field: str):
    """radar top-K series, n=20 (close_n20) if banked else n=5 (per-seed metrics)."""
    means, los, his = [], [], []
    for k in KS:
        if _USE_N20:
            vals = aud.arr(_N20, f"results.{fghz}.radar.{_KEY[field]}{k}_nf")
        else:
            vals = [aud.val(metrics_path("radar", f"closerange_{fghz}GHz", s),
                            f"{field}.{k}.nf") for s in SEEDS]
        c = mean_ci(vals)
        means.append(c["mean"]); los.append(c["mean"] - c["lo"]); his.append(c["hi"] - c["mean"])
    return means, [los, his]


def main():
    plt = setup_mpl()
    aud = Audit("fig3_topk")
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(7.16, 2.7))   # IEEE double-column
    for fghz, label, color, mk in FREQS:
        m, err = series(aud, fghz, "coverage")
        axc.errorbar(KS, m, yerr=err, marker=mk, color=color, capsize=2, lw=1.3,
                     markersize=4, label=label)
        m, err = series(aud, fghz, "dbloss")
        axd.errorbar(KS, m, yerr=err, marker=mk, color=color, capsize=2, lw=1.3,
                     markersize=4, label=label)
    for ax in (axc, axd):
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_xlabel("Shortlist size $K$"); ax.legend()
    axc.set_ylabel("Near-field top-$K$ coverage"); axc.set_ylim(0, 1.0)
    axc.set_title("(a) Genie coverage vs $K$")
    axd.set_ylabel("Near-field beam-focusing loss (dB)"); axd.set_ylim(bottom=0)
    axd.axhline(0, color="0.4", lw=0.7); axd.set_title("(b) dB-loss vs genie")
    fig.tight_layout()
    save_fig(fig, "fig3_topk_coverage_dbloss")
    aud.write()


if __name__ == "__main__":
    main()
