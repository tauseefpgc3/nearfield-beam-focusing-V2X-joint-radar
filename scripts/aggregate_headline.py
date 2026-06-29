"""Aggregate the headline sweep into the main results table + statistics (Batch 8).

Reads results/indomain/{modality}_{dtag}_seed{seed}_metrics.json across seeds and reports:
  - close-range headline table (radar/LiDAR/fusion x {60,140}): mean +/- 95% CI
  - paired t-test radar vs fusion on K=8 NF dB-loss (+ Cohen's d), per frequency
  - O1 far-field anchor (radar): polar-vs-DFT genie gap (does-no-harm) + top-1
  - the more-info-no-gain honesty note

    python scripts/aggregate_headline.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.statistics import mean_ci, paired_t_test

IND = os.path.join(_ROOT, "results", "indomain")


def load(modality, dtag):
    """Return {seed: metrics} for a (modality, dtag)."""
    out = {}
    for p in glob.glob(os.path.join(IND, f"{modality}_{dtag}_seed*_metrics.json")):
        m = json.load(open(p))
        out[m["seed"]] = m
    return out


def col(runs, path):
    """Extract a metric (dotted path) across seeds, sorted by seed."""
    vals = []
    for seed in sorted(runs):
        v = runs[seed]
        for key in path.split("."):
            v = v[key]
        vals.append(v)
    return np.array(vals, float)


def fmt(runs, path):
    if not runs:
        return "   n/a    "
    c = mean_ci(col(runs, path))
    return f"{c['mean']:.3f}±{(c['hi']-c['mean']):.3f}"


def main():
    print("=" * 96)
    print("HEADLINE TABLE — close-range PRIMARY (mean ± 95% CI over seeds; range/gain on "
          "near-field y_r>0)")
    print("=" * 96)
    hdr = (f"{'freq':>5} {'model':>7} | {'top1_th':>11} {'top1_rng':>11} {'rng_nf':>11} | "
           f"{'cov@4nf':>11} {'K4_NFdB':>11} {'K8_NFdB':>11}")
    for f in (60, 140):
        print(f"\n-- {f} GHz --")
        print(hdr)
        for model in ("radar", "lidar", "fusion"):
            r = load(model, f"closerange_{f}GHz")
            n = len(r)
            print(f"{f:>5} {model:>7} | {fmt(r,'top1_theta'):>11} {fmt(r,'top1_range'):>11} "
                  f"{fmt(r,'range_nf'):>11} | {fmt(r,'coverage.4.nf'):>11} "
                  f"{fmt(r,'dbloss.4.nf'):>11} {fmt(r,'dbloss.8.nf'):>11}  (n={n})")

    print("\n" + "=" * 96)
    print("SIGNIFICANCE — radar-alone vs fusion on K=8 NF dB-loss (paired across seeds)")
    print("=" * 96)
    for f in (60, 140):
        ra = load("radar", f"closerange_{f}GHz")
        fu = load("fusion", f"closerange_{f}GHz")
        seeds = sorted(set(ra) & set(fu))
        if len(seeds) < 2:
            print(f"  {f} GHz: insufficient paired seeds (n={len(seeds)})"); continue
        a = np.array([ra[s]["dbloss"]["8"]["nf"] for s in seeds])  # radar
        b = np.array([fu[s]["dbloss"]["8"]["nf"] for s in seeds])  # fusion
        t = paired_t_test(a, b)
        print(f"  {f} GHz (n={len(seeds)}): radar {a.mean():.3f} dB vs fusion {b.mean():.3f} dB"
              f" | mean diff {t['mean_diff']:+.3f} dB | t={t['t']:.2f} p={t['p']:.4f} "
              f"Cohen_d={t['cohen_d']:.2f}")
        better = "radar better" if t["mean_diff"] < 0 else "fusion better"
        print(f"         -> {better}; "
              f"{'significant' if t['p']<0.05 else 'NOT significant'} at .05")

    print("\n" + "=" * 96)
    print("O1 FAR-FIELD ANCHOR (radar) — does-no-harm: polar-vs-DFT genie gap should be ~0")
    print("=" * 96)
    for f in (60, 140):
        r = load("radar", f"synthetic_{f}GHz")
        if not r:
            print(f"  {f} GHz: n/a"); continue
        print(f"  {f} GHz (n={len(r)}): top1_theta {fmt(r,'top1_theta')} | "
              f"polar-vs-DFT genie (all) {fmt(r,'polar_vs_dft_genie_db.all')} dB | "
              f"K8 dB(all) {fmt(r,'dbloss.8.all')}")

    print("\n" + "=" * 96)
    print("HONESTY NOTE (docs/07 B4): fusion sees radar+LiDAR (MORE information) than radar-alone")
    print("(radar only), yet does not improve K=8 NF dB-loss. More information, no gain —")
    print("this asymmetry strengthens the negative finding; reported, not hidden.")


if __name__ == "__main__":
    main()
