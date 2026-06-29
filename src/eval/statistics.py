"""Self-contained statistics for cross-seed comparisons (Paper 5, docs/07 B4).

Mean +/- 95% CI, paired t-test, paired Cohen's d (d_z). Implemented here (not imported
from any other project); uses scipy only for the Student-t distribution function.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def mean_ci(x: Sequence[float], conf: float = 0.95) -> dict:
    """Mean and (two-sided) t-based confidence interval across seeds."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return dict(mean=m, lo=m, hi=m, sd=0.0, n=n)
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, n - 1))
    return dict(mean=m, lo=m - tcrit * se, hi=m + tcrit * se, sd=sd, n=n)


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired t-test of a vs b across seeds + paired Cohen's d (d_z) + 95% CI of mean diff.

    Returns t, p (two-sided), df, mean_diff (a-b), cohen_d, ci95 (of mean diff), n.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    if n < 2:
        raise ValueError("need >= 2 pairs")
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        t = 0.0 if md == 0 else float("inf")
        p = 1.0 if md == 0 else 0.0
    else:
        t = md / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
    cohen_d = md / sd if sd > 0 else float("inf") if md != 0 else 0.0
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return dict(t=float(t), p=p, df=n - 1, mean_diff=md, cohen_d=float(cohen_d),
                ci95=(md - tcrit * se, md + tcrit * se), n=n)
