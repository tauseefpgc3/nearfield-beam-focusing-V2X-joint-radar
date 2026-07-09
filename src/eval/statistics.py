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


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test of paired a vs b (non-parametric).

    Returns statistic, p, n. Uses scipy.stats.wilcoxon (exact for small n, else normal
    approx with zero_method='wilcox'). All-zero differences -> p=1.0 (no evidence).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = a - b
    if np.allclose(d, 0.0):
        return dict(statistic=0.0, p=1.0, n=int(d.size))
    res = stats.wilcoxon(a, b)
    return dict(statistic=float(res.statistic), p=float(res.pvalue), n=int(d.size))


def cohen_d_ci(a: Sequence[float], b: Sequence[float], conf: float = 0.95,
               n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired Cohen's d (d_z) with a bootstrap CI (resample pairs, seeded/deterministic)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        db = d[idx]
        s = db.std(ddof=1)
        boots[i] = db.mean() / s if s > 0 else 0.0
    lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return dict(d=dz, lo=float(lo), hi=float(hi), n=n, n_boot=n_boot)


def tost_paired(a: Sequence[float], b: Sequence[float], margin: float,
                alpha: float = 0.05) -> dict:
    """Paired TOST equivalence test: is mean(a-b) within +/- margin?

    Two one-sided t-tests: H0_lower: mean_diff <= -margin, H0_upper: mean_diff >= +margin.
    Equivalence (a ~ b within margin) is concluded at level alpha iff BOTH are rejected,
    i.e. p_tost = max(p_lower, p_upper) < alpha, equivalently the (1-2*alpha) CI of the
    mean difference lies entirely within [-margin, +margin].

    Returns mean_diff, p_lower, p_upper, p_tost, equivalent (bool), margin, and the
    (1-2*alpha) CI used for the decision.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    if n < 2:
        raise ValueError("need >= 2 pairs")
    md = float(d.mean())
    se = float(d.std(ddof=1)) / np.sqrt(n)
    df = n - 1
    if se == 0:
        p_lower = 0.0 if md > -margin else 1.0
        p_upper = 0.0 if md < margin else 1.0
    else:
        p_lower = float(stats.t.sf((md - (-margin)) / se, df))   # H0: md <= -margin
        p_upper = float(stats.t.cdf((md - margin) / se, df))     # H0: md >= +margin
    p_tost = max(p_lower, p_upper)
    tcrit = float(stats.t.ppf(1 - alpha, df))                    # (1-2alpha) CI
    ci = (md - tcrit * se, md + tcrit * se)
    return dict(mean_diff=md, p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
                equivalent=bool(p_tost < alpha), margin=margin, ci=ci, n=n, alpha=alpha)


def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], conf: float = 0.95,
                        n_boot: int = 10000, seed: int = 0) -> dict:
    """Per-sample paired bootstrap of mean(a-b): mean_diff, CI, two-sided p, n.

    Resamples the paired differences (with replacement, seeded/deterministic). The
    two-sided p is 2*min(frac(boot<=0), frac(boot>=0)) — the standard bootstrap sign test.
    Used for D-019 (per-user paired dB-loss within a split).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    md = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(mean_diff=md, lo=float(lo), hi=float(hi), p=float(min(p, 1.0)),
                n=n, n_boot=n_boot)


def bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    """Bonferroni correction over a family of m p-values.

    Returns m, corrected p-values (min(p*m, 1)), reject flags (p < alpha/m), and
    alpha_corrected = alpha/m.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    corrected = np.minimum(p * m, 1.0)
    return dict(m=int(m), alpha_corrected=alpha / m,
                corrected=corrected.tolist(),
                reject=[bool(x) for x in (p < alpha / m)])
