"""Tests for src/eval/statistics.py (validate against scipy reference)."""
import os
import sys

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.statistics import mean_ci, paired_t_test


def test_paired_t_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.5, 5)
    b = rng.normal(0.5, 0.5, 5)
    got = paired_t_test(a, b)
    t_ref, p_ref = stats.ttest_rel(a, b)
    assert abs(got["t"] - t_ref) < 1e-9
    assert abs(got["p"] - p_ref) < 1e-9
    assert got["df"] == 4


def test_cohen_d_sign_and_magnitude():
    a = [2, 2, 2, 2, 2]; b = [1, 1, 1, 1, 1]
    got = paired_t_test(a, b)
    assert got["mean_diff"] == 1.0
    assert got["cohen_d"] == float("inf")   # zero variance of differences, nonzero mean


def test_mean_ci_brackets_mean():
    x = [0.50, 0.55, 0.48, 0.52, 0.49]
    r = mean_ci(x)
    assert r["lo"] < r["mean"] < r["hi"]
    assert abs(r["mean"] - np.mean(x)) < 1e-12
    # 95% CI half-width matches t * se
    se = np.std(x, ddof=1) / np.sqrt(5)
    assert abs((r["hi"] - r["mean"]) - stats.t.ppf(0.975, 4) * se) < 1e-9
