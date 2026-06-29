"""Tests for the Gate-0 verification report (docs/07 Checklist G)."""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.polar import fraunhofer_distance

_DATA = os.path.join(_ROOT, "data", "processed")
_NEEDED = ["closerange_60GHz_full.h5", "closerange_140GHz_full.h5",
           "synthetic_60GHz_full.h5", "synthetic_140GHz_full.h5"]


def test_fraunhofer_uses_nside128_not_total():
    """Bug guard: d_F must use the per-side N_side=128, not the 16384 total elements."""
    d60 = fraunhofer_distance(60e9, N_side=128)
    d140 = fraunhofer_distance(140e9, N_side=128)
    assert 38.0 < d60 < 42.0, f"d_F@60GHz should be ~40.3 m, got {d60:.2f}"
    assert 16.0 < d140 < 18.5, f"d_F@140GHz should be ~17.3 m, got {d140:.2f}"
    # the forbidden total-N convention is kilometres
    assert fraunhofer_distance(60e9, N_side=16384) > 1000.0


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    if not all(os.path.exists(os.path.join(_DATA, f)) for f in _NEEDED):
        pytest.skip("processed datasets not present; run generation first")
    from scripts import gate0_verification as g0
    out = tmp_path_factory.mktemp("gate0") / "report.md"
    g0.run(max_samples=150, out_path=str(out))
    assert out.exists()
    return out.read_text(encoding="utf-8")


def test_report_generated(report):
    assert len(report) > 1000


def test_dF_uses_nside128_in_report(report):
    assert "N_side=128" in report
    assert "40.3" in report and "17.3" in report     # the two d_F values appear


def test_three_way_fractions_present(report):
    assert "frac(d<d_F)" in report
    assert "frac(y_r_gated>0)" in report
    assert "frac(genie adv>=3 dB)" in report


def test_both_datasets_separate_not_blended(report):
    assert "PRIMARY" in report and "ANCHOR" in report
    assert "closerange" in report and "synthetic" in report
    # both PRIMARY frequencies listed separately in the G0.1 table (not one blended row)
    assert report.count("PRIMARY (near-field)") >= 2
    assert report.count("ANCHOR (far-field, O1)") >= 2


def test_all_five_probes_present(report):
    for probe in ["G0.1", "G0.2", "G0.3", "G0.4", "G0.5"]:
        assert probe in report, f"{probe} missing from report"


def test_committed_report_exists():
    """The official report must be committed (Checklist G/F)."""
    p = os.path.join(_ROOT, "results", "gate0", "report.md")
    assert os.path.exists(p), "results/gate0/report.md not committed"
    assert os.path.getsize(p) > 1000
