"""Tests for the polar / DFT codebook (docs/07 checklist A1, A2, E2).

Test-first: these encode the acceptance criteria from 07_CODE_AUDIT_CHECKLIST.md
and the session precondition (Fraunhofer convention, N disambiguation). Run with:

    pytest tests/test_polar_codebook.py -v
"""
import os
import sys

import numpy as np
import pytest

# Make ``src`` importable without installation.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.dft import generate_dft_codebook
from src.codebook.polar import (
    compute_beam_focusing_gain_ratio,
    compute_optimal_label,
    fraunhofer_distance,
)
from src.codebook.polar import generate_polar_codebook
from src.codebook.steering import planar_steering, spherical_steering

# ---- shared constants (128x128 UPA, D-011 resolved) --------------------------
N_ELEMENTS = 16384
N_SIDE = 128
B_THETA = 64
B_R = 8
SPACING = 0.5
C = 299_792_458.0
# True per-side (D-002) d_F at 128x128. (Supervisor ballpark ~86/~38/~16 used the
# (128/32)^2=16 scaling; the exact (127/31)^2=16.78 formula gives these — see D-011.)
FREQS = {28e9: 86.35, 60e9: 40.29, 140e9: 17.27}


# ============================================================================
# Precondition: Fraunhofer distance (the gating bug)
# ============================================================================
def test_fraunhofer_uses_per_side_not_total():
    """If N_side=16384 (total) were used, d_F would be kilometres. Guard against it."""
    d_f_correct = fraunhofer_distance(28e9, N_side=N_SIDE, spacing=SPACING)
    d_f_bug = fraunhofer_distance(28e9, N_side=N_ELEMENTS, spacing=SPACING)
    assert d_f_correct < 200.0, "per-side d_F must be ~tens of metres, not km"
    assert d_f_bug > 1000.0, "sanity: the forbidden total-N convention IS kilometres"


@pytest.mark.parametrize("f_c,target", FREQS.items())
def test_fraunhofer_matches_plan_targets(f_c, target):
    """Per-side convention must reproduce 86.35/40.29/17.27 m within 5% (fail loud)."""
    d_f = fraunhofer_distance(f_c, N_side=N_SIDE, spacing=SPACING)
    rel = abs(d_f - target) / target
    assert rel < 0.05, f"d_F={d_f:.4f} m off target {target} m by {rel:.1%} (>5%)"


# ============================================================================
# A1 — Polar codebook generation
# ============================================================================
@pytest.fixture(scope="module")
def codebook_28():
    return generate_polar_codebook(N_ELEMENTS, 28e9, B_theta=B_THETA, B_r=B_R,
                                   spacing=SPACING)


def test_codebook_shape(codebook_28):
    cb, angles, ranges = codebook_28
    assert cb.shape == (B_THETA * B_R, N_ELEMENTS) == (512, 16384)
    assert angles.shape == (B_THETA,)
    assert ranges.shape == (B_R,)


def test_codebook_unit_norm(codebook_28):
    cb, _, _ = codebook_28
    norms = np.linalg.norm(cb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_range0_is_far_field(codebook_28):
    _, _, ranges = codebook_28
    assert np.isinf(ranges[0]), "range[0] must be far-field (r -> inf)"
    assert np.all(np.isfinite(ranges[1:]))
    # Cui & Dai: r_j decreases as j increases (1/r linear in j).
    assert np.all(np.diff(ranges[1:]) < 0)


def test_angles_uniform_in_sine(codebook_28):
    _, angles, _ = codebook_28
    diffs = np.diff(np.sin(angles))
    assert np.allclose(diffs, diffs[0], atol=1e-9), "sin(angles) must be uniform"


def test_codewords_distinct(codebook_28):
    """E2 (codewords distinct). See docs/decisions.md D-007 for the rank discussion.

    At the 128x128 UPA the far-field block (64 j=0 codewords, elevation-0 -> y-modes
    only) has rank min(B_theta, N_side) = 64 (now limited by the number of angles, not
    N_side). The near-field codewords add range curvature, enriching the rank well
    beyond 64. The real E2 intent — codewords distinct (no duplicates, none parallel)
    — holds. The exact total rank is reported by report_codebook_rank.py.
    """
    cb, _, _ = codebook_28
    # No exact duplicate codewords.
    assert np.unique(np.round(cb, 8), axis=0).shape[0] == cb.shape[0]
    # None parallel: max off-diagonal mutual coherence strictly below 1.
    gram = np.abs(cb.conj() @ cb.T)
    np.fill_diagonal(gram, 0.0)
    assert gram.max() < 0.999, "codewords must not be parallel/duplicated"
    # Far-field block: rank limited by the number of angles (64 <= N_side=128).
    far_field = cb[0::B_R]
    assert np.linalg.matrix_rank(far_field) == min(B_THETA, N_SIDE)
    # Near-field codewords enrich the rank beyond the far-field block.
    assert np.linalg.matrix_rank(cb) > min(B_THETA, N_SIDE)


# ============================================================================
# E2 — far-field codeword equals a true DFT codeword
# ============================================================================
def test_far_field_slice_equals_dft(codebook_28):
    cb, _, _ = codebook_28
    dft = generate_dft_codebook(N_ELEMENTS, B_theta=B_THETA)
    assert dft.shape == (B_THETA, N_ELEMENTS)
    # j=0 column for every angle i is at flat index i*B_r + 0.
    far_field = cb[0::B_R]
    assert far_field.shape == dft.shape
    assert np.allclose(far_field, dft, atol=1e-6), "polar j=0 must equal DFT codeword"


# ============================================================================
# A1/A2 — far-field user -> j=0 ; near-field user (r<d_F) -> j>0
# ============================================================================
def test_far_field_user_optimal_label_j0(codebook_28):
    cb, angles, _ = codebook_28
    i = 20
    h = planar_steering(angles[i], N_side=N_SIDE, spacing=SPACING, f_c=28e9)
    (y_theta, y_r), gain = compute_optimal_label(h, cb, B_THETA, B_R)
    assert y_theta == i
    assert y_r == 0, "planar (far-field) wave must select the j=0 codeword"


def test_near_field_user_optimal_label_jpos(codebook_28):
    cb, angles, ranges = codebook_28
    d_f = fraunhofer_distance(28e9, N_side=N_SIDE, spacing=SPACING)
    i, j = 20, 3
    assert ranges[j] < d_f, "fixture sanity: chosen range must be near-field"
    h = spherical_steering(angles[i], ranges[j], N_side=N_SIDE, spacing=SPACING,
                           f_c=28e9)
    (y_theta, y_r), gain = compute_optimal_label(h, cb, B_THETA, B_R)
    assert y_theta == i
    assert y_r == j, "near-field wave must select a j>0 (range-focused) codeword"
    assert y_r > 0


def test_optimal_label_in_valid_range(codebook_28):
    cb, angles, ranges = codebook_28
    h = spherical_steering(angles[5], ranges[2], N_side=N_SIDE, spacing=SPACING,
                           f_c=28e9)
    (y_theta, y_r), gain = compute_optimal_label(h, cb, B_THETA, B_R)
    assert 0 <= y_theta < B_THETA
    assert 0 <= y_r < B_R
    # Exact codebook match -> gain is the maximum N.
    assert gain == pytest.approx(N_ELEMENTS, rel=1e-6)


def test_gain_ratio_self_is_one(codebook_28):
    cb, angles, ranges = codebook_28
    h = spherical_steering(angles[10], ranges[4], N_side=N_SIDE, spacing=SPACING,
                           f_c=28e9)
    (yt, yr), _ = compute_optimal_label(h, cb, B_THETA, B_R)
    f_opt = cb[yt * B_R + yr]
    assert compute_beam_focusing_gain_ratio(f_opt, f_opt, h) == pytest.approx(1.0)
    # A DFT (far-field) beam on a near-field user loses gain: ratio < 1.
    f_dft = cb[yt * B_R + 0]
    ratio = compute_beam_focusing_gain_ratio(f_dft, f_opt, h)
    assert 0.0 <= ratio < 1.0


# ============================================================================
# Consistency: fraunhofer_distance vs the saved codebook metadata
# ============================================================================
def test_saved_codebook_dF_matches_fraunhofer():
    """E2/B2: the d_F stored with a generated codebook must equal fraunhofer_distance."""
    path = os.path.join(_ROOT, "data", "codebooks", "polar_codebook_28GHz.npz")
    if not os.path.exists(path):
        pytest.skip("codebook not generated yet; run scripts/generate_codebook.py")
    data = np.load(path, allow_pickle=True)
    d_f_expected = fraunhofer_distance(28e9, N_side=N_SIDE, spacing=SPACING)
    assert float(data["d_F"]) == d_f_expected, "stored d_F must be identical"
    assert data["codebook"].shape == (512, 16384)

