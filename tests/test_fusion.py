"""SensingPolarPredictor (fusion) tests (docs/07 A5, docs/10 D4)."""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.sensing_polar import (BRANCHES, SensingPolarPredictor,
                                       confidence_combine)
from src.train.loss import polar_loss

B_THETA, B_R = 64, 8
LOGIT_KEYS = [f"{b}_{k}_logits" for b in BRANCHES for k in ("theta", "range")]


def _inputs(n=4):
    return torch.randn(n, 2, 256, 128), torch.randn(n, 2, 150, 100)


def test_param_count_in_range():
    p = sum(x.numel() for x in SensingPolarPredictor().parameters())
    assert 2_000_000 <= p <= 20_000_000, f"param count {p:,} outside [2M,20M]"


def test_no_batchnorm():
    m = SensingPolarPredictor()
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                   for mod in m.modules()), "BatchNorm present — use GroupNorm (D9)"


def test_six_logits_shapes_no_nan():
    m = SensingPolarPredictor(B_theta=B_THETA, B_r=B_R).eval()
    xr, xl = _inputs()
    out = m(xr, xl)
    assert set(out.keys()) == set(LOGIT_KEYS), "must emit 6 logit tensors (3 branches x 2)"
    for b in BRANCHES:
        assert out[f"{b}_theta_logits"].shape == (4, B_THETA)
        assert out[f"{b}_range_logits"].shape == (4, B_R)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_backward_runs():
    m = SensingPolarPredictor()
    xr, xl = _inputs()
    out = m(xr, xl)
    yt = torch.randint(0, B_THETA, (4,)); yr = torch.randint(0, B_R, (4,))
    loss = sum(polar_loss({"theta_logits": out[f"{b}_theta_logits"],
                           "range_logits": out[f"{b}_range_logits"]}, yt, yr)[0]
               for b in BRANCHES)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_fused_responds_to_lidar():
    """D4: the fused branch must actually use LiDAR — changing/zeroing it changes output."""
    torch.manual_seed(0)
    m = SensingPolarPredictor().eval()
    xr, xl = _inputs()
    with torch.no_grad():
        base = m(xr, xl)["fused_theta_logits"]
        other = m(xr, torch.randn_like(xl))["fused_theta_logits"]
        zeroed = m(xr, xl, lidar_dropout=True)["fused_theta_logits"]
    assert not torch.allclose(base, other, atol=1e-4), "fused output ignores LiDAR content"
    assert not torch.allclose(base, zeroed, atol=1e-4), "fused output ignores LiDAR dropout"


def test_confidence_combine_valid_distribution():
    m = SensingPolarPredictor().eval()
    xr, xl = _inputs()
    with torch.no_grad():
        p_theta, p_r, w = confidence_combine(m(xr, xl), return_weights=True)
    assert torch.allclose(p_theta.sum(1), torch.ones(4), atol=1e-4)
    assert torch.allclose(p_r.sum(1), torch.ones(4), atol=1e-4)
    assert (p_theta >= 0).all() and (p_r >= 0).all()
    for head in ("theta", "range"):
        assert abs(sum(w[head].values()) - 1.0) < 1e-4   # weights normalized over branches


def test_overfit_small_batch():
    torch.manual_seed(0)
    m = SensingPolarPredictor()
    xr = torch.randn(24, 2, 48, 32); xl = torch.randn(24, 2, 40, 32)
    yt = torch.randint(0, B_THETA, (24,)); yr = torch.randint(0, B_R, (24,))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = None
    for step in range(500):
        opt.zero_grad()
        out = m(xr, xl)
        loss = sum(polar_loss({"theta_logits": out[f"{b}_theta_logits"],
                               "range_logits": out[f"{b}_range_logits"]}, yt, yr)[0]
                   for b in BRANCHES)
        loss.backward(); opt.step()
        if step == 0:
            first = float(loss.detach())
    with torch.no_grad():
        p_theta, p_r = confidence_combine(m(xr, xl))
        acc = ((p_theta.argmax(1) == yt) & (p_r.argmax(1) == yr)).float().mean()
    assert float(loss.detach()) < 0.5 * first
    assert acc > 0.9, f"fusion overfit joint acc {float(acc):.2f} too low"
