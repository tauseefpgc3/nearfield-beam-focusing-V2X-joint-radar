"""Radar encoder + heads tests (docs/07 A5, A6).

A5: param count in [2M, 20M]; forward produces correct-shape logits with no NaN;
    loss.backward() runs.
A6: the model CAN learn — loss decreases / it overfits a small fixed batch.
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.lidar_encoder import LidarPolarModel
from src.models.radar_encoder import RadarPolarModel
from src.train.loss import polar_loss

B_THETA, B_R = 64, 8


def test_param_count_in_range():
    m = RadarPolarModel(B_theta=B_THETA, B_r=B_R)
    p = sum(x.numel() for x in m.parameters())
    assert 2_000_000 <= p <= 20_000_000, f"param count {p:,} outside [2M, 20M]"


def test_no_batchnorm():
    """D9: must use GroupNorm, not BatchNorm (running stats leak in both-out test)."""
    m = RadarPolarModel()
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                   for mod in m.modules()), "BatchNorm present — use GroupNorm (D9)"


def test_forward_shapes_no_nan():
    m = RadarPolarModel(B_theta=B_THETA, B_r=B_R).eval()
    x = torch.randn(4, 2, 256, 128)
    out = m(x)
    assert out["theta_logits"].shape == (4, B_THETA)
    assert out["range_logits"].shape == (4, B_R)
    assert torch.isfinite(out["theta_logits"]).all()
    assert torch.isfinite(out["range_logits"]).all()


def test_backward_runs():
    m = RadarPolarModel()
    x = torch.randn(4, 2, 256, 128)
    yt = torch.randint(0, B_THETA, (4,))
    yr = torch.randint(0, B_R, (4,))
    loss, parts = polar_loss(m(x), yt, yr)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert parts["l_theta"] > 0 and parts["l_range"] > 0


def test_overfit_small_batch():
    """Capacity / no-label-bug check: overfit a fixed random batch -> loss collapses."""
    torch.manual_seed(0)
    m = RadarPolarModel()
    x = torch.randn(32, 2, 32, 32)          # small spatial for a fast CPU test
    yt = torch.randint(0, B_THETA, (32,))
    yr = torch.randint(0, B_R, (32,))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = None
    for step in range(250):
        opt.zero_grad()
        loss, _ = polar_loss(m(x), yt, yr)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
    last = float(loss.detach())
    assert last < 0.5 * first, f"did not overfit (first={first:.3f}, last={last:.3f})"
    # joint accuracy on the memorized batch should be high (no label/gradient bug)
    with torch.no_grad():
        out = m(x)
        acc = ((out["theta_logits"].argmax(1) == yt) &
               (out["range_logits"].argmax(1) == yr)).float().mean()
    assert acc > 0.9, f"overfit joint acc {float(acc):.2f} too low"


# ============================== LiDAR branch ================================
def test_lidar_param_count_in_range():
    p = sum(x.numel() for x in LidarPolarModel(B_theta=B_THETA, B_r=B_R).parameters())
    assert 2_000_000 <= p <= 20_000_000, f"lidar param count {p:,} outside [2M, 20M]"


def test_lidar_no_batchnorm():
    m = LidarPolarModel()
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                   for mod in m.modules()), "BatchNorm present — use GroupNorm (D9)"


def test_lidar_forward_shapes_no_nan():
    m = LidarPolarModel(B_theta=B_THETA, B_r=B_R).eval()
    x = torch.randn(4, 2, 150, 100)          # 2-channel BEV
    out = m(x)
    assert out["theta_logits"].shape == (4, B_THETA)
    assert out["range_logits"].shape == (4, B_R)
    assert torch.isfinite(out["theta_logits"]).all()
    assert torch.isfinite(out["range_logits"]).all()


def test_lidar_backward_runs():
    m = LidarPolarModel()
    x = torch.randn(4, 2, 150, 100)
    loss, _ = polar_loss(m(x), torch.randint(0, B_THETA, (4,)), torch.randint(0, B_R, (4,)))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_lidar_overfit_small_batch():
    torch.manual_seed(0)
    m = LidarPolarModel()
    x = torch.randn(32, 2, 48, 32)
    yt = torch.randint(0, B_THETA, (32,))
    yr = torch.randint(0, B_R, (32,))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = None
    for step in range(450):
        opt.zero_grad()
        loss, _ = polar_loss(m(x), yt, yr)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
    with torch.no_grad():
        out = m(x)
        acc = ((out["theta_logits"].argmax(1) == yt) &
               (out["range_logits"].argmax(1) == yr)).float().mean()
    assert float(loss.detach()) < 0.5 * first
    assert acc > 0.9, f"lidar overfit joint acc {float(acc):.2f} too low"
