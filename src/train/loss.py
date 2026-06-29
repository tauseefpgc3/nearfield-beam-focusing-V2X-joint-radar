"""Polar classification loss (Paper 5, docs/02 §4.4).

L = CE(theta_logits, y_theta) + lambda_r * CE(range_logits, y_r), lambda_r = 1.0.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def polar_loss(out: dict, y_theta: torch.Tensor, y_r: torch.Tensor,
               lambda_r: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """Joint angle+range cross-entropy.

    Parameters
    ----------
    out : dict
        Model output with 'theta_logits' (B, B_theta) and 'range_logits' (B, B_r).
    y_theta, y_r : torch.Tensor
        Integer labels (B,).
    lambda_r : float
        Range-loss weight (configs/base.yaml train.lambda_r).

    Returns
    -------
    total : torch.Tensor (scalar)
    parts : dict with 'l_theta', 'l_range' (floats).
    """
    l_theta = F.cross_entropy(out["theta_logits"], y_theta)
    l_range = F.cross_entropy(out["range_logits"], y_r)
    total = l_theta + lambda_r * l_range
    return total, {"l_theta": float(l_theta.detach()), "l_range": float(l_range.detach())}
