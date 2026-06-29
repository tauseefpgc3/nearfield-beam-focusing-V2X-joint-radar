"""Polar codebook classification heads (Paper 5, docs/02 §4.4).

Two independent heads over the shared feature z: an angle head (-> B_theta logits) and a
range head (-> B_r logits). The joint score is p_theta(i) * p_r(j) (conditional-
independence factorization, validated in ablation A6).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PolarHead(nn.Module):
    """Angle + range classification heads over a d_z feature."""

    def __init__(self, d_z: int = 128, B_theta: int = 64, B_r: int = 8,
                 hidden: int = 256):
        super().__init__()
        self.angle = nn.Sequential(
            nn.Linear(d_z, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, B_theta))
        self.range = nn.Sequential(
            nn.Linear(d_z, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, B_r))

    def forward(self, z: torch.Tensor) -> dict:
        """Returns {'theta_logits': (B, B_theta), 'range_logits': (B, B_r)}."""
        return {"theta_logits": self.angle(z), "range_logits": self.range(z)}
