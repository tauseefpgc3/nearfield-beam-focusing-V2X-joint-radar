"""Sensing-aided polar predictor: dual encoder + cross-attention fusion + 3 branches.

docs/02 §4.3-4.6 / §5.1. Three branches (radar-only, LiDAR-only, fused) each with angle
+ range heads -> 6 logit tensors. Inference uses a confidence-weighted ensemble (C3):
per-branch, per-head confidence = max(softmax); the final per-head distribution is the
confidence-weighted average of the three branches. This is what protects angle: if the
LiDAR/fused branch is unconfident on angle, the radar branch dominates.

GroupNorm throughout (no BatchNorm, D9), so a zeroed modality (degradation / both-out)
stays uninformative and the ensemble fails safe.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import CrossAttentionFusion
from .heads import PolarHead
from .lidar_encoder import LidarEncoder
from .radar_encoder import RadarEncoder

BRANCHES = ("radar", "lidar", "fused")


class SensingPolarPredictor(nn.Module):
    def __init__(self, d_z: int = 128, B_theta: int = 64, B_r: int = 8):
        super().__init__()
        self.radar_encoder = RadarEncoder(in_ch=2, d_z=d_z)
        self.lidar_encoder = LidarEncoder(in_ch=2, d_z=d_z)
        self.fusion = CrossAttentionFusion(d_z=d_z)
        self.head_radar = PolarHead(d_z, B_theta, B_r)
        self.head_lidar = PolarHead(d_z, B_theta, B_r)
        self.head_fused = PolarHead(d_z, B_theta, B_r)

    def forward(self, x_radar: torch.Tensor, x_lidar: torch.Tensor,
                radar_dropout: bool = False, lidar_dropout: bool = False) -> dict:
        """Returns 6 logit tensors: {branch}_{theta,range}_logits for 3 branches.

        ``radar_dropout`` / ``lidar_dropout`` zero that modality's feature (train-time
        augmentation §5.3; also used to simulate a missing sensor at eval).
        """
        z_r = self.radar_encoder(x_radar)
        z_l = self.lidar_encoder(x_lidar)
        if radar_dropout:
            z_r = torch.zeros_like(z_r)
        if lidar_dropout:
            z_l = torch.zeros_like(z_l)
        z_f = self.fusion(z_r, z_l)

        out = {}
        for name, z, head in (("radar", z_r, self.head_radar),
                              ("lidar", z_l, self.head_lidar),
                              ("fused", z_f, self.head_fused)):
            h = head(z)
            out[f"{name}_theta_logits"] = h["theta_logits"]
            out[f"{name}_range_logits"] = h["range_logits"]
        return out


def confidence_combine(out: dict, return_weights: bool = False):
    """Confidence-weighted ensemble of the 3 branches (C3, §5.1), per head.

    For each head, branch confidence c = max(softmax); final p = sum(c*p)/sum(c).

    Returns (p_theta, p_r) probability tensors; optionally a weights dict with the
    mean per-branch confidence weight for each head (diagnostic: does angle down-weight
    LiDAR?).
    """
    def _combine(kind: str):
        probs, confs = [], []
        for b in BRANCHES:
            p = F.softmax(out[f"{b}_{kind}_logits"], dim=1)
            probs.append(p)
            confs.append(p.max(dim=1, keepdim=True).values)        # (B,1)
        C = torch.cat(confs, dim=1)                                # (B,3)
        w = C / C.sum(dim=1, keepdim=True).clamp_min(1e-12)        # normalized weights
        p_final = sum(w[:, i:i + 1] * probs[i] for i in range(len(BRANCHES)))
        return p_final, w

    p_theta, w_theta = _combine("theta")
    p_r, w_r = _combine("range")
    if not return_weights:
        return p_theta, p_r
    weights = {
        "theta": {b: float(w_theta[:, i].mean()) for i, b in enumerate(BRANCHES)},
        "range": {b: float(w_r[:, i].mean()) for i, b in enumerate(BRANCHES)},
    }
    return p_theta, p_r, weights
