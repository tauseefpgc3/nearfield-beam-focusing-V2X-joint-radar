"""LiDAR encoder + single-modality polar classifier (Paper 5, docs/02 §4.2).

docs/02 §4.2: "Same CNN on 2-channel BEV." So `LidarEncoder` reuses the exact radar
encoder architecture (4 GroupNorm conv blocks 32->64->128->256 -> GAP -> projection ->
d_z), applied to the (2, 150, 100) BEV (point-count + mean-intensity). GroupNorm, not
BatchNorm (D9). This mirrors Batch 5 for a fair single-modality radar-vs-LiDAR comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import PolarHead
from .radar_encoder import RadarEncoder


class LidarEncoder(RadarEncoder):
    """Identical 4-block GroupNorm CNN as the radar encoder, on the 2-channel BEV."""
    # Architecture is shared verbatim with RadarEncoder (docs/02 §4.2).


class LidarPolarModel(nn.Module):
    """LiDAR-only (angle, range) classifier: BEV encoder + polar heads."""

    def __init__(self, in_ch: int = 2, d_z: int = 128, B_theta: int = 64, B_r: int = 8):
        super().__init__()
        self.encoder = LidarEncoder(in_ch=in_ch, d_z=d_z)
        self.head = PolarHead(d_z=d_z, B_theta=B_theta, B_r=B_r)

    def forward(self, x_lidar: torch.Tensor) -> dict:
        """Returns {'theta_logits': (B,B_theta), 'range_logits': (B,B_r)}."""
        return self.head(self.encoder(x_lidar))
