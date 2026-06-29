"""Cross-attention fusion of radar + LiDAR features (Paper 5, docs/02 §4.3).

z_fused = CrossAttn(query=z_radar, key/value=z_lidar), single head, residual + 2-layer
MLP. GroupNorm (no running stats; D9), so a zeroed modality stays uninformative for the
both-out fail-safe.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_GN_GROUPS = 8


class CrossAttentionFusion(nn.Module):
    """Single-head cross-attention: radar queries LiDAR; residual + MLP."""

    def __init__(self, d_z: int = 128, mlp_hidden: int = 256):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_z, num_heads=1, batch_first=True)
        self.norm1 = nn.GroupNorm(_GN_GROUPS, d_z)
        self.mlp = nn.Sequential(
            nn.Linear(d_z, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, d_z))
        self.norm2 = nn.GroupNorm(_GN_GROUPS, d_z)

    def forward(self, z_radar: torch.Tensor, z_lidar: torch.Tensor) -> torch.Tensor:
        q = z_radar.unsqueeze(1)                     # (B,1,d_z)
        kv = z_lidar.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)           # attends radar->lidar
        z = self.norm1(z_radar + attn_out.squeeze(1))   # residual on the radar query
        z = self.norm2(z + self.mlp(z))                 # 2-layer MLP, residual
        return z
