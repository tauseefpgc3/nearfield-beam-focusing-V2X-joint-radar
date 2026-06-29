"""Radar encoder + single-modality polar classifier (Paper 5, docs/02 §4.1).

`RadarEncoder`: 4 conv blocks (32->64->128->256) -> GAP -> projection -> z in R^d_z.
`RadarPolarModel`: encoder + polar heads, the radar-only (angle, range) baseline.

**GroupNorm, not BatchNorm (deliberate, docs/10 D9):** BatchNorm running statistics leak
typical-input information across samples, which corrupts the both-sensors-out degradation
test downstream (the model must fail safe to near-uniform when both modalities are
dropped). GroupNorm has no running stats, so a zeroed/garbage input stays uninformative.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import PolarHead

_GN_GROUPS = 8  # divides 32/64/128/256


def _conv_block(in_ch: int, out_ch: int, n_conv: int = 2) -> nn.Sequential:
    """n_conv x (Conv3x3 -> GroupNorm -> ReLU), then 2x downsample."""
    layers: list[nn.Module] = []
    c = in_ch
    for _ in range(n_conv):
        layers += [nn.Conv2d(c, out_ch, 3, padding=1, bias=False),
                   nn.GroupNorm(_GN_GROUPS, out_ch), nn.ReLU(inplace=True)]
        c = out_ch
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class RadarEncoder(nn.Module):
    """Radar map encoder. Input (B, in_ch, 256, 128) -> (B, d_z)."""

    def __init__(self, in_ch: int = 2, d_z: int = 128,
                 widths: tuple = (32, 64, 128, 256)):
        super().__init__()
        # Deeper (3-conv) blocks at the two high-capacity stages -> ~2-3 M params,
        # within the docs/07 A5 sanity band [2 M, 20 M].
        n_convs = (2, 2, 3, 3)
        blocks, c = [], in_ch
        for w, nc in zip(widths, n_convs):
            blocks.append(_conv_block(c, w, nc))
            c = w
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(c, 512), nn.ReLU(inplace=True),
            nn.Linear(512, d_z),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)        # (B, widths[-1])
        return self.proj(x)               # (B, d_z)


class RadarPolarModel(nn.Module):
    """Radar-only (angle, range) classifier: encoder + polar heads."""

    def __init__(self, in_ch: int = 2, d_z: int = 128, B_theta: int = 64, B_r: int = 8):
        super().__init__()
        self.encoder = RadarEncoder(in_ch=in_ch, d_z=d_z)
        self.head = PolarHead(d_z=d_z, B_theta=B_theta, B_r=B_r)

    def forward(self, x_radar: torch.Tensor) -> dict:
        """Returns {'theta_logits': (B,B_theta), 'range_logits': (B,B_r)}."""
        return self.head(self.encoder(x_radar))
