"""Deterministic seeding for Python / NumPy / PyTorch (docs/05, docs/07 Checklist C)."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch CPU+CUDA RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
