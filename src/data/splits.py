"""Trajectory-grouped train/val/test splits (Paper 5, docs/04 Step 7, docs/07 B1).

Whole trajectories are assigned to a single split (no within-trajectory leakage). The
split is seeded and regenerable.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def trajectory_grouped_split(trajectory_ids, ratios=(0.7, 0.1, 0.2), seed: int = 11
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices into (train, val, test) by whole trajectories.

    Trajectories are shuffled (seeded) and greedily assigned so the cumulative sample
    fractions approximate ``ratios`` — guaranteeing no trajectory spans two splits.

    Parameters
    ----------
    trajectory_ids : array-like (N,)
        Per-sample trajectory id.
    ratios : tuple
        (train, val, test) fractions, summing to 1.
    seed : int
        RNG seed for the trajectory shuffle.

    Returns
    -------
    (train_idx, val_idx, test_idx) : int arrays, disjoint, covering all N samples.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    traj = np.asarray(trajectory_ids)
    n = traj.shape[0]
    uniq = np.unique(traj)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    counts = {int(t): int(np.sum(traj == t)) for t in uniq}
    assign = {}
    acc = 0
    for t in uniq:
        frac = acc / n
        if frac < ratios[0]:
            assign[int(t)] = 0
        elif frac < ratios[0] + ratios[1]:
            assign[int(t)] = 1
        else:
            assign[int(t)] = 2
        acc += counts[int(t)]

    split_of = np.array([assign[int(t)] for t in traj])
    train_idx = np.where(split_of == 0)[0]
    val_idx = np.where(split_of == 1)[0]
    test_idx = np.where(split_of == 2)[0]
    # no trajectory may appear in two splits
    for a, b in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)):
        assert not (set(traj[a].tolist()) & set(traj[b].tolist())), "trajectory leak"
    return train_idx, val_idx, test_idx
