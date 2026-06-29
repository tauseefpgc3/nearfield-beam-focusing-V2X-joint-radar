"""Detailed (angle, range) evaluation metrics (Paper 5, docs/07 B3).

Reports overall top-1, the **near-field-subset range accuracy** (the metric that matters:
range top-1 conditioned on true y_r > 0, not inflated by the far-field majority), and the
range confusion matrix + per-class recall (do deep bins j>=3 ever get predicted?).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def detailed_eval(model, loader, device, B_r: int = 8) -> dict:
    model.eval()
    yt_all, pt_all, yr_all, pr_all = [], [], [], []
    for x, yt, yr in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
        pt_all.append(out["theta_logits"].argmax(1).cpu())
        pr_all.append(out["range_logits"].argmax(1).cpu())
        yt_all.append(yt)
        yr_all.append(yr)
    YT = torch.cat(yt_all); PT = torch.cat(pt_all)
    YR = torch.cat(yr_all); PR = torch.cat(pr_all)

    theta = float((PT == YT).float().mean())
    rng = float((PR == YR).float().mean())
    joint = float(((PT == YT) & (PR == YR)).float().mean())
    nf = YR > 0                                    # near-field subset
    n_nf = int(nf.sum())
    range_nf = float((PR[nf] == YR[nf]).float().mean()) if n_nf else float("nan")
    far = YR == 0
    range_far = float((PR[far] == YR[far]).float().mean()) if int(far.sum()) else float("nan")

    cm = np.zeros((B_r, B_r), dtype=int)
    for t, p in zip(YR.numpy(), PR.numpy()):
        cm[t, p] += 1
    recall = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")
              for i in range(B_r)]
    support = [int(cm[i].sum()) for i in range(B_r)]
    return dict(theta=theta, range=rng, joint=joint, range_nf=range_nf, n_nf=n_nf,
                range_far=range_far, range_cm=cm.tolist(), range_recall=recall,
                range_support=support, n=int(YR.numel()))


def print_range_report(name: str, m: dict) -> None:
    print(f"\n--- {name} range report ---")
    print(f"  overall: theta {m['theta']:.3f} | range {m['range']:.3f} | joint {m['joint']:.3f}")
    print(f"  range top-1 (all)            = {m['range']:.3f}")
    print(f"  range top-1 (near-field y_r>0)= {m['range_nf']:.3f}  (n={m['n_nf']})  <-- THE METRIC")
    print(f"  range top-1 (far y_r==0)      = {m['range_far']:.3f}")
    print(f"  per-bin range recall (j=0..7): "
          f"{[round(r,3) if r==r else None for r in m['range_recall']]}")
    print(f"  per-bin support             : {m['range_support']}")
    print("  range confusion matrix (rows=true j, cols=pred j):")
    for i, row in enumerate(m["range_cm"]):
        print(f"    j={i}: {row}")
