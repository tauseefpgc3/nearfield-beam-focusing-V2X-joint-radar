"""E10 consistency fix: absolute realized NF gain on a COMMON near-field sample set.

The 3 dB gate is codebook-dependent, so each B_r gates a slightly different NF set. This
re-evaluates ALL B_r on the SAME users (near-field under the finest B_r=16 gate) and the
SAME reference genie (best over the B_r=16 codebook). For each B_r the realized gain is
the best ACTUAL channel gain among that B_r model's top-K shortlist (over its own B_r
codebook); absolute gain (dB) = 10*log10(realized / finest_genie). Eval-only, no retrain.

    python scripts/ablate_br_common.py
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.splits import trajectory_grouped_split
from src.eval.statistics import mean_ci
from src.models.radar_encoder import RadarPolarModel

B_THETA, KS, SEEDS, B_RS = 64, (4, 8), [11, 22, 33], [4, 8, 16]


def codebook(B_r, device):
    cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                              f"polar_codebook_60GHz_Br{B_r}.npz"))["codebook"]
    return torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)


def main():
    device = torch.device("cuda")
    f16 = os.path.join(_ROOT, "data", "processed", "closerange_60GHz_Br16_full.h5")
    with h5py.File(f16, "r") as hf:
        traj = hf["trajectory_id"][:]
        yr16 = hf["y_r_gated"][:]          # finest gate defines the common NF set
        nf_counts = {b: int((h5py.File(os.path.join(_ROOT, "data", "processed",
                     f"closerange_60GHz_Br{b}_full.h5"), "r")["y_r_gated"][:] > 0).sum())
                     for b in B_RS}
    print(f"NF set size per B_r gate (full data, differ!): {nf_counts}")
    cb16 = codebook(16, device)

    abs_db = {b: {k: [] for k in KS} for b in B_RS}
    n_common = []
    for seed in SEEDS:
        _, _, te = trajectory_grouped_split(traj, seed=seed)
        te = np.sort(te)
        common = te[yr16[te] > 0]          # near-field under finest gate, in test
        n_common.append(len(common))
        with h5py.File(f16, "r") as hf:
            xr = torch.from_numpy(hf["radar_features"][common].astype(np.float32))
            chan = hf["channel"][common]
        h_all = torch.from_numpy(chan).to(device, dtype=torch.complex64)
        finest_genie = torch.empty(len(common), device=device)
        for s in range(0, len(common), 300):
            finest_genie[s:s+300] = (cb16 @ h_all[s:s+300].T).abs().pow(2).amax(0)

        for B_r in B_RS:
            cbc = codebook(B_r, device)
            model = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_r).to(device)
            ck = os.path.join(_ROOT, "checkpoints", f"closerange_60GHz_Br{B_r}_radar",
                              f"seed{seed}.pt")
            model.load_state_dict(torch.load(ck, map_location=device,
                                             weights_only=False)["model"])
            model.eval()
            real = {k: torch.empty(len(common), device=device) for k in KS}
            with torch.no_grad():
                for s in range(0, len(common), 256):
                    e = min(s + 256, len(common))
                    h = h_all[s:e]
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = model(xr[s:e].to(device))
                        pth = out["theta_logits"].float().softmax(1)
                        prg = out["range_logits"].float().softmax(1)
                    joint = (pth.unsqueeze(2) * prg.unsqueeze(1)).reshape(e - s, B_THETA * B_r)
                    topk = joint.topk(max(KS), 1).indices
                    gains = (cbc @ h.T).abs().pow(2).T
                    for k in KS:
                        real[k][s:e] = gains.gather(1, topk[:, :k]).max(1).values
            for k in KS:
                db = (10 * torch.log10(real[k] / finest_genie.clamp_min(1e-30))).mean()
                abs_db[B_r][k].append(float(db))

    print(f"common NF test-set sizes (B_r=16 gate): {n_common}")
    print("\nABSOLUTE realized NF gain vs COMMON finest (B_r=16) genie, identical users "
          "(higher=better, dB):")
    print(f"{'B_r':>4} | {'K=4 abs gain':>16} | {'K=8 abs gain':>16}")
    summ = {}
    for B_r in B_RS:
        c4, c8 = mean_ci(abs_db[B_r][4]), mean_ci(abs_db[B_r][8])
        summ[B_r] = c8["mean"]
        print(f"{B_r:>4} | {c4['mean']:+.2f} +/- {c4['hi']-c4['mean']:.2f} | "
              f"{c8['mean']:+.2f} +/- {c8['hi']-c8['mean']:.2f}")
    best = max(summ, key=summ.get)
    print(f"\nBEST absolute realized gain @ K=8 (on identical users): B_r = {best} "
          f"({summ[best]:+.2f} dB)")

    # Bank the common-NF-set cross-check as JSON (Table 4 cross-check, eval-only).
    out = dict(nf_counts_per_gate=nf_counts, common_nf_test_sizes=n_common,
               seeds=SEEDS, ref_genie="B_r=16 finest codebook", best_b_r=best)
    for B_r in B_RS:
        out[f"br{B_r}"] = {}
        for k in KS:
            c = mean_ci(abs_db[B_r][k])
            out[f"br{B_r}"][f"K{k}_abs_db"] = dict(
                mean=c["mean"], lo=c["lo"], hi=c["hi"], per_seed=abs_db[B_r][k])
    op = os.path.join(_ROOT, "results", "ablation", "common_nf_absgain.json")
    import json
    json.dump(out, open(op, "w"), indent=2)
    print(f"Wrote {os.path.relpath(op, _ROOT)}")


if __name__ == "__main__":
    main()
