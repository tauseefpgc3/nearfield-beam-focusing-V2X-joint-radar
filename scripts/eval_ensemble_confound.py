"""Confound check (eval-only): is the soft-avg fusion gain the LiDAR MODALITY or just
generic 3-model ENSEMBLING? (Reframed §5.4, follows D-018.)

For each frequency and each evaluation seed s (the 20 D-018 seeds), we compare, on s's own
trajectory-grouped TEST split, near-field K=8 beam-focusing dB-loss for:
  (a) radar-alone   -- the single radar model of seed s (pulled from rule_sweep_n20.json).
  (b) radar-3-ens   -- a radar-ONLY ensemble of 3 radar models (seeds s, s+1, s+2 in the
                       circular ordering below), combined with the SAME C3 confidence
                       weighting used by the multimodal soft-avg (per head: weight =
                       max-softmax confidence, normalised across the 3 members). This
                       matches the 3-branch count of the multimodal soft-avg.
  (c) multimodal soft-avg -- the C3 fusion (radar/lidar/fused), pulled from rule_sweep_n20.

SEED-TRIPLET ASSIGNMENT (deterministic, documented): the 20 seeds are ordered
  [11,22,33,44,55,66,77,88,99,110,121,132,143,154,165,176,187,198,209,220];
the triplet for the seed at index i is {seeds[i], seeds[(i+1)%20], seeds[(i+2)%20]}.

SPLIT PROTOCOL / HONEST CAVEAT (state in the paper): each ensemble member was trained on
its OWN seed's trajectory-grouped split, so members s+1/s+2 may have trained on trajectories
that fall in seed s's TEST split -> the radar ensemble (b) is potentially LEAKAGE-INFLATED.
This makes the (c)-vs-(b) test CONSERVATIVE against the fusion hypothesis: if the multimodal
(c), which is a single clean model on its own test split, still beats the leakage-favoured
(b), the LiDAR contribution is airtight; if (c) ~= (b), the gain may be generic ensembling.

Tests: (b) vs (a) and the key (c) vs (b) -- paired t, Wilcoxon, Cohen's d + bootstrap CI.
Banks results/fusion_rules/ensemble_confound.json. NO training.

    python scripts/eval_ensemble_confound.py
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.splits import trajectory_grouped_split
from src.eval.statistics import cohen_d_ci, mean_ci, paired_t_test, wilcoxon_signed_rank
from src.models.radar_encoder import RadarPolarModel

B_THETA, B_R, K = 64, 8, 8
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165,
         176, 187, 198, 209, 220]
FREQS = ["60", "140"]


def load_radar(fghz, seed, device):
    m = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
    ck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_radar", f"seed{seed}.pt")
    m.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    m.eval()
    return m


def c3_weighted(pths, prgs):
    """Per-head C3 confidence-weighted soft average over the member models."""
    PTH = torch.stack(pths); PRG = torch.stack(prgs)               # (M,b,·)
    wth = PTH.max(-1, keepdim=True).values
    wrg = PRG.max(-1, keepdim=True).values
    wth = wth / wth.sum(0, keepdim=True).clamp_min(1e-12)
    wrg = wrg / wrg.sum(0, keepdim=True).clamp_min(1e-12)
    return (wth * PTH).sum(0), (wrg * PRG).sum(0)                  # (b,64), (b,8)


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n20 = json.load(open(os.path.join(_ROOT, "results", "fusion_rules", "rule_sweep_n20.json")))
    assert n20["seeds"] == SEEDS, "seed order mismatch with rule_sweep_n20"

    out = {"metric": "near-field K=8 beam-focusing dB-loss vs genie (lower=better)",
           "seeds": SEEDS, "freqs": FREQS,
           "triplet_rule": "circular: seed[i], seed[(i+1)%20], seed[(i+2)%20]",
           "caveat": "radar-3-ensemble members trained on their own seed splits -> "
                     "(b) may be leakage-inflated; (c)-vs-(b) is conservative for fusion",
           "results": {}}

    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                                  f"polar_codebook_{fghz}GHz.npz"))["codebook"]
        cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)
        a = list(n20["results"][fghz]["radar_alone"]["per_seed"])   # (a) single radar
        c = list(n20["results"][fghz]["soft_avg"]["per_seed"])      # (c) multimodal soft-avg
        b = []                                                      # (b) radar-3-ensemble
        for i, s in enumerate(SEEDS):
            triplet = [SEEDS[i], SEEDS[(i + 1) % 20], SEEDS[(i + 2) % 20]]
            with h5py.File(data, "r") as hf:
                traj = hf["trajectory_id"][:]
                _, _, te = trajectory_grouped_split(traj, seed=s)   # seed s's OWN test split
                te = np.sort(te)
                xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
                yr = hf["y_r_gated"][te].astype(np.int64)
                chan = hf["channel"][te]
            nf = yr > 0
            models = [load_radar(fghz, ms, device) for ms in triplet]
            n = len(te); dbl = np.zeros(n)
            with torch.no_grad():
                for st in range(0, n, 256):
                    e = min(st + 256, n)
                    h = torch.from_numpy(chan[st:e]).to(device, dtype=torch.complex64)
                    xb = xr[st:e].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        pths = [m(xb)["theta_logits"].float().softmax(1) for m in models]
                        prgs = [m(xb)["range_logits"].float().softmax(1) for m in models]
                    p_theta, p_r = c3_weighted(pths, prgs)
                    joint = (p_theta.unsqueeze(2) * p_r.unsqueeze(1)).reshape(e - st, 512)
                    idx = joint.topk(K, 1).indices
                    gains = (cbc @ h.T).abs().pow(2).T
                    genie = gains.max(1).values
                    real = gains.gather(1, idx).max(1).values
                    dbl[st:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
            b.append(float(dbl[nf].mean()))
            print(f"  {fghz}GHz s{s} (triplet {triplet}): (a)={a[i]:.2f} (b)={b[-1]:.2f} "
                  f"(c)={c[i]:.2f}")

        def battery(x, y):     # x - y (>0 => y lower/better)
            pt = paired_t_test(x, y); wc = wilcoxon_signed_rank(x, y)
            dd = cohen_d_ci(x, y, seed=0)
            return dict(mean_x=mean_ci(x)["mean"], mean_y=mean_ci(y)["mean"],
                        mean_diff=pt["mean_diff"], paired_t_p=pt["p"], wilcoxon_p=wc["p"],
                        cohen_d=dd["d"], cohen_d_ci95=[dd["lo"], dd["hi"]])
        # Leakage audit: fraction of each eval seed's NF-test trajectories that ensemble
        # members (s+1, s+2) TRAINED on -> quantifies why (b) is not a valid baseline.
        with h5py.File(data, "r") as hf:
            traj = hf["trajectory_id"][:]; yrg = hf["y_r_gated"][:]
        leak = []
        for i, s in enumerate(SEEDS):
            _, _, te = trajectory_grouped_split(traj, seed=s); te = np.sort(te)
            nf_traj = set(traj[te[yrg[te] > 0]].tolist()); cont = set()
            for ms in (SEEDS[(i + 1) % 20], SEEDS[(i + 2) % 20]):
                tr_m, _, _ = trajectory_grouped_split(traj, seed=ms)
                cont |= set(traj[tr_m].tolist()) & nf_traj
            leak.append(len(cont) / max(len(nf_traj), 1))
        out["results"][fghz] = {
            "a_radar_alone": dict(per_seed=a, **mean_ci(a)),
            "b_radar_ens3": dict(per_seed=b, **mean_ci(b)),
            "c_multimodal_softavg": dict(per_seed=c, **mean_ci(c)),
            "b_leakage_frac": dict(per_seed=[round(x, 4) for x in leak],
                                   mean=float(np.mean(leak))),
            "b_vs_a": battery(a, b),          # ensembling effect (leakage-inflated)
            "c_vs_b": battery(b, c)}          # (b) leakage-inflated -> NOT a clean modality test

    od = os.path.join(_ROOT, "results", "fusion_rules")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "ensemble_confound.json"), "w"), indent=2)
    print("Wrote results/fusion_rules/ensemble_confound.json")
    for fghz in FREQS:
        r = out["results"][fghz]
        print(f"  [{fghz}GHz] (a)={r['a_radar_alone']['mean']:.2f} "
              f"(b)={r['b_radar_ens3']['mean']:.2f} (c)={r['c_multimodal_softavg']['mean']:.2f} "
              f"| c-vs-b p={r['c_vs_b']['paired_t_p']:.3f} W={r['c_vs_b']['wilcoxon_p']:.3f} "
              f"d={r['c_vs_b']['cohen_d']:+.2f}")


if __name__ == "__main__":
    main()
