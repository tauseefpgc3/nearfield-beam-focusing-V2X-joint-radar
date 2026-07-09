"""D-019 Stage 1 eval: clean same-split radar-ensemble control (leakage-free).

For each split seed s in {11,22,33} x {60,140} GHz:
  LEAKAGE ASSERTION (HARD-FAIL): every member (3 inits) was trained on s's OWN train split,
    so none may see any test trajectory of s. The computed leakage fraction MUST be 0.000;
    otherwise sys.exit(2) and the chain stops.
  (b') same-split radar-3-ensemble : C3 confidence-weighted soft-average of the 3 inits.
  (c') multimodal soft-avg         : the existing fusion_warmstart model for split s (no retrain).
  Per-user NF K=8 dB-loss on s's held-out test set -> per-sample paired bootstrap
    (mean delta = dbloss_b' - dbloss_c', 95% CI, two-sided p) + sign-consistency across splits.

Banks results/fusion_rules/ensemble_confound_clean.json. NO training. GPU required.
Exit codes: 0 ok; 2 leakage!=0 (hard-fail); 3 missing checkpoint.

    python scripts/eval_d019_clean.py
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.eval_ensemble_confound import c3_weighted
from src.data.splits import trajectory_grouped_split
from src.eval.statistics import bootstrap_mean_diff, mean_ci
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import BRANCHES, SensingPolarPredictor

B_THETA, B_R, K = 64, 8, 8
SPLITS = [11, 22, 33]
INITS = [201, 202, 203]
FREQS = ["60", "140"]


def die(code, msg):
    print(f"\n*** D-019 HARD-FAIL (exit {code}): {msg} ***", flush=True)
    sys.exit(code)


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA NOT AVAILABLE — STOP.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = {"metric": "near-field K=8 dB-loss vs genie (lower=better)", "splits": SPLITS,
           "inits": INITS, "freqs": FREQS,
           "delta_def": "mean_delta = dbloss(b' radar-3-ens) - dbloss(c' multimodal); "
                        ">0 => c' lower/better => LiDAR beyond ensembling", "results": {}}

    for fghz in FREQS:
        data = os.path.join(_ROOT, "data", "processed", f"closerange_{fghz}GHz_full.h5")
        cb = np.load(os.path.join(_ROOT, "data", "codebooks",
                                  f"polar_codebook_{fghz}GHz.npz"))["codebook"]
        cbc = torch.from_numpy(np.conj(cb)).to(device, dtype=torch.complex64)
        per_split = {}
        for s in SPLITS:
            with h5py.File(data, "r") as hf:
                traj = hf["trajectory_id"][:]
                tr, _, te = trajectory_grouped_split(traj, seed=s)
                te = np.sort(te)
                xr = torch.from_numpy(hf["radar_features"][te].astype(np.float32))
                xl = torch.from_numpy(hf["lidar_features"][te].astype(np.float32))
                yr = hf["y_r_gated"][te].astype(np.int64)
                chan = hf["channel"][te]
            nf = yr > 0
            # --- LEAKAGE ASSERTION: NF-test trajectories vs members' train trajectories ---
            train_traj = set(traj[tr].tolist())      # all 3 inits share this exact train set
            nf_traj = set(traj[te[nf]].tolist())
            leak = len(nf_traj & train_traj) / max(len(nf_traj), 1)
            print(f"  {fghz}GHz split{s}: leakage={leak:.3f} (nf users={int(nf.sum())})")
            if leak != 0.0:
                die(2, f"{fghz}GHz split{s} leakage {leak:.6f} != 0.000")

            # members (3 inits, same split)
            members = []
            for k in INITS:
                ck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_d019",
                                  f"split{s}_init{k}.pt")
                if not os.path.exists(ck):
                    die(3, f"missing member checkpoint {os.path.relpath(ck, _ROOT)}")
                m = RadarPolarModel(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
                m.load_state_dict(torch.load(ck, map_location=device,
                                             weights_only=False)["model"])
                m.eval(); members.append(m)
            # (c') multimodal fusion for split s
            fck = os.path.join(_ROOT, "checkpoints", f"closerange_{fghz}GHz_fusion_warmstart",
                               f"seed{s}.pt")
            if not os.path.exists(fck):
                die(3, f"missing fusion checkpoint {os.path.relpath(fck, _ROOT)}")
            fmodel = SensingPolarPredictor(d_z=128, B_theta=B_THETA, B_r=B_R).to(device)
            fmodel.load_state_dict(torch.load(fck, map_location=device,
                                              weights_only=False)["model"])
            fmodel.eval()

            n = len(te); db_b = np.zeros(n); db_c = np.zeros(n)
            with torch.no_grad():
                for st in range(0, n, 256):
                    e = min(st + 256, n)
                    h = torch.from_numpy(chan[st:e]).to(device, dtype=torch.complex64)
                    xb = xr[st:e].to(device)
                    gains = (cbc @ h.T).abs().pow(2).T
                    genie = gains.max(1).values
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        pths = [m(xb)["theta_logits"].float().softmax(1) for m in members]
                        prgs = [m(xb)["range_logits"].float().softmax(1) for m in members]
                        fo = fmodel(xb, xl[st:e].to(device))
                        fpt = torch.stack([F.softmax(fo[f"{br}_theta_logits"].float(), 1)
                                           for br in BRANCHES])
                        fpr = torch.stack([F.softmax(fo[f"{br}_range_logits"].float(), 1)
                                           for br in BRANCHES])
                    # (b') C3 over 3 radar inits
                    pth_b, pr_b = c3_weighted(pths, prgs)
                    # (c') C3 soft-avg over 3 fusion branches
                    wth = fpt.max(-1, keepdim=True).values; wrg = fpr.max(-1, keepdim=True).values
                    pth_c = (wth / wth.sum(0, keepdim=True).clamp_min(1e-12) * fpt).sum(0)
                    pr_c = (wrg / wrg.sum(0, keepdim=True).clamp_min(1e-12) * fpr).sum(0)
                    for p_t, p_r, dst in ((pth_b, pr_b, db_b), (pth_c, pr_c, db_c)):
                        joint = (p_t.unsqueeze(2) * p_r.unsqueeze(1)).reshape(e - st, 512)
                        real = gains.gather(1, joint.topk(K, 1).indices).max(1).values
                        dst[st:e] = (10 * torch.log10(genie / real.clamp_min(1e-30))).cpu().numpy()
            b_nf = db_b[nf].tolist(); c_nf = db_c[nf].tolist()
            bs = bootstrap_mean_diff(b_nf, c_nf, seed=0)   # b' - c'
            per_split[str(s)] = dict(
                leakage=leak, n_nf_users=int(nf.sum()),
                b_radar_ens3_mean=mean_ci(b_nf)["mean"], c_multimodal_mean=mean_ci(c_nf)["mean"],
                mean_delta_b_minus_c=bs["mean_diff"], ci95=[bs["lo"], bs["hi"]],
                bootstrap_p=bs["p"])
            print(f"    (b')={per_split[str(s)]['b_radar_ens3_mean']:.2f} "
                  f"(c')={per_split[str(s)]['c_multimodal_mean']:.2f} "
                  f"delta(b-c)={bs['mean_diff']:+.2f} [{bs['lo']:+.2f},{bs['hi']:+.2f}] p={bs['p']:.3f}")
        # sign-consistency of (c' better) across splits: delta>0 means c' lower/better
        signs = [per_split[str(s)]["mean_delta_b_minus_c"] > 0 for s in SPLITS]
        out["results"][fghz] = dict(per_split=per_split,
                                    c_better_in_all_splits=bool(all(signs)),
                                    c_worse_in_all_splits=bool(all(not x for x in signs)))

    od = os.path.join(_ROOT, "results", "fusion_rules")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "ensemble_confound_clean.json"), "w"), indent=2)
    print("Wrote results/fusion_rules/ensemble_confound_clean.json (all leakage=0.000)")


if __name__ == "__main__":
    main()
