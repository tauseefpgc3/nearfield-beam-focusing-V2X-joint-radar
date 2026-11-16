"""D-031 design selection (validation only): per sensor, head x lr from the seed-11 pilots by
mean validation sweep loss over M in {8,16,32,64} and both frequencies; learning check
(final training loss below the prior level). Writes results/phase3/design_networks.json."""
import json, os, sys
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "phase2"))
from p5_e23_sweep import sweep_loss, shapes
P2 = os.path.join(ROOT, "data", "processed", "phase2"); R3 = os.path.join(ROOT, "results", "phase3")


def val_score(est, G, ug, rings):
    return float(np.mean([min(sweep_loss(G[f], est, ug[f], rings[f], A, R).mean() for A, R in shapes(M)) for f in (60, 140) for M in (8, 16, 32, 64)]))


def main():
    with h5py.File(os.path.join(P2, "split.h5"), "r") as hf: va = np.sort(hf["val"][:])
    with h5py.File(os.path.join(P2, "users.h5"), "r") as hf: xy = hf["ant_xyz"][:, :2][va]
    G, ug, rings = {}, {}, {}
    for f in (60, 140):
        with h5py.File(os.path.join(P2, f"channels_{f}GHz.h5"), "r") as hf: G[f] = hf["gains"][:][va]; ug[f] = hf["u_grid"][:]; rings[f] = hf["rings"][:]
    out = {}
    for s in ("radar", "lidar", "fused"):
        rows = {}
        for h in ("P", "H"):
            for lr in ("0.001", "0.0003"):
                tag = f"{s}_{h}_lr{float(lr):g}_s11"; est = np.load(os.path.join(R3, "val_pred", tag + ".npy"))
                hist = [json.loads(l) for l in open(os.path.join(R3, "logs", tag + ".log")) if l.startswith("{")]
                e = np.hypot(*(est - xy).T)
                rows[f"{h}|{lr}"] = dict(val_sweep=val_score(est, G, ug, rings), val_err_med=float(np.median(e)), val_err_p90=float(np.percentile(e, 90)),
                                         final_train_loss=hist[-1]["train_loss"], prior=hist[-1]["prior"], below_prior=hist[-1]["train_loss"] < hist[-1]["prior"])
        best = min(rows, key=lambda k: rows[k]["val_sweep"])
        out[s] = dict(candidates=rows, chosen=best, learning_check_pass=rows[best]["below_prior"])
        print(s, json.dumps(out[s], indent=1))
    json.dump(out, open(os.path.join(R3, "design_networks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
