"""Independent verification: why does radar-alone 60 GHz seed 209 differ from the banked value?
Re-score at several batch sizes / precisions and list the users whose K=8 loss changes."""
import os, sys, h5py, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.data.splits import trajectory_grouped_split
from src.models.radar_encoder import RadarPolarModel
z = np.load(os.path.join(ROOT, "results", "verify", "gains_60GHz.npz")); G = z["G32"].astype(float); yrg = z["y_r_gated"]
_, _, te = trajectory_grouped_split(z["trajectory_id"], seed=209); te = np.sort(te); nf = yrg[te] > 0
with h5py.File(os.path.join(ROOT, "data", "processed", "closerange_60GHz_full.h5"), "r") as hf: X = torch.from_numpy(hf["radar_features"][te])
m = RadarPolarModel().cuda().eval(); m.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "closerange_60GHz_radar", "seed209.pt"), map_location="cpu", weights_only=False)["model"])
out = {}
for tag, bs, fp16 in (("b128", 128, True), ("b256", 256, True), ("b64", 64, True), ("fp32_b256", 256, False)):
    sl, js = [], []
    with torch.no_grad():
        for b in range(0, len(te), bs):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                o = m(X[b:b + bs].cuda())
            pt = o["theta_logits"].float().softmax(1); pr = o["range_logits"].float().softmax(1)
            j = (pt[:, :, None] * pr[:, None, :]).reshape(len(pt), 512); js.append(j.cpu().numpy()); sl.append(j.topk(8, 1).indices.cpu().numpy())
    sl = np.concatenate(sl); J = np.concatenate(js)
    L = 10 * np.log10(G[te].max(1) / np.take_along_axis(G[te], sl, 1).max(1)); out[tag] = (L, sl, J)
    print(tag, "NF K8 = %.4f" % L[nf].mean(), "all = %.4f" % L.mean())
for tag in ("b256", "b64", "fp32_b256"):
    d = np.where(np.abs(out[tag][0] - out["b128"][0]) > 1e-6)[0]
    print(tag, "vs b128: users differing", [(int(te[i]), bool(nf[i]), round(out["b128"][0][i], 2), round(out[tag][0][i], 2)) for i in d][:10])
    for i in d[:3]:
        Jb = np.sort(out["b128"][2][i])[::-1][:10]; print("   joint scores top10 (b128):", np.round(Jb, 6))
