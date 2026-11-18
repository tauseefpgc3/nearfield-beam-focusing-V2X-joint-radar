"""Phase 2a diagnostic (moved from the session scratchpad under D-030). Can RadarPolarModel overfit 64 training users at 20 dB (fp32 vs fp16)?"""
import sys, os, h5py, numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
from src.models.radar_encoder import RadarPolarModel
from src.train.loss import polar_loss
P="data/processed/phase2"
with h5py.File(f"{P}/split.h5") as hf: tr=np.sort(hf["train"][:]); si=hf["snr_idx"][:]
ix=tr[si[tr]==0][:64]
with h5py.File(f"{P}/radar_snr20.h5") as hf: X=torch.from_numpy(hf["radar_features"][ix]).cuda()
with h5py.File(f"{P}/channels_60GHz.h5") as hf: yt=torch.from_numpy(hf["y_theta"][:][ix].astype(np.int64)).cuda(); yr=torch.from_numpy(hf["y_r"][:][ix].astype(np.int64)).cuda()
print("input stats: mean %.3f std-across-users of RA ch %.4f" % (X.mean(), X[:,1].std(0).mean()))
print("label prior: n distinct theta", len(set(yt.tolist())))
torch.manual_seed(0)
for prec in ("fp32","fp16"):
    m=RadarPolarModel(d_z=128,B_theta=256,B_r=8).cuda(); opt=torch.optim.Adam(m.parameters(),1e-4)
    sc=torch.amp.GradScaler("cuda")
    for it in range(400):
        opt.zero_grad()
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=prec=="fp16"):
            o=m(X); l=polar_loss(o,yt,yr)[0]
        sc.scale(l).backward(); sc.step(opt); sc.update()
        if it%100==99:
            with torch.no_grad(): o=m(X)
            print(prec,it+1,"loss %.3f acc_theta %.2f"%(float(l),float((o["theta_logits"].argmax(1)==yt).float().mean())), "emb std across users %.4f"%float(m.encoder(X).std(0).mean()))
