"""Phase 2a diagnostic (moved from the session scratchpad under D-030). Radar picker position error at 20 dB and loss vs a lateral offset of the true position."""
import sys, os, json, h5py, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT,"scripts","phase2"))
import p3_eval as E
P="data/processed/phase2"
with h5py.File(f"{P}/split.h5") as hf: te=np.sort(hf["test"][:])
with h5py.File(f"{P}/radar_geometry.h5") as hf: Rt=hf["R_true"][:][te]; St=hf["sin_true"][:][te]
with h5py.File(f"{P}/users.h5") as hf: ant=hf["ant_xyz"][:][te]
X,lo,hi=E.load_radar(20,te); db=E.ra_db(X,lo,hi)
res=json.load(open("results/phase2/gate_results.json"))
print("chosen", {f:res["freqs"][f]["picker_chosen"] for f in ("60","140")}, "bias", res["bias"])
for kind in ("peak","centroid"):
    R,S=E.estimate(db,kind)
    eR=R-Rt; eA=np.rad2deg(np.arcsin(np.clip(S,-1,1))-np.arcsin(St))
    lat=np.abs(Rt*(S-St))
    print(kind,"|dR| med %.2f p90 %.2f m; |dAng| med %.2f p90 %.2f deg; lateral err med %.2f m; Rtrue med %.1f"%(np.median(abs(eR)),np.percentile(abs(eR),90),np.median(abs(eA)),np.percentile(abs(eA),90),np.median(lat),np.median(Rt)))
# loss for picker fed true geometry on test, and with lateral errors of size 0.25/0.5/1 m
with h5py.File(f"{P}/channels_140GHz.h5") as hf: G=hf["gains"][:][te]; ug=hf["u_grid"][:]; rings=hf["rings"][:]
for off in (0,0.1,0.25,0.5,1.0):
    S2=np.clip(St+off/Rt,-1,1); oa,orr=E.to_codeword(Rt,S2,ug,rings)
    print("140 GHz, true R, lateral offset %.2f m: K1 %.2f K8 %.2f dB"%(off,E.loss_of(G,E.picker_lists(oa,orr,"c1")[:,:1]).mean(),E.loss_of(G,E.picker_lists(oa,orr,"c1")).mean()))
