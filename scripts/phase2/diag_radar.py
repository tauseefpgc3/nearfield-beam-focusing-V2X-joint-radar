"""Diagnostic (training users only): why is the RA peak rarely at the vehicle?"""
import sys, os, h5py, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import p1_radar as P
dev = P.dev
OUT = P.OUT
ant = h5py.File(OUT + "/users.h5")["ant_xyz"][:]; psi = h5py.File(OUT + "/users.h5")["psi"][:]
tr = np.sort(h5py.File(OUT + "/split.h5")["train"][:])[:300]
g = h5py.File(OUT + "/radar_geometry.h5"); Rt = g["R_true"][:]; St = g["sin_true"][:]
ns = torch.arange(256, device=dev, dtype=torch.float64); nc = torch.arange(64, device=dev, dtype=torch.float64); nv = torch.arange(12, device=dev, dtype=torch.float64)
def cube(i, veh=True, clut=True):
    R, S, V, A = P.reflectors(i, ant[i, :2], psi[i]); m = np.r_[[veh] * 6, [clut] * 12]
    R, S, V, A = (torch.tensor(z[m], device=dev) for z in (R, S, V, A))
    ph = 2*np.pi*(R[:,None,None,None]/0.1/256)*ns + 2*np.pi*(V[:,None,None,None]/64)*nc[:,None] + np.pi*S[:,None,None,None]*nv[:,None,None]
    return (A[:,None,None,None]*torch.exp(1j*ph)).sum(0)
def err(feat_ra, idx):
    RA = feat_ra.clone(); RA[:, :3, :] = -1e9
    k, a = np.divmod(RA.reshape(len(idx), -1).argmax(1).cpu().numpy(), 101)
    R = k * 0.1; S = np.sin(P.ANG[a])
    return np.hypot(R*S - Rt[idx]*St[idx], R*np.sqrt(1-S**2) - Rt[idx]*np.sqrt(1-St[idx]**2))
for tag, veh, clut, snr in (("vehicle only, no noise", 1, 0, None), ("veh+clutter, no noise", 1, 1, None), ("veh+clutter, 20 dB", 1, 1, 20)):
    cs = []
    for i in tr:
        c = cube(i, veh, clut)
        if snr is not None:
            r = np.random.default_rng(1234 + i); s2 = 256*64*12/10**(snr/10)
            c = c + torch.tensor(np.sqrt(s2/2)*(r.standard_normal(c.shape)+1j*r.standard_normal(c.shape)), device=dev)
        cs.append(c.to(torch.complex64))
    feat, _ = P.process(torch.stack(cs))
    e = err(feat[:, 1, :, :101], tr)
    print(f"{tag:26s} within 3 m: {(e<3).mean():.3f} | median err {np.median(e):.2f} m")
# check: where does a single on-axis scatterer at 10 m land?
R = torch.tensor([10.0], device=dev, dtype=torch.float64); S = torch.tensor([0.3], device=dev, dtype=torch.float64)
ph = 2*np.pi*(R[:,None,None,None]/0.1/256)*ns + np.pi*S[:,None,None,None]*nv[:,None,None]
feat, _ = P.process(torch.exp(1j*ph).sum(0, keepdim=True).to(torch.complex64))
k, a = np.divmod(int(feat[0, 1, :, :101].argmax()), 101)
print("single point R=10 m, sin=0.3 -> peak range", k*0.1, "m, angle", np.rad2deg(P.ANG[a]), "deg (expect", np.rad2deg(np.arcsin(0.3)), ")")
