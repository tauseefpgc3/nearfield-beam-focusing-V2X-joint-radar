"""Sensor-level diagnostic on 300 TRAINING users (no method, no test data): fraction of RA
peaks within 3 m of the vehicle antenna, by RA processing and nominal SNR."""
import sys, os, h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import p1_radar as P
dev = P.dev; U = h5py.File(P.OUT + "/users.h5"); ant = U["ant_xyz"][:]; psi = U["psi"][:]
tr = np.sort(h5py.File(P.OUT + "/split.h5")["train"][:])[:300]
g = h5py.File(P.OUT + "/radar_geometry.h5"); Rt = g["R_true"][:]; St = g["sin_true"][:]
ns, nc, nv = (torch.arange(n, device=dev, dtype=torch.float64) for n in (256, 64, 12))
wr = torch.hann_window(256, periodic=False, device=dev); wd = torch.hann_window(64, periodic=False, device=dev)
steer = torch.exp(-1j*np.pi*nv[None]*torch.sin(torch.tensor(P.ANG, device=dev))[:, None])
clean = []
for i in tr:
    R, S, V, A = (torch.tensor(z, device=dev) for z in P.reflectors(i, ant[i, :2], psi[i]))
    ph = 2*np.pi*(R[:,None,None,None]/0.1/256)*ns + 2*np.pi*(V[:,None,None,None]/64)*nc[:,None] + np.pi*S[:,None,None,None]*nv[:,None,None]
    clean.append((A[:,None,None,None]*torch.exp(1j*ph)).sum(0))
clean = torch.stack(clean)
rng = np.random.default_rng(99)
noise = torch.tensor(rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape), device=dev) / np.sqrt(2)
def ra_maps(cube):
    x = torch.fft.fft(cube*wr, dim=-1); x = torch.fft.fftshift(torch.fft.fft(x*wd[:, None], dim=-2), dim=-2)
    bf = (torch.einsum("am,bmdk->badk", steer.to(x.dtype), x).abs()**2)          # (B,101,64,256)
    return {"sum_doppler (D-029)": bf.sum(2), "max_doppler": bf.amax(2), "zero_doppler": bf[:, :, 32]}
def within3(m):
    m = m.transpose(1, 2).clone(); m[:, :3] = -1
    k, a = np.divmod(m.reshape(len(tr), -1).argmax(1).cpu().numpy(), 101); R = k*0.1; S = np.sin(P.ANG[a])
    e = np.hypot(R*S - Rt[tr]*St[tr], R*np.sqrt(1-S**2) - Rt[tr]*np.sqrt(1-St[tr]**2)); return (e < 3).mean()
for snr in (40, 30, 20, 10, 0):
    s = np.sqrt(256*64*12/10**(snr/10)); res = {}
    for b in range(0, 300, 50):
        for k, v in ra_maps(clean[b:b+50] + s*noise[b:b+50]).items(): res.setdefault(k, []).append(v)
    print(f"integrated SNR {snr:3d} dB: " + " | ".join(f"{k} {within3(torch.cat(v)):.2f}" for k, v in res.items()))
