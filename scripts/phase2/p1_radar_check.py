"""Sensor-level report (final setup): fraction of users for whom the vehicle is the strongest
range-angle return (RA peak, range bins >= 3, within 3 m of the antenna), all 6000 users."""
import h5py, json, numpy as np
P = "data/processed/phase2/"
g = h5py.File(P + "radar_geometry.h5"); Rt = g["R_true"][:]; St = g["sin_true"][:]
ang = np.deg2rad(np.arange(-75, 75.0001, 1.5)); out = {}
for s in (20, 10, 0):
    X = h5py.File(P + f"radar_snr{s}.h5")["radar_features"]; hit = []
    for b in range(0, 6000, 500):
        RA = X[b:b + 500, 1, :, :101].copy(); RA[:, :3] = -1
        k, a = np.divmod(RA.reshape(len(RA), -1).argmax(1), 101); R = k * 0.1; S = np.sin(ang[a]); i = slice(b, b + 500)
        hit.append(np.hypot(R * S - Rt[i] * St[i], R * np.sqrt(1 - S**2) - Rt[i] * np.sqrt(1 - St[i]**2)) < 3)
    out[str(s)] = float(np.concatenate(hit).mean()); print(s, "dB: vehicle strongest return for", round(out[str(s)], 3), "of users")
json.dump(out, open("results/phase2/checks_radar.json", "w"), indent=1)
