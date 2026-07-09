# Table 1 — In-domain near-field results (n=20)

| Freq | Predictor | Angle top-1 | NF range top-1 | NF cov@4 | NF dB-loss@8 |
|---|---|---|---|---|---|
| 60 GHz | Soft-avg (recommended) | 78.9±2.3 | 46.4±2.4 | 72.7±3.0 | 0.37±0.08 |
| 60 GHz | Radar-alone | 74.0±2.1 | 38.8±2.6 | 64.2±3.6 | 0.73±0.20 |
| 140 GHz | Soft-avg (recommended) | 84.1±1.9 | 15.1±1.7 | 51.6±4.4 | 0.91±0.25 |
| 140 GHz | Radar-alone | 77.5±6.2 | 13.3±1.7 | 42.7±6.8 | 2.04±1.06 |

20-seed mean±95%CI (D-018). NF = near-field subset (3 dB gate). Soft-avg (C3) is the recommended multimodal predictor; radar-alone is the single-sensor operating point. LiDAR-alone is **not** retrained in the n=20 extension (it remains an n=5 reference — see the n=5 Table 1 variant).
