# Table 7 — Sensor-driven polar vs DFT-only codebook (near-field K=8 dB-loss vs polar genie, 5 seeds, mean±95%CI)

| Codebook (sensor-driven) | 60 GHz | 140 GHz |
|---|---|---|
| Polar (angle×range, ours) | 0.85±0.66 | 1.60±0.64 |
| DFT-only (angle, j=0 slice) | 8.17±0.80 | 7.00±0.27 |
| Penalty (DFT − polar) | +7.31±0.51 | +5.40±0.65 |

Lower is better. Restricting the same model to the angle-only DFT slice (range=∞) cannot focus in range and pays the listed near-field penalty; the polar codebook recovers it.
