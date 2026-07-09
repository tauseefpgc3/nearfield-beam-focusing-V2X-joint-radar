# Table 7 — Sensor-driven polar vs DFT-only codebook (near-field K=8 dB-loss vs polar genie, 5 seeds, mean±95%CI)

| Codebook (sensor-driven) | 60 GHz | 140 GHz |
|---|---|---|
| Polar (angle×range, ours) | 0.73±0.20 | 2.04±1.06 |
| DFT-only (angle, j=0 slice) | 8.00±0.22 | 7.23±0.32 |
| Penalty (DFT − polar) | +7.26±0.25 | +5.18±0.81 |

Lower is better. Restricting the same model to the angle-only DFT slice (range=∞) cannot focus in range and pays the listed near-field penalty; the polar codebook recovers it.
