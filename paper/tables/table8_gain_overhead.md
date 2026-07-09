# Table 8 — Beam-focusing gain vs RF-measurement overhead (n=20), mean±95%CI

| Method | # RF meas. | 60 GHz dB-loss | 140 GHz dB-loss |
|---|---|---|---|
| Exhaustive polar sweep (genie) | 512 | 0.00 | 0.00 |
| Hierarchical sweep (angle→range) | 71 | 2.56±0.22 | 2.03±0.12 |
| Ours soft-avg (recommended), K=1 | 1 | 4.53±0.71 | 6.94±0.73 |
| Ours soft-avg (recommended), K=2 | 2 | 2.67±0.48 | 5.27±0.53 |
| Ours soft-avg (recommended), K=4 | 4 | 1.12±0.26 | 2.65±0.50 |
| Ours soft-avg (recommended), K=8 | 8 | 0.37±0.08 | 0.91±0.25 |
| Ours radar-alone, K=1 | 1 | 6.43±0.76 | 9.34±2.50 |
| Ours radar-alone, K=2 | 2 | 4.10±0.67 | 7.27±1.82 |
| Ours radar-alone, K=4 | 4 | 2.04±0.43 | 4.16±1.53 |
| Ours radar-alone, K=8 | 8 | 0.73±0.20 | 2.04±1.06 |

Lower dB-loss is better. The recommended soft-avg predictor reaches near-genie focusing with a K=8 shortlist — ~9× fewer RF measurements than the 71-measurement hierarchical sweep, at lower loss.
