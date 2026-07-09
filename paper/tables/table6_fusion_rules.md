# Table 6 — Fusion combination-rule robustness (NF K=8 dB-loss, 20 seeds, mean±95%CI; paired t-test vs radar-alone)

| Combination rule | 60 GHz dB-loss | p vs radar | 140 GHz dB-loss | p vs radar |
|---|---|---|---|---|
| Radar-alone (reference) | 0.73±0.20 | — | 2.04±1.06 | — |
| Soft-avg (C3 confidence-weighted) | 0.37±0.08 | 0.001* | 0.91±0.25 | 0.016* |
| Argmax-confidence (both heads) | 0.57±0.14 | 0.079 | 1.36±0.43 | 0.100 |
| Radar angle + best-range (D-015, single-sensor rule) | 0.55±0.15 | 0.030* | 1.49±0.43 | 0.152 |
| Radar angle + radar range | 0.67±0.24 | 0.518 | 1.42±0.44 | 0.096 |
| Temperature C3 (T=4) | 0.38±0.09 | 0.001* | 0.91±0.32 | 0.010* |
| Temperature C3 (T=8) | 0.44±0.11 | 0.002* | 0.98±0.33 | 0.016* |

Lower dB-loss is better; `p` is the paired t-test of each rule vs the radar-alone reference. The pre-registered single-sensor rule (D-015, radar angle + best-range) has p=0.030/0.152 (60/140 GHz). **7 of 12** rule×frequency comparisons reach p<0.05 (Soft-avg (C3 confidence-weighted) @60GHz, Soft-avg (C3 confidence-weighted) @140GHz, Radar angle + best-range (D-015, single-sensor rule) @60GHz, Temperature C3 (T=4) @60GHz, Temperature C3 (T=4) @140GHz, Temperature C3 (T=8) @60GHz, Temperature C3 (T=8) @140GHz); the lowest (0.001) gives Bonferroni-adjusted ≈0.01. See Table 9/Table 11 Panel B for the full battery and the clean modality-vs-ensembling control.
