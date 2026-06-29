# Table 6 — Fusion combination-rule robustness (NF K=8 dB-loss, 5 seeds, mean±95%CI; paired t-test vs radar-alone)

| Combination rule | 60 GHz dB-loss | p vs radar | 140 GHz dB-loss | p vs radar |
|---|---|---|---|---|
| Radar-alone (reference) | 0.85±0.66 | — | 1.60±0.64 | — |
| Soft-avg (C3 confidence-weighted) | 0.55±0.22 | 0.254 | 1.20±0.35 | 0.061 |
| Argmax-confidence (both heads) | 0.79±0.34 | 0.704 | 1.66±0.98 | 0.901 |
| Radar angle + best-range (D-015, headline) | 0.70±0.45 | 0.243 | 1.70±0.64 | 0.469 |
| Radar angle + radar range | 0.85±0.66 | 1.000 | 1.60±0.64 | 1.000 |
| Temperature C3 (T=4) | 0.53±0.25 | 0.153 | 1.07±0.52 | 0.044* |
| Temperature C3 (T=8) | 0.61±0.31 | 0.169 | 1.12±0.77 | 0.147 |

Lower dB-loss is better; `p` is the paired t-test of each rule vs the radar-alone reference. The pre-registered headline rule (radar angle + best-range, D-015) is **not significant** at either frequency (p=0.243/0.469). Some C3/temperature variants have lower means, but only **1 of 12** rule×frequency comparisons reaches p<0.05 (temp4 at 140 GHz, p=0.044), which does **not** survive multiple-comparison correction (Bonferroni-adjusted ≈0.53). Conclusion: no combination rule is a statistically robust improvement over radar-alone — the radar-primary result is insensitive to the rule.
