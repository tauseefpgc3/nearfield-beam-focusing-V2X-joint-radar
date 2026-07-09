# Table 9 — Radar vs fusion, per-seed losses + statistics (near-field K=8 dB-loss, n=20 seeds; D-018)

**Panel A — per-seed loss (dB):**

| Seed | 60: radar | 60: D-015 | 60: soft-avg | 140: radar | 140: D-015 | 140: soft-avg |
|---|---|---|---|---|---|---|
| 11 | 0.50 | 0.60 | 0.67 | 2.00 | 1.62 | 1.50 |
| 22 | 1.18 | 0.71 | 0.31 | 2.31 | 2.59 | 1.39 |
| 33 | 0.61 | 0.64 | 0.54 | 1.13 | 1.25 | 0.88 |
| 44 | 1.62 | 1.28 | 0.77 | 1.24 | 1.56 | 1.29 |
| 55 | 0.36 | 0.29 | 0.45 | 1.34 | 1.50 | 0.93 |
| 66 | 0.82 | 0.50 | 0.31 | 0.95 | 0.80 | 0.28 |
| 77 | 0.51 | 0.67 | 0.47 | 0.17 | 0.37 | 0.23 |
| 88 | 1.04 | 0.62 | 0.36 | 1.77 | 1.55 | 0.42 |
| 99 | 0.49 | 0.38 | 0.38 | 2.99 | 2.43 | 1.37 |
| 110 | 0.28 | 0.18 | 0.16 | 0.16 | 0.69 | 0.48 |
| 121 | 0.79 | 0.51 | 0.31 | 0.95 | 1.35 | 1.10 |
| 132 | 0.15 | 0.30 | 0.25 | 1.95 | 1.10 | 0.57 |
| 143 | 0.53 | 0.52 | 0.38 | 6.70 | 4.18 | 2.08 |
| 154 | 1.07 | 1.40 | 0.55 | 1.58 | 1.25 | 0.48 |
| 165 | 0.19 | 0.20 | 0.13 | 0.16 | 0.37 | 0.31 |
| 176 | 0.70 | 0.50 | 0.43 | 0.56 | 1.15 | 0.68 |
| 187 | 1.76 | 0.41 | 0.15 | 9.56 | 2.71 | 1.73 |
| 198 | 0.34 | 0.16 | 0.27 | 0.78 | 0.51 | 0.51 |
| 209 | 0.83 | 0.53 | 0.33 | 2.20 | 1.06 | 0.79 |
| 220 | 0.87 | 0.64 | 0.25 | 2.35 | 1.84 | 1.24 |

**Panel B — aggregate tests (radar-alone vs rule):**

| Comparison (freq) | radar / rule | paired-t p | Wilcoxon p | Cohen d [95% CI] | TOST±0.5 p | equiv? |
|---|---|---|---|---|---|---|
| radar vs D-015 (60 GHz) | 0.73 / 0.55 | 0.030 | 0.017 | +0.52 [+0.21, +1.01] | 0.000 | yes |
| radar vs soft-avg (60 GHz) | 0.73 / 0.37 | 0.001 | 0.001 | +0.83 [+0.56, +1.32] | 0.081 | no |
| radar vs D-015 (140 GHz) | 2.04 / 1.49 | 0.152 | 0.231 | +0.33 [-0.02, +0.61] | 0.553 | no |
| radar vs soft-avg (140 GHz) | 2.04 / 0.91 | 0.016 | 0.001 | +0.59 [+0.43, +1.08] | 0.921 | no |

**Verdict (n=20):** soft-avg (C3) fusion has **significantly lower loss than radar-alone at 60 GHz (p=0.001), 140 GHz (p=0.016)**; it **survives Bonferroni** (12 comparisons): 60GHz:soft_avg, 60GHz:temp4, 60GHz:temp8. Radar-primary (D-015) is TOST-**equivalent** to radar-alone (within ±0.5 dB) at 60 GHz. Per D-018 the reported outcome is whatever this powered study yields; if soft-avg is significant after correction the paper reframes to 'a soft ensemble gives a small significant gain; radar-primary is an equivalence-tested single-sensor operating point.'
