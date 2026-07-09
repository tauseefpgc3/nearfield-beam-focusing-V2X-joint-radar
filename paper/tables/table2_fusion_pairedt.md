# Table 2 — Fusion vs radar-alone (NF K=8 dB-loss, n=20 seeds, D-018)

| Freq | Comparison | radar / rule | paired-t p | Wilcoxon p | Cohen d [95% CI] | TOST±0.5 | sig? |
|---|---|---|---|---|---|---|---|
| 60 GHz | vs Soft-avg (C3) | 0.73 / 0.37 | 0.001 | 0.001 | +0.83 [+0.56, +1.32] | — | yes |
| 60 GHz | vs D-015 radar-primary | 0.73 / 0.55 | 0.030 | 0.017 | +0.52 [+0.21, +1.01] | equiv | yes |
| 140 GHz | vs Soft-avg (C3) | 2.04 / 0.91 | 0.016 | 0.001 | +0.59 [+0.43, +1.08] | — | yes |
| 140 GHz | vs D-015 radar-primary | 2.04 / 1.49 | 0.152 | 0.231 | +0.33 [-0.02, +0.61] | — | no |

Lower dB-loss is better. At n=20, soft-avg (C3) is significantly below radar-alone at both frequencies (and survives Bonferroni — any-rule-significant=True); D-015 radar-primary is TOST-equivalent to radar-alone at 60 GHz. See the D-019 clean control (Table 11 Panel B): the soft-avg gain is largely generic ensembling, not the LiDAR modality.
