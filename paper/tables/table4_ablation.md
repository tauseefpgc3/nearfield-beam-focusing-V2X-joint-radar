# Table 4 — Range-codebook resolution ablation (E10, 3 seeds, mean±95%CI)

| B_r | NF dB-loss@4* | NF dB-loss@8* | NF range top-1 (%)* | Populated bins | Abs. gain@8 (dB)† |
|---|---|---|---|---|---|
| 4 | 0.73±0.76 | 0.24±0.12 | 56.7±7.8 | 3.0/4 | -2.35±1.88 |
| 8 (kept) | 1.81±1.52 | 0.66±0.71 | 39.4±9.7 | 7.0/8 | -1.05±0.95 |
| 16 | 5.32±12.12 | 3.24±7.17 | 35.0±5.8 | 12.0/16 | -3.24±7.17 |

\*Per-gate dB-loss and top-1 range are **not comparable across B_r**: each is measured on that B_r's own gated near-field set against its own (coarser) genie, so a smaller B_r is flattered.

†The comparable metric: **absolute** realized beam-focusing gain at K=8 on a **common** near-field set (identical users), referenced to the finest B_r=16 genie. On this metric **B_r=8 wins** (-1.05 dB) and is kept (D-016) — B_r=4 loses by under-resolving range, B_r=16 by sparse, noisy bins.

_Note: this table stays at 3 seeds — the far-field / range-codebook models are **not retrained in the n=20 extension** (which covered only the 60/140 GHz close-range radar + fusion predictors); the genie/absolute-gain reference columns are seed-independent._
