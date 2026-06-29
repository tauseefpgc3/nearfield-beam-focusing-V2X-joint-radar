# Table 3 — DeepSense6G real-data validation (top-1 beam accuracy, mean±95%CI)

| Scenario | Link | Modality | Seeds | n(test) | Top-1 beam (%) |
|---|---|---|---|---|---|
| Sc 31 | V2I (RSU) | Radar | 5 | 1374 | 23.05±2.38 |
| Sc 31 | V2I (RSU) | LiDAR | 5 | 1374 | 10.78±2.12 |
| Sc 31 | V2I (RSU) | Fusion | 5 | — | 22.55±2.26 |
| Sc 36 | V2V | Radar | 5 | 1602 | 17.14±5.40 |

**Zero-shot transfer** (random chance = 1.56%):

| Transfer | n | Top-1 beam (%) |
|---|---|---|
| real V2I -> real V2V | 8267 | 1.59±0.37 |
| synthetic close-range -> real V2I | 7012 | 0.84±0.75 |

In-domain rows: model trained and tested within scenario (trajectory-grouped split). Zero-shot rows collapse to ≈chance, showing the learned features are scene-specific (no free real-world transfer).
