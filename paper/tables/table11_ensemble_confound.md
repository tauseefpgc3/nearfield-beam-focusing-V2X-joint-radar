# Table 11 — Modality vs ensembling confound check (NF K=8 dB-loss)

**Panel A — n=20 eval-only cross-seed ensemble (leaky, invalid):**

| Freq | (a) radar | (b) radar×3 ens | (c) multimodal | (b) leakage | (c)−(b) p [W] | (c)−(b) d |
|---|---|---|---|---|---|---|
| 60 GHz | 0.73±0.20 | 0.13±0.02 | 0.37±0.08 | 89% | 0.000 [0.000] | -1.61 |
| 140 GHz | 2.04±1.06 | 0.26±0.09 | 0.91±0.25 | 89% | 0.000 [0.000] | -1.20 |

**Panel A verdict:** **Inconclusive — the eval-only radar ensemble (b) is invalid.** Its members were trained on other seeds' splits, and **~89% of each eval seed's near-field test trajectories fall in >=1 member's training set**, so (b) has effectively memorised the test data (its 0.13/0.26 dB is a leakage artifact, not honest ensembling). The apparent '(b) beats (c)' therefore does **not** show that ensembling explains the gain. The leakage-free evidence is the D-019 clean control (Panel B) and the degradation test (n=20): dropping LiDAR raises joint-top-1 error (clean 0.655 → lidar-out 0.604), while radar dominates (radar-out 0.337).


**Panel B — D-019 clean same-split control (leakage-free):**

| Freq | Split | (b') radar×3 ens | (c') multimodal | Δ(b−c) [95% CI] | boot p | leakage |
|---|---|---|---|---|---|---|
| 60 GHz | 11 | 0.69 | 0.67 | +0.02 [-0.14, +0.22] | 0.841 | 0.000 |
| 60 GHz | 22 | 0.20 | 0.31 | -0.11 [-0.32, +0.05] | 0.214 | 0.000 |
| 60 GHz | 33 | 0.70 | 0.54 | +0.16 [-0.10, +0.44] | 0.248 | 0.000 |
| 140 GHz | 11 | 1.89 | 1.50 | +0.39 [-0.43, +1.26] | 0.372 | 0.000 |
| 140 GHz | 22 | 1.65 | 1.39 | +0.26 [-0.05, +0.67] | 0.115 | 0.000 |
| 140 GHz | 33 | 0.28 | 0.88 | -0.60 [-1.20, -0.09] | 0.022 | 0.000 |

**Panel B verdict (D-019):** **Mixed across splits/frequencies** — reported as mixed per D-019 (no clean, consistent modality effect). All leakage entries are 0.000 (assertion enforced in code).
