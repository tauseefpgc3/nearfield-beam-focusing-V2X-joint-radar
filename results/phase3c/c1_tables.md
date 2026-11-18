## Headline (varied test, all users): varied-trained net better

| f, M | bootstrap Δ [CI] | seed Δ [CI] | Holm p | Wilcoxon p |
|---|---|---|---|---|
| 60|8 | -2.91 [-3.51, -2.33] | -2.91 [-2.97, -2.84] | 2.0e-26 | 1.9e-06 |
| 60|16 | -2.53 [-3.07, -2.01] | -2.53 [-2.58, -2.47] | 2.0e-26 | 1.9e-06 |
| 140|8 | -2.93 [-3.53, -2.34] | -2.93 [-2.99, -2.86] | 1.7e-26 | 1.9e-06 |
| 140|16 | -2.55 [-3.10, -2.03] | -2.55 [-2.61, -2.49] | 2.0e-26 | 1.9e-06 |

### varied test set, 60 GHz, all (n = 1193): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 6.14 [5.52, 6.79] | 4.43 [3.89, 5.01] | 2.22 [1.81, 2.66] | 0.89 [0.65, 1.15] | 0.72 |
| LiDAR net, single vehicle | 2.43 [2.36, 2.50] | 1.30 [1.24, 1.37] | 0.59 [0.55, 0.64] | 0.24 [0.21, 0.27] | 0.86 |
| LiDAR net, varied-trained | 3.23 [3.17, 3.30] | 1.91 [1.85, 1.96] | 0.90 [0.86, 0.95] | 0.33 [0.30, 0.36] | 0.82 |

### varied test set, 140 GHz, all (n = 1193): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 6.21 [5.59, 6.88] | 4.49 [3.95, 5.08] | 2.26 [1.84, 2.70] | 0.87 [0.62, 1.13] | 0.72 |
| LiDAR net, single vehicle | 2.48 [2.41, 2.55] | 1.34 [1.28, 1.40] | 0.61 [0.56, 0.66] | 0.23 [0.20, 0.25] | 0.86 |
| LiDAR net, varied-trained | 3.29 [3.22, 3.35] | 1.94 [1.88, 2.00] | 0.92 [0.88, 0.96] | 0.32 [0.29, 0.35] | 0.83 |

### varied test set, 60 GHz, adv_ge3 (n = 195): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 16.38 [14.46, 18.34] | 12.56 [10.59, 14.57] | 8.34 [6.57, 10.25] | 4.65 [3.40, 6.04] | 0.30 |
| LiDAR net, single vehicle | 11.05 [10.80, 11.29] | 6.60 [6.38, 6.83] | 3.04 [2.84, 3.25] | 1.25 [1.13, 1.38] | 0.43 |
| LiDAR net, varied-trained | 13.22 [12.97, 13.47] | 8.79 [8.51, 9.06] | 4.24 [3.97, 4.51] | 1.62 [1.44, 1.80] | 0.35 |

### varied test set, 140 GHz, adv_ge3 (n = 47): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 25.67 [22.59, 28.55] | 22.46 [18.95, 25.76] | 15.12 [11.14, 19.07] | 7.15 [3.95, 10.66] | 0.06 |
| LiDAR net, single vehicle | 19.41 [18.82, 20.00] | 14.71 [14.08, 15.34] | 9.20 [8.62, 9.77] | 3.09 [2.74, 3.44] | 0.21 |
| LiDAR net, varied-trained | 22.80 [22.35, 23.25] | 18.75 [18.23, 19.28] | 11.67 [11.11, 12.23] | 4.43 [3.85, 5.01] | 0.12 |

### varied test set, 60 GHz, d_ge_2m (n = 1137): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 5.29 [4.70, 5.88] | 3.68 [3.17, 4.21] | 1.70 [1.34, 2.09] | 0.61 [0.41, 0.83] | 0.75 |
| LiDAR net, single vehicle | 1.69 [1.63, 1.75] | 0.75 [0.70, 0.80] | 0.25 [0.21, 0.29] | 0.10 [0.08, 0.13] | 0.89 |
| LiDAR net, varied-trained | 2.41 [2.35, 2.48] | 1.23 [1.18, 1.29] | 0.50 [0.46, 0.53] | 0.16 [0.14, 0.18] | 0.86 |

### varied test set, 140 GHz, d_ge_2m (n = 1137): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 5.33 [4.73, 5.93] | 3.70 [3.18, 4.23] | 1.70 [1.34, 2.09] | 0.57 [0.38, 0.79] | 0.76 |
| LiDAR net, single vehicle | 1.70 [1.64, 1.76] | 0.75 [0.70, 0.80] | 0.24 [0.20, 0.28] | 0.09 [0.07, 0.11] | 0.89 |
| LiDAR net, varied-trained | 2.42 [2.35, 2.49] | 1.23 [1.17, 1.28] | 0.49 [0.45, 0.52] | 0.15 [0.12, 0.17] | 0.86 |

### varied test set: horizontal position error (cm), median / p90

| Method | all | adv ≥ 3 dB (60 GHz) | ≥ 2 m |
|---|---|---|---|
| Geometric | 23.4 / 225.6 | 30.3 / 167.7 | 22.7 / 227.1 |
| LiDAR net, single vehicle | 18.8 / 43.0 | 23.0 / 63.8 | 18.3 / 41.2 |
| LiDAR net, varied-trained | 20.1 / 46.6 | 29.2 / 105.3 | 19.4 / 43.5 |

### standard test set, 60 GHz, all (n = 1193): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 1.81 [1.42, 2.23] | 1.46 [1.11, 1.85] | 1.04 [0.75, 1.36] | 0.62 [0.42, 0.85] | 0.92 |
| LiDAR net, single vehicle | 0.87 [0.81, 0.93] | 0.61 [0.56, 0.66] | 0.39 [0.34, 0.43] | 0.19 [0.16, 0.22] | 0.96 |
| LiDAR net, varied-trained | 2.27 [2.18, 2.37] | 1.39 [1.31, 1.46] | 0.74 [0.68, 0.79] | 0.29 [0.25, 0.33] | 0.88 |

### standard test set, 140 GHz, all (n = 1193): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 1.88 [1.48, 2.31] | 1.52 [1.16, 1.92] | 1.07 [0.76, 1.39] | 0.63 [0.41, 0.86] | 0.92 |
| LiDAR net, single vehicle | 0.92 [0.86, 0.98] | 0.65 [0.60, 0.70] | 0.40 [0.36, 0.45] | 0.18 [0.15, 0.21] | 0.96 |
| LiDAR net, varied-trained | 2.32 [2.22, 2.42] | 1.42 [1.35, 1.50] | 0.76 [0.71, 0.81] | 0.29 [0.25, 0.33] | 0.88 |

### standard test set, 60 GHz, adv_ge3 (n = 195): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 9.06 [7.12, 11.07] | 7.85 [6.02, 9.78] | 5.81 [4.29, 7.47] | 3.44 [2.31, 4.67] | 0.66 |
| LiDAR net, single vehicle | 4.05 [3.77, 4.32] | 2.78 [2.56, 3.01] | 1.86 [1.66, 2.07] | 0.97 [0.82, 1.13] | 0.79 |
| LiDAR net, varied-trained | 10.10 [9.53, 10.67] | 6.36 [5.93, 6.80] | 3.48 [3.18, 3.77] | 1.45 [1.27, 1.64] | 0.48 |

### standard test set, 140 GHz, adv_ge3 (n = 47): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 22.51 [18.10, 26.78] | 20.38 [15.86, 24.73] | 15.48 [11.20, 19.69] | 10.58 [7.11, 14.23] | 0.28 |
| LiDAR net, single vehicle | 12.92 [12.20, 13.63] | 9.08 [8.37, 9.79] | 5.74 [5.09, 6.38] | 2.41 [1.90, 2.93] | 0.41 |
| LiDAR net, varied-trained | 21.06 [20.26, 21.87] | 17.08 [16.40, 17.76] | 10.67 [10.12, 11.23] | 3.84 [3.39, 4.30] | 0.18 |

### standard test set, 60 GHz, d_ge_2m (n = 1137): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 0.86 [0.60, 1.13] | 0.59 [0.37, 0.84] | 0.40 [0.22, 0.60] | 0.25 [0.12, 0.40] | 0.95 |
| LiDAR net, single vehicle | 0.37 [0.33, 0.41] | 0.26 [0.23, 0.30] | 0.15 [0.12, 0.19] | 0.08 [0.06, 0.11] | 0.98 |
| LiDAR net, varied-trained | 1.49 [1.41, 1.58] | 0.76 [0.69, 0.82] | 0.35 [0.31, 0.39] | 0.15 [0.12, 0.18] | 0.91 |

### standard test set, 140 GHz, d_ge_2m (n = 1137): loss vs genie, dB [95% CI]

| Method | M=8 | M=16 | M=32 | M=64 | within 1 dB (M=8) |
|---|---|---|---|---|---|
| Geometric | 0.89 [0.62, 1.17] | 0.62 [0.39, 0.87] | 0.39 [0.21, 0.60] | 0.22 [0.10, 0.37] | 0.96 |
| LiDAR net, single vehicle | 0.38 [0.34, 0.42] | 0.28 [0.24, 0.31] | 0.15 [0.12, 0.18] | 0.07 [0.05, 0.09] | 0.98 |
| LiDAR net, varied-trained | 1.50 [1.41, 1.59] | 0.76 [0.69, 0.82] | 0.34 [0.30, 0.38] | 0.14 [0.11, 0.17] | 0.91 |

### standard test set: horizontal position error (cm), median / p90

| Method | all | adv ≥ 3 dB (60 GHz) | ≥ 2 m |
|---|---|---|---|
| Geometric | 6.6 / 58.9 | 5.3 / 117.1 | 6.4 / 23.2 |
| LiDAR net, single vehicle | 3.1 / 9.3 | 6.1 / 45.4 | 2.9 / 7.6 |
| LiDAR net, varied-trained | 12.9 / 34.7 | 20.3 / 103.2 | 12.5 / 31.4 |

## Secondary: varied-trained − single-vehicle (bootstrap Δ dB [CI], verdict), all users

| test set | f, M | Δ | verdict |
|---|---|---|---|
| varied | 60, 8 | +0.80 [+0.58, +1.02] | varied-trained worse |
| varied | 60, 16 | +0.60 [+0.42, +0.79] | varied-trained worse |
| varied | 60, 32 | +0.31 [+0.19, +0.43] | varied-trained worse |
| varied | 60, 64 | +0.09 [+0.03, +0.16] | varied-trained worse |
| varied | 140, 8 | +0.80 [+0.58, +1.02] | varied-trained worse |
| varied | 140, 16 | +0.60 [+0.41, +0.78] | varied-trained worse |
| varied | 140, 32 | +0.31 [+0.19, +0.44] | varied-trained worse |
| varied | 140, 64 | +0.09 [+0.03, +0.16] | varied-trained worse |
| standard | 60, 8 | +1.41 [+1.19, +1.64] | varied-trained worse |
| standard | 60, 16 | +0.78 [+0.60, +0.96] | varied-trained worse |
| standard | 60, 32 | +0.35 [+0.23, +0.48] | varied-trained worse |
| standard | 60, 64 | +0.10 [+0.04, +0.16] | varied-trained worse |
| standard | 140, 8 | +1.40 [+1.19, +1.63] | varied-trained worse |
| standard | 140, 16 | +0.77 [+0.59, +0.96] | varied-trained worse |
| standard | 140, 32 | +0.35 [+0.23, +0.49] | varied-trained worse |
| standard | 140, 64 | +0.11 [+0.04, +0.17] | varied-trained worse |