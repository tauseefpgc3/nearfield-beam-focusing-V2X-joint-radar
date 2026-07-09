# Table 12 — Future-beam prediction vs horizon (DeepSense radar, 5 seeds, mean±95%CI)

| Scenario | Δ (frames) | Horizon (s) | Disp (m) | Top-1 (%) | Cov@4 (%) | Cov@8 (%) |
|---|---|---|---|---|---|---|
| V2I (Sc31) (anchor) | 0 | 0.0 | 0.00 | 23.08±2.91 | 55.6 | 69.7 |
| V2I (Sc31) | 1 | 0.1 | 0.39 | 22.98±2.72 | 55.5 | 68.1 |
| V2I (Sc31) | 2 | 0.2 | 0.77 | 23.39±3.05 | 55.6 | 69.9 |
| V2I (Sc31) | 3 | 0.3 | 1.16 | 23.83±2.64 | 55.8 | 70.3 |
| V2V (Sc36) (anchor) | 0 | 0.0 | 0.00 | 15.80±5.93 | 35.6 | 46.4 |
| V2V (Sc36) | 1 | 0.1 | 0.79 | 16.41±5.68 | 37.2 | 48.9 |
| V2V (Sc36) | 2 | 0.2 | 1.07 | 17.11±6.66 | 37.0 | 48.4 |
| V2V (Sc36) | 3 | 0.3 | 1.49 | 17.02±6.57 | 37.7 | 49.1 |

Δ=0 is the same-frame anchor (reproduces the banked DeepSense accuracy). Δ>0 predicts the beam that many frames ahead (0.1 s/frame nominal). Achieved trend over 0→3 frames (0→0.3 s): V2I flat (23.1%→23.8%); V2V flat (15.8%→17.0%). Top-1 accuracy and top-K coverage are essentially **stable across the 0–0.3 s horizon** (within 95% CI of the same-frame anchor) — at these short horizons the beam is highly predictable from the current frame, so the predictor operates under genuine mobility without loss.
