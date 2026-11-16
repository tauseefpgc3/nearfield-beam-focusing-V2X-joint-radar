# Design Decisions — Paper 5 (Sensing-Aided Near-Field Beam Focusing)

This file records non-obvious implementation choices that are *not* derivable from
the plan documents alone, with their justification. Update it whenever a convention
is fixed in code.

---

## D-001 — `N` disambiguation (the gating precondition)

The symbol `N` is overloaded in the plan (`02_SYSTEM_MODEL.md` uses `N = 32×32 = 1024`
for the steering-vector dimension *and* implicitly `N = 32` in the geometry that sets
`d_F`). In code these are **two distinct quantities**:

> **GEOMETRY UPDATE (D-011 resolved, supervisor-approved): the array is now 128×128.**
> All numbers below were updated from the original 32×32. `N_side = 128`,
> `N_elements = 16384`. Reason: 32×32 had **zero** near-field users in O1 at 60/140 GHz
> (verified `frac(d<d_F)=0.0000`); 128×128 lifts `d_F` to ~40/~17 m, past the ~8 m min
> user distance, creating a genuine near-field regime. See D-011 for the full rationale.

- `N_elements = 16384` — total UPA elements; the dimension of every steering vector
  (sum over all 16384 element positions). This is the `N` in
  `generate_polar_codebook(N=...)` and the steering-vector length.
- `N_side = 128` — per-side element count of the 128×128 UPA. This sets the **physical
  aperture**.

**Rule:** the Fraunhofer aperture `D` and the `(N-1)^2` term in the polar range
sampling use `N_side` (per-side geometry), **never** `16384`. Using `16384` for `D`
makes `d_F` kilometres and false-passes Gate-0 — a forbidden bug.

---

## D-002 — Fraunhofer aperture convention: **per-side**

`fraunhofer_distance(f_c, N_side=128, spacing=0.5)` uses the **per-side** aperture:

```
D   = (N_side - 1) * spacing * lambda      # physical aperture of one side, metres
d_F = 2 * D^2 / lambda
```

(The alternative diagonal convention `D = sqrt(2)*(N_side-1)*spacing*lambda` gives
~2× larger `d_F`.)

**Computed d_F at the 128×128 array** (c = 299792458 m/s, spacing = 0.5λ,
`D = 127*0.5*λ`):

| f_c    | lambda (m) | d_F (per-side, D-002) | supervisor ballpark |
|--------|-----------|------------------------|---------------------|
| 28 GHz | 0.010707  | **86.35 m**            | ~86 m               |
| 60 GHz | 0.004997  | **40.29 m**            | ~38 m               |
| 140 GHz| 0.002141  | **17.27 m**            | ~16 m               |

**Ballpark discrepancy (flagged for review):** the supervisor's `~38/~16 m` used the
`(128/32)^2 = 16` scaling from the old 32×32 values; the *exact* per-side formula keeps
the `-1`, i.e. `(127/31)^2 = 16.78`, giving `40.29/17.27 m` (~6–8% higher). The locked
D-002 formula is authoritative, so the **true targets are 86.35/40.29/17.27 m**; the
test `test_fraunhofer_matches_plan_targets` asserts these within 5%. Both are well past
the ~8 m min user distance, so the near-field-regime goal holds regardless. The old
`5.07/2.36/1.01 m` targets (32×32) are obsolete.

**Signature note (deliberate deviation from `05_REPO_STRUCTURE.md`):** docs/05
declares `fraunhofer_distance(f_c, N=1024, spacing=0.5)`. The session precondition
overrides docs/05 and mandates the `N_side` parameter; the default is now `128`. This is
the single intentional signature change, made because following docs/05 literally would
be incorrect.

---

## D-003 — Polar range sampling uses `N_side`

From `02_SYSTEM_MODEL.md` §1.3 (Cui & Dai 2022):

```
1 / r_j = j / ( beta_delta * (N_side - 1)^2 * lambda / 8 ),   j = 0..B_r-1
=>  r_j  = beta_delta * (N_side - 1)^2 * lambda / (8 * j)      (j >= 1)
    r_0  = +inf   (far-field; the DFT bin)
```

`(N-1)^2` here is `(N_side - 1)^2 = 127^2 = 16129`, **not** `16383^2`. `r_0 = inf` makes
the `j=0` column the far-field DFT codebook, so the polar codebook subsumes DFT.

---

## D-004 — Array geometry: 128×128 UPA in the y–z plane, user at elevation 0

- Elements lie in the y–z plane, boresight along +x, half-wavelength spacing,
  **centered at the origin** (the array centre is the phase reference — see E2.3).
  `y_m = (m - (N_side-1)/2)*d`, `z_l = (l - (N_side-1)/2)*d`, `d = spacing*lambda`.
  Element index `n = m*N_side + l` (row-major, y outer / z inner) — used identically
  in `steering.py` and `dft.py`.
- The user/target is parameterised by azimuth `theta` and range `r` at **elevation 0**:
  `p_u = (r*cos(theta), r*sin(theta), 0)`, consistent with `02_SYSTEM_MODEL.md`'s 2-D
  `p_u = r(cos θ, sin θ)`. The codebook samples one angle (azimuth) per `02` §1.3.

Consequence: in the **far-field limit** the wavefront is planar and the phase depends
only on `y_m` (elevation 0), so the z-rows are degenerate — this is the correct
far-field UPA response and equals the DFT codeword (E2). In the **near-field** the
spherical term `||p_n - p_u||` depends on `z_l` too (curvature), so all 16384 elements
carry distinct phase — the near-field signature the polar codebook exploits.

---

## D-005 — Beamforming gain uses the matched (conjugate) inner product

`02_SYSTEM_MODEL.md` writes the per-codeword gain as `|h^T f|^2` (plain transpose).
Taken literally with `f = a_NF/sqrt(N)` and `h = a_NF`, `h^T f = (1/sqrt N) Σ a_n^2`
has random phase and does **not** peak at the matched codeword — it would break every
label. The physically correct beamforming gain is the **matched filter**
`|f^H h|^2 = |conj(f) · h|^2`, which peaks at `N` when `f ∝ a_NF(theta_u, r_u)`.

We implement `gain = |conj(codebook) @ h|^2`. The plan's `h^T f` is shorthand that
assumes the channel and beamformer carry opposite-sign phase conventions (equivalent
to conjugating one of them); we make the conjugation explicit. Documented here rather
than applied silently.

---

## D-006 — Codeword / label ordering

Flat codeword index `k = i * B_r + j` where `i` = angle index (0..B_theta-1),
`j` = range index (0..B_r-1). Therefore `(y_theta, y_r) = divmod(k, B_r)`. The
codebook returned by `generate_polar_codebook` has shape `(B_theta*B_r, N) = (512, 16384)`
in this order. `compute_optimal_label` inverts with the same convention.

---

## D-007 — Codebook numerical rank is now **512 (full rank)** at 128×128

Audit `07` E2 says "codewords distinct: `matrix_rank(codebook)` near `min(B, N)`".
At the **128×128** UPA (elevation 0, D-004) the measured rank at 60 GHz is **512 = full
rank** (vs 299 at the old 32×32), decomposed as:

- far-field block (64 `j=0` codewords): rank **64** — now limited by the number of
  angles `B_theta=64` (≤ `N_side=128`), not by `N_side`;
- near-field block (448 `j≥1` codewords): rank **448** — all near-field codewords are
  linearly independent in the 16384-dim space.

All 512 codewords are pairwise distinct (0 duplicate rows; max off-diagonal mutual
coherence ≈ 0.92 < 1). The 16384-dim element space (vs 1024 before) is what lets the
codebook reach full rank — a direct benefit of the 128×128 geometry. This satisfies E2
("rank near `min(B, N)`") cleanly; the old 32×32 rank-299 caveat is resolved.

---

## D-008 — Synthetic channel = near-field reconstruction from DeepMIMO geometry

**Decision (user-approved, this phase):** DeepMIMO v4 generates **far-field/planar** channels
(its `steering_vec` takes only azimuth/elevation, no range — confirmed in source). Using
those channels directly would force every polar label to `j=0` (a planar wave's best
polar codeword is always the far-field one), collapsing `y_r` and making Gate-0 G0.2
meaningless. Paper 5's channel model (`02_SYSTEM_MODEL.md` §1.2) is the **near-field**
model `h = Σ αₗ·a_NF(θₗ, rₗ)`, so the stored `channel` is **reconstructed** from DeepMIMO
ray-tracing geometry:

```
h = Σ_ℓ αₗ · spherical_steering(θₗ_az, rₗ)        (src/data/deepmimo_loader.py)
  αₗ      = sqrt(power_linearₗ) · exp(j·phaseₗ)     (DeepMIMO per-path power & phase)
  θₗ_az   = AoD azimuth of path ℓ, rotated into the array-boresight frame
  rₗ      = path length = c · delayₗ  (LoS path length == user range; D-008a proxy)
```

- **Matched-filter consistency (D-005):** `h` is built with the *same* `spherical_steering`
  used to build the codebook, so channel and codebook live in the same azimuth×range
  representation and `compute_optimal_label`'s `|fᴴh|²` is exact. Elevation is dropped to
  match the azimuth×range codebook scope (locked decision — no elevation sampling).
- **D-008a — per-path range proxy:** `rₗ = c·delayₗ` (total path length). Exact for the
  LoS path (which dominates the focusing label); for NLoS it is the apparent-source
  distance proxy rather than the BS-to-first-bounce distance. **Flagged for review**
  — switchable to the `inter_pos`-based BS-side bounce distance
  (`range_via_inter_pos`). *Verified on real O1 (BS3, user 1234):* the two agree for LoS
  (293.5 vs 293.4 m) but diverge for NLoS (e.g. a 2-bounce path: `c·delay`=304 m vs
  `inter_pos`=16 m). For O1's geometry the choice does **not** create near-field labels —
  even the smallest first-bounce distance (~9 m) is ≫ `d_F` (2.4 m @ 60 GHz). Default
  kept as `c·delay` per the approved decision.

- **Verified field semantics (real O1):** azimuth uses **`_aod_az_rot`** (array-boresight
  frame; raw `aod_az` is the global frame — the BS array has a real default orientation,
  e.g. user-0 raw −88.6° vs rot −1.6°). `power` is in **dBW** (negative) → amplitude
  `sqrt(10**(dBW/10))`. `delay` in seconds. LoS extraction validated: dominant-path AoD
  vs geometric azimuth diff **0.00°**, `c·delay` vs 3-D distance diff **0.09 m**.
- **Paths outside the array FoV** (|boresight-relative azimuth| > 90°, i.e. behind the
  array) cannot be represented by the codebook; such paths are dropped with a count
  logged (no silent drop). Users with no in-FoV path are skipped and counted.

## D-009 — Synthetic radar/LiDAR parameters (proxy sensors; flagged)

`04_DATASET_SETUP.md` §6 calls the synthetic sensors a "proof-of-concept stand-in" and
gives the recipe (Gaussian reflector at a range bin + seeded clutter; LiDAR points on a
user bounding box + ground + seeded scatter) but not every constant. Choices made here
(module constants, not magic numbers), **flagged for review as tunable**:

- Radar: `RANGE_RES_M = 1.2 m` → 256 range bins span 307.2 m (O1 user distances reach
  ~292 m; 0.6 m would clamp ~25% of users). Range bin = `round(d/1.2)`, clamped to
  [0,255] with a warning if `d` exceeds max range. RDM = (256 range × 128 Doppler) via
  2-D FFT; RAM = (256 range × 64 angle) via DoA beamforming over `n_rx=4`. Stacked
  feature (2,256,128): channel 0 = RDM, channel 1 = RAM zero-padded 64→128.
- LiDAR: ROI `x∈[-20,40]`, `y∈[-20,20]`, `z∈[-2,2]` m, voxel 0.4 m → BEV (150×100);
  channel 0 = point count, channel 1 = mean intensity. User modelled as a 4.5×2×1.5 m
  box of points + ground plane + background scatter.
- All randomness (clutter, scatter, noise) takes an explicit seed (see D-010).

## D-010 — Synthesis seeds

Radar and LiDAR synthesis are seeded **per sample** as `seed = BASE_SEED + sample_index`,
with `BASE_SEED_RADAR = 20240601` and `BASE_SEED_LIDAR = 20240602` (distinct streams so
the two modalities are not correlated by a shared RNG). The base seeds are stored in each
HDF5 file (`seed_radar_base`, `seed_lidar_base`) for reproducibility (G0.5).

## D-011 — DeepMIMO O1 user selection & subsampling

`dm.load('o1_60')` returns a **MacroDataset** of 54 `(tx_set, rx_set)` pairs (18 BSs ×
3 user grids). We select **BS3** (`tx_set_id=3`, at `[235.5, 589.5, 6] m`) and the main
user grid (`rx_set_id=0`, ~497,931 users), then **uniformly subsample to `N_CAP_USERS`
= 5000** via a regular index stride (preserves the spatial/distance distribution). Full
generation of all 498k users is infeasible (~130 GB of sensor tensors); 5000 fits the
~8 GB budget (docs/04) and is ample for the distribution previews. Increase `--max-users`
for the final training set once the near-field scope is resolved at Gate-0.

**Original finding (32×32, this phase):** the BS sits at **z = 6 m** while users are on the
ground, so the minimum BS→user distance is ≥ ~8 m ≫ `d_F` (2.4 m @ 60 GHz, 1.0 m @
140 GHz). **No users were near-field** (`frac(d<d_F)=0.0000`), so polar labels collapsed
to the far-field bin.

### D-011 RESOLVED (supervisor-approved): array 32×32 → **128×128**

The 32×32 finding above was escalated and the supervisor approved enlarging the array to
**128×128** (`N_side=128`, `N_elements=16384`). This raises `d_F` to **40.3 m @ 60 GHz /
17.3 m @ 140 GHz** (D-002) — both well past the ~8 m minimum user distance — creating a
genuine near-field regime at **both** frequencies. This is a deliberate near-field design
choice; the 32×32 "no near-field" result is reported honestly in the paper, not hidden.
The azimuth×range codebook scope (64×8=512) is unchanged — no elevation sampling added;
near-field elevation curvature is justified in the paper text, not in the codebook.
BS3 / rx_set 0 / `N_CAP_USERS=5000` selection is retained.

---

## D-012 — Memory-safe channel generation (the 16384-element trap)

At `N_elements=16384`, each channel is `16384 × 16 B = 256 KB`; 5000 channels = 1.3 GB,
and the full per-file sensor payload (channel+radar+lidar) is ~3.2 GB — too much to hold
in RAM as a Python list before writing. Generation therefore **streams to HDF5 in chunks**
(`CHUNK_USERS=500`) via `StreamingH5Writer`: resizable datasets are extended per chunk, so
peak RAM is bounded by one chunk (~0.3 GB) plus the codebook (512×16384 complex ≈ 134 MB),
not the whole dataset. Under memory pressure, **reduce `CHUNK_USERS`** — never silently
subsample users or downcast precision. Peak resident memory is measured (Windows
`PeakWorkingSetSize`) and logged per scenario.

---

## D-013 — Gain-gated near-field label (convention-independent)

**Decision (supervisor-approved):** the worry that `c·delay` (D-008a) manufactures
spurious short ranges was **disproven** — `c·delay` = total path length ≥ LoS, so
`minCdel ≥ LoS` always; there are no spurious short ranges. The real issue: ~5 pp of the
raw `frac(y_r>0)=31%` is *soft* (razor-thin <0.5 dB quantization, or labels that flip
under the `inter_pos` convention). Rather than switch range convention to chase a cleaner
count, near-field is defined by the **physically meaningful, convention-independent**
genie polar-vs-DFT gain margin computed from the actual channel:

```
adv_dB(h) = 10*log10( max_k |f_polar_k^H h|^2  /  max_m |f_dft_m^H h|^2 )
y_r := (original c·delay bin) if adv_dB >= GATE_DB else 0   # GATE_DB = 3.0 dB (G0.2)
```

`GATE_DB` lives in `configs/base.yaml` (`labeling.gain_gate_db = 3.0`).
`scripts/apply_gain_gate.py` **relabels only** — channels/radar/lidar/distance and the
raw `y_r` are untouched; it adds `y_r_gated`, `polar_adv_db`, and the `gain_gate_db`
attr to each HDF5. Reporting (`report_distributions.py`) shows the regime three ways:
`frac(d<d_F)`, raw `frac(y_r>0)`, and `frac(adv>=3 dB)` (the anchor), plus the 0.5 dB
marginal band.

**Flagged for review (open):** the gate sets `y_r:=0` per the instruction but leaves
`y_theta` unchanged; for ~150–190 sub-gate samples/file the kept `y_theta` differs from
the DFT-optimal angle. Whether gated-to-far-field samples should also take the
DFT-optimal `y_theta` is a labeling choice to confirm before Gate-0.

**`inter_pos` semantics (confirmed, for §III wording):** `range_via_inter_pos` =
distance from the BS (TX) to the **first interaction point** along the departure ray
(`inter_pos[path]` is ordered TX→RX; `finite[0]` is 16.2 m from BS vs 228.7 m for the
last point). Since the BS array is the transmitter, this is the **BS-side first /
nearest scatterer along the departure ray** — *not* "last bounce before a receiving
array."

---

## D-014 — PRE-REGISTERED close-range synthetic V2X scenario (Path B)

**Supervisor-approved, written BEFORE any generation. This distribution is a FIXED
design choice. It will NOT be tuned to improve any model result; re-rolling geometry to
chase a number is forbidden. Whatever near-field fraction / accuracy falls out is
reported as-is.**

**Motivation (paper §V.A):** O1-at-128×128 is free-flow street V2X — far-field-dominated
and kept as the real-world anchor. But intersections, EV charging stations, parking, and
pickup/drop-off zones place vehicles at close range where the near-field dominates — a
legitimate, common V2X regime O1 does not represent. We study it with a controlled
close-range distribution and report results on BOTH datasets.

**Scenario:** charging-station / intersection RSU.

**Pre-registered distribution (locked):**
| Parameter | Value | Justification |
|---|---|---|
| RSU (BS array) height | 2.0 m | low-mount close-range unit (not the 6 m street pole) |
| Vehicle antenna height | 1.5 m | passenger-vehicle roof; 0.5 m vertical offset, negligible vs range |
| User horizontal distance `d` | **Uniform[1, 20] m** | covers the near-field span; 100% < d_F@60 GHz, ~85% < d_F@140 GHz |
| User azimuth `θ` | **Uniform[−60°, +60°]** | 120° frontal approach arc of an intersection/charging RSU |
| Users per frequency `N` | **6000** | healthy 70/10/20 × 5-seed per-bin counts |
| Frequencies | 60, 140 GHz | same as O1 |
| Array | 128×128 UPA (D-004) | identical to O1 |

**Channel synthesis (LoS + seeded weak NLoS), via the SAME D-008 reconstruction:**
- LoS path: `(θ_u, d)`, gain 1.
- `K ~ Uniform{2,3,4}` seeded NLoS scatterers: `az_k ~ U(−90°,90°)`, `r_k ~ U(1,25) m`,
  `|gain_k| ~ 10^(−U(10,20 dB)/20)` (10–20 dB below LoS), random phase.
- `h = LoS + Σ_k gain_k · spherical_steering(az_k, r_k)` (D-008), label by matched filter
  (D-005) then the 3 dB gain gate (D-013).

**Sensors (E1 correspondence):** radar (2,256,128) and LiDAR (2,150,100) are synthesized
from the **same per-sample user position** `(θ_u, d)` as the LoS path. NLoS scatterers are
channel-only multipath (radar clutter / lidar scatter are independently seeded, as in O1).

**Seeding (deterministic):** position + NLoS RNG = `np.random.default_rng(SCENARIO_SEED +
idx)` with `SCENARIO_SEED = 20240603` (distinct from the radar/lidar D-010 streams
`BASE_SEED_RADAR/LIDAR + idx`). All three streams stored in the HDF5.

**trajectory_id:** synthetic spatial grouping `az_bin(2°)·100 + dist_bin(1 m)` so later
trajectory-grouped 70/10/20 × 5-seed splits have no within-cell leakage.

**Files:** `data/processed/closerange_{60,140}GHz_full.h5`.

---

## D-013 RESOLVED — gated samples take the DFT-optimal angle

The open item in D-013 (gated `y_r:=0` left a possibly-wrong `y_theta`) is resolved:
when a sample is gated to far-field, `y_theta` is **recomputed as the DFT-optimal angle**
`argmax_i |f_dft_i^H h|^2`. `apply_gain_gate.py` stores `y_theta_gated` alongside
`y_r_gated`; the gated label is `(y_theta_gated, y_r_gated)`. Applied to ALL datasets
(O1 re-gated; close-range gated). Raw `y_theta`/`y_r` preserved.

---

## D-015 — PRE-REGISTERED decision: one-modality vs two-modality paper

**Written BEFORE the warmstart run; outcome reported either way. This rule is fixed now
and cannot become post-hoc.**

Context: the combiner sweep proved fusion's only shortfall is that the
jointly-trained in-fusion radar branch (angle 0.746) underperforms the standalone radar
model (0.768), which erases LiDAR's real but small range gain. Standard fix (docs/10 D4):
**warmstart the radar branch from the Batch-5 checkpoint** (frozen first — cleanest test),
preserve its angle, and let the LiDAR branch + cross-attention + combiner add range.

**Decision rule.** On close-range 60 GHz, seed-11 **TEST near-field subset (y_r>0)**, using
the **radar-angle + best-range** combiner (the established best), the warmstarted fusion is
reported as **PRIMARY (two-modality paper) IF AND ONLY IF it meets BOTH:**
1. **K=8 NF beam-focusing dB-loss ≤ 0.50** (radar-alone), **AND**
2. **top-1 theta ≥ 0.768** (no angle regression vs standalone radar).

If it does **not** meet both, **radar-alone is PRIMARY** and fusion is reported as the
analyzed-neutral result.

**Run protocol (one configuration, no sweep, no target-chasing):** init radar encoder +
radar head from `checkpoints/closerange_60GHz_radar/seed11.pt`; **freeze the radar branch
first** (cleanest test); train LiDAR branch + cross-attn + fused head at lr 1e-4; same
modality dropout, split, epochs as before. Only if frozen fails to train the rest, fall
back to radar-branch lr 1e-5, and report which was used and why. Do not try multiple
warmstart variants and pick the best.

---

## D-016 — DeepSense real-data role (label-trust gate result, this phase)

DeepSense6G has **no full channel** (per-DFT-beam power only), so polar labels are
reconstructed and low-trust (G0.4). Scenario 31 (dev format, V2V, 7012 frames, 52
sequences, V2V range 10.8–84.8 m) was processed and gated:

- **G0.3 angle non-degeneracy: PASS** — all 64/64 angle bins populated (y_theta histogram
  max 469 / min 2), not collapsed.
- **G0.4 angle agreement: 1.000** — the raw power-file argmax matches the CSV `unit1_beam`
  for every frame; the **angle label is trustworthy** (it comes directly from which DFT
  beam had max power).
- **y_r = 0 for all frames (far-field)** — DeepSense provides no near-field range info.
  This is **expected, not a failure**.

**Decision:** DeepSense is used for **radar → angle real-data validation + far-field
beam-selection** only. **Near-field beam-focusing-gain validation remains synthetic-only**
(stated in the paper §V.A / §VI.B). DeepSense is never used to claim near-field gain, and
(per G0.4) is validation-only — not a near-field training set.

### D-016 addendum — radar cube-axis bug

The first DeepSense radar pass gave only ~12% top-1 angle (below published radar-aided
baselines). Diagnosis: the DeepSense radar cube is **(rx=4, chirps=256, samples=250)** —
range/fast-time is the LAST axis (confirmed: range-FFT sharpness 25.3 over axis-2 vs 10.5
over axis-1). The processor had transposed it and run the range FFT over the *chirps*
axis, producing a **structureless RA** (angle-peak NaN/constant). Fixing the axis (no
transpose; `radar_preprocessing(cube, n_rx=4, n_chirps=256, n_samples=250)`) **raised
radar top-1 angle from 12.4% to 23.1% ± 2.4** (5 seeds), into the published baseline range.
This was a **processing bug, not a sim-to-real limit** — corrected, not tuned. Residual
genuine limit: the 4-RX radar's coarse angular resolution caps real top-1 ~23% (K=8
coverage ~68%). Zero-shot synthetic→real still fails (~0.8% < random) regardless — real
training remains necessary.

## D-017 — Range-codebook resolution: keep **B_r = 8** (E10 ablation result)

The range-codebook resolution B_r was ablated over {4, 8, 16} (3 seeds {11,22,33}; E10,
`scripts/ablate_br_*.py`). The **per-gate** near-field dB-loss is NOT comparable across B_r
(each B_r gates a different near-field set against its own, coarser genie — a smaller B_r is
flattered). The comparable metric is the **absolute** realized beam-focusing gain at K=8 on
a **common** near-field set (identical users), referenced to the finest B_r=16 genie
(`scripts/ablate_br_common.py` → `results/ablation/common_nf_absgain.json`):

| B_r | abs. realized NF gain @K=8 (dB, vs common B_r=16 genie) |
|---|---|
| 4  | −2.35 ± 1.88 |
| **8**  | **−1.05 ± 0.95 (best)** |
| 16 | −3.24 ± 7.17 |

**Decision:** keep **B_r = 8**. B_r=4 loses by under-resolving range; B_r=16 by sparse,
noisy bins. **CAVEAT (must travel with the number):** this rests on **n=3 seeds** with
**overlapping CIs** (8-vs-4 not statistically separated) — it is the best-achieved,
**directional** result reported honestly (anti-HARK), NOT a significant win. The system
model is unchanged (B_θ×B_r = 64×8 = 512). Supersedes the earlier mislabel of this choice
as "D-016" (D-016 is the DeepSense real-data role; it does not concern B_r).

## D-018 — PRE-REGISTERED: 20-seed radar-vs-fusion powered comparison (written BEFORE training)

Reviewer flagged the fusion comparison as under-powered (n=5). We commit, **before running any
of the new training**, to a 20-seed study and to reporting whatever it yields.

- **Seeds:** the 5 existing {11,22,33,44,55} plus 15 new {66,77,88,99,110,121,132,143,154,
  165,176,187,198,209,220} = **20 total**, at **60 and 140 GHz**, close-range scenario.
- **Config:** identical to seeds 11–55 — same hyperparameters (Adam lr 1e-4, wd 1e-5, 50
  epochs, batch 16, fp16, CosineAnnealing), same **trajectory-grouped splits regenerated per
  seed**, same **warm-start-fusion-from-radar** procedure (`sweep_taskA.py`, resumable). GPU
  required (no CPU fallback); checkpoints saved with scrubbed (repo-relative) args.
- **Primary endpoint:** near-field beam-focusing loss @ K=8, **radar-alone vs soft-avg (C3)**,
  per frequency.
- **Secondary endpoint:** radar-alone vs **D-015 radar-primary**, **TOST ±0.5 dB equivalence**.
- **Tests (all reported):** paired t-test AND Wilcoxon signed-rank; Cohen's d_z with
  bootstrap 95% CI; TOST ±0.5 dB; **Bonferroni** over the full rule×frequency sweep.
- **Decision rule, committed now (whatever n=20 shows is what the paper reports):**
  - If soft-avg is **significant after Bonferroni** → paper reframes to *"the soft ensemble
    yields a small significant gain; radar-primary is an equivalence-tested single-sensor
    operating point."*
  - If **not significant** → the negative result stands, now adequately powered.
- **Forbidden:** no retuning, no seed selection, no stopping early on a favourable interim
  result. Missed expectations are reported, never retuned (anti-HARK).
- **Banking:** n=20 stats → `results/fusion_rules/taskA_stats_n20.json` (the n=5
  `taskA_stats.json` is **not** overwritten); Table 9 + macros regenerate at n=20; the
  headline radar/fusion K=8 numbers are reported n=5-vs-n=20 side by side.

## D-019 — PRE-REGISTERED: clean same-split radar-ensemble control (written BEFORE training)

The n=20 eval-only radar-3-ensemble (Table 11) was ~89% test-leakage-contaminated (members
trained on other seeds' splits). This is the leakage-FREE replacement, committed before any
training.

- **Design:** for each split seed s in {11,22,33} x {60,140} GHz, train **3 radar models on
  s's OWN train split**, varying ONLY the weight-init seed (inits {201,202,203}); identical
  hyperparameters/data to the banked radar runs otherwise. All 3 members are evaluated on
  s's held-out test set -> no member ever sees a test trajectory of its split.
- **LEAKAGE ASSERTION (hard-fail):** the computed fraction of s's NF-test trajectories that
  appear in ANY member's train set must be **0.000**; if not, the chain stops.
- **Endpoint:** NF K=8 dB-loss, per split: (b') same-split radar-3-ensemble (C3
  confidence-weighted soft-average of the 3 inits, both heads) vs (c') the multimodal
  soft-avg (existing fusion checkpoint for split s; no retraining), on the SAME test set.
- **Inference:** per-sample paired bootstrap over the shared NF test users within each split
  (mean delta = dbloss_b' - dbloss_c', 95% CI, two-sided p), plus sign-consistency across
  the 3 splits.
- **Interpretation key (committed):**
  - (c') > (b') consistently (c' lower loss) -> the LiDAR modality contributes beyond
    generic ensembling.
  - (c') ~ (b') -> the gain is substantially generic ensembling; the paper says so and notes
    the radar-3-ensemble costs 3x radar compute vs a single fused pass.
  - mixed across splits -> reported as mixed.
- **Decision rule:** whatever the data shows is reported; no retuning, no split selection.
- **Banking:** `results/fusion_rules/ensemble_confound_clean.json`; Table 11 gains Panel B.

## D-020 — Mobility future-beam prediction on real DeepSense (motivation + protocol)

**Motivation (TVT):** vehicles in Sc31 (V2I) / Sc36 (V2V) move across consecutive frames;
predicting the beam Delta frames AHEAD from sensors at frame t demonstrates the predictor
under genuine mobility, not just same-frame estimation.

- **Data:** existing processed DeepSense radar features (no new downloads). Build
  (X_t -> y_{t+Delta}) pairs for Delta in {0,1,2,3} **within each trajectory only** (frames
  are contiguous/ordered within a trajectory; never pair across trajectory boundaries).
  Respect the banked trajectory-grouped splits; assert no trajectory is in both train and
  test of any split.
- **Frame interval / horizon:** DeepSense6G nominal ~10 Hz -> 0.1 s/frame. Measured mean
  per-frame range change: Sc31 ~0.39 m, Sc36 ~0.79 m (V2V faster). So Delta in {1,2,3} maps
  to ~0.1/0.2/0.3 s ahead (Sc31 ~0.4/0.8/1.2 m; Sc36 ~0.8/1.6/2.4 m displacement).
- **Train:** radar branch only, 5 seeds {11,22,33,44,55}, per (scenario, Delta), same
  hyperparameters as the banked DeepSense runs (per-(scenario,Delta) separate models — kept
  identical across Delta for comparability).
- **ANCHOR TRIPWIRE (hard-fail):** Delta=0 must reproduce the banked top-1 accuracies
  (23.05% Sc31, 17.14% Sc36) within the 5-seed 95% CI. Evaluate the Delta=0 anchor FIRST;
  if it fails, stop the chain and do not train Delta in {1,2,3}.
- **Report/bank:** per (scenario, Delta) top-1 + top-K coverage @ {4,8}, 5-seed mean+/-95%CI
  -> `results/mobility/future_beam.json`; table + accuracy-vs-horizon figure; new macros.
- Report achieved numbers as-is; no retuning.

---

# TVT major revision (VT-2026-05521), Phase 1 — eval-only diagnostics

Entries D-021..D-028 are written and committed **BEFORE** any of the Part-B diagnostics
are run. All are **eval-only**: banked checkpoints + banked data, no retraining, no data
regeneration, no edits to existing banked JSON. Outputs go to `results/revision/`, one
JSON per item; scripts live in `scripts/revision/`. Whatever each diagnostic shows is
reported as-is, including results that weaken the paper. No metric, subset, seed set or
baseline definition below may be changed after seeing results; any deviation forced by
an implementation problem is reported as a deviation, with the reason.

**Context discovered in Part A (before this entry):** the banked fusion checkpoints for
the original seeds {11,22,33,44,55} were trained with the radar branch FROZEN
(`args.freeze_radar=True`), the 15 new seeds {66..220} with it UNFROZEN
(`sweep_taskA.py` never passes `--freeze-radar`). The n=20 pool therefore mixes two
training configurations. This is reported; it does not change any definition below.

## D-021 — Common conventions + B1 subset dependence

**Conventions (apply to D-021..D-027 unless stated).**
- Data `data/processed/closerange_{60,140}GHz_full.h5`; labels `y_theta_gated`,
  `y_r_gated`. Seeds = the 20 D-018 seeds. Test set of seed s =
  `trajectory_grouped_split(trajectory_id, seed=s)[2]` (identical to the banked evals).
- Gains `G = |conj(codebook) @ h|^2` over all 512 codewords (complex64 on GPU, as in the
  banked evals). Polar genie = `max_k G`. Loss of a shortlist S:
  `10 log10(genie / max_{k in S} G_k)` dB. **Genie-DFT loss** =
  `10 log10(genie / max over the 64 j=0 codewords)`.
- Inference under fp16 autocast (as banked). Joint score = `p_theta(i) * p_r(j)`; top-K
  shortlist = the K highest joint scores.
- Predictors: **radar-alone** = standalone `checkpoints/closerange_{f}GHz_radar/seed{s}.pt`;
  **soft-avg** = C3 `combine("soft_avg")` on `closerange_{f}GHz_fusion_warmstart/seed{s}.pt`;
  **radar-primary** = `combine("radar_ang+bestR")` on the same checkpoint;
  **DFT-restricted** = radar-alone's top-K *angles* mapped to their j=0 codewords
  (definition of `scripts/eval_dft_baseline.py`).
- **User sets** (within each seed's test set): (i) all close-range test users;
  (ii) `distance < d_F` (d_F = 40.29 m / 17.27 m); (iii) `y_r_gated > 0` — the >=3 dB
  genie-advantage subset used in the paper (equivalently `polar_adv_db >= 3.0`);
  (iv) complement of (iii).
- **Aggregation:** per-seed mean over the set, then mean and t-based 95% CI over the 20
  seeds. Also reported: pooled-over-seeds median, and set sizes (per-seed mean/min/max and
  number of unique users across the 20 seeds).
- **Reproduction tripwire:** B1 set (iii), K=8, radar-alone and soft-avg per-seed values
  must reproduce `results/fusion_rules/rule_sweep_n20.json` within 0.01 dB. If not, stop
  and report before interpreting anything.

**B1.** Loss vs the polar genie at K in {1,2,4,8} for soft-avg, radar-alone,
DFT-restricted, plus the genie-DFT loss (K-independent), on sets (i)-(iv), both
frequencies. Output `results/revision/B1_subsets.json`.

## D-022 — B2: does the range head matter?

For soft-avg (primary) and radar-alone (secondary), on sets (i)-(iv):
- At K=4 and K=8: mean number of distinct angles and distinct ranges in the shortlist;
  fraction of samples whose K=8 shortlist is exactly one angle with all 8 ranges.
- Baseline (a) **angle + range sweep**: the predictor's top-1 angle (`argmax p_theta`) with
  all 8 range codewords measured (8 measurements); loss = genie / max_j G(i_hat, j).
  Compared against the predictor's own K=8 shortlist (also 8 measurements).
- Baseline (b) **range prior**: the predictor's `p_theta` with `p_r` replaced by the
  seed's training-set empirical `y_r_gated` distribution (train split of that seed, no
  smoothing); top-K at K in {1,2,4,8}.
Same loss definition, sets and aggregation as D-021. Output `results/revision/B2_range_head.json`.

## D-023 — B3: primary-endpoint tests on the 15 new seeds alone

Inputs: the banked per-seed NF K=8 losses in `results/fusion_rules/rule_sweep_n20.json`
(no recomputation of the endpoint). Seed sets: **new15** = {66,...,220};
**all20** (recomputed, must match `taskA_stats_n20.json`); **orig5** = {11,...,55}
(context). For each set and frequency: radar-alone vs soft-avg and radar-alone vs
radar-primary — paired t, Wilcoxon signed-rank (scipy), Cohen's d_z with 10,000-resample
bootstrap 95% CI (seed 0), TOST +/-0.5 dB (alpha 0.05); Bonferroni over the full
6-rule x 2-frequency family (m=12), all exactly as `scripts/eval_taskA_stats.py`. Note:
new15 = unfrozen-radar configuration only; orig5 = frozen. Output
`results/revision/B3_new_seeds.json`.

## D-024 — B4: mobility persistence (DeepSense Sc31, Sc36)

Seeds: the 5 banked mobility seeds {11,22,33,44,55} (only these exist; 20 not available).
- **Fixed sample set** per (scenario, seed): source frames t (original-file indexing,
  pairs rebuilt exactly as `build_mobility_pairs.py`) such that frame t+3 exists in the same
  trajectory AND t's trajectory is in the TEST split of every Delta in {1,2,3} dataset
  under that seed (splits recomputed per Delta dataset exactly as `eval_mobility.py`).
  The same set is used for all horizons.
- Per Delta in {1,2,3}: beam-change rate `P(y_{t+Delta} != y_t)`; persistence top-1
  `P(y_t == y_{t+Delta})`; our top-1 (the Delta-trained model
  `mobility_sc{S}_delta{Delta}_radar/seed{s}.pt`) overall, on changed frames and on
  unchanged frames. Supplementary: persistence coverage at K in {4,8} (the K beam indices
  nearest to y_t, ties to the lower index) vs our coverage@K; the Delta=0 model's top-1
  against y_{t+Delta}; change rate over all within-trajectory pairs of the whole file.
- 5-seed mean and 95% CI. Output `results/revision/B4_mobility.json`.

## D-025 — B5: learning-free baselines on the synthetic data

Same metrics (loss vs polar genie at K in {1,2,4,8}), sets (i)-(iv), seeds, aggregation as D-021.
- **(a) Radar peak picking.** RA map = `radar_features[1, :, :64]` (256 range x 64 angle;
  angle grid identical to the codebook's, verified in A4). Peak (rb, a) = argmax.
  d_hat = max(rb,1) * 1.2 m. j_hat = argmin_j |1/r_j - 1/d_hat| with 1/r_0 = 0 (uniform-in-1/r
  polar grid). Shortlist: all 512 codewords ranked by L1 index distance
  |i - a| + |j - j_hat|; ties by smaller |j - j_hat|, then smaller |i - a|, then lower flat
  index. Supplementary (a2): peak angle with all 8 ranges (K=8 only).
- **(b) 1-NN on raw sensor features.** Reference set = the seed's train split.
  (b1) radar features flattened (65,536-dim), Euclidean; (b2) radar flattened and LiDAR
  BEV flattened concatenated, each block divided by its mean train-set L2 norm, Euclidean.
  Shortlist = joint codewords `(y_theta_gated, y_r_gated)` of the nearest neighbours in
  distance order, de-duplicated, until K unique.
Output `results/revision/B5_learning_free.json`.

## D-026 — B6: both-sensors-dropped check

All 20 seeds x both frequencies, fusion checkpoints, C3 soft-avg, on each seed's test set.
- Primary mechanism = the paper's both-out (`radar_dropout=True, lidar_dropout=True`,
  encoder features zeroed). Secondary = raw input tensors set to zero.
- Report: joint top-1 accuracy (60 GHz must reproduce the banked 1.26% n=20 mean of
  `results/degradation_closerange60_n20.json`); frequency of the most common joint label in
  the test set; number of distinct predicted joint classes; the predicted class(es) and
  their test-set frequency; angle-only and range-only top-1; mean entropy of p_theta and p_r
  vs log(64) and log(8).
Output `results/revision/B6_both_dropped.json`.

## D-027 — B7: calibration

ECE with 15 equal-width confidence bins (confidence = max softmax, correctness = top-1),
all test users of each seed, 20 seeds x both frequencies, angle head and range head
separately. Predictors: fusion-model radar / LiDAR / fused branches, the C3 soft-avg
ensemble, and the standalone radar model. Conditions: clean; radar dropped
(`radar_dropout=True`); LiDAR dropped (`lidar_dropout=True`) — standalone radar clean only.
Also report accuracy and mean confidence. 20-seed mean and 95% CI. Output
`results/revision/B7_calibration.json`.

## D-028 — B8: cost

Parameter counts (total) and single-sample (batch=1) inference latency, excluding sensor
preprocessing, for: radar branch alone (`RadarPolarModel`); full three-branch
`SensingPolarPredictor` + C3 combine; a 3x radar ensemble (3 `RadarPolarModel` forward
passes, sequential, + C3 soft-average). GPU: fp32 and fp16-autocast, 50 warm-up + 300 timed
iterations with `cuda.synchronize`, median and p90. CPU: fp32, default thread count and 1
thread, 10 warm-up + 100 timed. Supplementary: CPU time of `radar_preprocessing` on one
(4,256,256) cube. Hardware recorded. Output `results/revision/B8_cost.json`.
