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

---

# TVT major revision (VT-2026-05521), Phase 2a — corrected setup + gate experiment

## D-029 — PRE-REGISTERED: corrected close-range setup and 5-seed gate experiment (written 2026-11-16, BEFORE any data generation)

**Why.** Phase 1 and its independent verification (`scripts/verify/`, `results/verify/`)
showed the D-014 setup cannot support the paper's claims: the synthetic radar places the
target at the exact true range bin on the codebook's own angle grid (a learning-free peak
picker beats the learned models); the 128x128 UPA with an azimuth-only codebook loses
~18 dB once the D-014 heights are applied; 64 angles under-sample the aperture (genie
~5.6 dB below the matched-filter bound). This entry fixes a corrected setup and a small
gate experiment: does a learned predictor beat a geometric radar baseline? Either answer
is reported as it comes out.

**Freeze rule.** Every value below is fixed once tag `revision-plan-v1` exists. A change
forced later is recorded here as a dated deviation with its reason. No setting is tuned
after seeing how the methods compare.

**Decision taken before any data (2026-11-16):** the instruction specified 128 angles.
An analytic, data-free sweep of the codebook (pure line-of-sight points on a deterministic
(u, r) grid, `scripts/phase2/p0_codebook_property.py`) gives a genie-vs-matched-filter gap
of 1.38 / 1.30 dB (60 / 140 GHz) at 128 angles (critical sampling; 3.9 dB at mid-beam),
~1.8 dB once the expected multipath share (~0.5 dB) is added, so the 1 dB check would fail
by construction. **Angular sampling is raised to 2x oversampling (256 angles)** to remove
the scalloping loss (analytic LoS gap 0.45 / 0.35 dB, ~0.8–0.9 dB with multipath).
Decided by the PI before any data was generated. The 1 dB stop rule is unchanged.

### Parameters

| Group | Parameter | Value | Physical reason |
|---|---|---|---|
| Frame | axes | x = array boresight (horizontal), y = horizontal along the array, z = up, ground z = 0 | one right-handed frame for channel and sensors |
| Array | type | horizontal ULA along y, **128 elements**, spacing λ/2, centred at (0, 0, **2.0 m**) | a horizontal ULA's response depends only on (u = p_y/‖p‖, r = ‖p‖), so a (u, r) codebook stays exact with heights |
| Array | carriers | 60 and 140 GHz | as before |
| Array | d_F = 2D²/λ, D = 127·λ/2 | 40.3 m / 17.3 m | users 1–20 m are near-field-capable |
| Users | count / seed | 6000 per frequency (same users at both), `SCENARIO_SEED = 20261116`, `default_rng(SCENARIO_SEED + idx)` | same size as D-014; new stream so nothing is reused from tuned data |
| Users | horizontal distance d | U[1, 20] m | D-014 range |
| Users | azimuth θ (from array centre, horizontal) | U[−60°, +60°] | D-014 arc |
| Users | antenna position | (d cosθ, d sinθ, **1.5 m**) — vehicle roof centre | D-014 vehicle antenna height |
| Users | vehicle heading ψ | U[0, 2π) | vehicles approach from any direction; affects radar/LiDAR shape only |
| Channel | model | h = Σ_ℓ g_ℓ · a(p_ℓ), a_n(p) = exp(−j2π/λ(‖p − q_n‖ − ‖p − q_0‖)), q_n element n, q_0 array centre | existing spherical-wave model, now in 3D |
| Channel | LoS | p_0 = user antenna, g_0 = 1 | unit LoS as before |
| Channel | reflections | K ~ U{2,3,4} point scatterers; horizontal range U[1, 25] m, azimuth U[−90°, 90°], height U[0.5, 3.0] m; \|g\| = 10^(−U[10,20]/20), phase U[0, 2π) | D-014 multipath, with plausible heights (vehicles, poles, facades) |
| Codebook | angles | **B_θ = 256**, u_i = (2i − 255)/256 (uniform in sine) | 2x oversampled vs the 128-element aperture (see decision above) |
| Codebook | ranges | B_r = 8: r_0 = ∞ (DFT), r_j = 1.2·127²·λ/(8j), j = 1..7 → 60 GHz 12.09…1.73 m, 140 GHz 5.18…0.74 m | same Cui–Dai rule (β_Δ = 1.2) |
| Codebook | codeword / order | f_{i,j} = a(u_i, r_j)/√128, flat k = 8i + j | as D-006 |
| Labels | definition | G_k = \|f_k^H h\|² (float64); label = smallest k with G_k ≥ (1 − 1e-12)·max G; (y_θ, y_r) = divmod(k, 8) | raw genie argmax, deterministic ties; **no gain gate** |
| Labels | advantage | adv = 10 log10(max_k G / max_{j=0} G) | genie polar-over-DFT advantage |
| Check | genie vs bound | gap_u = 10 log10(‖h_u‖² / max_k G_k), mean over all 6000 users | **STOP and report if the mean exceeds 1 dB at either frequency** |
| Radar | carrier / array | 77 GHz FMCW, 3TX×4RX TDM-MIMO → **12-element virtual horizontal ULA**, λ_r/2 | a typical single-chip automotive radar |
| Radar | mount | phase centre at (0, **+0.50 m**, **1.20 m**), i.e. offset (0, +0.5, −0.8) m from the comms array centre, boresight +x | separate radar unit on the same pole; the offset is known to all methods |
| Radar | range | bandwidth 1.5 GHz → ΔR = 0.10 m; 256 fast-time samples → 0–25.6 m | resolves 1–20 m users with margin |
| Radar | slow time | 64 chirps; static scene (all vehicles v = 0) | close-range users are static, as before |
| Radar | angle grid | **101 bins uniform in angle, −75° … +75°, step 1.5°**, conventional beamforming on the 12 virtual elements (sin φ = p_y/‖p‖ for a horizontal ULA) | not the codebook grid (uniform in angle, different count); covers the ±60° arc as seen from the offset radar |
| Radar | target | vehicle as **6 point scatterers**: body-frame offsets uniform in the 4.5 m × 2.0 m footprint, heights U[0.3, 1.5] m, rotated by ψ about the vehicle centre; amplitudes U[0.3, 1.0] normalised to Σa² = 1, phases U[0, 2π) | extended target with several dominant scattering centres; none is guaranteed at the antenna |
| Radar | clutter | 12 point reflectors, amplitude 0.25·U[0.3, 1.0] (relative to the vehicle total), slant range U[1, 25] m, sin(azimuth) of U[−90°, 90°], Doppler U[−16, 16] bins | the existing clutter model, rescaled to the new range span |
| Radar | SNR | **20 dB = primary condition; 10 and 0 dB = stress tests**, reported separately and not used for the headline comparison or for picker selection. Each level is defined as the vehicle's coherent-integration SNR: noise variance per raw complex sample σ² = (256·64·12)/10^(SNR/10) with Σa² = 1 (window losses not included) | a per-raw-sample definition would leave every level > 45 dB after integration and test nothing |
| Radar | noise seeds | geometry/clutter seed `BASE_RADAR = 20261117 + idx` (shared across SNR); noise seed `BASE_RADAR + 1_000_003·(1 + SNR-index) + idx` | SNR changes the noise only |
| Radar | processing | Hann window in range and Doppler; range FFT 256; Doppler FFT 64 (fftshift); RD = Σ_elements \|·\|² (256×64); RA = Σ_Doppler \|beamformed\|² (256×101); 10·log10(·+1e-12), then min–max to [0, 1] per map; tensor (2, 256, 128): ch0 = RD zero-padded to 128 columns, ch1 = RA zero-padded to 128 columns | the existing two-channel input shape and encoder |
| LiDAR | model | existing `simulate_point_cloud` model (400 car / 1500 ground / 600 background points, same intensities), sensor at the comms array (0, 0, 2.0 m): ground z = −2.0 m, car box 4.5×2.0×1.5 m with its base on the ground, rotated by ψ; `lidar_preprocessing` unchanged | keep the model, apply the heights |
| LiDAR | seed | `BASE_LIDAR = 20261118 + idx` | deterministic |
| Split | blocks | 15 azimuth sectors of 8° (−60…60°) × 4 distance bands ([1, 5.75), [5.75, 10.5), [10.5, 15.25), [15.25, 20] m) = 60 contiguous blocks | test users form contiguous regions, not interleaved |
| Split | assignment | **deterministic rotation**: sectors s = 0..14 (s = 0 is −60…−52°); in band b = 0..3, test = {(b + 5k) mod 15, k = 0,1,2}, val = {(b + 3 + 5k) mod 15, k = 0,1}, rest train | held-out sectors rotate across bands: no sector is test in more than one band and every sector is in training in ≥ 3 bands |
| Split | guard | **training** users within 0.5 m (horizontal) of any test user are dropped (validation is not guarded; it is never trained on) | no test user has a training neighbour inside 0.5 m; expected sizes ≈ 46 % train / 13 % val / 20 % test of users |
| Split | sharing | one fixed split for both frequencies and all seeds | a seed changes only the weight init (and the data order / dropout draws that follow from it) |
| Training | models | `RadarPolarModel` and `SensingPolarPredictor` (3 branches, cross-attention, C3), d_z = 128, heads 256 × 8 | the paper's architectures |
| Training | configuration (all seeds) | Adam lr 1e-4, wd 1e-5, cosine over 50 epochs, batch 16, fp16; loss CE(θ) + CE(r); modality dropout p = 0.15 each (never both); three-branch model warm-started from the same-seed radar-alone checkpoint with the **radar branch unfrozen**; final-epoch checkpoint used (no early stopping) | the established recipe, one configuration |
| Training | SNR | each train/val user is assigned one SNR from {20, 10, 0} by `default_rng(2030)`; test is evaluated at each SNR separately | one deployed model must work without knowing the SNR |
| Seeds | | 11, 22, 33, 44, 55; both frequencies | gate experiment only |

### Methods compared (same fixed test set, same gains)

- **(a) radar-alone CNN** and **(b) three-branch soft-average (C3)**: joint score p_θ(i)·p_r(j) computed in float64; top-K = the K largest scores, ties broken by lower flat index.
- **(c) geometric picker** (reads only the radar input tensor ch1, its per-frame dB normalisation constants — radar-derived, stored with the tensor — and known constants). The aim is the strongest reasonable geometric baseline. Two position estimates:
  - **E1 strongest return:** peak = argmax of the RA map over range bins ≥ 3 and all angles, refined by 3-point parabolic interpolation in range and in angle → slant range R̂, radar angle φ̂.
  - **E2 intensity-weighted centroid:** the target cluster = RA cells within ±15 range bins (±1.5 m) and ±8 angle bins (±12°) of the peak, with power ≥ peak − 10 dB and 4-connected to the peak cell; weights = linear power; centroid of the cells' horizontal radar-frame coordinates (R cos φ, R sin φ) → R̂ = norm of the centroid, sin φ̂ = centroid y / R̂.
  - **Bias correction (optional, per estimate):** constants b_R = median(R_true − R̂) and b_s = median(sin φ_true − sin φ̂) over the training users at their assigned SNR, where (R_true, φ_true) is the true antenna position seen from the radar; applied as R̂ + b_R, clip(sin φ̂ + b_s, −1, 1). Fitted separately for E1 and E2; training set only.
  - Conversion (all variants): slant range R̂ and radar angle φ̂ → radar-frame position with p_y = R̂ sin φ̂, Δz = 1.5 − 1.2 m (known antenna height), p_x = √max(R̂² − p_y² − Δz², 0); add the mount offset → comms-frame (û, r̂); î = nearest u_i, rings ranked by \|1/r_j − 1/r̂\| (1/r_0 = 0).
  - **c1 range sweep at the peak angle:** K-list = (î, ring rank 1..K).
  - **c2 2 angles × 4 ranges:** angles î and the second-nearest u_i to û; order (ring rank, angle rank): K = 1 → (î, ring 1); K = 2 → both angles at ring 1; K = 4 → 2 × 2; K = 8 → 2 × 4.
  - Variants: {E1, E2} × {no bias, bias-corrected} × {c1, c2} = 8.
  - **Selection on the validation set only**, per frequency: the variant with the lowest validation all-user loss averaged over K ∈ {1, 2, 4, 8} **at 20 dB** (the primary condition) is "(c)" and is used unchanged at 10 and 0 dB; all 8 are reported as supplementary.
- **(d) two-stage sweep** (channel measurements, noise-free; SNR-independent): measure all 256 far-field (j = 0) beams, take the best angle, then the 7 remaining rings at that angle (263 measurements); loss = genie / best of those 263.

### Metrics, sets, statistics

- Loss vs genie: L = 10 log10(max_k G / max_{k ∈ S} G) for shortlist S, K ∈ {1, 2, 4, 8}.
- Sets (test users): **all users (primary)**; **adv ≥ 3 dB (secondary)**; **sensitivity cut: horizontal d ≥ 1.6 m** (users closer than 1.6 m excluded; the 60 GHz deepest ring is 1.73 m).
- Each SNR (20, 10, 0 dB) separately.
- Learned methods: per-seed means, then 5-seed mean with t-based 95% CI.
- **Paired per-user bootstrap**, (a) vs (c) and (b) vs (c): Δ_u = (5-seed mean loss of the learned method for user u) − (picker loss for user u); 10,000 resamples of test users, `default_rng(0)`, percentile 95% CI of mean Δ; also the number of seeds whose mean beats the picker.
- Also reported: the fraction of users with adv ≥ 3 dB (all 6000 and test), the mean genie-DFT loss over all users (all 6000 and test), the genie-vs-bound gap (mean, median, p95), the split sizes and guard drops, and the selected picker variant.

### How the result will be read (fixed now)

- For each (frequency, SNR, K, set) cell: **learned better** if the bootstrap CI of Δ lies entirely below 0; **picker better** if entirely above 0; otherwise **no detectable difference**. Every cell is reported.
- **Gate verdict** (primary cell = all users, **20 dB** SNR, K = 4 and K = 8, both frequencies), for (b) vs (c); 10 and 0 dB cells are stress-test results and do not enter the verdict:
  - "learned beats the geometric baseline" if (b) is better in all four of those cells;
  - "geometric baseline beats learned" if (c) is better in all four;
  - otherwise "mixed", stated with the cells.
- Effect sizes are reported in dB with their CIs; no threshold for "practically relevant" is claimed beyond that.
- **Not in this phase:** no 20-seed campaign, no edits to `main.tex` or banked files.

### Amendment 1 (2026-11-16, PI instruction; before the tag and before any data)

1. Radar SNR: 20 dB is primary; 10 and 0 dB are stress tests, reported separately, not used for the headline. Consequently the gate cell moves from 10 dB to 20 dB, and picker selection uses 20 dB validation loss only (the original criterion averaged the three SNRs, which would let the stress tests choose the headline baseline).
2. Picker: added the intensity-weighted centroid (E2) and a training-set constant bias correction for either estimate; 8 variants, selected on validation only.
3. Split: data-free check `scripts/phase2/p0_split_check.py` (deterministic 0.05° × 0.01 m grid, each point labelled with its line-of-sight nearest codeword). **Result for the original `default_rng(2029)` assignment: FAIL** — sector 0 (−60…−52°) was held out in all four bands, sectors 0, 3, 10 were test in two bands, and 18 angle classes (17–26, 185–192) occurred only in test. **Replaced by the deterministic rotation above: PASS** — at both frequencies every angle class (184 in test, 222 in train) and every ring in test (60 GHz rings 1–7; 140 GHz rings 0–5) also occurs in training. Joint (angle, ring) classes are not all covered (60 GHz: 296 of 469 test joint classes absent from training; 140 GHz: 259 of 421), which the factorised angle/range heads tolerate; reported as a property of the block split. The guard now applies to training users only. After generation the same check is repeated on the real labels; any failure is reported as a deviation, without re-splitting.

### Deviation 1 (2026-11-16, PI decision; after a sensor check, before any method was run)

**What changed (radar only):**
1. **Range–angle map from the zero-Doppler bin.** RA = |beamformed|² at the zero-Doppler bin (index 32 of the fftshifted 64-bin Doppler axis) instead of power summed over all 64 Doppler bins. RD map, windows, angle grid, normalisation and tensor layout unchanged.
2. **6 of the 12 clutter reflectors are static.** The first 6 clutter reflectors in draw order have zero Doppler; their Doppler value is still drawn (so the random stream and every position/strength draw are unchanged) and then set to 0. Strength and position distributions unchanged; the other 6 keep Doppler U[−16, 16] bins.

**Reasons (physical).** (1) A static target's energy sits in one Doppler bin; summing power over all 64 bins adds the noise of 63 empty bins (~18 dB) and is not correct coherent processing for a static scene. (2) Real roadside clutter (poles, signs, kerbs, parked cars) is mostly static, so it must remain in the zero-Doppler map; a scene where all clutter is Doppler-separated would be unrealistically clean.

**How it was found.** A sensor-level check on training users only (`scripts/phase2/diag_radar2.py`): under the frozen processing the vehicle was the strongest range–angle return for 9% of 300 training users at the primary 20 dB condition. No model had been trained and no method had been run or compared.

**Statements.** (1) Decided after a sensor check and before any method was run. (2) The choice was made on physical grounds, not on the peak-rate figures (the check compared processing options, but the adopted combination — zero-Doppler plus static clutter — was chosen for physical realism and was not itself evaluated before this entry). (3) **The radar specification is closed after this deviation:** no further changes unless a check shows an outright bug, which will be reported to the PI before any change.

### Files

Scripts `scripts/phase2/`; data `data/processed/phase2/`; checkpoints `checkpoints/phase2/`;
results `results/phase2/` (`gate_results.json`, `checks.json`).

## D-030 — EXPLORATORY: Phase 2b feasibility check (written 2026-11-16, before any Phase 2b run)

**Status: exploratory.** Nothing in this entry is pre-registered as confirmatory, and no
confirmatory claim will be made from it. Its purpose is to establish whether any
sensing-aided scheme can approach the genie on the D-029 setup, after the Phase 2a gate
(`results/phase2/gate_results.json`) showed every sensing method 14–26 dB from the genie
at K ≤ 8, the radar-alone CNN not training (loss at the label prior in all 10 runs) and the
three-branch model's loss flat across SNR. Results are reported as they come out.

**Kept frozen:** the D-029 users, channels, gains, codebook (256 × 8), labels, radar and
LiDAR data, split, and Phase 2a checkpoints. No D-029 setting changes.

**Common tool — local sweep of M beams** (noise-free channel measurements, as method (d)):
from a position estimate p̂ (horizontal x, y; antenna height 1.5 m known) compute
r̂ = ‖p̂ − (0, 0, 2)‖, û = ŷ / r̂; take the A codebook angles nearest û (by |u_i − û|) and
the R rings ranked by |1/r_j − 1/r̂| (1/r_0 = 0), sweep all A × R = M beams, keep the best.
Loss = 10 log10(genie / best of the M). M ∈ {8, 16, 32, 64}; shapes A × R with R ≤ 8,
A ∈ {M/8, …, M} in powers of 2 (M = 8: 1×8, 2×4, 4×2, 8×1; M = 64: 8×8, 16×4, 32×2, 64×1).
**The shape is chosen per (frequency, error source, M) by the lowest mean validation loss**;
test is reported with that shape. The 263-measurement sweep (d) is the reference.

- **E1 Radar ablation.** The five Phase 2a three-branch checkpoints per frequency, test
  set, 20/10/0 dB: loss at K ∈ {1, 2, 4, 8} with the normal input vs with the radar tensor
  replaced by zeros (and, for completeness, with the LiDAR tensor zeroed). The difference
  shows whether the model uses the radar.
- **E2 Accuracy needed (no learning).** True antenna position plus i.i.d. Gaussian error,
  σ ∈ {0.02, 0.05, 0.1, 0.2, 0.5, 1.0} m per horizontal axis (x, y), `default_rng(2032)`
  (one draw per user and σ, validation and test drawn from the same stream, by user index).
  Shape chosen per σ and M on validation.
- **E3 Accuracy achieved.** Position-error distribution (range = error along the
  array-to-user horizontal direction, sideways = perpendicular horizontal component;
  median, p90, RMS) of (i) the Phase 2a selected radar picker estimate (centroid + training
  bias) at 20 dB, and (ii) a geometric LiDAR estimator on the raw point cloud (regenerated
  deterministically from the D-029 LiDAR seeds): drop points with z < −1.8 m (ground at
  −2.0 m) or z > 0.0 m; 2D histogram at 0.5 m; take the densest cell; collect points within
  3.0 m of it; two iterations of re-centring on the median and re-collecting; vehicle
  centre = midpoint of the cluster's extent along its two principal axes (PCA; heading
  recovered up to 180°, which is irrelevant here); antenna = vehicle centre + known offset
  (zero: the D-029 antenna is the roof centre). Each estimate then seeds the same local
  sweep, shape chosen on validation with that estimator.
- **E4 Can the radar network learn?** `RadarPolarModel`, seed 11, 60 GHz, training and
  validation users all at 20 dB, fp32, Adam lr ∈ {1e-3, 3e-4}, weight decay 1e-5, cosine
  over 50 epochs, batch 16, final epoch. Reported: training loss vs the label-prior entropy
  H(y_θ) + H(y_r) of the training labels, validation angle accuracy (exact and ±1 class),
  ring accuracy, and validation sideways position error |u_pred − u_true|·r_true. Test is
  not used in E4.

**Also:** the two Phase 2a diagnostic scripts (overfit check, picker-error check) are
moved into `scripts/phase2/`. No edits to `main.tex` or banked files.

**Files.** Scripts `scripts/phase2/p5_*.py`; results `results/phase2/feasibility/`.

## D-031 — PRE-REGISTERED: Phase 3a realistic LiDAR, location-aware networks, 5-seed test (written 2026-11-16, before any Phase 3a data)

**Why.** Phase 2b (D-030, exploratory) found that a LiDAR-seeded local sweep reached ~0.9 dB
at 32 beams, but with an idealised LiDAR (points through the whole car volume, no
occlusion, antenna at the box centre), and that the global-average-pooled radar CNN cannot
learn (one class for all users). This phase makes the LiDAR realistic, repairs the
networks with a location-aware design, and tests whether the result survives.

**Kept frozen (D-029):** array, codebook, channels, gains, labels, users (antenna positions
and headings psi), split (train 3005 / val 793 / test 1193), radar data and radar spec, and
the Phase 2a radar picker (centroid + training bias, selected in Phase 2a).

**Protocol.** Tag `revision-plan-v2` on this entry before any LiDAR data. All design choices
(estimator thresholds, network head and learning rate, sweep shape) are made on training
and validation users only. When fixed, they are recorded in a dated addendum below,
committed and tagged `revision-design-v2`; only then is the test set evaluated, once.

### New parameters

| Group | Parameter | Value | Physical reason |
|---|---|---|---|
| LiDAR sensor | type | 64-channel mechanical spinning LiDAR, Hesai Pandar64-class (simplified to uniform channel spacing) | typical roadside-unit LiDAR |
| LiDAR sensor | mount | (0, 0, 2.0 m), at the comms array, no tilt | same pole as D-029 |
| LiDAR sensor | vertical FOV / channels | −25° … +15°, 64 channels uniform (0.635° step) | Pandar64 FOV |
| LiDAR sensor | horizontal resolution | 0.2° (1800 azimuths per revolution, 10 Hz), full 360° | Pandar64 at 10 Hz |
| LiDAR sensor | range | 0.3 – 200 m | Pandar64 |
| LiDAR sensor | range noise | N(0, 0.02 m) along each ray | Pandar64 range accuracy ±2 cm |
| LiDAR sensor | return dropout | each return independently lost with p = 0.10 | dark / specular surfaces, glass |
| LiDAR scene | rendering | ray casting: each ray returns its first hit among the vehicle and the ground plane (world z = 0), so only surfaces visible from the sensor return points (occlusion and self-shadowing) | realistic visibility |
| LiDAR scene | background | 600 clutter points uniform in x ∈ [−20, 40], y ∈ [−20, 20], z ∈ [0, 4] m (world), as the existing model; removed if inside the vehicle or if the sensor-to-point segment crosses the vehicle; same range noise and dropout | kept from the existing model, now occluded |
| LiDAR scene | intensities | vehicle U(0.4, 1.0), ground U(0, 0.2), background U(0, 0.5) | existing model |
| Vehicle | shape | two boxes: body 4.5 m (L) × 1.8 m (W), z 0 – 1.0 m; cabin 2.4 m × 1.6 m, z 1.0 – 1.5 m, centred 0.4 m behind the body centre | passenger-car silhouette; the rear-biased cabin breaks front/back symmetry, as on real cars |
| Vehicle | frame | origin = body footprint centre, xi forward along heading psi (frozen D-029 psi), eta left | |
| Vehicle | antenna | fixed at (xi, eta, z) = (−1.3 m, 0, 1.5 m): roof-mounted shark-fin at the rear of the cabin roof, on the centreline | typical roof antenna; 1.3 m from the vehicle centre |
| Vehicle | placement | antenna position = frozen D-029 `ant_xyz`; vehicle centre = antenna − R(psi)·(−1.3, 0) | keeps channels and labels unchanged |
| LiDAR | seed | `default_rng(20261119 + idx)` (background, intensities, noise, dropout, in that order) | deterministic |
| LiDAR BEV (network input) | grid | 0.2 m cells, x ∈ [−3.2, 22.4) m (128), y ∈ [−20.8, 20.8) m (208) | covers every user (x ≥ 0.56, \|y\| ≤ 16.9 m) at the accuracy the sweep needs |
| LiDAR BEV | channels | log(1 + point count); max point height above ground in the cell / 2 m (0 if empty); mean intensity | occupancy, shape (cabin) and reflectivity |
| Radar BEV (network input) | warp | Phase 2a 20 dB range–angle channel (frozen data), bilinearly resampled onto the same BEV grid: each cell (x, y) at the known antenna height 1.5 m → slant range and sin(phi) seen from the radar (0, 0.5, 1.2) → RA bin coordinates; cells outside the RA grid = 0 | puts radar and LiDAR in one Cartesian frame so a location-aware network can read position directly |

**Known inconsistency (frozen radar spec):** the D-029 radar scatterers are drawn around the
antenna position (body centred on the antenna); the LiDAR vehicle is centred 1.3 m from the
antenna. The radar spec is closed (D-029 deviation 1), so it is not changed. This favours the
radar methods (no antenna offset to recover).

### Methods (each seeds the same local sweep)

- **Local sweep (from D-030):** horizontal estimate → (u, r) with the known 1.5 m height;
  A nearest angles × R nearest rings (in 1/r), A·R = M ∈ {8, 16, 32, 64}, shapes
  A ∈ {M/8, M/4, M/2, M}; best of the M noise-free measurements; loss vs genie.
  **Shape chosen per (method, frequency, M, seed) on validation.**
- **Radar picker:** Phase 2a selected variant, unchanged.
- **LiDAR geometric estimator** (raw point cloud): ground removal by height; clustering of
  the remaining points into connected occupied cells; the vehicle cluster selected by size;
  heading from L-shape (rectangle) fitting of the cluster footprint over the visible faces;
  box placed from the visible faces using the known body size 4.5 × 1.8 m; front/back from
  the cabin (points above the body top lie on the rear-biased cabin); antenna = centre +
  known offset. A fixed fallback for frames with no valid cluster. Thresholds (ground
  height, cell size, minimum cluster size, maximum extent, cabin height) and the fallback
  are tuned on train/val only and recorded in the design addendum.
- **Networks** (radar-only: radar BEV; LiDAR-only: LiDAR BEV; fused: both, concatenated at
  the input), all with two coordinate channels (x, y) appended: a U-Net on the BEV grid
  (4 levels, widths 32/64/128/256, GroupNorm(8), ReLU, bilinear upsampling with skip
  connections) giving one logit map at full resolution — no global pooling. Output options,
  chosen on validation:
  - **P (position):** soft-argmax of the map → (x, y); smooth-L1 loss in metres.
  - **H (cell scores):** cross-entropy against a Gaussian target (sigma = 0.2 m) centred on
    the true antenna; estimate = soft-argmax within the 5 × 5 cells around the arg-max cell.
  Training: AdamW, weight decay 1e-4, cosine schedule, 40 epochs, batch 16, fp32, final
  epoch; training users with radar at 20 dB; no augmentation. The network predicts the
  antenna position, which does not depend on the carrier, so one model per seed serves
  both frequencies.
  - **Design selection (seed 11 only, validation only):** head ∈ {P, H} × lr ∈ {1e-3, 3e-4},
    per sensor configuration; criterion = mean validation sweep loss over M ∈ {8, 16, 32,
    64} and both frequencies. The chosen configuration is then trained for seeds 11–55.
  - **Learning check (before the 5-seed runs):** the chosen configuration's final training
    loss must fall below the prior level — P: smooth-L1 of always predicting the training
    median position; H: cross-entropy of the mean training target map. **If any model
    fails, stop and report.**

### Test evaluation (once, after `revision-design-v2`)

- Seeds 11, 22, 33, 44, 55; 60 and 140 GHz; radar at 20 dB.
- Loss vs genie by M ∈ {8, 16, 32, 64}: all test users (primary), adv ≥ 3 dB (secondary).
  Networks: 5-seed mean with t-based 95% CI.
- Position error of each method: range (along the array-to-user horizontal direction) and
  sideways (perpendicular): median and p90 of the absolute error (networks: pooled over
  seeds, plus the per-seed range).
- Paired per-user bootstrap (10,000 resamples, `default_rng(0)`, percentile 95% CI) of the
  per-user loss difference (network: 5-seed mean per user):
  - radar network − radar picker; LiDAR network − LiDAR geometric estimator;
  - fused network − best single-sensor method, where "best" is the single-sensor method
    (radar picker, LiDAR geometric, radar network, LiDAR network) with the lowest
    validation loss at that (frequency, M).
  Reading: "network better" if the CI is entirely below 0, "worse" if entirely above,
  otherwise "no detectable difference".
- References: the 263-beam two-stage sweep; the Phase 2b sigma-vs-M curve (D-030 E2).
- Also reported: the fraction of frames where the LiDAR estimator used its fallback; the
  number of users whose vehicle footprint covers the pole (0, 0) (frozen users, reported
  only).

**Files.** Scripts `scripts/phase3/`; data `data/processed/phase3/`; checkpoints
`checkpoints/phase3/`; results `results/phase3/`. No edits to `main.tex` or banked files.

### D-031 design addendum (2026-11-17, validation only; before any test evaluation)

Fixed on training and validation data only. The test split has not been loaded by any
Phase 3 script before this point. The frozen designs follow.

- **LiDAR geometric estimator** (`scripts/phase3/lidar_estimator.py`). Pipeline: ground removal
  (z ∈ [0.15, 1.7] m) → 0.2 m occupancy grid, 8-connected clusters, largest with diagonal ≤ 6 m and
  ≥ 20 points → heading by L-shape closeness fitting on body-side points (z ≤ Z_SIDE, ≥ 30 points;
  else the min-area rectangle) → box from the visible faces with the known 4.5 × 1.8 m footprint →
  front/back from cabin-level points (z ≥ Z_CAB) in the rear-cabin zone ξ ∈ (−1.6+δ, −0.8−δ) under
  each heading hypothesis (needs a lead of ≥ N_CAB) → antenna = centre − 1.3 m · forward.
  - Tuned on validation by a 64-point grid. Result: AMBIG = near (unresolved frames take the
    hypothesis nearer the sensor), Z_CAB = 1.1, N_CAB = 3, δ = 0.05, TOL = 0.2, Z_SIDE = 0.6.
  - Fallback: the training-median antenna position (9.12, 0.68) m.
  - Validation results: mean sweep loss 1.10 dB; error median 0.065 m, p90 0.58 m; fallback 0%;
    front/back resolved 92%.
- **Networks** (`scripts/phase3/train_net.py`, seed-11 pilots, head {P, H} × lr {1e-3, 3e-4}).
  Every sensor chose **H, lr 1e-3** (Gaussian-target cross-entropy; local soft-argmax readout).

  | Sensor | Validation sweep (dB) | Validation error median / p90 (m) | Final train loss / prior |
  |---|---|---|---|
  | Radar | 7.05 | 0.82 / 2.09 | 4.88 / 8.69 |
  | LiDAR | 0.50 | 0.030 / 0.080 | 2.84 / 8.69 |
  | Fused | 0.33 | 0.028 / 0.074 | 2.84 / 8.69 |

  - The learning check passes for every chosen configuration.
  - Radar P stayed at the prior at both learning rates: 9.9357 / 9.9350 against 9.9357. It was
    not chosen.
  - Seeds 22, 33, 44 and 55 are trained with the chosen configuration. Seed 11 reuses its pilot.
- **Evaluation** (`scripts/phase3/eval_test.py`, frozen at this tag): sweep shape chosen per
  method, frequency, M and seed on validation; the remaining rules are as pre-registered above.
- Records: `results/phase3/design_lidar_estimator.json` and `results/phase3/design_networks.json`.

## D-029 deviation 2 — radar vehicle scatterers centred on the antenna (bug, leak) (2026-11-17)

**Found by** the Phase 3b audit (`scripts/verify3/`, `results/verify3/a1_a2.json`, item A1a).
**Bug.** `scripts/phase2/p1_radar.py` drew the 6 vehicle scatterers in a box centred on the
**true antenna position** (ξ ~ U(−2.25, 2.25), η ~ U(−1, 1) around `ant_xy`). In the Phase 3
scene the antenna sits 1.3 m behind the body centre, so the radar saw a car centred on the
answer. The evidence:
- The scatterers' power-weighted centre lies on the antenna (mean ξ = 0.007 m).
- The radar net's mean error along the heading is −0.03 m.
- The radar net's median distance to the antenna is 0.90 m, against 1.53 m to the body centre.

This is a leak of the antenna position into the radar input. It affects the radar picker, the
radar network and the fused network. The LiDAR inputs and the geometric estimator are not
affected (audit A1b, A1d, A6).
**Fix (PI decision, 2026-11-17).** Draw the 6 scatterers in the car body: a 4.5 × 1.8 m footprint
around the **true body centre** c = antenna + 1.3 m · (cos ψ, sin ψ), the same body as the LiDAR
scene. The draws are ξ ~ U(−2.25, 2.25) and η ~ U(−0.9, 0.9); height, amplitude and phase are
unchanged. The rng and the draw order are unchanged (same number and order of draws). Nothing
else in the radar spec changes:
- array, waveform, the 12 clutter reflectors (6 static), SNR definition and noise seeds;
- zero-Doppler range-angle processing;
- the BEV warp.

The fixed data goes to `data/processed/phase3b/`; the Phase 2a files stay as the record.
**Consequence.** The Phase 3a radar-picker, radar-network and fused-network results
(`results/phase3/test_results.json`) are superseded. The Phase 3a LiDAR network and
geometric-estimator results are unaffected; audit A4 reproduces them exactly.

## D-032 — PRE-REGISTERED: Phase 3b, 20-seed campaign on the fixed radar, stress, varied vehicles (written 2026-11-17, before any Phase 3b data)

**Frozen at `revision-design-v2`:**
- array, codebook, channels, users and split; the radar spec (with deviation 2 above); the
  LiDAR spec;
- the geometric estimator and its tuned parameters;
- the network architecture (coordinate U-Net, H head, local soft-argmax) and its training
  settings (AdamW, lr 1e-3, wd 1e-4, cosine, 40 epochs, batch 16, fp32);
- the sweep (A × R = M, shape chosen per method, frequency, M and seed on validation).

There is no re-tuning. The radar and fused networks are retrained on the fixed radar with
exactly these settings. `train_net.py` gains only path and tag arguments (`--radar_bev`,
`--lidar_bev`, `--suffix`), with defaults unchanged.

**Test-set history.** The frozen test split was also used for evaluation in Phases 2a
(D-029) and 2b (D-030). No Phase 3 design choice used it: audit A3 checked every Phase 3
design script, and only `eval_test.py`, run once after `revision-design-v2`, reads it.

### Methods

| Name | What |
|---|---|
| radar picker | Phase 2a centroid picker on the fixed 20 dB radar. The bias is re-estimated by the frozen rule: median of (true − estimate) in (R, sin φ) over training users at their assigned SNR. The output is converted to xy at antenna height 1.5 m. |
| LiDAR geometric | frozen estimator (`results/phase3/design_lidar_estimator.json`) |
| radar net | H, lr 1e-3, trained on the fixed 20 dB radar BEV |
| LiDAR net | H, lr 1e-3, the Phase 3a LiDAR BEV (seeds 11–55 reused) |
| fused net | H, lr 1e-3, fixed radar BEV + LiDAR BEV |
| LiDAR net (varied) | as the LiDAR net, trained on varied-vehicle BEVs (item 3a below), 5 seeds |

**Seeds (B1).** 20 seeds: 11, 22, …, 55 and 66, 77, …, 220 (step 11). A seed changes only
the weight initialisation and the mini-batch order. Data, split and noise are fixed.

**Learning check.** Each network's final training loss is compared with its prior level. If
the radar network does not get below the prior, that is reported as the result, and the
remaining items still run.

### Primary endpoint (B1)

**LiDAR net vs LiDAR geometric, all test users, M ∈ {8, 16}, at 60 and 140 GHz (4 cells).**
- (i) Paired per-user bootstrap of Δ = (20-seed mean per-user loss of the net) − (geometric
  loss), with 10,000 resamples, `default_rng(0)` and a percentile 95% CI.
- (ii) Seed-level test: for each seed s, Δ_s = mean test loss of seed s − mean test loss of the
  geometric estimator. Run a two-sided one-sample t-test of the 20 Δ_s against 0, with
  Wilcoxon signed-rank as a secondary.
- **"LiDAR net better"** requires, in all 4 cells, a bootstrap CI entirely below 0 **and** a
  Holm-adjusted t-test p < 0.05 (Holm over the 4 cells).
- **"LiDAR net worse"** requires the mirror condition. Anything else is "no detectable
  difference".

### Secondary (B2, B3)

**Tables (B2)** for every method, at 60 and 140 GHz and M ∈ {8, 16, 32, 64}:
- Loss vs genie:
  - networks: 20-seed mean with t-based 95% CI;
  - radar picker and geometric estimator: single deterministic value.
- Share of users within 1 dB.
- Position error: median and p90 of the range and sideways components.

**Subsets:**
- (a) all test users;
- (b) r < d_F, where r is the 3-D distance from the array centre to the antenna and
  d_F = 2D²/λ with D = 127λ/2. That is 40.3 m at 60 GHz (every user) and 17.3 m at 140 GHz;
- (c) adv ≥ 3 dB;
- (d) sensitivity cut, item 3c: users with horizontal distance ≥ 2 m, crossed with (a)–(c).

**Bootstraps** (as D-031):
- radar net − radar picker;
- LiDAR net − LiDAR geometric;
- fused net − best single-sensor method. The best single-sensor method is chosen at each
  (f, M) by the lowest validation loss among the radar picker, LiDAR geometric, radar net and
  LiDAR net.

The references are the 263-beam sweep and the Phase 2b error-vs-beams curve.

**Value of range focusing (B3).** Every sweep is repeated with the far-field ring only:
the A = M nearest angles at ring 0. Reported: the far-field-only loss and
Δ_focus = far-field-only − full, per method, f, M and subset. Positive values mean range
focusing helps. The genie's far-field-only loss (= adv) is given as a reference.

### Stress tests (B4): evaluation only, frozen models (all 20 seeds), shape from clean validation

| Setting | Change | Methods evaluated |
|---|---|---|
| LiDAR 32 channels | every 2nd of the 64 elevation channels (same −25…+15° FOV) | LiDAR net, fused net, geometric |
| LiDAR 16 channels | every 4th channel | same |
| LiDAR dropout 30% | P_drop 0.10 → 0.30 | same |
| LiDAR noise 5 cm | σ_r 0.02 → 0.05 m | same |
| radar 10 dB, 0 dB | fixed radar at the existing D-029 SNR levels | radar net, fused net, radar picker |
| fused, radar removed | radar channel set to 0 | fused net |
| fused, LiDAR removed | LiDAR channels set to 0 | fused net |

The test (and validation) scenes are re-rendered with the same per-user seeds; only the
stated parameter changes.

### Cost (B5)

For each network:
- the number of parameters;
- per-frame time at batch 1, on the GPU (RTX 5060 Laptop) and on the CPU:
  - preprocessing: LiDAR point cloud → BEV; radar cube → range-angle map → BEV warp;
  - inference.

For the geometric estimator: per-frame CPU time. All times are medians over 200 test frames,
after warm-up.

### Added items (PI, 2026-11-17)

**3a. Varied vehicles.** The vehicle is drawn per user with `default_rng(20261121 + idx)`:
- length L ~ U(3.8, 5.0) m; width W ~ U(1.65, 1.95) m;
- roof height H ~ U(1.35, 1.50) m. This is the compact-to-mid-size passenger-car range. The cap
  at 1.50 m is there because the frozen channels put the antenna phase centre at 1.5 m, so the
  fin rises 0–0.15 m above the roof;
- body box L × W × (2/3)H;
- cabin: length 0.533 L, width W − 0.2, from (2/3)H to H, rear edge at ξ = −0.356 L (the
  frozen template, scaled);
- antenna on the centreline at ξ_a ~ U(−0.34 L, −0.20 L) (template: −0.289 L). The car is
  placed so that this point coincides with the frozen antenna position, so users and channels
  are unchanged;
- LiDAR noise seeds are unchanged.

The steps:
1. Evaluate the frozen LiDAR net (20 seeds) and the frozen geometric estimator (which still
   assumes 4.5 × 1.8 m and 1.3 m) on the varied-vehicle test set. Evaluation only.
2. Train the 5-seed LiDAR net (varied), seeds 11–55, frozen settings, on the varied-vehicle
   training users.
3. Report it on the varied test set next to the single-vehicle results.

Radar and fused are not evaluated on varied vehicles, because the radar scene keeps the
standard body.

**3b. Antenna-offset error.** The car is re-rendered with the true antenna 1.3 + δ m behind the
body centre, for δ ∈ {−0.5, −0.2, −0.1, +0.1, +0.2, +0.5} m. The geometric estimator still
assumes 1.3 m, which is equivalent to its assumed offset being wrong by δ. Evaluated: the
geometric estimator and the frozen LiDAR net; sweep loss at all M and both frequencies, all
test users.

**3c.** Every B2 table is also given with users closer than 2 m (horizontal) excluded.

**Files.** Scripts `scripts/phase3b/`; data `data/processed/phase3b/`; checkpoints
`checkpoints/phase3/` with suffix `_r2` (fixed radar) or `_var` (varied vehicles); results
`results/phase3b/`. The Phase 3a and 3b scripts and results and `scripts/verify3` /
`results/verify3` are committed to the release repo at the end. No edits to `main.tex` or
banked files.

## D-033 — PRE-REGISTERED: Phase 3c, varied vehicles at 20 seeds, DeepSense LiDAR check, paper assets (written 2026-11-17, before any Phase 3c training)

**Frozen at `revision-design-v2`** (with D-029 deviation 2 and D-032): array, codebook,
channels, users, split, the LiDAR and radar specs, the varied-vehicle generator of D-032 item
3a, the geometric estimator and its tuned parameters, the network architecture (coordinate
U-Net, H head, local soft-argmax) and its training settings (AdamW, lr 1e-3, wd 1e-4,
cosine, 40 epochs, batch 16, fp32, final epoch), and the sweep (A × R = M, shape chosen on
validation). There is no re-tuning.

**Test-set history.** The frozen simulated test split has been evaluated in Phases 2a, 2b,
3a and 3b. C1 evaluates it again with the same frozen rules; no Phase 3c design choice uses
it. The DeepSense test split (C2) is new and is evaluated once, after training.

### C1. Varied vehicles, 20 seeds (headline)

**Training.** LiDAR net (varied), seeds 66, 77, …, 220 (15 runs), on the varied-vehicle
BEVs (`data/processed/phase3b/lidar_bev_var.h5`), exactly as the 5 seeds 11–55 of D-032.
Learning check: final training loss below the prior.

**Methods**
- LiDAR geometric (frozen; still assumes 4.5 × 1.8 m and the 1.3 m antenna offset);
- LiDAR net, single vehicle (20 seeds, D-032);
- LiDAR net, varied-trained (20 seeds).

**Sweep shape.** Chosen per method, frequency, M and seed on validation users of the
method's own training condition: the standard validation set for the geometric estimator
and the single-vehicle net, the varied validation set for the varied-trained net (as D-032
item 3a).

**Test sets**
- varied-vehicle test set (main): the antenna position on the car is unknown to the
  methods;
- standard test set (identical vehicles): the antenna position on the car is known.

**Reported** for each test set, f ∈ {60, 140} GHz, M ∈ {8, 16, 32, 64}, and subsets all
users, adv ≥ 3 dB, horizontal distance ≥ 2 m:
- loss vs genie: networks as the 20-seed mean with t-based 95% CI; the geometric estimator as
  a single value with a per-user bootstrap 95% CI;
- share within 1 dB;
- position error: median and p90 of the horizontal Euclidean error, and of its range and
  sideways components.

**How it is read.**
- Headline comparison: varied-trained net vs geometric, on the varied test set, all users,
  M ∈ {8, 16}, both frequencies (4 cells). Same rule as the D-032 primary endpoint: paired
  per-user bootstrap (10,000, `default_rng(0)`) of the 20-seed mean minus geometric, plus a
  seed-level one-sample t-test with Holm over the 4 cells. "Better" needs every CI below 0
  and every Holm p < 0.05; "worse" is the mirror; else "no detectable difference".
- Secondary: varied-trained vs single-vehicle net (paired bootstrap of the two 20-seed
  means, and a Welch t-test on the seed means), and both nets vs geometric, in every
  reported cell. These are read by CI only.

### C2. Real-data check (DeepSense 6G, LiDAR)

**Source** (copied read-only from the lab share; the share is left untouched; the data are
not committed to the release repo, which has `data/deepsense_raw/` in `.gitignore`):
`\\TAUSEEFLABPC\d\DataSet\deepsense and Deepmimo\Deepsense\`

| Scenario | Type | Source files | LiDAR |
|---|---|---|---|
| 8 | V2I | `scenario8.zip` (28,738 entries; 4,044 LiDAR frames) | 2-D scan (range, angle), about 16 m range |
| 31 | V2I | `scenario31_new.zip` (35,074 entries; 7,013 LiDAR frames) | in archive |
| 36 | V2V | `Scenario36/`: `scenario36.csv/.mat/.p`, `scenario36_lidar.zip.001–.002`, `scenario36_pwr.zip.001`, `scenario36_gps.zip.001` | in archive |

The scenario-36 radar, camera and satellite archives (about 112 GB) are not copied: they are
not used, and they would not fit on the local disk.

**Task.** Per frame, predict the best beam = arg-max of the measured beam-power vector
(64-beam codebook; angle only, no near-field range labels). If a scenario has several
arrays, the label is the best (array, beam) pair over all of them and the output covers all
pairs.

**Split.** Per scenario, whole sequences (`seq_index`) are assigned at random,
`default_rng(20261118)`, to train/val/test in order until each holds about 60/20/20% of
the frames. No sequence spans two splits.

**Input (adapt only the grid).** LiDAR points in the sensor frame → BEV with the D-031
channels (log count, max height / 2, mean intensity). A channel the sensor does not provide
(height, intensity on a 2-D scan) is set to 0. ROI, cell size and grid size (divisible by
8) are fixed per scenario from the training frames only and recorded in a dated design
addendum before any C2 training. The coordinate channels are normalised to the new grid.

**Network (adapt only the output).** The coordinate U-Net trunk is unchanged. The output
conv has 1 + B channels (B = number of beams or array-beam pairs): channel 0 is the
location logit map, softmax over cells = p(cell); channels 1..B are per-cell beam logits
ℓ(cell, b). Beam probability P(b) = Σ_cells p(cell) · softmax_b ℓ(cell, b). The network
thus localises the user and reads the beam at that location. Loss: cross-entropy of
log P(b) vs the best beam. Training settings as frozen (AdamW, lr 1e-3, wd 1e-4, cosine,
40 epochs, batch 16, fp32, final epoch). Seeds 11, 22, …, 220 (20 seeds). Learning check:
final training loss below the prior (the entropy of the training-beam marginal).

**Baselines**
- most common beam: the K most frequent training labels;
- 1-NN on LiDAR features: the BEV log-count channel, 4 × 4 average-pooled, L2-normalised;
  the nearest training frame (Euclidean); its measured power vector ranks the top-K;
- geometric estimator → nearest beam, only if calibration allows. It needs a 3-D cloud with
  a determinable ground plane and sensor height, and a beam-angle map. The beam-angle
  offset is fitted on training frames only (one boresight offset, least circular error).
  If any of these is missing, it is reported as not applicable with the reason.

**Metrics**, per scenario, on the test split:
- top-K accuracy and mean power loss vs the best beam, 10 log10(P_best / max over the
  top-K of P), at K ∈ {1, 3, 5, 8} (median also given);
- beam-change rate: the fraction of consecutive frame pairs within a sequence whose label
  changes, for the ground truth and for each method's top-1 prediction.

**How it is read.** Network: 20-seed mean with t-based 95% CI. Net vs each baseline:
sequence-cluster bootstrap (resample test sequences, 10,000, `default_rng(0)`) of the
difference in top-1 accuracy and in K = 1 and K = 3 power loss → better / worse / no
detectable difference. This is a real-data check, not a near-field claim.

The earlier DeepSense results (radar network, D-016 to D-024) are superseded for the paper
and are not used for any claim.

### C3. Paper assets

One script, `scripts/paper_v2/build_assets.py`, regenerates everything from banked JSONs
(`results/phase2/feasibility/`, `results/phase3b/`, `results/phase3c/`) into
`paper_assets_v2/`:
- `canonical_macros_v2.tex` (macro names letters only);
- table `.tex` files;
- figure PDFs.

`scripts/paper_v2/trace_check.py` checks that every number in the tables, macros and figure
labels traces to a banked JSON value. No edits to `main.tex` or banked files.

**Files.** Scripts `scripts/phase3c/`, `scripts/paper_v2/`; data
`data/processed/phase3c/`; checkpoints `checkpoints/phase3/` (suffix `_var`) and
`checkpoints/phase3c/`; results `results/phase3c/`.

### D-033 design addendum (written 2026-11-17, before any C2 training; training frames and file formats only)

**Copy check.** All 9 copied files match the source sizes byte for byte (`scenario8.zip`
910,993,649; `scenario31_new.zip` 8,614,298,522; scenario-36 LiDAR parts 8,522,825,728 +
394,361,348; power 88,623,841; GPS 12,535,384; index files). The extracted archives give
4,043 / 7,012 / 24,799 frames (LiDAR + beam power) for scenarios 8 / 31 / 36.

**Labels.** Best beam = arg-max of the measured power vector (0-based). The CSV beam indices
of scenarios 8 and 31 are 1-based (CSV = arg-max + 1 for 100% of frames). Scenario 36 has four
64-beam arrays: label = array × 64 + beam over 256 pairs; it equals the CSV
`unit1_overall-beam` for 100% of frames.

**Splits** (`default_rng(20261118)`, whole sequences):

| Scenario | LiDAR | Frames train / val / test | Sequences train / val / test |
|---|---|---|---|
| 8 (V2I) | 2-D scan, about 255 points per frame | 2,432 / 849 / 762 | 50 / 17 / 16 |
| 31 (V2I) | 3-D, about 16,500 points, x/y/z (z up) + intensity | 4,219 / 1,592 / 1,201 | 31 / 13 / 8 |
| 36 (V2V) | 3-D Ouster, about 15,300 valid points, x/y/z (mm) + reflectivity | 15,000 / 5,000 / 4,799 | 75 / 25 / 24 |

**Grid rule** (from every 5th training frame): horizontal range clip R_c = min(99th percentile
of training-point range, 51.2 m); ROI = [0.5, 99.5] percentile box of x and y within R_c;
cell = the smallest of 0.2, 0.25, 0.4, 0.5, 0.8, 1.0, 1.25 m with both sides ≤ 128 cells
(sides rounded up to a multiple of 8). The cap of 128 keeps the scenario-36 inputs at about
2.4 GB in RAM and its 20 seeds at about 10 GPU-hours. Height = z − z_ground (2nd percentile of
training z within R_c), clipped at 0, / 2; intensity / training 99th percentile, clipped to 1.

| Scenario | R_c (m) | Cell (m) | Grid | z_ground (m) |
|---|---|---|---|---|
| 8 | 14.2 | 0.2 | 96 × 120 | – (2-D: height and intensity channels 0) |
| 31 | 51.2 | 0.8 | 104 × 104 | −0.98 |
| 36 | 51.2 | 0.8 | 96 × 96 | −2.44 |

Values are in `results/phase3c/c2_design_ds{8,31,36}.json`.

**Geometric estimator.**
- Scenario 8: not applicable (2-D scan; the frozen estimator needs a 3-D cloud).
- Scenario 36: not applicable (V2V: the LiDAR moves with the receiving car; there is no heading
  for a LiDAR-to-GPS azimuth offset; four arrays).
- Scenario 31: applicable.
  - The frozen estimator and parameters run on clouds shifted by −z_ground.
  - Beam angles θ_b = circular median of the GPS azimuth of the user (unit 2) seen from the
    base station (unit 1, local east-north), over training frames with label b.
  - One LiDAR-to-GPS azimuth offset = circular median of (LiDAR azimuth − GPS azimuth) over
    non-fallback training frames.
  - Top-K = the K beams whose θ_b is nearest the corrected LiDAR azimuth. Frames where the
    estimator falls back use the most-common-beam ranking.

**Code.**
- `scripts/phase3c/ds_prepare.py`, `train_ds.py` (output adaptation exactly as D-033),
  `chain_ds.sh` and `eval_c2.py`.
- A 2-epoch smoke run (scenario 8, seed 1, checkpoint deleted) checked that the code runs;
  no setting was changed after it.
- Training order: scenario 8, then 31, then 36, after C1.

## D-033 results note (after all Phase 3c runs; 2026-11-18)

Recorded after the runs. No pre-registered design setting was changed; the items below are
evaluation-side deviations and incidents, disclosed here.

**Interrupted run.** The training queue was stopped for a laptop restart while scenario 36,
seed 209 was at epoch 30. No checkpoint is written before the final epoch, so the run was
repeated from the start with the same seed and settings. The log line "FAILED ds36_s209" in
`results/phase3c/logs/lane_ds.out` is that stop.

**Non-finite power vectors (deviation).** 79 scenario 31 frames have NaN entries in the raw
DeepSense power vector: 49 in train, 17 in validation and 13 in test. Scenarios 8 and 36 have none.
- `eval_c2.py` excludes these frames from scoring and from the training pool of the baselines;
  the counts are in `c2.json` (`excluded_nonfinite_power`).
- Their labels (arg-max with NaN) were still used when the network was trained, about 1% of
  scenario 31 training frames. Retraining was not done.

**Learning check (clarification).** "train loss < prior" alone is met trivially by a run that sits
at the prior. `eval_c2.py` also reports the relative drop and `learned` = final train loss
< 0.95 × prior.
- Scenario 8: 20/20 learned (drop 37–79%).
- Scenario 36: 20/20 learned (drop 66–87%).
- Scenario 31: 0/20 learned (drop ≤ 0.5%); the network predicts the training beam marginal and
  matches the most-common-beam baseline. This is reported as a negative result; no tuning was
  done, since the design is frozen.

**Headline reading.**
- C1: varied-trained net better than the geometric estimator in all four headline cells. The
  single-vehicle net is better than the varied-trained net in every cell (secondary).
- C2: the network is better than both baselines on scenario 8. On scenario 31 there is no learning.
  On scenario 36 there is no detectable top-1 difference from 1-NN or the most common beam, and
  the network is worse than the most common beam at top-3 and above.

**Trace check.** `scripts/paper_v2/trace_check.py` treats the dataset name "DeepSense 6G" as
text. Final verdict: PASS (518 values, 72 keys, every table and figure number traced).

## D-034 — PRE-REGISTERED: Phase 3d, verification, gap-filling, reproducibility (written 2026-11-18, before any Part B or C run)

**Frozen** as in D-033: simulator, split, sensors, varied-vehicle generator, geometric
estimator, the location networks (architecture, H head, training settings) and the sweep rule
(A × R = M, shape chosen on validation). Only B6 trains new networks; Part A may retrain a
DeepSense scenario only if A3 finds a bug.

**Bug-fix rule.** Any fix is a dated deviation in this log. Results are reported before and
after the fix, and the old numbers are kept.

**Paths.** Part A scripts in `scripts/verify4/`, Part B in `scripts/phase3d/`, Part C in
`scripts/repro/`. Outputs in `results/verify4/` and `results/phase3d/`.

**Pre-analysis observations (code reading only, before writing this entry).**
- The varied-vehicle renderer keeps each user's antenna at its frozen position and places
  the car around it (car centre = antenna − ant_xi · heading).
- The channel model has the antenna and point reflectors only; the car body is not in the
  channel.
- The paper assets mix position-error definitions: per-axis range/sideways medians in the
  macros and the required-accuracy figure, Euclidean elsewhere.
- Scenario 36 CSV rows carry per-device file timestamps that differ by up to about 2 s
  (power vs LiDAR) within one row.

### Part A — verification (fresh scripts; each item gets PASS / FAIL with evidence)

**A1 Varied-vehicle labels.**
- Check 1: the training target of every `_var` checkpoint is `users.h5` ant_xyz[:, :2], and
  the BEV file is `lidar_bev_var.h5` (checkpoint args).
- Check 2: for every user, the varied car centre + ant_xi · heading equals ant_xyz to
  1e-9 m.
- Check 3: the template antenna (car centre − 1.3 m · heading) differs from it by
  |ant_xi + 1.3|.
- Check 4: in the rendered varied cloud, the vehicle-box returns surround the car's own
  centre.
- Shown by hand for 5 training users (fixed list: the first 5 training indices).
- PASS if all four checks hold.

**A2 Channel and genie, standard vs varied.**
- Check that the gain matrices used for both test sets are the same array, and that the
  generator has no vehicle-geometry input.
- Report the 263-beam full-sweep loss on the varied test set (all, adv ≥ 3 dB, ≥ 2 m;
  both frequencies).
- "Identical" if the max |ΔG| is 0. Either way, the result goes in the paper's limitations
  (no body blockage or scattering).

**A3 DeepSense sanity, scenarios 31 and 36.**
- **Frames:** 10 test frames each, `default_rng(34)`. Render the BEV grid with the
  receiver's GPS position projected into the LiDAR frame.
  - Scenario 31: ENU about unit 1, rotated by one azimuth.
  - Scenario 36: unit 2 relative to unit 1, in unit 1's heading frame, with heading from
    unit 1's GPS track.
- **Rotation:** fitted on training frames by maximising the mean LiDAR point count within
  2 m of the projected position, over 0–359° in 1° steps.
- **Time alignment:** repeat the fit with the GPS (and, separately, the beam label) shifted
  by k ∈ [−10, 10] frames within a sequence. The best k should be 0 ± 1.
- **Label check:** circular correlation between the label's beam angle and the GPS bearing,
  at k = 0 vs the best k.
- **PASS:** receiver inside the grid on ≥ 9 of 10 frames, best lag within ±1 for both
  checks, and the fitted rotation stable across sequences (circular SD < 10°).
- **On FAIL:**
  - Find the cause. If it is a pipeline bug, fix it, record a deviation and retrain that
    scenario (same 20 seeds, same design and grid rule), then report before and after.
  - If the cause is the dataset itself (e.g. device clock offsets), report it and do not
    retrain.
- The rendered images derive from licensed data: they are kept locally and only summary
  statistics are committed.

**A4 Position error definition.**
- List the definition used in every existing report.
- From now on, all paper assets use the horizontal Euclidean error ‖p̂ − p‖ (median, p90).
- The required-accuracy curve (Phase 2b, E2) is re-expressed in the same measure: the
  injected errors are regenerated exactly (`default_rng(2032)`, same order). The rerun must
  reproduce the banked per-axis medians before the Euclidean medians are banked.
- Per-axis components may appear only as clearly labelled supplementary values.

**A5 Independent reproduction.**
- A new script with its own network definition, decoder, beam set, loss and shape
  selection, importing nothing from phase3/3b/3c/verify3 evaluation code. It loads the
  checkpoints with strict=True; the simulator and the geometric estimator are called as
  black boxes.
- Cells: single-vehicle LiDAR net (20 seeds), geometric estimator and 263-beam sweep, on
  the standard and varied test sets, M ∈ {8, 16, 32, 64}, 60/140 GHz, all users.
- PASS if every cell matches the banked value (`results_b.json`, `c1.json`) within
  0.005 dB.

**A6 DeepSense cluster bootstrap.**
- Report G (test sequences) next to every real-data CI: 16 / 8 / 24 for scenarios
  8 / 31 / 36.
- **Validity check (Monte Carlo):**
  - Pool the per-frame network-minus-baseline differences on the validation and test
    sequences of each scenario; the population mean is the pooled mean.
  - Draw G sequences without replacement (G = that scenario's test count), compute the
    percentile cluster-bootstrap CI (2,000 resamples), repeat 1,000 times
    (`default_rng(36)`), and report the coverage.
- **Rule:** if coverage < 0.93, real-data CIs in the paper switch to the cluster-robust
  t-interval with G − 1 df, and both are reported.

### Part B — gap-filling (evaluation only, except B6)

- **B1 1-NN.**
  - Features: the BEV channel 0 (LiDAR log-count; radar 20 dB range-angle BEV), 4 × 4
    average-pooled and L2-normalised.
  - The nearest training user (Euclidean) gives the position estimate, which then goes
    through the frozen sweep with the shape chosen on validation (validation queries
    against the training set). Test at 20 dB radar.
- **B2 Radar ensemble.**
  - Six disjoint triples of radar seeds, in seed order: (11, 22, 33), …, (176, 187, 198).
  - Ensemble map = mean of the three softmax maps, decoded with the frozen local
    soft-argmax on its log.
  - Report: mean ± t-CI over the 6 ensembles; paired bootstrap vs the 20-seed single radar
    net (read by CI); cost = 3 × measured inference + preprocessing (CUDA and CPU, batch 1).
- **B3 Sweep shapes.**
  - Tabulate, for every method, frequency and M, the shapes chosen on validation (per-seed
    counts).
  - Add a fixed-shape baseline: M/8 nearest angles × all 8 rings (at M = 8: the top
    predicted angle × 8 rings), for every method.
- **B4 Acquisition time.**
  - Formula: M × T_beam + sensor latency + preprocessing + inference, against 263 × T_beam.
  - T_beam (primary): one OFDM symbol with normal CP at 120 kHz SCS = 8.92 µs (NR FR2
    numerology, one reference symbol per beam).
  - T_beam (sensitivity): 960 kHz SCS = 1.12 µs (NR FR2-2), and an SSB of 4 symbols at
    120 kHz = 35.7 µs.
  - Sensor latency: LiDAR at 10 Hz, mean frame age 50 ms, worst case 100 ms. Radar assumed
    20 Hz: 25 ms mean, 50 ms worst.
  - Processing: from `cost.json`.
  - Report air-time overhead and time-to-beam separately.
- **B5 Fresh test set.**
  - Draw candidate users with the frozen generator at new indices 1,000,000 + k (new
    channel, radar, LiDAR and noise seeds). Keep a candidate only if it falls in a test block
    (role 2) and lies ≥ 0.5 m from every training user. Stop at 1,200.
  - Generate channels (60/140 GHz), fixed radar at 20 dB only, and the standard-vehicle
    LiDAR.
  - Evaluate once, with shapes from the original validation: all frozen models (radar
    picker, geometric, radar/LiDAR/fused nets, varied-trained net), B1, B2, B6, and the
    263-beam sweep.
  - Report next to the original test set (all, adv ≥ 3 dB, ≥ 2 m). No decision uses it.
- **B6 Direct codeword classifier (trains).**
  - Same LiDAR U-Net backbone; separate per frequency; label = genie codeword (2,048
    classes); cross-entropy loss; frozen optimiser settings.
  - Candidate heads: (G) global average pool of the deepest encoder features → linear;
    (D) full decoder → global average and max pool → linear.
  - Grid: head ∈ {G, D} × lr ∈ {1e-3, 3e-4}, seed 11, per frequency.
  - Selection: lowest validation mean loss at M = 8, with the sweep = the top-M codewords
    by logit.
  - Then seeds 11–55 (5 seeds) of the chosen setting, tested once at M ∈ {8, 16, 32, 64}.
  - Comparison with the LiDAR location net: 5-seed vs 20-seed means, paired bootstrap, read
    by CI.
  - Learning check: final training loss < 0.95 × prior.

### Part C — reproducibility

- **C1 Scripts and fresh-clone rerun.**
  - Commit every local script the results depend on, including all of `scripts/phase2`.
  - In a fresh clone (with checkpoints from the C2 package), regenerate users, channels,
    split and LiDAR, then the single-vehicle LiDAR net and geometric rows of the standard
    test-set loss table (60 GHz, all users, M = 8–64).
  - PASS if the clone matches the banked values within 0.005 dB.
- **C2 Zenodo package.**
  - Contents: every checkpoint cited by the paper, a README, and a SHA-256 manifest. No
    DeepSense data.
  - The DeepSense-trained weights go in a separate folder, so they can be left out.
  - Report the file list and total size. The upload is left to the author.
- **C3 Near-field subset.** Varied test set, adv ≥ 3 dB, M = 8–64, both frequencies, every
  method (as in B5) plus the 263-beam sweep.

**End.** Regenerate the paper assets and rerun the trace check.

### D-034 deviations (dated)

**2026-11-18 — A3 lag window ±10 → ±20 frames.** The change was made in the A3 script before any A3 run, but this note was only written at 18:40, after the scenario 31 A3 run (which already used ±20) and as the scenario 36 run started. (The times first written here, 19:05 and 19:20, were wrong; corrected at 18:50.)
- The scenario 36 CSV shows constant per-device file-timestamp offsets relative to the row
  time: power −0.655 s, LiDAR +1.29 s (±0.02), radar +1.31 s, GPS and camera ≈ 0.
- A true LiDAR misalignment of that size (about 13 frames) would lie outside ±10, so the
  search window is widened.
- The best lag within ±10 is still reported.

**2026-11-18, 18:40 — B5 index offset 1,000,000 → 10,000,000.** The script used 10,000,000 from its first run; this note was written as that generation started (18:40).
- Every generator stream is seeded as base + user index, with bases 20261116 (geometry and
  channel), 20261117 (radar) and 20261119 (LiDAR), and radar noise at
  20261117 + 1,000,003 · (1 + s).
- With index 1,000,000 + k, the new users' geometry seeds (21261116 + k) would equal the
  original users' 20 dB radar-noise seeds (21261120 + i).
- With 10,000,000 + k, no new stream seed equals any original one.

### D-034 results (2026-11-18)

**Bug fixes and deviations recorded during the run (before / after):**
- **.gitignore excluded src/data (C1).** `data/` also matched `src/data/`, so a fresh clone
  failed at p1_lidar_split (`ModuleNotFoundError: src.data`). Before: C1 FAIL at step 2.
  Fix: anchor to `/data/` and commit src/data (90e52fd). After: C1 PASS.
- **Non-finite power rows in DeepSense scenario 31 (Phase 3c C2).** 79 frames have NaN
  power; they are excluded in eval_c2 (`excluded_nonfinite_power`), as recorded in the D-033 note.
- **Deviation note times corrected (3c92496).** The A3 lag-window note was first stamped
  19:05/19:20 but was written at 18:40; the A3 scenario 31 run preceded it.
- **Paper assets.** Internal decision IDs removed from table captions (the trace check
  flagged "034" as an untraced number). No result changed.

**A — verification**

| Item | Verdict | Evidence |
|---|---|---|
| A1 varied labels | PASS | All 20 _var nets use lidar_bev_var.h5 with labels = each car's own antenna (error 3.6e-15 m); 5 users checked by hand (a1_a2.json) |
| A2 channel/genie | PASS (identical) | No vehicle body in the channel model. 263-beam sweep on the varied set: 0.67 / 0.09 dB (60 / 140 GHz, all), 4.11 / 2.27 dB (adv ≥ 3 dB) |
| A3 DeepSense alignment | FAIL on the pre-registered criteria; no pipeline bug; no retrain | s31: LiDAR and labels mutually aligned, receiver GPS ~0.2 s behind; receiver inside the grid 4/10 (66.8% of all test frames), visible 47.6%. s36: no stable LiDAR-to-GPS rotation (contrast ~1.2, per-sequence SD 83°); projection unreliable |
| A4 position error | Inconsistency found and fixed | Per-axis medians were used in the PosErr macros and fig_required_accuracy; Euclidean elsewhere. Assets now use Euclidean (median, p90) everywhere |
| A5 reproduction | PASS | 34 headline cells from checkpoints; max diff 1.3e-15 dB |
| A6 cluster bootstrap | Under-covers at G = 8 (s31, 0.874) | Rule switch to cluster-t for s8 and s31; s8 1-NN K=1 loss cell under-covers with both (0.77–0.79). With t, two s31 calls become not significant: vs 1-NN loss K=3 [−0.35, +0.03] dB and vs geometric loss K=1 [−0.15, +0.03] dB (a6b_realdata_ci.json, tab_realdata_cmp) |

**B — gap-filling** (60 GHz, M = 8, standard test set unless stated)
- B1 1-NN: LiDAR 22.55 dB (+21.7 dB vs the LiDAR net), radar 32.18 dB (+12.3 dB vs the radar net).
- B2 radar ensemble (3 models): +0.19 [0.01, 0.38] dB vs a single radar net (test), +0.03 [−0.15, 0.21] (fresh); 2.9× the processing (9.6 vs 3.3 ms).
- B3 sweep shape: every method and seed chooses M angles × 1 ring; the fixed M/8 × 8 shape is worse (LiDAR net 1.77 vs 0.87 dB).
- B4 acquisition time (120 kHz, 8.92 µs/beam): M = 8 air time 0.071 ms vs 2.35 ms for the 263-beam sweep; mean latency is dominated by the sensor frame age (LiDAR net 53.8 ms, radar net 28.3 ms).
- B5 fresh set (1,200 users, evaluated once): replicates the test set (LiDAR net 0.93 vs 0.87 dB; sweep 0.63 vs 0.67 dB).
- B6 direct codeword classifier: chosen G|1e-3 (60 GHz) and D|3e-4 (140 GHz); all 10 runs learned. Test M = 8: 17.46 dB vs 0.87 for the LiDAR net (+16.6 [16.1, 17.1] dB); fresh +16.4; varied +15.0; 140 GHz +17.3. Not a bug (b6_diagnostic.json): labels equal the genie argmax; train loss 1.50 dB; 34% of test codewords never occur in training under the block split.

**C — reproducibility**
- C1 PASS: fresh clone at 90e52fd regenerated the LiDAR-net and geometric rows (60 GHz, M = 8–64), max diff 2.2e-16 dB.
- C2: zenodo_package/ (not committed): location_nets.zip 80 files 642.3 MB; design_selection.zip 19 / 152.5 MB; codeword_classifier.zip 16 / 149.6 MB; deepsense_lidar_nets/deepsense_lidar_nets.zip 60 / 475.0 MB (separate; DeepSense licence); README.md, SHA256SUMS.txt, manifest.json. Total 1.42 GB. Upload left to the author.
- C3: results_d.json, set "varied", subset adv_ge3 (LiDAR net 11.05 / 6.60 / 3.04 / 1.25 dB at 60 GHz vs sweep 4.10).

**Assets:** regenerated (new: tab_realdata_cmp, tab_baselines, tab_fixed_shape, tab_acq_time); trace check PASS (787 values, 113 keys).

## D-035 — Phase 3e: sensor age and vehicle speed (evaluation only) — pre-registered 2026-11-18

**Scope.** No training. Frozen checkpoints and frozen static predictions only
(results/phase3d/preds_d.npz). Frozen sweep shapes: the static validation choice per seed,
as in eval_d. Tag `revision-plan-v6` is set on the commit that adds this entry, before any
Phase 3e run.

**Methods.**
- LiDAR net: single-vehicle net, H head, lr 1e-3, 20 seeds.
- Geometric estimator: frozen D-031 estimator.
- Both are evaluated on the standard test set and the varied test set (1,193 users each).
- Reference: the 263-beam sweep.

### E1 — stale estimate

**Timing.** τ_i = a_i + p_m.
- a_i is the LiDAR frame age, uniform on [0, 100 ms] (10 Hz, sweep instant uniform within the
  frame period). It is drawn once per test user from `default_rng(2035)` in sorted test-index
  order and shared by all methods.
- p_m is the measured processing time from results/phase3d/b4_timing.json (`processing_ms`):
  3.76 ms for the LiDAR net, 2.67 ms for the geometric estimator.
- Air time (M × 8.92 µs, at most 0.57 ms) is excluded: at most 1.1 cm at 20 m/s.
- Reported: mean and 95th percentile of τ per method.

**Motion.**
- Speeds v ∈ {0, 2, 5, 10, 15, 20} m/s along each car's heading ψ.
- True antenna at sweep time: p + v·τ_i·(cos ψ, sin ψ, 0).
- Reflectors are static (positions and complex gains unchanged, as in the channel model).
- The channel is recomputed with p1_channels.channel and the same codebook. The genie is the
  best of the 2048 codewords on the moved channel.

**Estimate.** The static estimate from the LiDAR frame at t0 (preds_d), unchanged.

**Metric.** Loss vs genie (dB) on the moved channel.
- M ∈ {8, 16, 32, 64}; 60 and 140 GHz.
- Subsets: all users, and users with static adv ≥ 3 dB. Membership is fixed across speeds.
- Networks: 20-seed mean with a t-interval over seeds. Geometric: per-user bootstrap.

**Reference sweep.** Beams chosen on the channel at t0, used at t0 + 263 × 8.92 µs (2.35 ms).

**Bug check.** v = 0 must reproduce the results_d.json cells to 1e-9 dB. Otherwise stop and debug.

**Read.**
1. Primary: the loss increase over v = 0 at M = 8, standard test set, all users, per frequency,
   with a paired per-user bootstrap CI (10,000 resamples, seed 0).
2. Break-even speed: the smallest v in the grid at which the method's mean loss exceeds the
   reference sweep's loss at that speed, per frequency, M and subset.

### E2 — motion compensation from two LiDAR frames

**Second frame.** An earlier frame at t0 − 0.1 s, with the car at p − v·0.1·(cos ψ, sin ψ).
- Same vehicle (standard or varied) and same static background clutter: the original stream
  `LV.SEED + i`, first 1,800 draws.
- Fresh range noise, intensities and dropout from `default_rng(40000000 + 10000·k + i)`,
  where k is the speed index 0–5. Rendered at v = 0 too, so the velocity estimate carries
  estimator noise at rest.

**Estimates.**
- The same frozen estimator on the earlier frame (each of the 20 nets with its own
  predictions; the geometric estimator).
- v̂ = (p̂(t0) − p̂(t0 − 0.1)) / 0.1, a 2-D vector.
- Projected position p̂(t0) + v̂·τ_i. No gating; the estimators' own errors and fallbacks are
  kept.

**Metrics.** As E1, plus:
- Horizontal Euclidean residual position error (median, p90): stale (E1) vs projected (E2),
  per speed.
- Speed-estimate error |v̂ − v| (median, p90).
- Paired E2 − E1 loss difference: M = 8, all users, both frequencies, both sets, per-user
  bootstrap.

**Read.** Compensation helps at a speed if the E2 − E1 CI lies below 0, and hurts if it lies
above 0.

### E3 — real data, exploratory and post-hoc

**Data.** DeepSense scenario 31 test frames (finite power).

**Subset.** Visible = the A3b definition: rotation from a3_ds31.json at lag 0, and at least
3 non-ground points within 2 m of the GPS-projected receiver. 566 of 1,188 frames.

**Methods.** The existing 20 networks and the baselines (most-common beam, 1-NN, geometric).
- Metrics as eval_c2: top-K (K = 1, 3, 5, 8) and mean loss (K = 1, 3).
- Network minus baseline, with the number of visible-subset sequences G, the cluster
  bootstrap and the cluster-t interval (the A6 rule for scenario 31).
- Reported next to the all-frames result.

**Label.** Exploratory and post-hoc. The subset was defined after A3b had already shown
network top-1 of 1.1% (visible) vs 13.8% (not visible). No confirmatory claim.

### Assets

New tables tab_motion and tab_realdata_visible, and figure fig_motion (loss vs speed, M = 8,
standard set, both frequencies, stale vs compensated, sweep reference). Then regenerate all
assets and run the trace check (must PASS).

### D-035 results (2026-11-18)

**Bug checks.**
- The renderer copy reproduces the stored BEVs exactly (standard and varied, 25 users each).
- Net inference matches preds_d to 7e-15 m.
- E1 at v = 0 reproduces results_d.json to 4.4e-16 dB.
- E3 all-frames top-1 equals c2.json exactly.

**Deviations.**
- None in the analyses.
- Asset-only fixes: on fig_motion the x tick label "2" is omitted, because pdftotext merged "0"
  and "2" into "02" (the 2 m/s point is still plotted). The caption says "p90" instead of
  "90th percentile".

**τ (ms).**

| Method | Mean | 95th percentile |
|---|---|---|
| LiDAR net | 52.7 | 98.8 |
| Geometric | 51.6 | 97.7 |

Users moved to within 1 m of the array: at most 4 (at v ≥ 10 m/s).

**E1 — stale estimate.** Standard test set, all users, M = 8, 60 GHz.
- LiDAR net loss: 0.87 / 1.27 / 3.23 / 7.00 / 10.18 / 12.68 dB at v = 0 / 2 / 5 / 10 / 15 / 20 m/s.
- Increase over static: +0.40 [0.26, 0.54] dB at 2 m/s; +2.36 dB at 5 m/s; +11.81 dB at 20 m/s.
- Geometric: 1.81 → 12.65 dB.
- 140 GHz: within 0.1 dB of 60 GHz.
- M = 64 is far more tolerant: LiDAR net 0.19 → 1.32 dB at 20 m/s.
- Median position error at the sweep instant (LiDAR net): 3.1 cm static, 26.9 cm at 5 m/s,
  104 cm at 20 m/s.
- Break-even vs the reference sweep: at M = 8 both methods are already above the sweep at rest
  (0.87 vs 0.67 dB). At 60 GHz, all users, the LiDAR net crosses at 2 m/s (M = 16),
  10 m/s (M = 32) and 15 m/s (M = 64). At 140 GHz it is above the sweep at every M and speed
  (the sweep is 0.09 dB).
- Geometric at M = 64 dips at low speed (0.62 → 0.41 dB). This is not a bug: its 140 large
  errors (> 0.3 m) lie about 1.2 m ahead along the heading, so forward motion partly cancels them.

**E2 — compensation from two frames 0.1 s apart.**
- LiDAR net, M = 8, 60 GHz, all users: 0.90 / 1.03 / 1.20 / 1.32 / 1.40 / 1.49 dB.
- E2 − E1: +0.04 [0.02, 0.05] dB at v = 0 (velocity noise costs slightly at rest);
  −0.24 [−0.38, −0.10] at 2 m/s; −11.19 [−11.85, −10.54] at 20 m/s.
- Residual median error: 3.4 → 4.8 cm (p90 10 → 17 cm).
- Speed-estimate error, median: 0.14–0.44 m/s.
- Geometric compensated: 1.85 → 2.15 dB. Varied set at 2 m/s: compensation hurts the geometric
  estimator (+0.35 [0.08, 0.62] dB) and helps from 5 m/s.
- On the users with adv ≥ 3 dB the residual loss is still large at 140 GHz
  (LiDAR net, compensated, 13–16 dB at M = 8).

**E3 — exploratory, post-hoc.** DeepSense scenario 31, car-visible frames
(n = 566, G = 8) vs all frames (n = 1,188).
- Network top-1: 1.1% (visible) vs 7.8% (all).
- Most-common beam: 0.5% vs 7.7%.
- 1-NN: 14.7% vs 9.3%.
- Geometric: 6.0% vs 4.5%.
- The network behaves like the most-common prior; its all-frames accuracy comes from frames
  without a visible car.

**Assets.** New: tab_motion, tab_motion_poserr, tab_realdata_visible, fig_motion. Trace check
PASS (1,038 values, 138 keys).

### D-035 addendum (2026-11-19) — two additions for the headline table, before any rerun

**Why.** The headline table (tab_headline) needs two things motion.json does not hold:
- the users ≥ 2 m subset at v > 0 (Phase 3e saved only all users and adv ≥ 3 dB);
- a 95% CI for the 263-beam sweep, which the "smallest M within the sweep's 95% CI" rule
  compares against.

**Additions.**
1. Subset d_ge_2m: horizontal distance of the static true antenna from the array ≥ 2 m (the
   eval_d definition). Membership is fixed across speeds, as for adv ≥ 3 dB. It is added to
   every E1/E2 loss cell and to the break-even read.
2. Sweep CI: per-user bootstrap of the sweep's mean loss (B.boot, 10,000 resamples, seed 0),
   per frequency, speed and subset, stored as `sweep263_ci`. The `sweep263` means are
   unchanged.

**Procedure.** Evaluation-only rerun of motion.py with the same frozen models, seeds, users,
τ draws and earlier-frame noise streams. Output goes to results/phase3e/motion_v2.json.
Every key and value of motion.json except runtime_s must reproduce exactly (bit-for-bit
equal floats); if any cell differs, stop and report without using the new file.

**Headline rule.** For each frequency and test set, at v = 0, 10 and 20 m/s, all users: the
smallest M in {8, 16, 32, 64} at which the compensated LiDAR net's mean loss is ≤ the upper end
of the sweep's 95% CI. "None" if no M qualifies.
of the sweep's 95% CI. "None" if no M qualifies.

### D-035 addendum results (2026-11-19)
- Rerun reproduced every one of the 4,135 values in motion.json exactly (runtime excluded); the
  d_ge_2m cells at v = 0 match results_d.json to 1.1e-16 dB. Output: results/phase3e/motion_v2.json.
- New asset tab_headline (compensated LiDAR net and geometric estimator vs the 263-beam sweep;
  v = 0, 10, 20 m/s; both sets and frequencies; M = 8–64; all / adv ≥ 3 dB / ≥ 2 m, with 95% CIs;
  air time; median residual position error). Trace check PASS (2,052 values, 187 keys).
- Smallest M at which the compensated LiDAR net (all users) is ≤ the sweep's upper 95% CI,
  at v = 0 / 10 / 20 m/s:
  - Standard, 60 GHz: 16 / 32 / 32.
  - Standard, 140 GHz: none / 64 / 64.
  - Varied, 60 GHz: 32 / 32 / 64.
  - Varied, 140 GHz: none / none / none.

## D-036 — Phase 4: two robustness experiments — pre-registered 2026-11-19 (signed off with amendments 1–5)

Tag `revision-plan-v7` is set on the commit that adds this entry, before any training or
evaluation. Frozen designs: the LiDAR network
and DeepSense pipeline at `revision-plan-v6`, the D-031 geometric estimator, and the channel,
array, codebook and LiDAR renderer as used in D-035. The existing simulated test set and the
Phase 3d fresh test set are not used for anything new.

---

### Step 0 — data check (done before this entry; commit ec6651a)

**Downloads.** All three are read only; members are copied to data/deepsense_raw/ex/s{S}/.

| | 32 (day) | 33 (night) | 34 (night) |
|---|---|---|---|
| File | scenario32.zip | scenario33.zip | scenario34_new.zip |
| Zip size | 4.07 GB | 5.16 GB | 5.66 GB |
| Frames (CSV rows = LiDAR = power = unit-2 GPS files) | 3,235 | 3,981 | 4,439 |
| Sequences | 15 | 18 | 31 |
| Rows with finite power | 3,214 | 3,943 | 4,389 |

- **Modalities:** unit-1 LiDAR (.ply x, y, z, intensity), 64-beam 60 GHz power, unit-1 GPS
  (static), unit-2 GPS per frame, radar, camera.
- **CSV:** the same columns as scenario 31.
- **Copied:** CSV, LiDAR, power and GPS only (3.44 / 4.24 / 4.89 GB).

**Labels.** Arg-max of the power vector equals the CSV beam index (minus 1) on 100% of rows in
all three scenarios. Beams seen in training (old split): 61 / 63 / 64.

**File pairing (decision 1).** On every CSV row of scenarios 31–34, the LiDAR file index equals
the power file index and the CSV index.
- Neither file type carries a capture timestamp (the .ply header holds only point fields; the
  power file holds only 64 numbers).
- The zip member times are export times, not capture times. LiDAR and power files of the same
  row differ by −7 to −10 min (31), −1.5 to −2.6 h (32), about −38 h (33) and −1 to −2 min (34).
  Within one sequence the LiDAR times spread over up to 70 min, although a sequence lasts
  7–40 s by the CSV.
- The CSV row step is a median of 0.092–0.094 s in all four scenarios (10 Hz, some 0.18 s gaps).
- **Decision (amendment 1):** the dataset's own row pairing (same index on every row) is
  accepted. Capture timestamps are not available, so LiDAR-to-label alignment for 32–34 cannot
  be independently verified. The paper states this. No frame is shifted.

**GPS alignment (A3 method; reported as a dataset limitation, decision 3).**

| | 31 (ref.) | 32 | 33 | 34 |
|---|---|---|---|---|
| Best LiDAR–GPS lag (frames) | −2 | −5 | −17 | +6 |
| Best label–GPS lag (frames) | −2 | 0 | −1 | +2 |
| Alignment score at lag 0 vs background | 12.4 vs 2.5 | 2.6 vs 0.9 | 3.4 vs 1.3 | 2.8 vs 0.9 |
| Per-sequence rotation spread | 34° | 20° | 113° | 18° |
| Receiver inside the grid (10 test frames) | 4 | 10 | 2 | 8 |
| Verdict | FAIL | FAIL | FAIL | FAIL |

The visible-frame shares under the A3b rule (old split) were 27%, 35% and 18%. They depend on
this alignment and are not used.

**Deviation (loader only).** Scenarios 32–34 were added to ds_prepare's path table and use
scenario 31's .ply reader:
- 32 and 33 have no top folder in the zip;
- 34's CSV is scenario34.csv, not _dev.

The A3 script uses the scenario-31 GPS reader for them. Nothing else changed.

---

### Experiment A — unseen real scenarios 32, 33, 34

**Pipeline.** As Phase 3c, unchanged:
- preprocessing and the grid rule (designed from the training frames of each fold);
- the LiDAR network design and settings (AdamW lr 1e-3, wd 1e-4, cosine, 40 epochs,
  batch 16, final epoch);
- labels = arg-max of power; only rows with finite power are used.

**Splits (pre-registered change).** The Phase 3c 60/20/20 sequence split gives scenario 32 a
single test sequence, so no cluster CI would exist. Scenarios 32–34 therefore use grouped
5-fold cross-validation over sequences:
- **Folds:** the unique sequences are shuffled with default_rng(20261118), the Phase 3c split
  seed. They are cut into 5 folds at cumulative frame fractions 0.2, 0.4, 0.6 and 0.8 (the
  Phase 3c cumulative-fraction rule). Every sequence is tested exactly once.
- **Validation within fold k:** the sequences of the other four folds are shuffled with
  default_rng(20261118 + k). Validation is the leading sequences up to 25% of their frames,
  which keeps Phase 3c's 60:20 train:validation ratio. The rest is training.
- **Seeds:** 5 per fold (11, 22, 33, 44, 55); 75 trainings in all.
- **Runs:** run j is seed j in every fold, so each test frame is predicted once per run.
- **CIs:** cluster over all sequences of the scenario (G = 15, 18, 31), with the cluster
  bootstrap (10,000 resamples, default_rng(0)) and cluster-t.

Scenario 31 keeps its banked Phase 3c single-split result (G = 8). The table states the
protocol per scenario.

**Learning check.** Per seed and fold: final training loss < 0.95 × the label-prior entropy.
- Reported as the count passing out of 25 per scenario.
- Failing runs stay in the results.

**Baselines.** Fitted per fold on the training frames:
- the most-common beam;
- 1-NN on LiDAR features (the eval_c2 feature).

The geometric estimator and the visible-frame subset are not run on 32–34 (decision 3).

**Metrics.** As eval_c2: top-1/3/5/8 accuracy, mean (and median) power loss vs the best beam at
K = 1, 3, 5, 8, and beam-change rate.
- Network: mean over the 5 runs.
- Network minus each baseline: paired per frame; cluster bootstrap and cluster-t, with G next
  to every CI.

**Read.**
- For each scenario and baseline, the network beats the baseline on a metric only if both the
  cluster-bootstrap and the cluster-t intervals for the difference exclude 0 in its favour.
- "Works on scenario S" = top-1 better than both the most-common beam and 1-NN.
- Each scenario is reported separately; no pooling.
- A scenario is unusable only if its LiDAR frames are empty or broken; GPS problems do not
  qualify, because the network does not use GPS. Nothing is substituted.

---

### Experiment B — multi-car simulation

**New parameters and their reasons.**

| Item | Value | Reason |
|---|---|---|
| Road | straight road through the user car's antenna position, along its frozen heading ψ: 4 lanes, 2 per direction (lanes 0–1 heading ψ, lanes 2–3 heading ψ + π) | keeps the frozen user distribution (position and heading as D-029), so the frozen net is tested on what it was trained for, while the surroundings become a real road |
| Lane width | 3.5 m | typical urban and arterial lane (3.0–3.75 m) |
| Lane order | lateral coordinate η to the left of ψ: lane 0 (kerb, direction ψ), 1 (ψ), 2 (ψ + π), 3 (kerb, ψ + π); lane j centre at η = 3.5·(j − ℓ_u) m | right-hand traffic |
| User lane | ℓ_u uniform over lanes 0, 1; user at the lane centre | the user drives with the traffic in its direction |
| Other vehicles per scene | K_o ~ U{0, …, 5}, independent per scene | 0 = empty-road control, inside the same test set; 5 cars within ±30 m is dense urban traffic |
| Vehicle mix (moving) | car 70%, van / SUV 20%, truck / bus 10% | urban mix with heavy vehicles, which occlude most |
| Car | L ~ U[3.8, 5.0], W ~ U[1.65, 1.95], H ~ U[1.35, 1.50] m; two-box shape as D-032 varied vehicles | the D-032 passenger-car range |
| Van / SUV | L ~ U[4.8, 6.0], W ~ U[1.90, 2.10], H ~ U[1.9, 2.6] m; single box | panel vans and large SUVs |
| Truck / bus | L ~ U[8, 12], W ~ U[2.40, 2.55], H ~ U[3.0, 3.8] m; single box | rigid trucks and buses; EU limit 2.55 m wide, 4 m high |
| User car (amendment 3) | drawn from the D-032 varied-vehicle distribution (L ~ U[3.8, 5.0], W ~ U[1.65, 1.95], H ~ U[1.35, 1.50] m, antenna at fraction U[−0.34, −0.20] of L from the centre; two-box shape), from default_rng(20270601 + i) in that draw order | multi-car scenes are tested with varied cars; both frozen nets (single-vehicle and varied-trained) are reported as baselines |
| Longitudinal placement | each moving vehicle: lane uniform over 0–3, centre at s ~ U[−30, 30] m along the road from the user, lateral jitter U[−0.3, 0.3] m, heading = lane direction + N(0, 2°) | ±30 m covers the LiDAR ROI ([−20, 40] × [−20, 20] m) along any heading; jitter as in real lane keeping |
| Minimum gap | ≥ 2.0 m bumper to bumper in the same lane; footprints must not overlap (any lane); no footprint within 1.0 m of the RSU pole (0, 0) | a stopped queue keeps about 2 m; nothing stands in the pole |
| Placement | rejection sampling, up to 200 draws per vehicle; if a vehicle cannot be placed, the scene has fewer (actual count recorded and used for the by-count results) | no forced overlaps |
| Parked vehicles (amendment 4) | each other vehicle is parked with probability 0.15: placed in a 2.5 m parking strip outside the kerb lane on its side (lane 0 side or lane 3 side, equally likely), heading = that side's lane direction, type car 80% / van 20% | kerbside parking is common on urban roads, and parked cars are frequent static occluders |
| Other-vehicle speeds (amendment 4) | moving, same direction as the user (lanes 0–1): speed max(0, v + U[−3, 3]) m/s along ψ; moving, opposite direction (lanes 2–3): speed max(0, v + U[−3, 3]) m/s along ψ + π (opposite sign); parked: 0 | vehicles in one traffic stream travel within a few m/s (about ±10 km/h) of each other; oncoming traffic on the same road shares the speed regime; parked cars do not move |
| Second frame | 0.1 s before t0 (one 10 Hz LiDAR period). All vehicles moved back by their own velocity × 0.1 s; background clutter static; fresh range noise and dropout | matches D-035 E2 |
| Delay τ | as D-035: frame age U[0, 100 ms] from default_rng(2036) per test user + the b4 processing time of the estimator + the measured median association time per frame (rules R1, R2) | same latency model |
| Channel | frozen: user LoS + 2–4 static point reflectors (p1_channels draw order with the new user seed); other vehicles are not in the radio channel | the brief fixes the channel model |
| Radio blockage (amendment 5, evaluation only) | per test scene at t0: blocked if the straight segment from the array (0, 0, 2.0 m) to the user antenna (x, y, 1.5 m) intersects any other vehicle box. Reported: blocked fraction overall, by number of other vehicles, and by vehicle type (share of scenes blocked by at least one vehicle of that type, and per-vehicle blocking rate by type) | **The channel model does not include this blockage, so results for blocked users are optimistic for every method, including the sweep.** Losses are also reported for blocked and unblocked users separately |
| LiDAR | frozen renderer (D-031), ray-cast against all vehicle boxes. Background points are removed if any vehicle occludes them | the occlusion is the point of the experiment |
| GPS report | reported user antenna position = truth + e, e ~ N(0, σ²I₂), σ ∈ {1, 3} m. The same e at t0 and t0 − 0.1 s | GNSS error decorrelates over tens of seconds, so it is effectively the same across 0.1 s; σ = 1 m good open-sky GNSS, 3 m urban |
| Scene seeds | users default_rng(20270101 + i), the p1_channels draw order; user car default_rng(20270601 + i); vehicles default_rng(20270201 + i); LiDAR noise default_rng(20270301 + i); second frame default_rng(20270401 + 1000·k + i) for speed index k; GPS error default_rng(20270501 + i) | new seeds, disjoint from all earlier ones |
| Users and split | 6,000 users, the D-029 spatial-block rule (15 sectors × 4 bands, deterministic rotation, 0.5 m training guard) | same split rule, new draws |

**Association (choosing the user car).**
- **Clusters:** connected components of non-ground LiDAR returns (z > 0.3 m) at 0.8 m linking
  distance, at least 10 points. With 600 background points over 2,400 m², random chaining at
  0.8 m is rare.
- **Gate:** a cluster is a candidate if its nearest return lies within 3σ + 0.5 m of the
  GPS-reported antenna position. The 2-D GPS error exceeds 3σ with probability e^(−4.5) = 1.1%;
  the antenna lies within 0.5 m of the user car's outer surface.
- **Selection:** the candidate nearest the GPS report. If there is no candidate, the user car is
  taken as unseen and the estimate falls back to the GPS report itself.

Candidate rules (only the rule is chosen on validation):
- **R0 none:** the frozen estimator on the raw scene.
- **R1 erase others:** remove the returns of all non-selected clusters. The selected cluster,
  ground and background stay, so the input resembles training.
- **R2 crop:** keep only returns within 3σ + 3 m of the GPS report (plus ground).

**Truth labels for the results.**
- **Association success:** the selected cluster holds ≥ 50% of the user car's returns. The
  renderer records which returns hit the user car.
- **Occlusion:** the number of rays whose first hit is the user car, divided by the same count
  with the user car rendered alone. Both counts are taken before noise and dropout, so they are
  deterministic.
  Classes: visible ≥ 0.5, partly occluded 0 < r < 0.5, fully occluded 0.

**Models.**
1. **Frozen baselines (amendment 3):**
   - the single-vehicle LiDAR net (lidar_H_lr0.001_s{seed}.pt, 20 seeds);
   - the varied-trained LiDAR net (…_var.pt, 20 seeds);
   - the D-031 geometric estimator.
   Each uses the association rule chosen for it on validation.
2. **Multi-car LiDAR net:**
   - The frozen design and settings (D-031 U-Net, H head, lr 1e-3, AdamW wd 1e-4, cosine,
     40 epochs, batch 16, final epoch).
   - Trained on the multi-car training scenes at v = 0, after association.
   - Each training scene's σ is assigned 1 or 3 by default_rng(2037).
   - One training run per candidate rule: R0, R1, R2 × 5 seeds (11, 22, 33, 44, 55).
   - Learning check: final loss < 0.95 × prior.
3. **Rule choice (each model):** on validation, by the mean loss at M = 8, 60 GHz, v = 0,
   averaged over σ ∈ {1, 3} and seeds. Only the chosen rule is tested, once.
4. **Sweep shapes:** the validation choice per seed (as D-035), on the new validation set.

**Metrics** (new test set).
- Loss vs genie at M = 8, 16, 32, 64, both frequencies, v ∈ {0, 10, 20} m/s, with two-frame
  motion compensation (D-035 E2), σ ∈ {1, 3}.
- Association success rate per σ (and per occlusion class).
- Loss by number of other vehicles (0–5) and by occlusion class.
- The 263-beam sweep with per-user bootstrap CI.
- Median residual position error.

CIs:
- Nets: t-interval over seeds (20 for the frozen net, 5 for the multi-car net).
- Geometric estimator and sweep: per-user bootstrap (10,000, seed 0).

**Read (amendment 2).**
1. **Primary:** the multi-car-trained LiDAR net (its chosen rule) vs the 263-beam sweep.
   - Condition: 60 GHz, M = 32, v = 10 m/s with two-frame motion compensation, σ = 3 m, all
     test users.
   - "Matches the sweep in traffic" if the net's mean loss is within the sweep's 95% CI or better,
     i.e. ≤ the upper end of the sweep's CI. Otherwise it does not.
   - The net's own 5-seed CI is reported next to it.
2. **Secondary:**
   - Frozen-vs-retrained: each frozen net minus the multi-car net at the primary condition;
     per-user paired difference of seed means, per-user bootstrap.
   - All other frequencies, M, speeds and σ.
   - Degradation with traffic: K_o = 0 vs K_o ≥ 1, two-sample bootstrap.
   - By occlusion class and by radio blockage.
   - Smallest M at which each compensated method is ≤ the sweep's upper 95% CI.

All results are reported whichever way they go. No comparison is made against the earlier
single-car test sets: different users.

**Assets.**
- The real-data table gains scenarios 32–34, with a protocol column (single split G = 8 for 31;
  5-fold CV for 32–34).
- New table tab_multicar (+ by-count / occlusion breakdown).
- Trace check must PASS.

**Cost estimate.**
- Experiment A: 75 trainings.
- Experiment B: generation of 6,000 scenes, plus 3 × 6 second frames on validation and test;
  15 trainings at about 11 min each.
- Roughly 1.5–2 days of compute in total.

### D-036 results (2026-11-20)

Everything below was run once, on the designs frozen at tag revision-plan-v7. Nothing was chosen
on test data. The existing simulated test set and the Phase 3d fresh test set were not used.

**Step 0 (data check; recorded before D-036, commit ec6651a).**

| | 32 | 33 | 34 |
|---|---|---|---|
| Frames | 3,235 | 3,981 | 4,439 |
| Sequences | 15 | 18 | 31 |
| CSV beam agrees with the power vector | 100% | 100% | 100% |
| A3 GPS-alignment verdict | FAIL | FAIL (rotation SD 113°, LiDAR lag −17 frames) | FAIL |

- No capture timestamps exist. The dataset's same-index row pairing is accepted (amendment 1), so
  LiDAR-to-label alignment for 32–34 cannot be independently verified.
- No GPS-based item (geometric estimator, visible-frames subset) was run on 32–34.

**Experiment A — DeepSense 32, 33, 34 (grouped 5-fold sequence CV, 5 seeds per fold).**

Setup:
- Every sequence was tested exactly once.
- Frames whose raw power vector contains NaN are excluded from scoring (frozen Phase 3c rule).
  Scored frames: 3,214 / 3,943 / 4,389.
- A "run" is one seed across all five folds, so each scenario has 5 runs.
- Network CIs below: sequence-cluster bootstrap, then cluster-robust t, over all G sequences.

Learning check (final training loss < 0.95 × label-prior entropy):

| Scenario | Models learned (of 25) | By fold |
|---|---|---|
| 32 | 6 | [1, 1, 3, 1, 0] |
| 33 | 14 | [2, 4, 2, 4, 2] |
| 34 | 0 | none |

The 34 models end at the prior (for example 3.7279 vs 3.7282), so they output the label prior.

| Scenario | G | Network top-1 | Most common | 1-NN | Net − most common, top-1 (pp) | Net − most common, loss K=1 (dB) | Net − 1-NN, top-1 (pp) |
|---|---|---|---|---|---|---|---|
| 32 | 15 | 12.0% [7.5, 16.3] [7.0, 17.0] | 13.4% | 6.2% | −1.4 [−2.5, −0.5], worse | +0.05 [+0.01, +0.10], worse | +5.8 [+1.7, +9.8], better |
| 33 | 18 | 11.9% [7.7, 17.1] [6.5, 17.4] | 16.4% | 5.1% | −4.5 [−10.3, −0.3], worse | +0.21 [+0.03, +0.47], worse | +6.8 [+1.8, +12.4], better |
| 34 | 31 | 7.5% [6.0, 9.1] [5.8, 9.2] | 7.5% | 5.5% | +0.0 [−0.1, +0.1], no detectable difference | −0.00 [−0.01, +0.01], none | +2.0 [−0.3, +4.3], none |

- **Read:** the network does not beat the most-common beam on any of the three unseen scenarios.
  It is worse on 32 and 33 and identical to it on 34. It beats 1-NN on 32 and 33.
- Ground-truth top-1 beam-change rate between consecutive frames: 61.8% / 60.8% / 61.3%.
- Full table: tab_realdata (rows added for 32–34) and tab_realdata_cv_cmp. File:
  results/phase4/a_results.json.
- **Note on the JSON:** the `verdict_boot` field under `network_cluster` compares a raw metric
  against 0 and has no meaning. It is not used in any table.

**Experiment B — multi-car simulation.**

Data and setup:
- 6,000 new users: train 2,962, val 793, test 1,244 (1,001 dropped by the guard).
- The new renderer reproduces the frozen one for a single standard car (max abs diff 0.0).
- Test scenes:
  - K_o = 0…5: 188 / 222 / 192 / 214 / 218 / 210 users. All 3,170 drawn vehicles were placed.
  - Vehicle types: car 1,929, van 558, truck 255, parked car 343, parked van 85.
  - LiDAR occlusion of the user car: visible 1,206, partly occluded 32, fully occluded 6.
- **Multi-car nets:** all 15 runs passed the learning check (final loss 2.84–2.86 vs prior 8.69).

Rule choice on validation (mean loss at M = 8, 60 GHz, v = 0, pooled over σ ∈ {1, 3}):

| Model | R0 | R1 | R2 | Chosen |
|---|---|---|---|---|
| Single-vehicle net | 11.28 | 3.94 | 6.29 | R1 |
| Varied-trained net | 13.96 | 4.88 | 8.09 | R1 |
| Multi-car net | 9.39 | 5.21 | 5.42 | R1 |
| Geometric | 14.54 | 8.36 | 9.77 | R1 |

Association success (selected cluster holds ≥ 50% of the user car's returns):

| | σ = 1 m | σ = 3 m |
|---|---|---|
| All test users | 95.0% (no candidate 0.4%) | 87.7% (no candidate 0.0%) |
| Visible | 95.6% | 88.9% |
| Partly occluded | 90.6% | 59.4% |
| Fully occluded | 0% | 0% |

Radio blockage (another vehicle on the array-to-antenna line; not in the channel model):
- 1.0% of test users (13): 0% at K_o ≤ 2, 0.9% / 2.3% / 2.9% at K_o = 3 / 4 / 5.
- Per-vehicle blocking rate: van 2.0%, truck 0.4%, parked van 1.2%, car and parked car 0%.
- **The channel model lacks this blockage, so results for blocked users are optimistic for every
  method, including the sweep.**

τ (frame age + b4 processing + median association time 16.2 ms):

| | Mean | 95th percentile |
|---|---|---|
| Nets | 70.4 ms | 114.8 ms |
| Geometric | 69.3 ms | 113.7 ms |

**Primary (amendment 2).** Multi-car-trained net (R1), 60 GHz, M = 32, v = 10 m/s, compensated,
σ = 3 m, all 1,244 test users:
- Net: 4.25 [4.02, 4.48] dB (5 seeds).
- 263-beam sweep: 0.76 [0.65, 0.87] dB.
- **It does not match the sweep in traffic.**

Secondary results at the primary condition, unless stated:

- **Frozen minus multi-car (per-user bootstrap):**
  - Single-vehicle net: −0.99 [−1.30, −0.69] dB. The frozen single-vehicle net is better than the
    retrained one.
  - Varied-trained net: −0.18 [−0.49, +0.12] dB, no detectable difference.
  - Geometric: +1.50 [+0.90, +2.13] dB, geometric worse.
- **Traffic effect, K_o ≥ 1 minus K_o = 0:**
  - +2.6 to +2.7 dB for every method. Multi-car net +2.71 [+1.77, +3.60].
  - K_o = 0: multi-car 1.95, single-vehicle 1.05, sweep 0.61 dB.
  - K_o ≥ 1: 4.66 / 3.65 / 0.78 dB.
- **By occlusion (multi-car net / single-vehicle net / sweep):**
  - Visible: 4.00 / 3.06 / 0.78 dB.
  - Partly occluded: 10.60 / 8.76 / 0.08 dB.
  - Fully occluded: 20.39 / 14.15 / 0.02 dB.
- **Radio-blocked users:** multi-car 16.66 dB, sweep 0.01 dB (n = 13; the channel has no blockage).
- **All conditions** (tab_multicar), multi-car net, σ = 3 m, 60 GHz:
  - v = 0: 6.98 / 5.35 / 3.59 / 1.97 dB at M = 8 / 16 / 32 / 64.
  - v = 20: 8.72 / 6.64 / 4.49 / 2.46 dB.
  - 140 GHz is within about 0.2 dB of 60 GHz for the nets, while the sweep falls to 0.16–0.35 dB.
- **Smallest M within the sweep's CI:** only the frozen single-vehicle net, σ = 1 m, 60 GHz, at
  v = 0 and v = 10 (M = 64). Every other method, σ, frequency and speed: none.
- **Median position error, σ = 3 m:** multi-car 26 / 34 cm at v = 0 / 10; single-vehicle
  23 / 25 cm. For the nets the 90th percentiles are 3.5–5.8 m: wrong associations.
- **Context for the level:** the error with no other vehicles (K_o = 0) is about the D-032
  varied-car level (20 cm median, varied-trained net). D-032 also found the single-vehicle net
  better than the varied-trained net on varied cars, as here.

Files: results/phase4/b_results.json; assets tab_multicar and tab_multicar_scene.

**Deviations (Phase 4).**
1. **Loader (pre-declared).** ds_prepare / a3 gained path and CSV-name entries for 32–34
   (S31_FMT). There are no logic changes.
2. **Clustering reimplemented after the first B test run crashed (2026-11-20).**
   - The first run ran out of memory: one dense test scene (46,710 above-ground points) produced
     a 1.37e9-pair list.
   - The run had finished validation and the v = 0 test frames; no results were written.
   - `clusters()` now uses an exact grid/union-find method with the original numbering.
   - Equivalence: identical labels on all 1,267 frames the original could run (every validation
     frame, plus the first 120 test users at v = 0 and v = 20, both frames). Six frames were too
     large for the original. Check: results/phase4/b_cluster_check.json.
   - The original is kept as `clusters_pairs`.
   - The full evaluation was rerun from the start. Validation reproduced the crashed run's numbers
     exactly.
   - The clustering is faster (median 14 vs 31 ms), so the association-time term in τ (pre-registered
     as measured) is 16.2 ms. It would have been about twice as large with the original method.
3. **Sweep shape choice.** Per seed on validation as pre-registered, using the chosen rule's
   validation estimates pooled over σ ∈ {1, 3}. D-036 did not say which σ.
4. **Association success.** Counted on the t0 frame at v = 0 with the shared cluster labels.
   Because the gate and selection are the same under R1 and R2, it is one number per σ.
5. **Parked vehicles.** Placed with the same 200-attempt rule (lateral jitter ±0.3 m, heading
   noise 2°) as moving vehicles. Each draws a speed offset that is never used, which keeps the
   draw order fixed.
6. **Asset-only.**
   - tab_realdata and tab_realdata_cv_cmp are wrapped in \resizebox to text width. tab_realdata was
     already 19 pt too wide before Phase 4 and 44 pt with the new rows.
   - The tab_multicar caption's K_o range is registered for the trace check.

**Not done.**
- Geometric estimator and visible-frames subset on 32–34: excluded by D-036.
- The Zenodo upload is still left to the user.

**Assets.** Trace check PASS (2,614 values, 219 keys, no skipped assets). The new and changed
tables compile without errors or overfull boxes.
