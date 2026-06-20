# HDR→SDR tone-map shootout (full) — 76 imazen-26 samples × 3 peak methods × 20 curves

**Date:** 2026-06-20
**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads)
**Samples processed:** 76 (33 UltraHDR JPEG + 43 HEIC) of 76 candidate files at `/home/lilith/work/codec-corpus/imazen-26/`
**Total runtime:** 2376.2s (39.6min)
**HdrToSdr fix verified:** commit `ca614df0` — HdrToSdr now rescales source-norm input to target-norm internally; the prior shootout's pre-multiplication workaround is removed.

## Methodology

For each sample we:
1. Walk `/home/lilith/work/codec-corpus/imazen-26/` recursively, filter to `.jpg/.jpeg/.heic`, probe each candidate (`DecodeRequest::probe()`), and keep the ones with `GainMapPresence::Available` or `supplements.gain_map == true`.
2. Decode the SDR base via `DecodeRequest::decode_full_frame()` (the producer's ground truth).
3. Reconstruct HDR via `DecodeRequest::reconstruct_hdr(None).decode_full_frame()` — works for both JPEG-UltraHDR (jpeg-ultrahdr feature) and HEIC (heic-decode feature). Returns linear-RGB f32 anchored at `1.0 = SDR diffuse white = 203 nits`.
4. Measure the source peak via three methods on the linear-RGB f32 buffer (BT.2408 anchor, MaxRgb reduction): `measure_max` (spec-literal max), `measure_robust` (99.99 % percentile — the production default in libplacebo / DaVinci), and `measure_max_smoothed` (3×1 box-filter max — defect-tolerant without committing to a percentile).
5. Apply each candidate curve (20 cells: 12 Möbius knee × 4kt × 3kg + BT.2446 A/B/C + BT.2408 + BT.2390 + Narkowicz + HableFilmic + AcesAp1) per peak method. For Möbius (HdrToSdr) the input is now source-normalized: `scratch = hdr * (203 / source_peak_nits)`, then `converter.apply_strip(strip)` handles the rescale internally. For non-Möbius curves the prior normalization is unchanged.
6. Compare against the SDR base in sRGB byte space (PSNR), CIE Lab (mean ΔE2000, % pixels with ΔE > 5), and normalised linear (max |Δ| per channel). The SDR base is the producer's ground truth — the curve closest to it matches the producer's aesthetic, NOT scene-referred fidelity.

## Headline verdict

On 76 gain-mapped samples (33 UltraHDR JPEG + 43 HEIC, all decoded successfully, 0 failures), **`bt2390` wins under every peak-measurement method** by a clear margin. Median ΔE2000: 8.34 (measure_max), **6.09 (measure_robust)**, 8.33 (measure_max_smoothed). Next-best curves: HableFilmic (11.19), BT.2446C (10.65 under measure_robust, 11.95 otherwise), AcesAp1 (12.70). Every Möbius knee cell sits at 16.6+ ΔE2000.

**Peak-measurement choice does NOT shift the headline curve ranking** — BT.2390 wins under all three methods. It does shift the *magnitude*: BT.2390's median ΔE2000 drops from 8.34 to 6.09 (-27%) when switching from `measure_max` to the production-default `measure_robust` (99.99 %ile). The same magnitude shift also benefits BT.2446C (-11%). Curves that don't actually use the measured peak (Narkowicz, HableFilmic, AcesAp1) are unaffected.

**Recommended HdrToSdr defaults** based on the production-default `measure_robust` peak: **`Bt2390 { source_peak, target_peak }`** (median ΔE2000 6.089) — a significant rejection of the libplacebo / mpv / VLC Möbius convention for this *producer-SDR matching* use case. If the intent is libplacebo-compatible HDR display rendering instead, keep Möbius with `knee_tone = 0.20, knee_gamut = 0.90` (the best in-family cell), accepting median ΔE2000 16.65.

## Per-curve summary by peak method

### `measure_max` — sorted by median ΔE2000 ascending

| Rank | Curve | Median PSNR (dB) | Median ΔE2000 | Median max\|Δ\| | Median %>ΔE5 |
|------|-------|------------------|---------------|-----------------|---------------|
| 1 | `bt2390` | 19.88 | 8.342 | 0.4342 | 82.55 |
| 2 | `hable_filmic` | 17.31 | 11.190 | 0.3717 | 96.32 |
| 3 | `bt2446c` | 17.22 | 11.947 | 0.6175 | 95.96 |
| 4 | `aces_ap1` | 15.53 | 12.699 | 0.5527 | 83.18 |
| 5 | `mobius_kt0.20_kg0.90` | 14.02 | 16.622 | 0.4709 | 98.70 |
| 6 | `mobius_kt0.20_kg0.95` | 14.02 | 16.647 | 0.4761 | 98.76 |
| 7 | `mobius_kt0.20_kg0.99` | 14.02 | 16.647 | 0.4766 | 98.76 |
| 8 | `mobius_kt0.30_kg0.99` | 13.47 | 17.561 | 0.5145 | 98.80 |
| 9 | `mobius_kt0.30_kg0.95` | 13.48 | 17.562 | 0.5142 | 98.80 |
| 10 | `mobius_kt0.30_kg0.90` | 13.51 | 17.564 | 0.5082 | 98.78 |
| 11 | `mobius_kt0.40_kg0.99` | 13.03 | 18.187 | 0.5539 | 98.86 |
| 12 | `mobius_kt0.40_kg0.95` | 13.03 | 18.191 | 0.5539 | 98.85 |
| 13 | `mobius_kt0.40_kg0.90` | 13.04 | 18.192 | 0.5472 | 98.83 |
| 14 | `bt2408` | 13.12 | 18.362 | 0.7577 | 98.23 |
| 15 | `bt2446b` | 12.85 | 18.411 | 0.7815 | 97.57 |
| 16 | `mobius_kt0.50_kg0.90` | 12.68 | 18.659 | 0.5850 | 98.85 |
| 17 | `mobius_kt0.50_kg0.99` | 12.66 | 18.663 | 0.5954 | 98.88 |
| 18 | `mobius_kt0.50_kg0.95` | 12.67 | 18.665 | 0.5933 | 98.86 |
| 19 | `narkowicz` | 11.21 | 22.385 | 0.6280 | 97.05 |
| 20 | `bt2446a` | 10.99 | 22.452 | 0.4215 | 99.01 |

### `measure_robust` — sorted by median ΔE2000 ascending

| Rank | Curve | Median PSNR (dB) | Median ΔE2000 | Median max\|Δ\| | Median %>ΔE5 |
|------|-------|------------------|---------------|-----------------|---------------|
| 1 | `bt2390` | 22.94 | 6.089 | 0.3545 | 70.08 |
| 2 | `bt2446c` | 18.03 | 10.653 | 0.5539 | 94.65 |
| 3 | `hable_filmic` | 17.31 | 11.190 | 0.3717 | 96.32 |
| 4 | `aces_ap1` | 15.53 | 12.699 | 0.5527 | 83.18 |
| 5 | `mobius_kt0.20_kg0.90` | 13.94 | 16.651 | 0.4723 | 98.71 |
| 6 | `mobius_kt0.20_kg0.99` | 13.94 | 16.674 | 0.4781 | 98.76 |
| 7 | `mobius_kt0.20_kg0.95` | 13.94 | 16.675 | 0.4772 | 98.76 |
| 8 | `mobius_kt0.30_kg0.99` | 13.43 | 17.577 | 0.5168 | 98.81 |
| 9 | `mobius_kt0.30_kg0.90` | 13.46 | 17.577 | 0.5091 | 98.79 |
| 10 | `mobius_kt0.30_kg0.95` | 13.43 | 17.578 | 0.5159 | 98.81 |
| 11 | `mobius_kt0.40_kg0.99` | 13.02 | 18.193 | 0.5564 | 98.86 |
| 12 | `mobius_kt0.40_kg0.95` | 13.02 | 18.196 | 0.5549 | 98.85 |
| 13 | `mobius_kt0.40_kg0.90` | 13.03 | 18.197 | 0.5492 | 98.83 |
| 14 | `bt2408` | 12.99 | 18.436 | 0.7586 | 98.28 |
| 15 | `bt2446b` | 12.83 | 18.534 | 0.7815 | 97.60 |
| 16 | `mobius_kt0.50_kg0.90` | 12.68 | 18.661 | 0.5853 | 98.85 |
| 17 | `mobius_kt0.50_kg0.99` | 12.66 | 18.666 | 0.5969 | 98.88 |
| 18 | `mobius_kt0.50_kg0.95` | 12.66 | 18.668 | 0.5938 | 98.87 |
| 19 | `narkowicz` | 11.21 | 22.385 | 0.6280 | 97.05 |
| 20 | `bt2446a` | 10.90 | 22.974 | 0.4390 | 99.24 |

### `measure_max_smoothed` — sorted by median ΔE2000 ascending

| Rank | Curve | Median PSNR (dB) | Median ΔE2000 | Median max\|Δ\| | Median %>ΔE5 |
|------|-------|------------------|---------------|-----------------|---------------|
| 1 | `bt2390` | 20.14 | 8.331 | 0.4239 | 80.67 |
| 2 | `hable_filmic` | 17.31 | 11.190 | 0.3717 | 96.32 |
| 3 | `bt2446c` | 17.39 | 11.909 | 0.6050 | 95.79 |
| 4 | `aces_ap1` | 15.53 | 12.699 | 0.5527 | 83.18 |
| 5 | `mobius_kt0.20_kg0.90` | 14.02 | 16.626 | 0.4713 | 98.70 |
| 6 | `mobius_kt0.20_kg0.95` | 14.01 | 16.651 | 0.4763 | 98.76 |
| 7 | `mobius_kt0.20_kg0.99` | 14.01 | 16.652 | 0.4769 | 98.76 |
| 8 | `mobius_kt0.30_kg0.99` | 13.47 | 17.563 | 0.5147 | 98.80 |
| 9 | `mobius_kt0.30_kg0.95` | 13.48 | 17.565 | 0.5145 | 98.80 |
| 10 | `mobius_kt0.30_kg0.90` | 13.50 | 17.566 | 0.5085 | 98.78 |
| 11 | `mobius_kt0.40_kg0.99` | 13.03 | 18.190 | 0.5541 | 98.86 |
| 12 | `mobius_kt0.40_kg0.95` | 13.03 | 18.193 | 0.5541 | 98.85 |
| 13 | `mobius_kt0.40_kg0.90` | 13.04 | 18.194 | 0.5475 | 98.83 |
| 14 | `bt2408` | 13.12 | 18.379 | 0.7577 | 98.23 |
| 15 | `bt2446b` | 12.85 | 18.417 | 0.7815 | 97.58 |
| 16 | `mobius_kt0.50_kg0.90` | 12.68 | 18.660 | 0.5852 | 98.85 |
| 17 | `mobius_kt0.50_kg0.99` | 12.66 | 18.665 | 0.5956 | 98.88 |
| 18 | `mobius_kt0.50_kg0.95` | 12.67 | 18.667 | 0.5935 | 98.86 |
| 19 | `narkowicz` | 11.21 | 22.385 | 0.6280 | 97.05 |
| 20 | `bt2446a` | 10.99 | 22.541 | 0.4219 | 99.03 |

## Möbius knee sweep (3 heatmaps, one per peak method)

Cells show median ΔE2000 across all processed samples. Lower = closer to producer's SDR.

### `measure_max`

| knee_tone \ knee_gamut | 0.90 | 0.95 | 0.99 |
|---|---|---|---|
| **0.20** | **16.622** | 16.647 | 16.647 |
| **0.30** | 17.564 | 17.562 | 17.561 |
| **0.40** | 18.192 | 18.191 | 18.187 |
| **0.50** | 18.659 | 18.665 | 18.663 |

**Best Möbius cell (measure_max):** `mobius_kt0.20_kg0.90` (median ΔE2000 = 16.622)

### `measure_robust`

| knee_tone \ knee_gamut | 0.90 | 0.95 | 0.99 |
|---|---|---|---|
| **0.20** | **16.651** | 16.675 | 16.674 |
| **0.30** | 17.577 | 17.578 | 17.577 |
| **0.40** | 18.197 | 18.196 | 18.193 |
| **0.50** | 18.661 | 18.668 | 18.666 |

**Best Möbius cell (measure_robust):** `mobius_kt0.20_kg0.90` (median ΔE2000 = 16.651)

### `measure_max_smoothed`

| knee_tone \ knee_gamut | 0.90 | 0.95 | 0.99 |
|---|---|---|---|
| **0.20** | **16.626** | 16.651 | 16.652 |
| **0.30** | 17.566 | 17.565 | 17.563 |
| **0.40** | 18.194 | 18.193 | 18.190 |
| **0.50** | 18.660 | 18.667 | 18.665 |

**Best Möbius cell (measure_max_smoothed):** `mobius_kt0.20_kg0.90` (median ΔE2000 = 16.626)

## Robustness to peak measurement choice

For each curve, the median ΔE2000 under each peak method. `Δ_max_method` = max ΔE2000 difference across methods (the curve's sensitivity).

| Curve | measure_max | measure_robust | measure_max_smoothed | Δ_max_method |
|---|---|---|---|---|
| `narkowicz` | 22.385 | 22.385 | 22.385 | 0.000 |
| `hable_filmic` | 11.190 | 11.190 | 11.190 | 0.000 |
| `aces_ap1` | 12.699 | 12.699 | 12.699 | 0.000 |
| `mobius_kt0.50_kg0.90` | 18.659 | 18.661 | 18.660 | 0.003 |
| `mobius_kt0.50_kg0.95` | 18.665 | 18.668 | 18.667 | 0.003 |
| `mobius_kt0.50_kg0.99` | 18.663 | 18.666 | 18.665 | 0.004 |
| `mobius_kt0.40_kg0.90` | 18.192 | 18.197 | 18.194 | 0.005 |
| `mobius_kt0.40_kg0.95` | 18.191 | 18.196 | 18.193 | 0.005 |
| `mobius_kt0.40_kg0.99` | 18.187 | 18.193 | 18.190 | 0.006 |
| `mobius_kt0.30_kg0.90` | 17.564 | 17.577 | 17.566 | 0.013 |
| `mobius_kt0.30_kg0.95` | 17.562 | 17.578 | 17.565 | 0.015 |
| `mobius_kt0.30_kg0.99` | 17.561 | 17.577 | 17.563 | 0.016 |
| `mobius_kt0.20_kg0.99` | 16.647 | 16.674 | 16.652 | 0.028 |
| `mobius_kt0.20_kg0.95` | 16.647 | 16.675 | 16.651 | 0.029 |
| `mobius_kt0.20_kg0.90` | 16.622 | 16.651 | 16.626 | 0.030 |
| `bt2408` | 18.362 | 18.436 | 18.379 | 0.075 |
| `bt2446b` | 18.411 | 18.534 | 18.417 | 0.123 |
| `bt2446a` | 22.452 | 22.974 | 22.541 | 0.523 |
| `bt2446c` | 11.947 | 10.653 | 11.909 | 1.294 |
| `bt2390` | 8.342 | 6.089 | 8.331 | 2.253 |

## Per-source-device breakdown (using `measure_robust`)

UltraHDR JPEG vs HEIC: same curve ranking, or different? Top-5 per class.

### UltraHDR JPEG (33 samples)

| Rank | Curve | Median ΔE2000 |
|---|---|---|
| 1 | `bt2390` | 5.150 |
| 2 | `bt2446c` | 9.434 |
| 3 | `hable_filmic` | 12.551 |
| 4 | `aces_ap1` | 14.150 |
| 5 | `mobius_kt0.20_kg0.90` | 17.958 |

### HEIC (43 samples)

| Rank | Curve | Median ΔE2000 |
|---|---|---|
| 1 | `bt2390` | 7.844 |
| 2 | `hable_filmic` | 8.329 |
| 3 | `aces_ap1` | 10.921 |
| 4 | `bt2446c` | 11.862 |
| 5 | `mobius_kt0.20_kg0.90` | 13.868 |

## Top-3 / bottom-3 montages per (curve, peak_method) cell

All saved under [`/mnt/v/output/zentone/shootout-full/`](http://172.23.240.1:3300/zentone/shootout-full/) as 3-panel images: producer SDR | our SDR | abs Δ×10.

File naming: `<sample-stem>__<curve_label>__<peak_method>__<best|worst>.png`.

## Verdict

**Recommended HdrToSdr default**: assuming the production-default `measure_robust` peak, the curve closest to producer-graded SDR across 76 samples is `bt2390` (median ΔE2000 6.089). Inside the Möbius family the best knee setting is `mobius_kt0.20_kg0.90` (median ΔE2000 16.651).

**Verdict change vs the prior 4-sample shootout.** The prior 4-sample shootout (`benchmarks/hdr_tone_map_shootout_2026-06-20.md`) found HableFilmic winning at median ΔE2000 4.5, with BT.2390 second at 6.9 and Möbius rolling off at 8.7+. On 76 samples the picture flips: **BT.2390 wins by a clear margin**, HableFilmic / BT.2446C / AcesAp1 form a second tier (11-13 ΔE2000), and every Möbius knee cell sits at 16+. The reason: BT.2390's EETF is a smooth Hermite-spline-shaped knee very close to what real HDR-playback receivers do, and at 76 samples its tendency to match producer-graded SDR shows up clearly that wasn't visible in the seine/apple-only 4-sample set.

**Möbius is the worst-performing curve family in this corpus.** Every knee setting (knee_tone × knee_gamut) sits above 16.6 ΔE2000 — significantly worse than BT.2390, HableFilmic, BT.2446C, AcesAp1. The Möbius family is the libplacebo HDR-playback default, which is the right choice for *display-side rendering* (where the priority is preserving HDR detail with smooth rolloff). For HDR-source → producer-SDR matching it under-compresses highlights, leaving the SDR result too bright vs the producer's hand-graded reference. Inside the family `knee_tone = 0.20` wins on every method and `knee_gamut` is near-irrelevant (the gamut soft-clip rarely triggers).

**Peak-measurement choice matters most for `bt2390` and `bt2446c`.** Curves with low `Δ_max_method` (narkowicz/hable_filmic/aces_ap1 = 0) are insensitive — they don't actually use the measured peak, just a fixed normalization. Möbius/BT.2408/BT.2446B have low sensitivity (< 0.12). BT.2390 is highly sensitive (2.25): `measure_robust` (99.99 %ile) drops it from 8.34 to 6.09 ΔE2000 — that's a big win for the production default. BT.2446C swings 1.29 in the opposite direction.

**Device split: UltraHDR JPEG vs HEIC.** Both classes pick `bt2390` first, but the rest of the order shifts. UltraHDR JPEG (33 Samsung s23u/s25u/zfold7 shots) likes BT.2446C second; HEIC (43 iPhone ip13/ip16 + Samsung s23u shots) likes HableFilmic second. Both come from very different tone-mapping pipelines (libultrahdr's BT.2446 vs Apple's Photonic-style curve), so a single default that matches both is necessarily a compromise.

**Suggested action for HdrToSdr defaults:** If the goal is to closely match producer-graded SDR across web-image workloads, the default curve should be `Bt2390 { source_peak, target_peak }` rather than Möbius. If the goal is libplacebo-compatible HDR *playback* (the original docstring framing), keep Möbius with `knee_tone=0.20, knee_gamut=0.90`. A clarifying decision is needed on which intent the wrapper serves — they aren't the same thing.

## Reproduce

```bash
nice -n19 cargo build -p zentone --release \
  --example hdr_tone_map_shootout_full --features hdr-shootout
nice -n19 cargo run -p zentone --release \
  --example hdr_tone_map_shootout_full --features hdr-shootout
```

Sample inputs: `/home/lilith/work/codec-corpus/imazen-26/**/{*.jpg,*.jpeg,*.heic}`.
Per-sample × per-cell montages (top-3 + bottom-3 per cell): `/mnt/v/output/zentone/shootout-full/...`.
Full CSV: `/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20.csv`.
Source: `examples/hdr_tone_map_shootout_full.rs`.

---

## gainforge curve sweep (addendum, 2026-06-20)

**Tested**: 10 cells × 3 peak methods × 76 samples = 2280 cell-evaluations. Runtime: 918.5s (15.3min). gainforge version: 0.5.0 (`Itu2408` formerly `Rec2408`).

**Goal**: test whether gainforge's `MappingColorSpace::{Jzazbz, Yrg}` color-space mapping beats zentone's per-RGB-channel tone-map (the BT.2390 winner of the main shootout, median ΔE2000 6.089 under `measure_robust`).

**Pipeline**: HDR linear f32 (anchored 1.0 = 203 nits) → PQ-encoded u8 → `gainforge::create_tone_mapper_rgb(BT2020-PQ → sRGB, method, wcs).tonemap_lane(pq_u8, srgb_u8)` → u8 sRGB → linearize → diff against the producer SDR base. PQ encoding uses two frames per gainforge's internal LUT convention:
- **Aces / Filmic / FilmicSpline**: `reference_display = 10000/203 ≈ 49.26` in gainforge's LUT, so PQ-encode in standard PQ-absolute frame (255 u8 = 10000 nits, our SDR-white u8 ≈ 148).
- **Itu2408**: `reference_display = 1.0` in gainforge's LUT, so PQ-encode in content-relative frame (255 u8 = `content_max_brightness` nits). Re-encoded per peak-method since `content_max_brightness = source_peak_nits`.

The u8 round-trip is necessary because gainforge's Yrg/Jzazbz factories don't expose a clean f32-linear-RGB entry point; the u8 PQ path is canonical and the quantization (256 PQ-encoded steps) is well below the ΔE differences being measured (>5).

### Per-cell summary by peak method

#### `measure_max` — sorted by median ΔE2000 ascending (BT.2390 reference: 8.342)

| Rank | Cell | WCS | Median PSNR (dB) | Median ΔE2000 | Δ vs BT.2390 | Median max\|Δ\| | Median %>ΔE5 |
|------|------|-----|------------------|---------------|--------------|-----------------|---------------|
| 1 | `gainforge_itu2408_rgb` | rgb | 20.12 | 8.453 | +0.111 | 0.5403 | 84.80 |
| 2 | `gainforge_filmic_spline_rgb` | rgb | 19.89 | 8.774 | +0.432 | 0.5207 | 77.05 |
| 3 | `gainforge_aces_yrg` | yrg | 17.89 | 9.875 | +1.533 | 0.6173 | 97.14 |
| 4 | `gainforge_filmic_rgb` | rgb | 17.75 | 9.877 | +1.535 | 0.6607 | 81.37 |
| 5 | `gainforge_aces_rgb` | rgb | 18.18 | 10.178 | +1.836 | 0.6110 | 96.99 |
| 6 | `gainforge_filmic_yrg` | yrg | 17.09 | 10.211 | +1.869 | 0.7079 | 80.12 |
| 7 | `gainforge_filmic_jzazbz` | jzazbz | 15.33 | 13.577 | +5.235 | 0.6814 | 92.52 |
| 8 | `gainforge_filmic_spline_jzazbz` | jzazbz | 14.97 | 15.148 | +6.806 | 0.5100 | 91.66 |
| 9 | `gainforge_aces_jzazbz` | jzazbz | 10.47 | 25.919 | +17.577 | 0.8657 | 97.34 |
| 10 | `gainforge_itu2408_jzazbz` | jzazbz | 9.25 | 29.628 | +21.286 | 0.5031 | 99.13 |

#### `measure_robust` — sorted by median ΔE2000 ascending (BT.2390 reference: 6.089)

| Rank | Cell | WCS | Median PSNR (dB) | Median ΔE2000 | Δ vs BT.2390 | Median max\|Δ\| | Median %>ΔE5 |
|------|------|-----|------------------|---------------|--------------|-----------------|---------------|
| 1 | `gainforge_itu2408_rgb` | rgb | 19.99 | 8.477 | +2.388 | 0.5403 | 84.65 |
| 2 | `gainforge_filmic_spline_rgb` | rgb | 19.89 | 8.774 | +2.685 | 0.5207 | 77.05 |
| 3 | `gainforge_aces_yrg` | yrg | 17.89 | 9.875 | +3.786 | 0.6173 | 97.14 |
| 4 | `gainforge_filmic_rgb` | rgb | 17.75 | 9.877 | +3.788 | 0.6607 | 81.37 |
| 5 | `gainforge_aces_rgb` | rgb | 18.18 | 10.178 | +4.089 | 0.6110 | 96.99 |
| 6 | `gainforge_filmic_yrg` | yrg | 17.09 | 10.211 | +4.122 | 0.7079 | 80.12 |
| 7 | `gainforge_filmic_jzazbz` | jzazbz | 15.33 | 13.577 | +7.488 | 0.6814 | 92.52 |
| 8 | `gainforge_filmic_spline_jzazbz` | jzazbz | 14.97 | 15.148 | +9.059 | 0.5100 | 91.66 |
| 9 | `gainforge_aces_jzazbz` | jzazbz | 10.47 | 25.919 | +19.830 | 0.8657 | 97.34 |
| 10 | `gainforge_itu2408_jzazbz` | jzazbz | 9.41 | 29.203 | +23.114 | 0.5031 | 99.06 |

#### `measure_max_smoothed` — sorted by median ΔE2000 ascending (BT.2390 reference: 8.334)

| Rank | Cell | WCS | Median PSNR (dB) | Median ΔE2000 | Δ vs BT.2390 | Median max\|Δ\| | Median %>ΔE5 |
|------|------|-----|------------------|---------------|--------------|-----------------|---------------|
| 1 | `gainforge_itu2408_rgb` | rgb | 20.08 | 8.467 | +0.133 | 0.5403 | 84.68 |
| 2 | `gainforge_filmic_spline_rgb` | rgb | 19.89 | 8.774 | +0.440 | 0.5207 | 77.05 |
| 3 | `gainforge_aces_yrg` | yrg | 17.89 | 9.875 | +1.541 | 0.6173 | 97.14 |
| 4 | `gainforge_filmic_rgb` | rgb | 17.75 | 9.877 | +1.543 | 0.6607 | 81.37 |
| 5 | `gainforge_aces_rgb` | rgb | 18.18 | 10.178 | +1.844 | 0.6110 | 96.99 |
| 6 | `gainforge_filmic_yrg` | yrg | 17.09 | 10.211 | +1.877 | 0.7079 | 80.12 |
| 7 | `gainforge_filmic_jzazbz` | jzazbz | 15.33 | 13.577 | +5.243 | 0.6814 | 92.52 |
| 8 | `gainforge_filmic_spline_jzazbz` | jzazbz | 14.97 | 15.148 | +6.814 | 0.5100 | 91.66 |
| 9 | `gainforge_aces_jzazbz` | jzazbz | 10.47 | 25.919 | +17.585 | 0.8657 | 97.34 |
| 10 | `gainforge_itu2408_jzazbz` | jzazbz | 9.25 | 29.578 | +21.244 | 0.5031 | 99.13 |

### Color-space effect: RGB vs Jzazbz vs Yrg

For each curve family, median ΔE2000 under `measure_robust` per working color space (lower is closer to producer SDR).

| Curve | RGB | Jzazbz | Yrg | Best WCS | Δ(RGB→best) |
|---|---|---|---|---|---|
| `aces` | 10.178 | 25.919 | 9.875 | **Yrg** | -0.303 |
| `filmic` | 9.877 | 13.577 | 10.211 | **RGB** | +0.000 |
| `itu2408` | 8.477 | 29.203 | n/a | **RGB** | +0.000 |
| `filmic_spline` | 8.774 | 15.148 | n/a | **RGB** | +0.000 |

### Verdict

**gainforge winner**: `gainforge_itu2408_rgb` (WCS=rgb) at median ΔE2000 8.477 under `measure_robust` — does NOT beat BT.2390 (6.089); deficit 2.388.

**Color-space ranking (count of curves where each WCS is best)**: rgb (3), yrg (1).

**HdrToSdr default recommendation**: see the `Δ vs BT.2390` column under `measure_robust` above. If no gainforge cell shows a `BEATS` marker, keep the BT.2390 default. If one or more do, the Jzazbz/Yrg color-space path may be worth pulling into zentone (separate conversation — would require zentone to learn the moxcms RGB↔Jzazbz/Yrg conversions, not just adopt a new curve).

**Tonemap thumbnails** (1024-pixel-wide PNG previews) saved under [`/mnt/v/output/zentone/shootout-full-gainforge/`](http://172.23.240.1:3300/zentone/shootout-full-gainforge/). Naming: `<sample-stem>__REF.png` (the producer-SDR reference) and `<sample-stem>__<cell_label>__<peak_method>.png` (each tonemap candidate). All cells × samples saved inline during the per-sample sweep (no end-of-run re-decode pass; bounded peak memory).

### Reproduce

```bash
nice -n19 cargo run -p zentone --release \
  --example hdr_tone_map_shootout_gainforge --features hdr-shootout
```

Side-car CSV: `/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20_gainforge.csv`.
Source: `examples/hdr_tone_map_shootout_gainforge.rs`.
