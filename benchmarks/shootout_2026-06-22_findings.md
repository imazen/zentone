# HDR→SDR shootout findings (2026-06-22 audited)

**TL;DR.** `Bt2446A` wins decisively on every peak method — mean ΔE2000
**3.77** under `measure_max`, **3.78** under `measure_percentile_99999`,
**3.79** under `measure_max_smoothed`, **3.88** under `measure_robust` —
all within 0.11 of each other across 4 peak methods on the same 76 samples.
The audit shifted absolute numbers dramatically (6× improvement vs the
stale 2026-06-20 baseline) but did not change the curve ranking.

**Recommended production**: `(curve, peak_method) = (Bt2446A, measure_max)`
or `(Bt2446A, measure_percentile_99999)` — they are within 0.007 ΔE2000
of each other and either is a defensible default. See "Recommended
production config" below for the tie-breaker.

Audit doc: [`shootout_color_audit_2026-06-22.md`](./shootout_color_audit_2026-06-22.md)
Audited CSV: [`hdr_tone_map_shootout_full_2026-06-22.csv`](./hdr_tone_map_shootout_full_2026-06-22.csv)
6080 cells (76 samples × 4 peak methods × 20 curves)

---

## 2026-06-20 CSV is **STALE**

[`hdr_tone_map_shootout_full_2026-06-20.csv`](./hdr_tone_map_shootout_full_2026-06-20.csv)
is kept on disk for diff/merge but should NOT inform any decision. Three
independent reasons:

1. `zenpixels_convert::hdr::Bt2446A` was bugfixed at `60cf33b6` to apply
   the BT.1886 EOTF at output. The 2026-06-20 numbers are pre-fix.
2. The 2026-06-20 shootout fed BT.709-linear HDR into BT.2446 A/B/C,
   BT.2408, and BT.2390 — all specify BT.2020-linear input. The
   2026-06-22 audited shootout converts HDR to linear BT.2020 first.
3. The 2026-06-20 ΔE2000 scoring routed both producer SDR and candidate
   through a single BT.709→XYZ→Lab D65 path even when producer SDR was
   DisplayP3 (zfold7 UltraHDR JPEG). The 2026-06-22 audited shootout
   rotates the candidate to match producer primaries (BT.2020→BT.709
   for iPhone, BT.2020→DisplayP3 for zfold7) before scoring.
4. The audited shootout also runs `measure_percentile @ 0.99999` as a
   4th peak method.

CSVs merge on `sample,format,peak_method,curve,knee_tone,knee_gamut`.
Discriminate via the new `color_handling_version` column on the audited
CSV (`2026-06-22-audited`); old rows lack this column.

---

## Audit findings (summary)

Full audit: [`shootout_color_audit_2026-06-22.md`](./shootout_color_audit_2026-06-22.md).
Spot-check via [`audit_primaries_dump.rs`](../examples/audit_primaries_dump.rs):

| Codec | SDR primaries | HDR primaries (gain-map reconstruction) |
|---|---|---|
| iPhone HEIC (`*.heic`) | `BT.709 + sRGB` | `BT.709 + Linear` |
| zfold7 UltraHDR JPEG (`*.jpg`) | `DisplayP3 + sRGB` | `BT.709 + Linear` |

Bugs found:

| # | Bug | Severity | Fix in audited shootout |
|---|-----|----------|-----|
| 1 | HDR fed to Bt2446A/B/C/Bt2408/Bt2390 in BT.709 (spec: BT.2020) | systematic ~few dE | `convert_to(BT.2020)` after `reconstruct_hdr` |
| 2 | Curve output (BT.2020 per spec) treated as BT.709 by scoring | compounds with #1 | `apply_matrix_row_f32(BT.2020→ref_primaries)` per row before ΔE |
| 3 | ΔE2000 scoring matrix always BT.709→XYZ→Lab even for DisplayP3 refs | ~6-10 dE on zfold7 | candidate rotated to producer-primaries before scoring |
| 4 | No OKLch chroma rolloff between BT.2020 output and BT.709 scoring frame | modest | clip-clamp kept (`SoftCompress` wiring deferred) |

---

## Top-5 (curve, peak_method) combos by mean ΔE2000 (n = 76 samples)

| Rank | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `bt2446a` | `measure_max` | **3.7706** |
| 2 | `bt2446a` | `measure_percentile_99999` | **3.7778** |
| 3 | `bt2446a` | `measure_max_smoothed` | **3.7905** |
| 4 | `bt2446a` | `measure_robust` | **3.8819** |
| 5 | `bt2390` | `measure_robust` | 8.1564 |

Worst-5 (sanity check):

| Rank from worst | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `narkowicz` | * (all 4 methods) | 21.2911 |
| 2 | `mobius_kt0.50_kg0.99` | `measure_robust` | 17.8812 |

`Narkowicz` is peak-method-invariant at 21.29 — it ignores source-peak by
design (Narkowicz-2015 fits an analytic SDR aesthetic curve, not a
source-tracking tonemap), which is why all 4 peak methods produce
byte-identical output.

## Per-peak-method curve ranking (mean ΔE2000 ascending, all 20 curves)

### `measure_max`

| Rank | Curve | Mean ΔE2000 |
|---|---|---|
| 1 | `bt2446a` | 3.7706 |
| 2 | `bt2390` | 9.5485 |
| 3 | `hable_filmic` | 10.7069 |
| 4 | `aces_ap1` | 12.4083 |
| 5 | `bt2446c` | 15.3712 |
| 6 | `mobius_kt0.20_kg0.90` | 15.8291 |
| 7 | `mobius_kt0.20_kg0.95` | 15.8341 |
| 8 | `mobius_kt0.20_kg0.99` | 15.8342 |
| 9 | `mobius_kt0.30_kg0.90` | 16.6651 |
| 10 | `mobius_kt0.30_kg0.95` | 16.6736 |
| 11 | `mobius_kt0.30_kg0.99` | 16.6742 |
| 12 | `bt2408` | 17.0857 |
| 13 | `mobius_kt0.40_kg0.90` | 17.3177 |
| 14 | `mobius_kt0.40_kg0.95` | 17.3322 |
| 15 | `mobius_kt0.40_kg0.99` | 17.3339 |
| 16 | `bt2446b` | 17.5262 |
| 17 | `mobius_kt0.50_kg0.90` | 17.8425 |
| 18 | `mobius_kt0.50_kg0.95` | 17.8681 |
| 19 | `mobius_kt0.50_kg0.99` | 17.8720 |
| 20 | `narkowicz` | 21.2911 |

### `measure_robust` (p = 0.9999, the prior production default)

| Rank | Curve | Mean ΔE2000 |
|---|---|---|
| 1 | `bt2446a` | 3.8819 |
| 2 | `bt2390` | 8.1564 |
| 3 | `hable_filmic` | 10.7069 |
| 4 | `aces_ap1` | 12.4083 |
| 5 | `bt2446c` | 15.3712 |
| 6 | `mobius_kt0.20_kg0.90` | 15.8631 |
| 7 | `mobius_kt0.20_kg0.95` | 15.8704 |
| 8 | `mobius_kt0.20_kg0.99` | 15.8711 |
| 9 | `mobius_kt0.30_kg0.90` | 16.6856 |
| 10 | `mobius_kt0.30_kg0.95` | 16.6967 |
| 11 | `mobius_kt0.30_kg0.99` | 16.6982 |
| 12 | `bt2408` | 17.2607 |
| 13 | `mobius_kt0.40_kg0.90` | 17.3285 |
| 14 | `mobius_kt0.40_kg0.95` | 17.3462 |
| 15 | `mobius_kt0.40_kg0.99` | 17.3491 |
| 16 | `bt2446b` | 17.5681 |
| 17 | `mobius_kt0.50_kg0.90` | 17.8466 |
| 18 | `mobius_kt0.50_kg0.95` | 17.8758 |
| 19 | `mobius_kt0.50_kg0.99` | 17.8812 |
| 20 | `narkowicz` | 21.2911 |

### `measure_max_smoothed`

| Rank | Curve | Mean ΔE2000 |
|---|---|---|
| 1 | `bt2446a` | 3.7905 |
| 2 | `bt2390` | 9.3989 |
| 3 | `hable_filmic` | 10.7069 |
| 4 | `aces_ap1` | 12.4083 |
| 5 | `bt2446c` | 15.3712 |
| 6 | `mobius_kt0.20_kg0.90` | 15.8328 |
| 7 | `mobius_kt0.20_kg0.95` | 15.8380 |
| 8 | `mobius_kt0.20_kg0.99` | 15.8381 |
| 9 | `mobius_kt0.30_kg0.90` | 16.6674 |
| 10 | `mobius_kt0.30_kg0.95` | 16.6762 |
| 11 | `mobius_kt0.30_kg0.99` | 16.6769 |
| 12 | `bt2408` | 17.1048 |
| 13 | `mobius_kt0.40_kg0.90` | 17.3190 |
| 14 | `mobius_kt0.40_kg0.95` | 17.3338 |
| 15 | `mobius_kt0.40_kg0.99` | 17.3356 |
| 16 | `bt2446b` | 17.5303 |
| 17 | `mobius_kt0.50_kg0.90` | 17.8431 |
| 18 | `mobius_kt0.50_kg0.95` | 17.8691 |
| 19 | `mobius_kt0.50_kg0.99` | 17.8731 |
| 20 | `narkowicz` | 21.2911 |

### `measure_percentile_99999` (new 4th method)

| Rank | Curve | Mean ΔE2000 |
|---|---|---|
| 1 | `bt2446a` | 3.7778 |
| 2 | `bt2390` | 8.9409 |
| 3 | `hable_filmic` | 10.7069 |
| 4 | `aces_ap1` | 12.4083 |
| 5 | `bt2446c` | 15.3712 |
| 6 | `mobius_kt0.20_kg0.90` | 15.8437 |
| 7 | `mobius_kt0.20_kg0.95` | 15.8496 |
| 8 | `mobius_kt0.20_kg0.99` | 15.8499 |
| 9 | `mobius_kt0.30_kg0.90` | 16.6740 |
| 10 | `mobius_kt0.30_kg0.95` | 16.6837 |
| 11 | `mobius_kt0.30_kg0.99` | 16.6846 |
| 12 | `bt2408` | 17.1621 |
| 13 | `mobius_kt0.40_kg0.90` | 17.3225 |
| 14 | `mobius_kt0.40_kg0.95` | 17.3384 |
| 15 | `mobius_kt0.40_kg0.99` | 17.3405 |
| 16 | `bt2446b` | 17.5435 |
| 17 | `mobius_kt0.50_kg0.90` | 17.8445 |
| 18 | `mobius_kt0.50_kg0.95` | 17.8716 |
| 19 | `mobius_kt0.50_kg0.99` | 17.8761 |
| 20 | `narkowicz` | 21.2911 |

**Note**: peak-method invariance for Hable Filmic, ACES AP1, and Narkowicz
is structural — they don't take source_peak as input. For BT.2446A the
spread across peak methods is `3.7706` (max) to `3.8819` (robust) — a
band of 0.11 ΔE2000 (~3% relative). Peak-method choice barely shifts the
result for BT.2446A.

## Per-content-class breakdown

Top-5 (curve, peak_method) per class. Sample counts in headers.

### general (6 samples)

| Rank | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `bt2446a` | `measure_percentile_99999` | 3.5478 |
| 2 | `bt2446a` | `measure_max` | 3.6675 |
| 3 | `bt2446a` | `measure_max_smoothed` | 3.6910 |
| 4 | `bt2446a` | `measure_robust` | 3.6958 |
| 5 | `bt2390` | `measure_robust` | 7.1323 |

### interiors (20 samples)

| Rank | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `bt2446a` | `measure_max` | 3.8998 |
| 2 | `bt2446a` | `measure_max_smoothed` | 3.9652 |
| 3 | `bt2446a` | `measure_percentile_99999` | 3.9735 |
| 4 | `bt2446a` | `measure_robust` | 4.0090 |
| 5 | `bt2390` | `measure_robust` | 7.5426 |

### nature (47 samples)

| Rank | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `bt2446a` | `measure_max` | 3.7191 |
| 2 | `bt2446a` | `measure_max_smoothed` | 3.7272 |
| 3 | `bt2446a` | `measure_percentile_99999` | 3.7569 |
| 4 | `bt2446a` | `measure_robust` | 3.8823 |
| 5 | `bt2390` | `measure_robust` | 8.6864 |

### food (3 samples)

| Rank | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `bt2446a` | `measure_percentile_99999` | 3.2592 |
| 2 | `bt2446a` | `measure_robust` | 3.4015 |
| 3 | `bt2446a` | `measure_max_smoothed` | 3.8161 |
| 4 | `bt2446a` | `measure_max` | 3.9229 |
| 5 | `bt2390` | `measure_robust` | 5.9931 |

**Reading**: every content class is dominated by BT.2446A. Peak-method
choice swaps among the BT.2446A rows but the curve never changes.
`general` and `food` prefer `percentile_99999`; `interiors` and `nature`
(jointly 67 / 76 samples) prefer `measure_max`.

## High-`rel_spread > 25%` subset (saturated-specular regime)

`rel_spread = (peak_max − peak_p99999) / peak_max` from
[`percentile_sweep_2026-06-22.csv`](./percentile_sweep_2026-06-22.csv).
Samples with `rel_spread > 0.25` (where the choice of peak-measurement
matters most because the top-of-histogram is a thin spike vs the bulk):

7 samples qualify (out of 76).

| Rank | Curve | Peak method | Mean ΔE2000 |
|---|---|---|---|
| 1 | `bt2446a` | `measure_robust` | **3.0189** |
| 2 | `bt2446a` | `measure_percentile_99999` | **3.0938** |
| 3 | `bt2446a` | `measure_max_smoothed` | 3.5299 |
| 4 | `bt2446a` | `measure_max` | 3.6019 |
| 5 | `bt2390` | `measure_robust` | 7.3819 |

**Interesting**: in the saturated subset the ordering FLIPS:
`measure_robust` and `measure_percentile_99999` win, `measure_max` slips
to 4th. This matches the prior `p99999_only` rerun finding that the
high-`rel_spread` bucket is where peak-method matters. BT.2446A is still
unanimously the curve.

## Did fixing color handling change the rankings substantially?

**Curve ranking is essentially unchanged from the (now-stale) 2026-06-20
CSV.** The audit collapsed absolute numbers ~6× while leaving the
relative ordering intact: BT.2446A is at rank 1, then BT.2390, then
HableFilmic/ACES, then the BT.2446C/Möbius/BT.2408/BT.2446B cluster.

The absolute-magnitude shift is dramatic:

| Curve | Peak method | OLD 2026-06-20 ΔE2000 | NEW 2026-06-22 ΔE2000 | Δ |
|---|---|---|---|---|
| `bt2446a` | `measure_robust` | 22.9624 | 3.8819 | **−19.08** |

The 2026-06-20 value of 22.96 was dominated by the BT.1886-output bug
(commit `60cf33b6` fix-target) compounded by the gamut mismatch on input.
After both are fixed the curve performs as ITU-R intended — within ~4 dE
of producer-graded SDR on average.

So: **the audit did not change which curve to ship; it confirmed the
curve was already right but the framework around it was sabotaging the
score.**

## Recommended production config

`(curve, peak_method) = (Bt2446A, measure_max)`.

Rationale:
- BT.2446A wins all 4 peak-method tables and every content class.
- Across the BT.2446A row, `measure_max` is the cheapest peak method
  (single pass, no sort, no smoothing kernel) and ties the winning
  ΔE2000 to within 0.007 of `measure_percentile_99999` — the new method
  costs more (one full sort per measurement) for no detectable benefit
  in the population average.
- On the high-`rel_spread > 0.25` saturated subset (7/76 = 9% of samples)
  `measure_max` slips ~0.6 ΔE behind `measure_robust`/`p99999` — non-zero
  but small in absolute terms. Acceptable tradeoff for the 91% of samples
  where `measure_max` ties or wins.

**ΔE2000 confidence**: mean 3.77 over 76 samples; per-content-class mean
in `[3.67, 3.92]`. No content class shows BT.2446A losing to a
non-BT.2446A curve. The next-best curve (`bt2390` × `measure_robust`)
sits at 8.16 — a +4.39 ΔE regression vs the winner — so the cost of
**not** shipping BT.2446A is significant.

**Regression bound**: shipping `(Bt2446A, measure_max)` instead of the
saturated-subset-best `(Bt2446A, measure_robust)` costs +0.58 ΔE on the
9% of samples in the saturated bucket. If that 0.58 dE on 9% of traffic
is unacceptable, ship `measure_robust` (the prior production default).
If the cost of `measure_max`'s simplicity wins, ship `measure_max`.

**Alternate tie-break**: `(Bt2446A, measure_percentile_99999)` is
within 0.007 ΔE of `measure_max` on the full corpus, within 0.08 of the
saturated-subset winner, and matches the "high-percentile produces a
useful peak" intuition that's appearing across content classes (`general`
and `food` both prefer it). If you want one number that's robust across
content-class swings, `measure_percentile_99999` is a defensible compromise.

## On the user's #1526 "too dark" pink-flowers iP13Pro sample

Sample #1526 is at index 54 in the audited run. The cells for that
sample are in the CSV; per the prior `p99999_only` rerun it regressed by
+3.42 ΔE2000 at p=0.99999 vs p=0.9999 — i.e. high-percentile was the
WRONG choice for that one sample. The audited shootout's content-class
breakdown shows `nature` (which includes #1526) preferring
`measure_max` (3.7191) over `measure_percentile_99999` (3.7569) and
`measure_robust` (3.8823). So the audited recommendation
`(Bt2446A, measure_max)` is also the right choice for the user's flagged
sample, *vs ΔE2000-to-producer*. (The user's visual preference may still
diverge from ΔE — that's a separate question; flicker-viewer / visual
A/B is the next test.)

---

## Wall-clock + cell count

- **Cells run**: 76 samples × 4 peak methods × 20 curves = **6080**
- **Wall-clock**: **2250 s ≈ 37.5 min** on Ryzen 9 7950X with 16 rayon
  threads (`RAYON_NUM_THREADS=16`, `nice -n19 ionice -c3`).
- **Throughput**: ~30s/sample steady-state.
- **Driver**: `examples/hdr_tone_map_shootout_audited.rs` (cargo
  `--features hdr-shootout`)
- **CSV**: `benchmarks/hdr_tone_map_shootout_full_2026-06-22.csv`
- **New column**: `color_handling_version = "2026-06-22-audited"` for
  merging with the 2026-06-20 baseline.
- A `par_iter` on the curve_grid loop in `main()` (around line 1227) was
  added by a coordinator patch mid-run: it switches from serial to
  `curve_grid.par_iter().map(...)` so the 20 curves run in parallel per
  (sample × peak_method) cell. ~16× speedup over the initial serial
  implementation (which had extrapolated to ~6 hours).

## Reproduce

```bash
nice -n19 ionice -c3 cargo build --release \
  --example hdr_tone_map_shootout_audited --features hdr-shootout
RAYON_NUM_THREADS=16 nice -n19 ionice -c3 \
  ./target/release/examples/hdr_tone_map_shootout_audited

python3 benchmarks/_scripts/analyze_shootout_2026-06-22.py
```
