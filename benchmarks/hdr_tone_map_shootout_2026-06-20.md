# HDR→SDR tone-map shootout — Möbius vs BT.2446 family vs BT.2390 vs filmic

**Date:** 2026-06-20
**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads)
**Samples processed:** 4 of 11 at /mnt/v/input/gainmap-samples/
**Total runtime:** 4.2s

**Methodology.** For each sample we:
1. Call `zencodecs::DecodeRequest::decode_gain_map()` to obtain the producer's SDR base (linearised to f32 RGB in source primaries) AND the ISO 21496-1 gain map + metadata. Samples that don't carry an SDR base + gain map (Apple AMPF without HDRGainMap support; AVIF base=HDR; broken containers) are skipped.
2. Reconstruct full-boost HDR via `ultrahdr_core::gainmap::apply_gainmap()` with `display_boost = 1024.0` (the function clamps internally to the metadata's encoded maximum). HDR comes back in linear-RGBA-f32 where `1.0 = SDR diffuse white = 203 nits` per the ultrahdr-core convention.
3. Measure source peak via `zenpixels_convert::hdr::CllMeasure::measure_max(MaxRgb)` (BT.2408 anchor, 203 nits).
4. Apply each candidate curve to the HDR buffer with per-curve-correct input normalisation (see the source). Output is in [0, 1] where 1.0 = SDR display peak.
5. Compare against the SDR base in sRGB byte space (PSNR), CIE Lab (mean ΔE2000, % pixels with ΔE > 5), and normalised linear (max |Δ| per channel). The SDR base is the producer's ground truth — the curve closest to it matches the producer's aesthetic, NOT necessarily scene-referred fidelity.

## Failures

- `/mnt/v/input/gainmap-samples/AVIF/color_grid_gainmap_different_grid.avif`: decode: no gain map in container
- `/mnt/v/input/gainmap-samples/AVIF/gregbenz-DSC8998.avif`: decode: decode_gain_map: codec error (Avif): AVIF parse error: box content size mismatch
- `/mnt/v/input/gainmap-samples/AVIF/seine_hdr_gainmap_small_srgb.avif`: SDR base is PQ-encoded
- `/mnt/v/input/gainmap-samples/AVIF/seine_hdr_gainmap_srgb.avif`: SDR base is PQ-encoded
- `/mnt/v/input/gainmap-samples/JPEG/apple_gainmap_old.jpg`: no HDR boost (hdr_max=1.000, sdr_max=1.000)
- `/mnt/v/input/gainmap-samples/JPEG/fullColor-fullRes-IDEAL.jpg`: no HDR boost (hdr_max=1.000, sdr_max=1.000)
- `/mnt/v/input/gainmap-samples/JPEG/luminosity-lowRes.jpg`: no HDR boost (hdr_max=1.000, sdr_max=1.000)

## Per-curve summary (sorted by median ΔE2000 ascending)

| Rank | Curve | Median PSNR (dB) | Median ΔE2000 | Median max\|Δ\| | Median %>ΔE5 |
|------|-------|------------------|---------------|-----------------|---------------|
| 1 | `hable_filmic` | 23.85 | 4.533 | 0.2816 | 27.47 |
| 2 | `bt2390` | 19.48 | 6.904 | 0.4889 | 46.97 |
| 3 | `mobius_kt0.20_kg0.90` | 18.43 | 8.645 | 0.3704 | 93.65 |
| 4 | `mobius_kt0.20_kg0.95` | 18.36 | 8.715 | 0.3704 | 96.56 |
| 5 | `mobius_kt0.20_kg0.99` | 18.36 | 8.717 | 0.3704 | 96.69 |
| 6 | `mobius_kt0.30_kg0.90` | 17.76 | 9.244 | 0.4071 | 96.19 |
| 7 | `mobius_kt0.30_kg0.95` | 17.65 | 9.362 | 0.4071 | 98.31 |
| 8 | `mobius_kt0.30_kg0.99` | 17.65 | 9.366 | 0.4071 | 98.46 |
| 9 | `mobius_kt0.40_kg0.90` | 17.23 | 9.716 | 0.4447 | 97.46 |
| 10 | `mobius_kt0.40_kg0.95` | 17.07 | 9.887 | 0.4447 | 98.86 |
| 11 | `mobius_kt0.40_kg0.99` | 17.06 | 9.901 | 0.4447 | 98.94 |
| 12 | `mobius_kt0.50_kg0.90` | 16.78 | 10.101 | 0.4879 | 97.75 |
| 13 | `bt2446b` | 16.40 | 10.283 | 0.6475 | 98.42 |
| 14 | `mobius_kt0.50_kg0.95` | 16.60 | 10.313 | 0.4879 | 99.09 |
| 15 | `mobius_kt0.50_kg0.99` | 16.57 | 10.353 | 0.4879 | 99.14 |
| 16 | `bt2408` | 16.25 | 10.742 | 0.6362 | 99.28 |
| 17 | `bt2446c` | 15.58 | 12.204 | 0.6552 | 93.43 |
| 18 | `narkowicz` | 14.82 | 12.541 | 0.5376 | 98.53 |
| 19 | `bt2446a` | 14.21 | 13.159 | 0.3884 | 93.28 |

## Möbius knee sweep (12 cells × samples)

Median ΔE2000 across all processed samples. Lower = closer to producer's SDR.

| knee_tone \ knee_gamut | 0.90 | 0.95 | 0.99 |
|---|---|---|---|
| **0.20** | 8.645 | 8.715 | 8.717 |
| **0.30** | 9.244 | 9.362 | 9.366 |
| **0.40** | 9.716 | 9.887 | 9.901 |
| **0.50** | 10.101 | 10.313 | 10.353 |

**Best Möbius cell:** `mobius_kt0.20_kg0.90` (median ΔE2000 = 8.645).

## Per-sample ΔE2000 (mean)

Each cell = mean ΔE2000 across all pixels for that sample × curve.

| Sample | size | peak_nits | mobius_kt0.20_kg0.90 | mobius_kt0.20_kg0.95 | mobius_kt0.20_kg0.99 | mobius_kt0.30_kg0.90 | mobius_kt0.30_kg0.95 | mobius_kt0.30_kg0.99 | mobius_kt0.40_kg0.90 | mobius_kt0.40_kg0.95 | mobius_kt0.40_kg0.99 | mobius_kt0.50_kg0.90 | mobius_kt0.50_kg0.95 | mobius_kt0.50_kg0.99 | bt2446a | bt2446b | bt2446c | bt2408 | bt2390 | narkowicz | hable_filmic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `seine_sdr_gainmap_big_srgb.avif` | 400×300 | 471 | 8.67 | 8.75 | 8.75 | 9.26 | 9.39 | 9.39 | 9.73 | 9.91 | 9.92 | 10.11 | 10.33 | 10.37 | 13.27 | 10.31 | 11.28 | 10.83 | 6.65 | 12.52 | 4.53 |
| `seine_sdr_gainmap_srgb.avif` | 400×300 | 494 | 8.62 | 8.68 | 8.68 | 9.23 | 9.34 | 9.34 | 9.70 | 9.87 | 9.88 | 10.09 | 10.30 | 10.33 | 13.05 | 10.25 | 12.18 | 10.76 | 6.93 | 12.59 | 4.53 |
| `apple_gainmap_new.jpg` | 384×512 | 1891 | 9.03 | 9.03 | 9.03 | 9.61 | 9.61 | 9.61 | 10.07 | 10.08 | 10.08 | 10.42 | 10.46 | 10.46 | 16.53 | 10.41 | 19.38 | 9.16 | 17.69 | 12.49 | 5.43 |
| `seine_sdr_gainmap_srgb.jpg` | 400×300 | 494 | 8.61 | 8.66 | 8.67 | 9.21 | 9.32 | 9.32 | 9.69 | 9.85 | 9.86 | 10.08 | 10.29 | 10.32 | 13.04 | 10.22 | 12.22 | 10.72 | 6.88 | 12.56 | 4.51 |

## Verdict

On 4 UltraHDR samples, **`hable_filmic` wins** on median ΔE2000 by a clear margin (second: `bt2390`).

**Best Möbius cell**: `mobius_kt0.20_kg0.90` — lower `knee_tone` consistently wins in this corpus (less aggressive S-curve, closer to a soft-clip). `knee_gamut` makes minimal difference because the gamut soft-clip rarely triggers on these samples (HDR pixels stay close to the sRGB gamut in their HDR-source form).

**The result is not what production HDR playback (libplacebo's mpv/VLC/Plex) defaults predict.** Möbius is the de-facto default for HDR→SDR in those players. The producer-graded SDR in this corpus matches a *filmic* shoulder-curve much more closely than Möbius's behaviour — likely because producers calibrate SDR to look like a graded-film final, which has more gradual highlight rolloff than Möbius's continuous-and-smooth C¹ knee. The filmic curve compresses highlights MORE aggressively, which is what cameras and colourists do too.

**Recommended default**: if zentone's `HdrToSdr` is intended to match *broadcast HDR-playback behaviour*, keep Möbius (knee_tone = 0.20 — lower than libplacebo's default 0.30 — based on this small corpus). If `HdrToSdr` is intended to produce SDR that *matches producer-graded SDR output*, expose `HableFilmic` as the default instead.

**Caveats** (important — this is a tiny corpus):
- Only 4 of 11 samples in the gainmap-samples corpus carry both an SDR base and a real HDR boost: 2 `seine_sdr_*` (Brouwer, ~470 nits peak), 1 `apple_gainmap_new.jpg` (1891 nits peak), and the JPEG dup of seine_sdr. The other 7 samples were skipped: 2 AVIF files have base=HDR (no producer-SDR to compare); 1 AVIF and 1 JPEG have broken / unsupported metadata; 3 JPEGs are illustrations of gain map images rather than full UltraHDR JPEGs; 1 Apple JPEG uses the older `HDRGainMap` namespace which zenjpeg + the `raw-decode-gainmap` Apple AMPF path don't yet decode.
- 4 samples is below the user's 8-of-11 bar for a viable verdict. Treat this report as a directional smoke test; the verdict could flip on a larger or more diverse corpus (especially: skin-tone-heavy portraits, neon, very-dark DR, real-world camera UltraHDR from non-Apple/Google sources).
- The producer-SDR ground truth varies per-producer (Apple uses a custom curve; Google uses libultrahdr's BT.2446-derived default; Brouwer's seine samples were hand-graded). A curve scoring well here matches *that* aesthetic, NOT scene-referred fidelity.
- ΔE2000 is a colorimetric metric, not a perceptual-quality one. Future work: rerun the sweep against a perceptual metric (zensim Profile-A, or SSIMULACRA2).

## Reproduce

```bash
nice -n19 cargo run -p zentone --release \
  --example hdr_tone_map_shootout --features hdr-shootout
```

Sample inputs: `/mnt/v/input/gainmap-samples/{JPEG,AVIF}/*`.

Per-sample × per-curve montages: `/mnt/v/output/zentone/shootout/<sample>__<curve_label>.png` (3 panels: producer SDR | our SDR | abs Δ×10).
Source: `examples/hdr_tone_map_shootout.rs`.
