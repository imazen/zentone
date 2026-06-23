# SoftCompress `knee` sweep — empirical default calibration (2026-06-23)

Sweep design:

- Corpus: 76 gain-mapped images from `/home/lilith/work/codec-corpus/imazen-26`.

- Pipeline: HDR (BT.2020 linear) → `Bt2446A(measure_max)` → BT.2020→BT.709 matrix → `SoftCompress(knee)`.

- Pre-clamp metrics. The final `clamp(0, 1)` is NOT applied before measurement.

- 13 knee values: 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00.

- Total cells: 13 × 76 = 988.

- Production-safety target: corpus-p90 `pixels_clipped_pct` < 0.1%.


Metrics:

- `pixels_clipped_pct` — fraction of pixels with any pre-clamp linear-BT.709 channel outside `[0, 1]`.

- `chroma_compression_pct` — mean OKLch C reduction vs. `knee=1.0` baseline (over chromatic pixels).

- `mean_de2000_vs_knee_1` — Lab D65 ΔE2000 against the same pipeline at `knee=1.0`.


## Aggregate corpus table

| knee | corpus_p50_clip% | corpus_p90_clip% | corpus_p99_clip% | corpus_max_clip% | mean_chroma_comp% | p90_chroma_comp% | mean_de2000 |
|------|-------------------|-------------------|-------------------|-------------------|--------------------|--------------------|--------------|
| 0.50 | 0.000000 | 0.0001 | 0.0230 | 0.0549 | 2.9424 | 5.8001 | 0.4210 |
| 0.60 | 0.000000 | 0.0002 | 0.0239 | 0.0579 | 1.6926 | 3.6271 | 0.2450 |
| 0.70 | 0.000000 | 0.0002 | 0.0252 | 0.0625 | 0.9020 | 2.0225 | 0.1298 |
| 0.75 | 0.000008 | 0.0003 | 0.0267 | 0.0685 | 0.6368 | 1.4797 | 0.0905 |
| 0.80 | 0.000017 | 0.0071 | 0.1052 | 0.1601 | 0.4404 | 1.1682 | 0.0613 |
| 0.85 | 0.000029 | 0.0089 | 0.1129 | 0.1602 | 0.3023 | 0.8122 | 0.0407 |
| 0.88 | 0.000038 | 0.0095 | 0.1198 | 0.1603 | 0.2451 | 0.6351 | 0.0323 |
| 0.90 | 0.000042 | 0.0096 | 0.1223 | 0.1604 | 0.2164 | 0.5000 | 0.0281 |
| 0.92 | 0.000111 | 0.0240 | 0.1466 | 0.1670 | 0.1927 | 0.3970 | 0.0247 |
| 0.94 | 0.000316 | 0.0308 | 0.1558 | 0.1781 | 0.1736 | 0.3321 | 0.0220 |
| 0.96 | 0.000530 | 0.0480 | 0.1734 | 0.1964 | 0.1604 | 0.3052 | 0.0202 |
| 0.98 | 0.001612 | 0.1576 | 0.4662 | 0.6061 | 0.1525 | 0.2901 | 0.0192 |
| 1.00 | 0.231156 | 2.4632 | 10.0968 | 10.2745 | 0.0000 | 0.0000 | 0.0000 |


## Pareto frontier (p90_clip < target, ranked by chroma compression)


| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |
|------|------------|---------------------|--------------|
| 0.96 | 0.0480 | 0.1604 | 0.0202 |
| 0.94 | 0.0308 | 0.1736 | 0.0220 |
| 0.92 | 0.0240 | 0.1927 | 0.0247 |
| 0.90 | 0.0096 | 0.2164 | 0.0281 |
| 0.88 | 0.0095 | 0.2451 | 0.0323 |
| 0.85 | 0.0089 | 0.3023 | 0.0407 |
| 0.80 | 0.0071 | 0.4404 | 0.0613 |
| 0.75 | 0.0003 | 0.6368 | 0.0905 |
| 0.70 | 0.0002 | 0.9020 | 0.1298 |
| 0.60 | 0.0002 | 1.6926 | 0.2450 |
| 0.50 | 0.0001 | 2.9424 | 0.4210 |



## Recommendation

**Recommended production default: `knee = 0.96`**

Rationale: it is the LARGEST knee value (so the LEAST chroma compression / desaturation) in the swept grid where the corpus-p90 pixels_clipped_pct stays under 0.1% — meaning at most 10% of corpus images have more than 0.1% of their pixels clip-overshoot the `[0, 1]` linear-BT.709 range. Tighter knees would cost more chroma without reducing clipping below the per-sample noise floor; looser knees would let visible clipping leak through.

At this default, the mean chroma compression is 0.1604%, the mean ΔE2000 vs. no-rolloff is 0.0202, and the corpus-p90 ΔE2000 is 0.0403.


**Comparison vs. current default `knee = 0.90`:** the current default is unnecessarily aggressive. Moving from `0.90` → `0.96` saves 0.0561% of chroma compression (26% relative reduction in desaturation cost), at the cost of a 0.0384% corpus-p90 clip increase (still under the 0.1% threshold). The `max_clip_pct` increases from 0.1604 → 0.1964 on the worst-case sample, both well below the no-rolloff `knee = 1.00` baseline of 10.2745.


### Per-content-class breakdown


**general** (6 samples)

| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |
|------|------------|---------------------|--------------|
| 0.50 | 0.0275 | 1.7215 | 0.2505 |
| 0.60 | 0.0290 | 0.9823 | 0.1508 |
| 0.70 | 0.0313 | 0.5449 | 0.0863 |
| 0.75 | 0.0343 | 0.3900 | 0.0615 |
| 0.80 | 0.0437 | 0.2677 | 0.0406 |
| 0.85 | 0.0488 | 0.1737 | 0.0238 |
| 0.88 | 0.0533 | 0.1299 | 0.0158 |
| 0.90 | 0.0550 | 0.1056 | 0.0113 |
| 0.92 | 0.0701 | 0.0852 | 0.0076 |
| 0.94 | 0.0748 | 0.0695 | 0.0047 |
| 0.96 | 0.0836 | 0.0600 | 0.0034 |
| 0.98 | 0.0867 | 0.0556 | 0.0031 |
| 1.00 | 1.3438 | 0.0000 | 0.0000 |

_general recommended knee: **0.98** (p90 clip 0.0867, chroma compression 0.0556)_


**interiors** (20 samples)

| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |
|------|------------|---------------------|--------------|
| 0.50 | 0.0000 | 2.3089 | 0.3388 |
| 0.60 | 0.0000 | 1.3582 | 0.1980 |
| 0.70 | 0.0000 | 0.7422 | 0.1043 |
| 0.75 | 0.0000 | 0.5329 | 0.0723 |
| 0.80 | 0.0084 | 0.3765 | 0.0487 |
| 0.85 | 0.0087 | 0.2634 | 0.0323 |
| 0.88 | 0.0087 | 0.2122 | 0.0252 |
| 0.90 | 0.0087 | 0.1837 | 0.0214 |
| 0.92 | 0.0088 | 0.1593 | 0.0183 |
| 0.94 | 0.0090 | 0.1390 | 0.0158 |
| 0.96 | 0.0511 | 0.1243 | 0.0142 |
| 0.98 | 0.1591 | 0.1153 | 0.0133 |
| 1.00 | 2.4038 | 0.0000 | 0.0000 |

_interiors recommended knee: **0.96** (p90 clip 0.0511, chroma compression 0.1243)_


**nature** (47 samples)

| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |
|------|------------|---------------------|--------------|
| 0.50 | 0.0009 | 3.4307 | 0.4837 |
| 0.60 | 0.0009 | 1.9648 | 0.2807 |
| 0.70 | 0.0010 | 1.0391 | 0.1484 |
| 0.75 | 0.0011 | 0.7308 | 0.1037 |
| 0.80 | 0.0071 | 0.5039 | 0.0706 |
| 0.85 | 0.0086 | 0.3464 | 0.0476 |
| 0.88 | 0.0091 | 0.2836 | 0.0384 |
| 0.90 | 0.0091 | 0.2537 | 0.0340 |
| 0.92 | 0.0238 | 0.2293 | 0.0305 |
| 0.94 | 0.0296 | 0.2098 | 0.0277 |
| 0.96 | 0.0379 | 0.1964 | 0.0257 |
| 0.98 | 0.1742 | 0.1883 | 0.0245 |
| 1.00 | 3.2900 | 0.0000 | 0.0000 |

_nature recommended knee: **0.96** (p90 clip 0.0379, chroma compression 0.1964)_


**food** (3 samples)

| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |
|------|------------|---------------------|--------------|
| 0.50 | 0.0000 | 1.9573 | 0.3281 |
| 0.60 | 0.0000 | 1.0796 | 0.1876 |
| 0.70 | 0.0000 | 0.5319 | 0.0956 |
| 0.75 | 0.0000 | 0.3502 | 0.0637 |
| 0.80 | 0.0000 | 0.2176 | 0.0399 |
| 0.85 | 0.0000 | 0.1275 | 0.0235 |
| 0.88 | 0.0000 | 0.0908 | 0.0169 |
| 0.90 | 0.0000 | 0.0718 | 0.0135 |
| 0.92 | 0.0000 | 0.0564 | 0.0109 |
| 0.94 | 0.0001 | 0.0445 | 0.0088 |
| 0.96 | 0.0001 | 0.0369 | 0.0075 |
| 0.98 | 0.0003 | 0.0334 | 0.0070 |
| 1.00 | 0.7591 | 0.0000 | 0.0000 |

_food recommended knee: **0.98** (p90 clip 0.0003, chroma compression 0.0334)_


### High-rel_spread subset (saturated specular tail)

Samples from the prior percentile sweep with `rel_spread > 25%` (n=6). These are the cases where the percentile-
based peak measurement diverges sharply from the max-RGB peak —
typically saturated specular highlights on flowers, water, metal. The question: does this tail need a tighter knee than the
general corpus?

| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |
|------|------------|---------------------|--------------|
| 0.50 | 0.0009 | 5.6181 | 0.7080 |
| 0.60 | 0.0010 | 3.2844 | 0.4095 |
| 0.70 | 0.0011 | 1.6372 | 0.2036 |
| 0.75 | 0.0011 | 1.0195 | 0.1281 |
| 0.80 | 0.0012 | 0.5321 | 0.0692 |
| 0.85 | 0.0013 | 0.1936 | 0.0280 |
| 0.88 | 0.0014 | 0.0827 | 0.0136 |
| 0.90 | 0.0014 | 0.0493 | 0.0084 |
| 0.92 | 0.0014 | 0.0269 | 0.0047 |
| 0.94 | 0.0015 | 0.0117 | 0.0021 |
| 0.96 | 0.0015 | 0.0050 | 0.0009 |
| 0.98 | 0.0016 | 0.0031 | 0.0006 |
| 1.00 | 0.2025 | 0.0000 | 0.0000 |

_High-rel_spread recommended knee: **0.98** (p90 clip 0.0016, chroma compression 0.0031)_


## Method notes


- SoftCompress is run in BT.709 (target primaries) per the production pipeline order: matrix-rotate first, then OKLch chroma rolloff. Out-of-gamut BT.709 chroma comes from the BT.2020→BT.709 matrix, *not* the tone-map.

- At `knee = 1.00` no rolloff is applied (`c <= knee * max_c` always holds in `SoftCompress::compress_planes`), so all clip% / chroma deltas at the baseline are zero by construction. The non-zero clip values at `knee = 1.00` are the pre-rolloff overshoot that the rolloff exists to mitigate.

- Clip is BOTH directions: any channel > 1.0 OR any channel < 0.0 counts (the BT.2020→BT.709 matrix can produce negative R/G/B for saturated BT.2020 primaries; the final clamp would zero them).

- Driver: `examples/softcompress_knee_sweep.rs`. Outer threads capped at 5 for the ≤5 GB memory contract.

- Recommended default knee, if any: **0.96**
