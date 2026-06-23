# HDR → SDR shootout v2 — tail-aware findings (2026-06-22)

**Headline.** Re-scoring the 6080-cell audited sweep with per-image ΔE percentiles and OKLab Euclidean ΔE, to surface tail behavior that `mean_de2000` averaged away.

- CSV: `benchmarks/hdr_tone_map_shootout_full_2026-06-22_v2.csv` (6080 rows × 23 cols)
- Unique (curve, peak) combos: 80
- metrics_version: `2026-06-22-percentiles-oklab`


## 1. Six-criteria top-5 grid

Note: ranks 1-4 across every metric are all `bt2446a × <peak method>` — they cluster within ~3% of each other. Rank 5 jumps to a different curve (typically `bt2390 × measure_robust` at ~2× the error). The top-1 winner is therefore a peak-method choice within `bt2446a`, not a curve choice.

### Top 5 by `mean_de2000` — Lab ΔE2000 mean (old default)

| rank | combo | mean_de2000 | de_ok_mean | de_ok_p95 | pct>de5 |
|---|---|---|---|---|---|
| 1 | bt2446a × measure_max | 3.7706 | 0.03651 | 0.06158 | 16.63 |
| 2 | bt2446a × measure_percentile_99999 | 3.7778 | 0.03680 | 0.06073 | 18.44 |
| 3 | bt2446a × measure_max_smoothed | 3.7905 | 0.03681 | 0.06172 | 16.96 |
| 4 | bt2446a × measure_robust | 3.8819 | 0.03829 | 0.06181 | 21.56 |
| 5 | bt2390 × measure_robust | 8.1564 | 0.08254 | 0.11816 | 66.94 |

### Top 5 by `de2000_p95` — Lab ΔE2000 p95 (per-image p95 → corpus mean)

| rank | combo | de2000_p95 | de_ok_mean | de_ok_p95 | mean_de2000 | pct>de5 |
|---|---|---|---|---|---|---|
| 1 | bt2446a × measure_percentile_99999 | 6.0977 | 0.03680 | 0.06073 | 3.7778 | 18.44 |
| 2 | bt2446a × measure_robust | 6.1720 | 0.03829 | 0.06181 | 3.8819 | 21.56 |
| 3 | bt2446a × measure_max | 6.2061 | 0.03651 | 0.06158 | 3.7706 | 16.63 |
| 4 | bt2446a × measure_max_smoothed | 6.2104 | 0.03681 | 0.06172 | 3.7905 | 16.96 |
| 5 | bt2390 × measure_robust | 12.8901 | 0.08254 | 0.11816 | 8.1564 | 66.94 |

### Top 5 by `de2000_p99` — Lab ΔE2000 p99 (per-image p99 → corpus mean)

| rank | combo | de2000_p99 | de_ok_mean | de_ok_p95 | mean_de2000 | pct>de5 |
|---|---|---|---|---|---|---|
| 1 | bt2446a × measure_percentile_99999 | 7.0854 | 0.03680 | 0.06073 | 3.7778 | 18.44 |
| 2 | bt2446a × measure_robust | 7.1653 | 0.03829 | 0.06181 | 3.8819 | 21.56 |
| 3 | bt2446a × measure_max | 7.1965 | 0.03651 | 0.06158 | 3.7706 | 16.63 |
| 4 | bt2446a × measure_max_smoothed | 7.2030 | 0.03681 | 0.06172 | 3.7905 | 16.96 |
| 5 | bt2390 × measure_robust | 13.7530 | 0.08254 | 0.11816 | 8.1564 | 66.94 |

### Top 5 by `pct_above_de5` — % pixels per image with ΔE2000 > 5

| rank | combo | pct_above_de5 | de_ok_mean | de_ok_p95 | mean_de2000 |
|---|---|---|---|---|---|
| 1 | bt2446a × measure_max | 16.6286 | 0.03651 | 0.06158 | 3.7706 |
| 2 | bt2446a × measure_max_smoothed | 16.9562 | 0.03681 | 0.06172 | 3.7905 |
| 3 | bt2446a × measure_percentile_99999 | 18.4362 | 0.03680 | 0.06073 | 3.7778 |
| 4 | bt2446a × measure_robust | 21.5640 | 0.03829 | 0.06181 | 3.8819 |
| 5 | bt2390 × measure_robust | 66.9351 | 0.08254 | 0.11816 | 8.1564 |

### Top 5 by `de_ok_mean` — OKLab ΔE mean

| rank | combo | de_ok_mean | de_ok_p95 | mean_de2000 | pct>de5 |
|---|---|---|---|---|---|
| 1 | bt2446a × measure_max | 0.03651 | 0.06158 | 3.7706 | 16.63 |
| 2 | bt2446a × measure_percentile_99999 | 0.03680 | 0.06073 | 3.7778 | 18.44 |
| 3 | bt2446a × measure_max_smoothed | 0.03681 | 0.06172 | 3.7905 | 16.96 |
| 4 | bt2446a × measure_robust | 0.03829 | 0.06181 | 3.8819 | 21.56 |
| 5 | bt2390 × measure_robust | 0.08254 | 0.11816 | 8.1564 | 66.94 |

### Top 5 by `de_ok_p95` — OKLab ΔE p95 (per-image p95 → corpus mean)

| rank | combo | de_ok_p95 | de_ok_mean | mean_de2000 | pct>de5 |
|---|---|---|---|---|---|
| 1 | bt2446a × measure_percentile_99999 | 0.06073 | 0.03680 | 3.7778 | 18.44 |
| 2 | bt2446a × measure_max | 0.06158 | 0.03651 | 3.7706 | 16.63 |
| 3 | bt2446a × measure_max_smoothed | 0.06172 | 0.03681 | 3.7905 | 16.96 |
| 4 | bt2446a × measure_robust | 0.06181 | 0.03829 | 3.8819 | 21.56 |
| 5 | bt2390 × measure_robust | 0.11816 | 0.08254 | 8.1564 | 66.94 |


## 2. Worst-tail samples under ΔE2000 p99 (top 4 combos)

### bt2446a × measure_percentile_99999

| sample (stem) | de2000_p99 | de_ok_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 20.278 | 0.1820 | 13.657 | 96.77 |
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 19.675 | 0.2171 | 12.616 | 79.57 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 19.383 | 0.1888 | 9.646 | 86.59 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 18.881 | 0.2036 | 13.523 | 97.15 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 17.345 | 0.1772 | 8.953 | 83.25 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 15.602 | 0.1789 | 6.945 | 63.98 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 13.913 | 0.1654 | 6.486 | 62.85 |
| 1526_nature_pink-flowers-garden_colorado_ip13pro_i... | 13.644 | 0.1361 | 8.201 | 75.20 |
| 1509_nature_ice-cave-opening_hornafjordur-iceland_... | 9.703 | 0.0991 | 2.186 | 19.11 |
| 1508_nature_ice-cave-interior_hornafjordur-iceland... | 9.353 | 0.1004 | 2.025 | 17.50 |

### bt2446a × measure_robust

| sample (stem) | de2000_p99 | de_ok_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 20.182 | 0.1813 | 13.589 | 96.67 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 19.383 | 0.1888 | 9.646 | 86.59 |
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 18.750 | 0.2074 | 12.011 | 78.87 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 17.700 | 0.1915 | 12.651 | 96.30 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 16.883 | 0.1727 | 8.648 | 81.59 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 15.064 | 0.1730 | 6.635 | 61.77 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 13.066 | 0.1560 | 5.929 | 60.59 |
| 1509_nature_ice-cave-opening_hornafjordur-iceland_... | 9.703 | 0.0991 | 2.186 | 19.11 |
| 1533_nature_purple-water-lily_koyama-kamiuchikawar... | 9.504 | 0.1018 | 2.766 | 7.60 |
| 1544_nature_purple-water-lily-flower_miyuki-beppu-... | 9.121 | 0.0939 | 4.771 | 38.56 |

### bt2446a × measure_max

| sample (stem) | de2000_p99 | de_ok_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 20.278 | 0.1820 | 13.657 | 96.77 |
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 19.757 | 0.2180 | 12.669 | 79.63 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 19.385 | 0.2088 | 13.892 | 97.44 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 19.383 | 0.1888 | 9.646 | 86.59 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 17.725 | 0.1808 | 9.203 | 84.39 |
| 1526_nature_pink-flowers-garden_colorado_ip13pro_i... | 16.795 | 0.1687 | 10.149 | 81.03 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 15.704 | 0.1800 | 7.004 | 64.38 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 14.091 | 0.1674 | 6.602 | 63.28 |
| 1508_nature_ice-cave-interior_hornafjordur-iceland... | 10.335 | 0.1108 | 2.295 | 18.78 |
| 1509_nature_ice-cave-opening_hornafjordur-iceland_... | 9.703 | 0.0991 | 2.186 | 19.11 |

### bt2446a × measure_max_smoothed

| sample (stem) | de2000_p99 | de_ok_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 20.278 | 0.1820 | 13.657 | 96.77 |
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 19.757 | 0.2180 | 12.669 | 79.63 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 19.383 | 0.1888 | 9.646 | 86.59 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 19.343 | 0.2083 | 13.861 | 97.42 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 17.433 | 0.1780 | 9.011 | 83.52 |
| 1526_nature_pink-flowers-garden_colorado_ip13pro_i... | 16.347 | 0.1641 | 9.876 | 80.27 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 15.693 | 0.1799 | 6.997 | 64.33 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 14.091 | 0.1674 | 6.602 | 63.28 |
| 1508_nature_ice-cave-interior_hornafjordur-iceland... | 10.144 | 0.1088 | 2.243 | 18.53 |
| 1509_nature_ice-cave-opening_hornafjordur-iceland_... | 9.703 | 0.0991 | 2.186 | 19.11 |


## 3. Worst-tail samples under OKLab ΔE p99 (top 4 combos)

### bt2446a × measure_percentile_99999

| sample (stem) | de_ok_p99 | de2000_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 0.2171 | 19.675 | 12.616 | 79.57 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 0.2036 | 18.881 | 13.523 | 97.15 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1888 | 19.383 | 9.646 | 86.59 |
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 0.1820 | 20.278 | 13.657 | 96.77 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 0.1789 | 15.602 | 6.945 | 63.98 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1772 | 17.345 | 8.953 | 83.25 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 0.1654 | 13.913 | 6.486 | 62.85 |
| 1521_nature_potted-tulips-garden_notre-dame-versai... | 0.1477 | 7.066 | 3.695 | 22.53 |
| 1526_nature_pink-flowers-garden_colorado_ip13pro_i... | 0.1361 | 13.644 | 8.201 | 75.20 |
| 1520_nature_tulips-in-garden_notre-dame-versailles... | 0.1344 | 6.885 | 3.346 | 13.49 |

### bt2446a × measure_robust

| sample (stem) | de_ok_p99 | de2000_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 0.2074 | 18.750 | 12.011 | 78.87 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 0.1915 | 17.700 | 12.651 | 96.30 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1888 | 19.383 | 9.646 | 86.59 |
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 0.1813 | 20.182 | 13.589 | 96.67 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 0.1730 | 15.064 | 6.635 | 61.77 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1727 | 16.883 | 8.648 | 81.59 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 0.1560 | 13.066 | 5.929 | 60.59 |
| 1521_nature_potted-tulips-garden_notre-dame-versai... | 0.1489 | 7.160 | 3.884 | 26.47 |
| 1520_nature_tulips-in-garden_notre-dame-versailles... | 0.1370 | 7.070 | 3.572 | 18.66 |
| 1515_nature_orange-flower-closeup_colorado_ip17pro... | 0.1137 | 9.094 | 5.171 | 62.33 |

### bt2446a × measure_max

| sample (stem) | de_ok_p99 | de2000_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 0.2180 | 19.757 | 12.669 | 79.63 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 0.2088 | 19.385 | 13.892 | 97.44 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1888 | 19.383 | 9.646 | 86.59 |
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 0.1820 | 20.278 | 13.657 | 96.77 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1808 | 17.725 | 9.203 | 84.39 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 0.1800 | 15.704 | 7.004 | 64.38 |
| 1526_nature_pink-flowers-garden_colorado_ip13pro_i... | 0.1687 | 16.795 | 10.149 | 81.03 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 0.1674 | 14.091 | 6.602 | 63.28 |
| 1521_nature_potted-tulips-garden_notre-dame-versai... | 0.1472 | 7.025 | 3.626 | 21.38 |
| 1520_nature_tulips-in-garden_notre-dame-versailles... | 0.1341 | 6.854 | 3.320 | 12.94 |

### bt2446a × measure_max_smoothed

| sample (stem) | de_ok_p99 | de2000_p99 | mean_de2000 | pct>de5 |
|---|---|---|---|---|
| 1522_nature_water-lilies-pond_colorado_ip13pro_iso... | 0.2180 | 19.757 | 12.669 | 79.63 |
| 1498_nature_yellow-flowers-garden-bed_colorado_ip1... | 0.2083 | 19.343 | 13.861 | 97.42 |
| 1523_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1888 | 19.383 | 9.646 | 86.59 |
| 1231_interiors_tiled-arched-interior_casa-batllo-b... | 0.1820 | 20.278 | 13.657 | 96.77 |
| 1539_nature_legs-in-water_felipe-carrillo-puerto-m... | 0.1799 | 15.693 | 6.997 | 64.33 |
| 1524_nature_yellow-flowers-garden_colorado_ip13pro... | 0.1780 | 17.433 | 9.011 | 83.52 |
| 1070_general_stone-temple-ruins_ip13pro_iso50-f1p5... | 0.1674 | 14.091 | 6.602 | 63.28 |
| 1526_nature_pink-flowers-garden_colorado_ip13pro_i... | 0.1641 | 16.347 | 9.876 | 80.27 |
| 1521_nature_potted-tulips-garden_notre-dame-versai... | 0.1472 | 7.025 | 3.626 | 21.38 |
| 1520_nature_tulips-in-garden_notre-dame-versailles... | 0.1341 | 6.857 | 3.323 | 12.99 |


## 4. Surprise findings

Under every new tail-aware metric, the top-5 combos are a subset or permutation of the top-5 under `mean_de2000`. The ranking *order* may shift, but no dark-horse combo emerges.


## 5. Recommended production combo

**Recommendation: `bt2446a × measure_percentile_99999`**

- de_ok_p95 = 0.06073
- de_ok_mean = 0.03680
- mean_de2000 = 3.7778
- de2000_p95 = 6.0977
- de2000_p99 = 7.0854
- pct_above_de5 = 18.44
- pct_above_de_ok_0p04 = 33.21


### Tiebreaker policy
1. **Primary: `de_ok_p95`** — caps the perceptual tail. OKLab is more perceptually uniform than CIE Lab D65, especially on saturated colors (the kinds of pixels the user could see were wrong in the flicker viewer). Capping the 95th-percentile pixel ΔE inside each image is the closest single number to 'no pixel will look obviously wrong'.
2. **Secondary: `de_ok_mean`** — overall fidelity once the tail is bounded.
3. **Tertiary: peak-method cost** — `measure_max` is the cheapest of the four (single pass, no smoothing/percentile/robust statistics).


## 6. Top-5 under the production policy

Combining primary + secondary + tertiary into a single sort:

| rank | combo | de_ok_p95 | de_ok_mean | mean_de2000 | pct>de5 | peak cost |
|---|---|---|---|---|---|---|
| 1 | bt2446a × measure_percentile_99999 | 0.06073 | 0.03680 | 3.7778 | 18.44 | 2 |
| 2 | bt2446a × measure_max | 0.06158 | 0.03651 | 3.7706 | 16.63 | 0 |
| 3 | bt2446a × measure_max_smoothed | 0.06172 | 0.03681 | 3.7905 | 16.96 | 1 |
| 4 | bt2446a × measure_robust | 0.06181 | 0.03829 | 3.8819 | 21.56 | 3 |
| 5 | bt2390 × measure_robust | 0.11816 | 0.08254 | 8.1564 | 66.94 | 3 |


## 7. Stability of the curve choice

- `mean_de2000`: top curve = `bt2446a`
- `de2000_p95`: top curve = `bt2446a`
- `de2000_p99`: top curve = `bt2446a`
- `de_ok_mean`: top curve = `bt2446a`
- `de_ok_p95`: top curve = `bt2446a`
