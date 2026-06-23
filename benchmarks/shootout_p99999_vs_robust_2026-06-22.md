# Bt2446A: percentile p=0.99999 vs p=0.9999 (`measure_robust`)

**Date:** 2026-06-22
**Source code:** `examples/shootout_p99999_only.rs` (env-driven; both runs from the same binary)
**Corpus:** 76 gain-mapped samples from `/home/lilith/work/codec-corpus/imazen-26/`

Both runs use the same Bt2446A pipeline as `hdr_tone_map_shootout_full.rs` and the same CIE2000 implementation (linear-RGB → CIE Lab D65 → CIE2000), so deltas measure ONLY the peak-percentile change.

## Headline

| metric | p=0.9999 (`measure_robust`) | p=0.99999 (candidate) | delta |
|---|---|---|---|
| mean ΔE2000 (n=76) | **3.5745** | **3.5129** | -0.0615 |
| samples improved (Δ < -0.05) | — | **42** | |
| samples regressed (Δ > +0.05) | — | **18** | |
| samples unchanged (|Δ| ≤ 0.05) | — | **16** | |
| p10 / p50 / p90 of Δ | — | -0.5210 / -0.0929 / +0.3241 | |
| max regression (single sample) | — | +3.4229 | |

## Decision criteria walk-through

- Mean ΔE2000 moved by **-0.0615** (<= -0.05 → improvement)
- Largest single-sample regression: **+3.4229** ΔE2000 (> +0.5 → blocking)

- Mean Δ on high-`rel_spread` samples (≥25%, n=7): **+0.2007**
- Mean Δ on low-`rel_spread` samples (<25%, n=69): **-0.0882**

## Top 5 wins (biggest mean ΔE2000 improvements)

| rank | sample | content | dE_robust | dE_p99999 | Δ | peak_robust (nits) | peak_p99999 (nits) | rel_spread% |
|---|---|---|---|---|---|---|---|---|
| 1 | `1496_nature_lake-with-boats_winthrop-maine_ip13pro_iso50-f1p5_fullsize` | nature | 2.422 | 1.169 | -1.253 | 403 | 468 | 24.39 |
| 2 | `1541_nature_blue-wildflower-grass_snaefellsbaer-iceland_ip16pro_iso64-` | nature | 5.020 | 3.847 | -1.173 | 525 | 577 | 12.65 |
| 3 | `1536_nature_purple-flowers-walkway_seward-alaska_ip16pro_iso80-f1p8_im` | nature | 2.313 | 1.241 | -1.072 | 629 | 757 | 25.12 |
| 4 | `1495_nature_lighthouse-on-island_york-maine_ip13pro_iso50-f1p5_fullsiz` | nature | 2.079 | 1.177 | -0.902 | 410 | 467 | 18.97 |
| 5 | `1493_nature_sunset-over-sea_lower-davie-vancouver-canada_s25u_iso64-f3` | nature | 5.852 | 5.095 | -0.758 | 484 | 516 | 10.04 |

## Top 5 regressions (biggest mean ΔE2000 increases)

| rank | sample | content | dE_robust | dE_p99999 | Δ | peak_robust (nits) | peak_p99999 (nits) | rel_spread% |
|---|---|---|---|---|---|---|---|---|
| 1 | `1526_nature_pink-flowers-garden_colorado_ip13pro_iso50-f1p5_img-4543_4` | nature | 5.104 | 8.526 | +3.423 | 764 | 1126 | 46.69 |
| 2 | `1498_nature_yellow-flowers-garden-bed_colorado_ip13pro_iso50-f1p5_full` | nature | 12.939 | 13.787 | +0.848 | 1398 | 1525 | 11.29 |
| 3 | `1504_nature_cloudy-sky-over-field_colorado_ip13pro_iso50-f1p5_img-0374` | nature | 3.410 | 4.154 | +0.745 | 682 | 740 | 11.31 |
| 4 | `1522_nature_water-lilies-pond_colorado_ip13pro_iso50-f1p5_img-3764_403` | nature | 12.004 | 12.612 | +0.608 | 1529 | 1640 | 7.11 |
| 5 | `1070_general_stone-temple-ruins_ip13pro_iso50-f1p5_img-8852_4032x3024.` | general | 5.901 | 6.476 | +0.575 | 891 | 948 | 7.19 |

## Full per-sample table (sorted by Δ ascending — best improvement first)

| sample | content | dE_robust | dE_p99999 | Δ | peak_robust | peak_p99999 | rel_spread% |
|---|---|---|---|---|---|---|---|
| `1496_nature_lake-with-boats_winthrop-maine_ip13pro_iso50-f1p5_fullsize` | nature | 2.422 | 1.169 | -1.253 | 403 | 468 | 24.39 |
| `1541_nature_blue-wildflower-grass_snaefellsbaer-iceland_ip16pro_iso64-` | nature | 5.020 | 3.847 | -1.173 | 525 | 577 | 12.65 |
| `1536_nature_purple-flowers-walkway_seward-alaska_ip16pro_iso80-f1p8_im` | nature | 2.313 | 1.241 | -1.072 | 629 | 757 | 25.12 |
| `1495_nature_lighthouse-on-island_york-maine_ip13pro_iso50-f1p5_fullsiz` | nature | 2.079 | 1.177 | -0.902 | 410 | 467 | 18.97 |
| `1493_nature_sunset-over-sea_lower-davie-vancouver-canada_s25u_iso64-f3` | nature | 5.852 | 5.095 | -0.758 | 484 | 516 | 10.04 |
| `1066_general_stone-building-facade_park-guell-barcelona_zfold7_iso80-f` | general | 2.725 | 2.100 | -0.625 | 630 | 821 | 34.58 |
| `1067_general_architectural-model-display_la-pedrera-barcelona_zfold7_i` | general | 2.199 | 1.667 | -0.532 | 723 | 872 | 24.45 |
| `1534_nature_red-flowers-closeup_ketchikan-alaska_ip16pro_iso125-f1p8_i` | nature | 3.841 | 3.320 | -0.521 | 654 | 684 | 10.04 |
| `1487_nature_glacier-over-water_kenai-peninsula-borough-alaska_s25u_iso` | nature | 5.821 | 5.323 | -0.498 | 517 | 538 | 10.86 |
| `1064_general_castle-bridge-moat_montjuic-castle-barcelona_zfold7_iso40` | general | 4.446 | 3.980 | -0.467 | 593 | 620 | 6.91 |
| `1527_nature_mountain-wildflowers-landscape_stikine-region-canada_ip16p` | nature | 2.428 | 1.980 | -0.448 | 745 | 844 | 21.74 |
| `1532_nature_red-poppy-flowers_taos-new-mexico_ip17pro_iso80-f1p8_img-6` | nature | 4.231 | 3.836 | -0.395 | 507 | 525 | 5.41 |
| `1540_nature_purple-flowers-garden_belltown-seattle-washington_ip16pro_` | nature | 2.097 | 1.745 | -0.351 | 813 | 952 | 18.86 |
| `1543_nature_water-lilies-pond_miyuki-beppu-japan_ip16pro_iso80-f1p8_im` | nature | 4.222 | 3.933 | -0.289 | 550 | 563 | 3.34 |
| `1531_nature_red-poppy-flowers_taos-new-mexico_ip17pro_iso80-f1p8_img-6` | nature | 3.793 | 3.530 | -0.263 | 493 | 506 | 7.16 |
| `1544_nature_purple-water-lily-flower_miyuki-beppu-japan_ip16pro_iso64-` | nature | 4.319 | 4.078 | -0.241 | 493 | 502 | 3.33 |
| `1488_nature_iceberg-ocean-landscape_yakutat-alaska_s25u_iso80-f1p7_202` | nature | 4.560 | 4.323 | -0.237 | 570 | 581 | 3.55 |
| `1492_nature_mountain-lake-landscape_whitehorse-canada_s25u_iso64-f1p7_` | nature | 1.892 | 1.661 | -0.231 | 760 | 849 | 13.83 |
| `1497_nature_pink-flower-on-vine_belltown-seattle-washington_ip16pro_is` | nature | 2.398 | 2.173 | -0.225 | 666 | 685 | 3.76 |
| `1503_nature_passionflower-on-vine_centrum-amsterdam-netherlands_ip16pr` | nature | 2.573 | 2.356 | -0.217 | 518 | 529 | 2.26 |
| `1242_interiors_storage-room-clutter_la-pedrera-barcelona_zfold7_iso640` | interiors | 3.886 | 3.672 | -0.213 | 582 | 593 | 3.96 |
| `1639_food_sashimi-on-plate_colorado_s25u_iso200-f1p7_20250425-214145_4` | food | 4.072 | 3.860 | -0.212 | 577 | 587 | 3.03 |
| `1520_nature_tulips-in-garden_notre-dame-versailles-france_ip15pro_iso8` | nature | 3.528 | 3.315 | -0.212 | 1128 | 1207 | 7.54 |
| `1515_nature_orange-flower-closeup_colorado_ip17pro_iso320-f1p8_img-142` | nature | 3.922 | 3.725 | -0.196 | 484 | 493 | 3.20 |
| `1494_nature_sunset-over-forest_colorado_zfold7_iso250-f1p7_20251110-17` | nature | 2.514 | 2.320 | -0.194 | 755 | 776 | 7.70 |
| `1521_nature_potted-tulips-garden_notre-dame-versailles-france_ip15pro_` | nature | 3.766 | 3.576 | -0.190 | 971 | 1007 | 4.33 |
| `1491_nature_ocean-sunset-clouds_unorganized-borough-alaska_s25u_iso100` | nature | 1.744 | 1.574 | -0.171 | 812 | 838 | 7.41 |
| `1638_food_sliced-fish-dish_colorado_s25u_iso800-f1p7_20250425-213810_4` | food | 3.582 | 3.416 | -0.166 | 595 | 604 | 3.09 |
| `1547_nature_orange-flowers-garden_colorado_ip16pro_iso80-f1p8_img-9652` | nature | 4.490 | 4.329 | -0.161 | 621 | 630 | 33.72 |
| `1232_interiors_stained-glass-windows_sagrada-familia-barcelona_zfold7_` | interiors | 4.419 | 4.273 | -0.146 | 623 | 631 | 4.01 |
| `1546_nature_underground-cave-lake_playa-del-carmen-mexico_ip13pro_iso2` | nature | 1.259 | 1.117 | -0.142 | 395 | 502 | 35.67 |
| `1227_interiors_stained-glass-window_casa-batllo-barcelona_zfold7_iso16` | interiors | 3.461 | 3.332 | -0.129 | 561 | 571 | 3.28 |
| `1235_interiors_floral-wall-mural_casa-vicens-barcelona_zfold7_iso1600-` | interiors | 3.372 | 3.249 | -0.123 | 578 | 587 | 2.86 |
| `1537_nature_purple-flowers-closeup_seward-alaska_ip16pro_iso100-f1p8_i` | nature | 1.452 | 1.330 | -0.121 | 673 | 687 | 4.54 |
| `1516_nature_pink-flower-in-bottle_colorado_ip17pro_iso50-f1p8_img-1509` | nature | 1.440 | 1.331 | -0.109 | 558 | 565 | 1.24 |
| `1236_interiors_ornate-arched-interior_casa-vicens-barcelona_zfold7_iso` | interiors | 3.381 | 3.278 | -0.103 | 605 | 612 | 3.20 |
| `1230_interiors_empty-tiled-pool_casa-batllo-barcelona_zfold7_iso640-f1` | interiors | 3.309 | 3.206 | -0.102 | 588 | 596 | 2.33 |
| `1245_interiors_circular-building-interior_la-pedrera-barcelona_zfold7_` | interiors | 3.338 | 3.237 | -0.102 | 606 | 613 | 2.57 |
| `1065_general_cathedral-model-workshop_sagrada-familia-barcelona_zfold7` | general | 4.110 | 4.017 | -0.093 | 626 | 631 | 2.34 |
| `1542_nature_orange-lily-flower_belltown-seattle-washington_ip16pro_iso` | nature | 2.060 | 1.972 | -0.089 | 991 | 1401 | 29.26 |
| `1241_interiors_living-room-sofa_la-pedrera-barcelona_zfold7_iso1000-f1` | interiors | 3.426 | 3.343 | -0.083 | 608 | 613 | 0.98 |
| `1228_interiors_stained-glass-windows_casa-batllo-barcelona_zfold7_iso2` | interiors | 3.034 | 2.961 | -0.074 | 605 | 612 | 3.20 |
| `1533_nature_purple-water-lily_koyama-kamiuchikawaracho-kyoto-japan_ip1` | nature | 2.371 | 2.345 | -0.026 | 383 | 384 | 0.78 |
| `1239_interiors_decorative-coffered-ceiling_casa-vicens-barcelona_zfold` | interiors | 3.882 | 3.861 | -0.020 | 595 | 596 | 1.16 |
| `1243_interiors_vintage-living-room_la-pedrera-barcelona_zfold7_iso400-` | interiors | 3.470 | 3.450 | -0.020 | 612 | 613 | 4.52 |
| `1240_interiors_bathroom-with-bathtub_casa-vicens-barcelona_zfold7_iso1` | interiors | 3.113 | 3.097 | -0.016 | 612 | 613 | 0.97 |
| `1069_general_illuminated-castle-night_cinderella-castle-orange_ip13pro` | general | 1.359 | 1.359 | +0.000 | 656 | 656 | 0.00 |
| `1231_interiors_tiled-arched-interior_casa-batllo-barcelona_zfold7_iso6` | interiors | 13.439 | 13.439 | +0.000 | 1000 | 1000 | 0.00 |
| `1237_interiors_person-standing-by-window_casa-vicens-barcelona_zfold7_` | interiors | 2.989 | 2.989 | +0.000 | 613 | 613 | 0.81 |
| `1244_interiors_painted-ceiling-columns_la-pedrera-barcelona_zfold7_iso` | interiors | 3.231 | 3.231 | +0.000 | 613 | 613 | 1.92 |
| `1246_interiors_woven-roof-structure_azulik-city-arts-francisco-uh-may_` | interiors | 1.164 | 1.164 | +0.000 | 389 | 389 | 1.77 |
| `1507_nature_ice-cave-waterfall_hornafjordur-iceland_ip13pro_iso200-f1p` | nature | 3.058 | 3.058 | +0.000 | 622 | 622 | 0.00 |
| `1509_nature_ice-cave-opening_hornafjordur-iceland_ip13pro_iso1000-f1p8` | nature | 2.160 | 2.160 | +0.000 | 685 | 685 | 0.00 |
| `1523_nature_yellow-flowers-garden_colorado_ip13pro_iso50-f1p5_img-3767` | nature | 9.681 | 9.681 | +0.000 | 1646 | 1646 | 0.00 |
| `1545_nature_stalactite-cave-lake_playa-del-carmen-mexico_ip13pro_iso16` | nature | 2.414 | 2.414 | +0.000 | 633 | 633 | 0.00 |
| `1528_nature_pink-blossom-branch_kawazu-japan_ip16pro_iso80-f1p8_img-49` | nature | 1.681 | 1.693 | +0.012 | 644 | 655 | 2.57 |
| `1229_interiors_lighted-archway-ceiling_casa-batllo-barcelona_zfold7_is` | interiors | 1.822 | 1.855 | +0.032 | 917 | 945 | 6.52 |
| `1234_interiors_cathedral-ceiling-columns_sagrada-familia-barcelona_zfo` | interiors | 1.881 | 1.925 | +0.045 | 958 | 970 | 4.20 |
| `1238_interiors_open-doorway-interior_casa-vicens-barcelona_zfold7_iso2` | interiors | 1.948 | 1.999 | +0.051 | 961 | 973 | 3.90 |
| `1640_food_birthday-cake-table_colorado_ip16pro_iso125-f1p8_img-2666_57` | food | 1.324 | 1.395 | +0.072 | 697 | 733 | 29.45 |
| `1502_nature_ice-cave-opening_hornafjordur-iceland_ip13pro_iso32-f1p8_i` | nature | 3.346 | 3.422 | +0.076 | 579 | 583 | 3.02 |
| `1519_nature_orange-rose-vase_colorado_ip17pro_iso200-f1p8_img-2700_396` | nature | 1.353 | 1.434 | +0.081 | 670 | 706 | 6.94 |
| `1233_interiors_cathedral-ceiling-interior_sagrada-familia-barcelona_zf` | interiors | 2.218 | 2.301 | +0.083 | 961 | 973 | 3.90 |
| `1525_nature_wildflowers-in-garden_colorado_ip13pro_iso50-f1p5_img-3861` | nature | 1.859 | 2.075 | +0.217 | 470 | 483 | 4.08 |
| `1530_nature_whale-tail-ocean_ip13pro_iso32-f2p8_img-5574_4032x3024.hei` | nature | 1.538 | 1.808 | +0.270 | 563 | 584 | 4.90 |
| `1524_nature_yellow-flowers-garden_colorado_ip13pro_iso50-f1p5_img-3841` | nature | 8.681 | 8.971 | +0.290 | 1189 | 1228 | 6.16 |
| `1539_nature_legs-in-water_felipe-carrillo-puerto-mexico_ip13pro_iso32-` | nature | 6.637 | 6.952 | +0.315 | 1135 | 1182 | 4.38 |
| `1508_nature_ice-cave-interior_hornafjordur-iceland_ip13pro_iso1000-f1p` | nature | 1.697 | 2.019 | +0.322 | 609 | 659 | 13.37 |
| `1489_nature_cruise-ship-deck_unorganized-borough-alaska_s25u_iso125-f1` | nature | 1.978 | 2.302 | +0.324 | 910 | 955 | 6.67 |
| `1490_nature_boat-wake-at-sunset_unorganized-borough-alaska_s25u_iso200` | nature | 1.778 | 2.181 | +0.403 | 894 | 948 | 9.42 |
| `1529_nature_whale-tail-ocean_ip13pro_iso32-f2p8_img-5573_4032x3024.hei` | nature | 1.434 | 1.878 | +0.444 | 525 | 573 | 10.56 |
| `1070_general_stone-temple-ruins_ip13pro_iso50-f1p5_img-8852_4032x3024.` | general | 5.901 | 6.476 | +0.575 | 891 | 948 | 7.19 |
| `1522_nature_water-lilies-pond_colorado_ip13pro_iso50-f1p5_img-3764_403` | nature | 12.004 | 12.612 | +0.608 | 1529 | 1640 | 7.11 |
| `1504_nature_cloudy-sky-over-field_colorado_ip13pro_iso50-f1p5_img-0374` | nature | 3.410 | 4.154 | +0.745 | 682 | 740 | 11.31 |
| `1498_nature_yellow-flowers-garden-bed_colorado_ip13pro_iso50-f1p5_full` | nature | 12.939 | 13.787 | +0.848 | 1398 | 1525 | 11.29 |
| `1526_nature_pink-flowers-garden_colorado_ip13pro_iso50-f1p5_img-4543_4` | nature | 5.104 | 8.526 | +3.423 | 764 | 1126 | 46.69 |

## Recommendation

**DO NOT SHIP** `DEFAULT_PERCENTILE = 0.99999`. Mean ΔE2000 changes by -0.0615 and the largest single-sample regression is +3.4229. Investigate which content classes regressed (most likely saturated-specular content where 99.999 over-trusts the peak).

## Reproduce

```bash
cd /home/lilith/work/zen/zentone
# Baseline (p=0.9999, equivalent to measure_robust):
SHOOTOUT_PERCENTILE=0.9999 SHOOTOUT_CSV_PATH=benchmarks/shootout_robust_2026-06-22.csv \
  cargo run --release --example shootout_p99999_only --features hdr-shootout
# Candidate (p=0.99999):
cargo run --release --example shootout_p99999_only --features hdr-shootout
# Compare:
python3 /tmp/build_comparison.py
```
