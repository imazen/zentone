# Flicker test for top-7 CLL-variance imazen-26 samples (2026-06-22)

Companion to `hdr_tone_map_shootout_full_2026-06-20.md` and
`hdr_tone_map_reshoot_2446_2026-06-20.md`. Tests visually whether the choice
of source-peak measurement (`measure_max` / `measure_robust` /
`measure_max_smoothed`) drives a noticeable Bt2446A render difference on the
samples where the three methods disagree the most — and whether the
container's `MaxCLL` / `HDRGainMapHeadroom` / `HDRCapacityMax` metadata gives
a different (and possibly better-matching) peak than any of the
measured-from-pixels methods.

The user observed visually that sample #1 (1526 pink flowers iPhone 13 Pro
HEIC) looks "too dark" under `measure_robust`. Hypothesis: the container's
cLLi metadata may give a better-matching peak than any of the three
`measure_*` methods.

## Outputs

- Full-resolution PNGs and per-sample flicker viewer:
  `/mnt/v/output/zentone/flicker-test/<stem>/` (REF + 3-4 Bt2446A variants).
- Top-level index: `http://172.23.240.1:3300/zentone/flicker-test/`

The Bt2446A pipeline is unchanged from `hdr_tone_map_reshoot_2446.rs`
(post-fix Bt2446A: linear-output after commit `60cf33b6`).

## Source-metadata extraction paths

| Container shape                                    | XMP / box used                 | Formula                                         |
|----------------------------------------------------|--------------------------------|-------------------------------------------------|
| Apple HEIC (iP16Pro-class, GainMapVersion=131072)  | `HDRGainMap:HDRGainMapHeadroom`| `peak_nits = 2^headroom * 203`                  |
| UltraHDR JPEG (Adobe gain map)                     | `HDRCapacityMax` (in `-ee` XMP)| `peak_nits = 2^HDRCapacityMax * 203`            |
| HEIF/AVIF with `clli` box                          | `MaxContentLightLevel`         | raw nits                                        |
| Apple HEIC (iP13Pro-class, GainMapVersion=65536)   | no headroom in XMP             | falls back to `.missing` marker                 |

Extraction is via `exiftool -ee` (the embedded-segment flag is required for
UltraHDR JPEG; HEIC primary XMP works without it). When no usable metadata is
present, a `<stem>__bt2446a__metadata.png.missing` marker file is emitted
instead of fabricating a value.

## Per-sample peak values

All in nits. Bt2446A is rendered at full resolution for each peak value.
"meta source" describes which XMP/box path surfaced the metadata peak.

| # | Stem (prefix)                          | dim (display)  | max  | robust | smoothed | metadata | meta source                       | spread (max/min) |
|---|----------------------------------------|----------------|-----:|-------:|---------:|---------:|-----------------------------------|-----------------:|
| 1 | 1526_pink-flowers (iP13Pro HEIC)       | 3024 x 4032    | 1433 |    764 |     1383 |        — | none (GainMapVersion=65536)       |             1.88 |
| 2 | 1546_underground-cave-lake (iP13Pro)   | 3024 x 4032    |  614 |    395 |      611 |        — | none (GainMapVersion=65536)       |             1.55 |
| 3 | 1547_orange-flowers (iP16Pro)          | 4284 x 5712    |  937 |    621 |      817 |     5461 | HDRGainMapHeadroom = 4.7495 stops |             8.79 |
| 4 | 1066_stone-building (zfold7 JPEG)      | 3000 x 4000    |  963 |    630 |      949 |     1000 | HDRCapacityMax    = 2.3000 stops  |             1.59 |
| 5 | 1640_birthday-cake (iP16Pro)           | 4284 x 5712    |  988 |    697 |      932 |     5926 | HDRGainMapHeadroom = 4.8676 stops |             8.50 |
| 6 | 1542_orange-lily (iP16Pro)             | 4284 x 5712    | 1401 |    991 |     1401 |    24285 | HDRGainMapHeadroom = 6.9025 stops |            24.51 |
| 7 | 1536_purple-flowers (iP16Pro)          | 4284 x 5712    |  840 |    629 |      839 |     3571 | HDRGainMapHeadroom = 4.1369 stops |             5.68 |

Notable observations from the extraction itself (before any visual comparison):

- **Two iP13Pro samples have no metadata path.** GainMapVersion=65536 is
  Apple's older gain-map format and doesn't carry `HDRGainMapHeadroom` —
  the only usable peak signal for those is the measured value. So for
  1526 (the user's pink-flowers test case), there's no metadata variant
  to flicker against; the comparison is between the three `measure_*`
  methods only.
- **iP16Pro `HDRGainMapHeadroom` values look very large.** 24,285 nits
  for 1542 is well above any consumer panel's peak; that's the gain-map
  metadata's worst-case display headroom, not the actual content's max
  light level. Whether Bt2446A *should* be parameterized with the
  display-cap headroom or the content-max is the open question this
  comparison opens up.
- **`HDRCapacityMax = 2.3` on 1066 gives a sensible 1000 nits** that
  matches the spread of `measure_max` (963) and `measure_robust` (630).
  This is the case where the metadata path looks most likely to win.

## Render performance

- Total: 45.1s for 7 samples (avg 6.4s/sample) on the local box.
- HEIC iP16Pro (24 MP) samples: ~7s each; iP13Pro (12 MP): ~3.5s; zfold7
  JPEG (12 MP): ~3.8s. Wall time is dominated by the decode +
  `reconstruct_hdr` step, not the Bt2446A call.

## Flicker viewer

Per-sample HTML at `/mnt/v/output/zentone/flicker-test/<stem>/flicker.html`:

- Vanilla-JS, no CDN, no build step.
- Two `<img>` elements toggled via `visibility` on a `setInterval` timer.
- Dropdown for "variant A" and "variant B"; both default to a useful pair
  (REF vs `bt2446a__robust`); each dropdown lists all available variants
  (REF + max + robust + smoothed [+ metadata]).
- Speed slider 50..2000 ms (default 500).
- Pause/play + swap-A/B buttons.
- Sticky toolbar; legend in fixed top-right corner showing which variant
  is currently visible.
- Top of toolbar shows the source path + the 4 peak values + the
  metadata-source label.
- Top-level `index.html` lists all 7 samples with thumbnails and links.

## Out of scope

- No fresh ΔE2000 metrics — the existing reshoot CSV at
  `benchmarks/hdr_tone_map_reshoot_2446_2026-06-20.csv` already has
  per-sample numbers for the three `measure_*` methods, and metadata
  variants don't carry over from there. Visual flicker is the
  ground-truth call here.
- No animated WebP/GIF flicker — the HTML viewer is more flexible (any
  pair, any interval).
- Code path changes — `apply_curve` and `Bt2446A` are unchanged from
  main@origin; this run reuses them as-is.

## Reproducing

```
nice -n19 cargo run -p zentone --release \
    --example flicker_test_render --features hdr-shootout
```

Output at `/mnt/v/output/zentone/flicker-test/`. Re-run is idempotent;
existing PNGs are overwritten.
