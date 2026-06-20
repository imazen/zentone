# Changelog

All notable changes to `zentone` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semver.

## [Unreleased]

### QUEUED BREAKING CHANGES

<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

- `gamut::apply_matrix_row` now takes `channels: u8` (was `usize`), matching the `ToneMap` trait and every other `channels` parameter in the crate, so a single channel-count value chains through the gamut → tone-map seam without a per-call-site cast (closes #23). Call sites passing an integer literal are unaffected; only those passing a `usize` *variable* need to switch to `u8`.

### Added

- `ToneMapCurve::Mobius { source_peak, knee }` — port of libplacebo's
  `mobius()` (the production HDR-playback default across mpv, VLC,
  FFmpeg, and Plex). `M(x) = scale · (x + a) / (x + b)` for `x > knee`,
  identity below; coefficients solve `M(knee) = knee`, `M(source_peak) =
  1.0`, `M'(knee) = 1.0` — continuous and C¹-smooth. Default knee
  `0.30` matches libplacebo's `linear_knee`. Input is per-channel
  normalized so `1.0` is the SDR target peak; pass `source_peak =
  source_peak_nits / target_peak_nits`. Per-channel evaluation
  re-associates the rational form to stay finite at `f32::MAX`.
- `gamut::soft_clip_knee(rgb, knee)` + `gamut::soft_clip_knee_strip` —
  hue-preserving rational knee rolloff. Identity below `knee` on every
  channel; above, a single per-pixel scale on the max channel pulls all
  channels toward `1.0`, preserving channel ratios (so hue is held
  exactly). Negatives clamped to 0 first (same convention as
  `soft_clip`). C¹-smooth at the knee; output guaranteed in `[0, 1]` for
  any non-negative input when `knee < 1`. Cheap enough to ship without a
  dedicated SIMD tier — LLVM auto-vectorises the inner loop.
- `HdrToSdr { source_peak_nits, target_peak_nits, knee_tone, knee_gamut }`
  — one-call HDR→SDR wrapper: Möbius tone-map → `soft_clip_knee`.
  Production defaults are `target_peak_nits = 100`, `knee_tone = 0.30`,
  `knee_gamut = 0.95`. Source peak is **caller-provided** — `zentone`
  stays dep-free of `zenpixels-convert` per the workspace architecture
  (zentone owns HDR algorithms, CLL measurement lives in
  `zenpixels_convert::hdr::measure`). Skips tone mapping when
  `source_peak_nits <= target_peak_nits` (input already fits target).
  Gamut conversion (e.g. BT.2020 → BT.709) is the caller's
  responsibility before `apply_strip`; this struct only handles
  luminance compression + gamut-edge soft-clip in the target gamut.
  Exposes `apply_strip(&mut [[f32; 3]])` and `apply_rgb([f32; 3])`.

### Changed

- Unified the last two internal `channels: usize` parameters to `u8` (the private `CellGrid::add_row` and a gainmap test helper), so every `channels` count crate-wide is now `u8`. No public-API impact; completes the #23 unification.
- Exclude `tests/` and `.gitignore` from the published crate package to reduce crate download size (~234 KB saved); `benches/` retained due to explicit `[[bench]]` targets in Cargo.toml.

### Fixed

- **`Bt2446A` — corrected two algorithm bugs against ITU-R BT.2446-1 §4
  and the libplacebo reference implementation.** Both silently
  miscalibrated the HDR→SDR mapping; the existing tests were loose
  enough not to catch them.
  - **`ρ_H` / `ρ_S` exponent**: `(L_peak / 10 000)^2.4` →
    `(L_peak / 10 000)^(1/2.4)`. Pre-fix `ρ_H` at 1000 nits was ~1.13
    (vs the correct ~13.26), which made the log-compression step a
    near-identity and let HDR-mid-grey land roughly half a stop too dark.
  - **G' channel YCbCr→RGB coefficients**:
    `(0.16455 / 0.6780) · Cb + (0.57135 / 0.6780) · Cr` →
    `0.16455 · Cb + 0.57135 · Cr`. The constants `0.16455` and `0.57135`
    are already `2·Kb·(1-Kb)/Kg` and `2·Kr·(1-Kr)/Kg` for BT.2020
    (already divided by `Kg = 0.6780`); dividing again made the green
    channel ~1.47× off and shifted hue on saturated content. Same bug
    in `simd::curves::bt2446a_tier` (SIMD splat and scalar tail both).
  - Added regression tests: `rho_hdr_matches_itu_reference_values`
    (pins `ρ_H ≈ 13.26` at 1000 nits and `33` at 10 000 nits per the
    spec), `libplacebo_parity_eetf_only` (compares the per-channel EETF
    against the published libplacebo formula across 12 sample points × 3
    peak configurations), `ycbcr_inverse_matrix_round_trips_at_y_tmo_passthrough`
    (pins the G' coefficient correctness independent of tone curve).
- Tone-map curves no longer emit non-finite (`NaN`/`±Inf`) output for finite input, and the scalar `map_rgb` now agrees with the SIMD `map_row` on extreme/negative inputs (found by the fuzz farm, zentone#21). `ExtendedReinhard` overflowed `l_in·(1 + l_in/l_max²)` to `+Inf` for large luminance (reassociated to keep intermediates finite); `ExtendedReinhard`/`ReinhardJodie`/`TunedReinhard` now clamp channels to non-negative linear light in both scalar and SIMD paths (the SIMD row path already clamped, so the two diverged on negatives). Regression: `tone_map::tests::all_curves_finite_and_parity_on_extreme_inputs` + `fuzz/regression/fuzz_curves_extreme_reinhard_zentone21`.

## [0.1.0] - 2026-04-26

First publish to crates.io.

### Changed

- Pipeline functions (`tonemap_pq_row_simd`, `tonemap_pq_rgba_row_simd`,
  `tonemap_hlg_row_simd`, `tonemap_hlg_rgba_row_simd`,
  `tonemap_pq_to_srgb8_row_simd`, `tonemap_pq_to_srgb8_rgba_row_simd`) take
  `&mut TonemapScratch` as the first argument and internally chunk the
  input by `scratch.chunk_size()`. Caps working-set memory regardless of
  input strip length and eliminates the remaining per-call `Vec`
  allocations in the RGBA + sRGB8 paths. Verified by
  `pipeline_memory_bounded_by_chunk_size` and
  `pipeline_chunk_size_invariance` in `tests/simd_parity.rs`.

### Added

- `TonemapScratch` (re-exported from crate root) — reusable scratch buffers
  + chunk-size policy for the SIMD pipelines. Default chunk size 4096
  pixels (~48 KiB working set, fits L2). `with_chunk_size(...)` for tuning.
- `gainmap` module: stable home for `LumaToneMap`, `LumaGainMapSplitter`,
  `SplitConfig`, `SplitStats`, `LumaFn`, `Bt2408Yrgb`, `ExtendedReinhardLuma`,
  and a built-in `HableFilmic` curve. Graduates from `experimental::gain_map`,
  no longer requires the `experimental` feature.
- `HableFilmic` — the GDC-2010 Uncharted 2 filmic curve. Useful zero-config
  default for HDR encoding without external content metadata.
- `simd::*` curve kernels migrated to `#[archmage::magetypes(...)]`. Lower
  LOC, broader arch coverage (V4/V3/NEON/WASM128/scalar where supported,
  V3+NEON+WASM128+scalar for transcendental-using curves) (26e9c56).
- `gamut::apply_matrix_row_simd`, `gamut::apply_matrix_row_simd_rgba` —
  3x3 RGB matrix over an RGB or RGBA strip, vectorized (46fbe67).
- `gamut::soft_clip_row_simd` — vectorized hue-preserving soft clip
  (46fbe67).
- `gamut::is_out_of_gamut_mask_simd` — per-pixel 1.0/0.0 mask, vectorized
  (46fbe67).
- `hlg::hlg_ootf_row_simd`, `hlg::hlg_inverse_ootf_row_simd` —
  chromaticity-preserving HLG OOTF forward/inverse over a strip (46fbe67).
- `hlg::hlg_ootf_approx_row_simd`, `hlg::hlg_inverse_ootf_approx_row_simd`
  — per-channel libultrahdr-compat OOTF over a strip (46fbe67).
- `pipeline::tonemap_pq_row_simd`, `pipeline::tonemap_pq_rgba_row_simd`
  — fused PQ → tone-map → BT.709 → soft-clip kernel for `&[[f32; 3]]` /
  `&[[f32; 4]]` strips (7ae5c05).
- `pipeline::tonemap_hlg_row_simd`, `pipeline::tonemap_hlg_rgba_row_simd`
  — fused HLG → tone-map → BT.709 → soft-clip kernel for f32 strips
  (7ae5c05).
- `pipeline::tonemap_pq_to_srgb8_row_simd`,
  `pipeline::tonemap_pq_to_srgb8_rgba_row_simd` — fused PQ → tone-map →
  sRGB-encoded `u8` strip kernels (7ae5c05).
- `ToneMap::map_strip_simd` — provided trait method that operates on a
  `&mut [[f32; 3]]` strip; default implementation calls `map_rgb` per
  pixel (7ae5c05).
- `Bt2408Tonemapper::map_strip_simd`, `Bt2446A::map_strip_simd`,
  `Bt2446B::map_strip_simd`, `Bt2446C::map_strip_simd`,
  `CompiledFilmicSpline::map_strip_simd` — SIMD overrides that
  short-circuit the per-pixel default (7f15b10).
- API tier section in the crate-level rustdoc landing page distinguishing
  hot-path strip APIs, reference per-pixel functions, and experimental
  surface (this release).
- `benches/pipeline_bench.rs` — scalar-vs-SIMD bench harness covering
  PR1 curve kernels (regression gate), PR2 building blocks
  (`apply_matrix_row_simd`, `soft_clip_row_simd`, `hlg_ootf_*_row_simd`),
  PR4 stateful curve `map_strip_simd` overrides at three strip widths
  (256/1024/4096), and one end-to-end `tonemap_pq_to_srgb8_row_simd`
  pair. Results checked into `benchmarks/zentone_simd_*.csv`. SIMD wins
  every cell, range 1.21× (BT.2446B) – 26.97× (HLG OOTF approx),
  median 2.25×.

### Changed

- `LumaToneMap` and the splitter API moved from `zentone::experimental::gain_map::*`
  to `zentone::*` (re-exported from the new `gainmap` module). The
  `experimental::gain_map::*` paths are removed; users on `experimental`
  feature gating should update their imports.
- `ToneMap` trait now exposes `map_strip_simd` as a provided method.
  Default behavior is preserved (per-pixel loop over `map_rgb`); existing
  implementors do not need to override it (7ae5c05).

### Fixed

- `gamut::soft_clip` mishandled the `r == g == b > 1` branch — only `r`
  and `g` were clamped, leaving `b` over-range. The new SIMD kernel always
  clamped correctly; the per-pixel reference now matches (46fbe67).
- `gamut::soft_clip` also mishandled `r >= g == b` with `r > 1` (e.g.
  HDR-saturated red after BT.709 conversion): previously mapped all three
  channels to `min(r, 1)` (white). Now routes through `clip_sorted` like
  every other branch and produces the correct hue-preserving clipped red
  (7ae5c05).
- ARM NEON regression in `simd::reinhard_jodie_*_tier`,
  `simd::narkowicz_*_tier`, and `simd::hable_*_tier`: extreme inputs
  (negative for ReinhardJodie, near `f32::MAX` for Narkowicz/Hable)
  produced NaN on aarch64. The trailing `.min(one).max(zero)` masks NaN
  to a finite value on x86 SSE/AVX (`_mm_min_ps` returns the second
  operand on NaN) but ARM `vminq_f32` propagates NaN, exposing
  -1/0 → ±Inf and Inf/Inf → NaN paths the scalar reference avoids via
  early-out. SIMD kernels now pre-clamp inputs to a finite, in-domain
  range matching the scalar's effective domain so NaN never enters the
  arithmetic. Caught by `tests/bruteforce_robustness.rs`.

### Removed

- `pipeline::tonemap_pq_to_linear_srgb` — superseded by
  `pipeline::tonemap_pq_row_simd` in this release.
- `pipeline::tonemap_pq_to_srgb8` — superseded by
  `pipeline::tonemap_pq_to_srgb8_row_simd` in this release.
- `pipeline::tonemap_hlg_to_linear_srgb` — superseded by
  `pipeline::tonemap_hlg_row_simd` in this release.

The `&[f32]` + `channels: u8` forms are no longer present; migrate to the
SIMD strip-form siblings (`&[[f32; 3]]` / `&[[f32; 4]]`).
