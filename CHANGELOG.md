# Changelog

All notable changes to `zentone` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semver.

## [Unreleased]

### QUEUED BREAKING CHANGES

<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

(none currently queued)

### Added

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

### Changed

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

### Deprecated

- `pipeline::tonemap_pq_to_linear_srgb` — superseded by
  `pipeline::tonemap_pq_row_simd` (this release).
- `pipeline::tonemap_pq_to_srgb8` — superseded by
  `pipeline::tonemap_pq_to_srgb8_row_simd` (this release).
- `pipeline::tonemap_hlg_to_linear_srgb` — superseded by
  `pipeline::tonemap_hlg_row_simd` (this release).

The `&[f32]` + `channels: u8` forms remain functional but produce scalar
code; migrate to the strip-form siblings for SIMD throughput.
