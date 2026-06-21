# Changelog

All notable changes to `zentone` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semver.

## [Unreleased]

### QUEUED BREAKING CHANGES

<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

- `gamut::apply_matrix_row` now takes `channels: u8` (was `usize`), matching the `ToneMap` trait and every other `channels` parameter in the crate, so a single channel-count value chains through the gamut → tone-map seam without a per-call-site cast (closes #23). Call sites passing an integer literal are unaffected; only those passing a `usize` *variable* need to switch to `u8`.

### Removed

- **`Bt2446A` (curve struct + `bt2446a_tier` SIMD strip kernel)** moved to
  [`zenpixels_convert::hdr::Bt2446A`](https://docs.rs/zenpixels-convert)
  (gated on the existing `hdr-experimental` feature). The convert crate is
  now the canonical home for ITU-R BT.2446 Method A; it lives next to the
  CLL measurement primitives that feed the HDR → SDR pipeline, and composes
  into a one-call `zenpixels_convert::hdr::HdrToSdr` wrapper that pairs the
  curve with a BT.2020 → BT.709 primary conversion + OKLch soft chroma
  compression. The byte-identical algorithm, the POW24 polynomial EOTF
  coefficients, and the strip-vs-scalar parity test moved with the kernel.
- **`HdrToSdr` (the zentone one-call wrapper) removed.** Use
  [`zenpixels_convert::hdr::HdrToSdr`](https://docs.rs/zenpixels-convert)
  instead — same `HdrToSdr::new(source_peak_nits)` constructor signature,
  same `apply_strip` / `apply_rgb` API, plus a richer pipeline (it now
  applies a BT.2020 → BT.709 primary matrix before the curve and uses an
  OKLch soft-compress instead of zentone's `soft_clip_knee_strip` at the
  output). Existing call sites only need to swap the import line.
- Bt2446A-only examples (`examples/bt2446a_throughput.rs`,
  `examples/hdr_tone_map_reshoot_2446.rs`) removed; throughput / shootout
  work for that curve now belongs in the convert crate. Multi-curve
  shootout examples (`hdr_tone_map_shootout`, `hdr_tone_map_shootout_full`)
  were updated to import `Bt2446A` from `zenpixels_convert::hdr` and still
  build under `--features hdr-shootout`.

### Changed

- **`HdrToSdr::new(source_peak_nits)` now uses `Bt2446A` instead of `Möbius` as
  the internal tone-mapping curve**, based on the 76-sample HDR shootout
  ([commit `5a25742`](../commit/5a25742) — `benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`).
  Median ΔE2000 to producer-graded SDR drops from ~16.65 to 3.17 — a ~5×
  improvement at the median under the production `measure_robust` peak
  method, across both UltraHDR JPEG and iPhone HEIC sources. The Möbius
  variant ([`ToneMapCurve::Mobius`](src/curves.rs)) stays selectable for
  callers who specifically need libplacebo-compatible HDR-playback
  behavior (mpv/VLC/FFmpeg parity); `HdrToSdr` targets the distinct
  "match what users see in their phone/camera gallery after the device
  ISP grades the SDR" intent. The `knee_tone` field is dropped because
  Bt2446A's EETF is fully parameterized by `(source_peak_nits,
  target_peak_nits)` and doesn't expose a Möbius-style knee. `knee_gamut`
  stays for the soft-clip-knee post-process.

### Performance

- **`Bt2446A::map_strip_simd` is ~2.3× faster** (111 → 258 Mpix/s
  measured on Zen 4 / AMD Ryzen 9 7950X w/ AVX-512; `RUSTFLAGS="-C target-cpu=native"`).
  The kernel now resolves to `f32x16` ops on AVX-512 hardware (was f32x8
  only), every transcendental switches to its `_unchecked` variant (inputs
  are pre-clamped to `pos_eps`), and the BT.1886 EOTF (`x^2.4`) uses a
  degree-7 monomial polynomial evaluated via Estrin's method in place of
  three serial `pow_midp` calls — critical-path latency drops from ~28 to
  ~12 cycles per channel. Approximation error 5.88e-5 vs `libm::powf`,
  10× inside the SIMD parity tolerance and ~3000× tighter than the
  perceptual delta the regression tests fence. Scalar `map_rgb` path
  keeps `libm::powf` for bit-exact reproducibility — only the SIMD strip
  kernel uses the polynomial. Bench harness:
  `examples/bt2446a_throughput.rs`, results:
  [`benchmarks/bt2446a_throughput_2026-06-20.md`](benchmarks/bt2446a_throughput_2026-06-20.md).
  Since `HdrToSdr` now defaults to `Bt2446A` on every web HDR→SDR
  conversion, this speedup is on the hot path for every served image.

### Investigation

- **Polyfit replacement for the last `pow_midp_unchecked(1/2.4)` in
  `Bt2446A::map_strip_simd`'s input transfer: tried, doesn't beat the
  baseline.** Investigation log: appended section to
  [`benchmarks/bt2446a_throughput_2026-06-20.md`](benchmarks/bt2446a_throughput_2026-06-20.md).
  Reproducible fitting tool at `examples/fit_pow_inv_24.rs` (gated on the
  `__polyfit-tools` dev feature). Summary: a sqrt-substituted piecewise
  monomial of `√x` for `x^(1/2.4)` can hit 3.4e-4 max abs error on
  `[0, 12]` as a 2-piece (eps=0.07, deg 7 + deg 10), but the BT.2446-A
  kernel's downstream `(b_p - y_p)` Cb/Cr cancellation amplifies that to
  5.2e-4 in the output — just over the 5e-4 × magnitude SIMD parity
  tolerance. A 3-piece (eps=0.05/0.30, deg 7/9/11) fixes the precision
  (2.16e-4 max output error) but regresses throughput from 258 → 203
  Mpix/s (-21%) because the 3 parallel Horner chains create register
  pressure and the deepest chain still has 11 FMA latency vs
  `pow_midp_unchecked`'s shorter dependency-chain depth on Zen 4 V4.
  Decision: keep `pow_midp_unchecked(1/2.4)` for the input transfer.

### Added

- `filmic_narkowicz` input-domain documentation + `narkowicz_input_domain_pins`
  test. Documents the published Krzysztof Narkowicz curve's expectation that
  scene mid-grey land at `x ≈ 0.18` (output ≈ 0.267) and pins the curve
  output at canonical input points (0.05 / 0.18 / 0.50 / 1.0 / 2.0) so future
  investigators don't re-derive these by hand. Also explains why
  `examples/hdr_tone_map_shootout_full.rs` scored Narkowicz at ΔE2000 ≈ 22.4
  on producer-graded SDR (the shootout's `* 2.03` input scale over-exposes
  the curve by 2x; and Narkowicz's filmic look — deep toe, saturated shoulder
  — diverges from camera-ISP-graded SDR even when correctly exposed). The
  curve itself is bit-exact per the published formula
  (`tests/cross_reference.rs::narkowicz_matches_reference`); no code change.
- `examples/hdr_tone_map_shootout_full.rs` (dev-only, `hdr-shootout` feature) — extended HDR→SDR
  shootout running on the 76-sample imazen-26 corpus subset that carries gain maps (UltraHDR JPEG +
  iPhone HEIC), against 3 source-peak measurement methods (`measure_max` / `measure_robust` /
  `measure_max_smoothed`) × 20 curve cells. CSV + bench markdown at
  `benchmarks/hdr_tone_map_shootout_full_2026-06-20.{csv,md}`. Honours the post-`ca614df0` HdrToSdr
  contract: source-normalized input fed straight into `apply_strip`. Dev-deps additions: `heic-decode`
  feature on the path-`zencodecs` dev-dep + `heic = { git = ".../heic" }` patch override (heic 0.2.0
  on crates.io is yanked).
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

- **`HdrToSdr` input normalization now matches its docstring** — the
  wrapper previously passed source-normalized input (`1.0 = source_peak_nits`,
  max ≈ 1.0) straight to `ToneMapCurve::Mobius`, which expects
  target-normalized input (`1.0 = target_peak_nits`, max = `source/target`
  ratio). Result: Möbius's knee + rolloff branch was effectively dead,
  because the input range stayed inside `[0, 1]` instead of stretching to
  `[0, peak]` where the rolloff is supposed to act. Surfaced by the HDR
  cross-curve shootout on real UltraHDR samples (commit `39737ee` — the
  validation example pre-multiplied by `peak` to work around the bug).
  Fix: `apply_strip` / `apply_rgb` now rescale source-norm input to
  target-norm before feeding Möbius (`px *= peak`), so the caller side of
  the API matches the documented contract. Existing tests updated to use
  source-normalized inputs; new `mobius_knee_actually_fires` regression
  test pins the bug.

- **`Bt2446A` — output is now linear-light, not gamma-encoded** (third
  algorithm bug against ITU-R BT.2446-1 §4, caught on the 76-sample
  imazen-26 HDR shootout where `Bt2446A` ranked dead last out of 20
  curves with median ΔE2000 ≈ 23 vs producer-graded SDR). The spec's
  §4.1 pipeline gamma-encodes `R/G/B` with `^(1/2.4)` at step 1, runs
  the tone curve in gamma + Y'Cb'Cr' domain, and emits gamma-encoded
  `R'_TMO G'_TMO B'_TMO` (Table 3's "Colour space conversion" note: the
  outputs are R'G'B' per BT.2020 Table 4 — primed = gamma). The `ToneMap`
  trait contract is linear-light in / linear-light out (matching
  `Bt2446B` / `Bt2446C` / `Bt2408`, all of which operate end-to-end in
  linear-light). Pre-fix, the gamma-encoded output was treated as
  linear by the consumer and then double-gamma-encoded into sRGB,
  pushing every pixel far too bright. Fix: apply the BT.1886 EOTF
  (`^2.4`) to each output channel at the end of `map_rgb` and the SIMD
  `bt2446a_tier` kernel — the same closing step libplacebo applies via
  `bt1886_eotf` in `tone_mapping.c:525`. New regression test
  `output_is_linear_light_not_gamma_encoded` pins mid-grey HDR 0.18 →
  linear SDR ≈ 0.37 (was 0.66 gamma) and guards the pre-fix value.
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
