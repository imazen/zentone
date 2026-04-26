# zentone ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zentone/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zentone?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/zentone?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/zentone?style=flat-square) ![MSRV](https://img.shields.io/crates/msrv/zentone?style=flat-square) ![license](https://img.shields.io/crates/l/zentone?style=flat-square)

HDR to SDR tone mapping curves in safe Rust. Classical curves (Reinhard, Hable, Narkowicz, ACES, AgX), ITU-R BT.2408 / BT.2446 EETFs, the darktable/Blender filmic spline, and an ISO 21496-1 / Apple Ultra HDR gain-map splitter.

zentone is a **curve library**, not a color pipeline. It expects linear-light f32 input and returns linear-light f32 output. Transfer-function decode/encode (PQ, HLG, sRGB), primary conversion, and ICC handling live elsewhere — `linear-srgb` for the math, `zenpixels-convert` for format negotiation. The `pipeline` module composes a few of those pieces into fused PQ/HLG → sRGB strip kernels for the common case. There's no codec dependency.

> **Active development as of April 2026.** Public APIs may rename or reorganize through the next few minor releases; anything under the `experimental` feature is explicitly unstable and may change without semver bumps. Pin minor versions and read `CHANGELOG.md` before upgrading. This is the first publish to crates.io — the in-development `&[f32]` + `channels: u8` pipeline forms were removed before release in favor of the SIMD strip-form APIs (`&[[f32; 3]]` / `&[[f32; 4]]`).

## Getting started

A per-pixel curve, no setup:

```rust
use zentone::{AgxLook, ToneMap, ToneMapCurve};

let sdr = ToneMapCurve::Agx(AgxLook::Default).map_rgb([2.5, 1.8, 0.4]);
```

A stateful, display-aware curve constructed once and applied to many rows:

```rust
use zentone::{Bt2408Tonemapper, ToneMap};

let tm = Bt2408Tonemapper::new(4000.0, 1000.0); // content peak, display peak (nits)
let mut row = vec![0.3_f32, 0.5, 0.2, 0.7, 0.1, 0.9];
tm.map_row(&mut row, 3); // 3 = RGB, 4 = RGBA (alpha preserved)
```

Fused SIMD strip pipeline — PQ EOTF → tone map → BT.2020→BT.709 → soft clip → sRGB OETF:

```rust
use zentone::{Bt2408Tonemapper, TonemapScratch, pipeline::tonemap_pq_to_srgb8_row_simd};

let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
let mut scratch = TonemapScratch::new();          // amortizes per-thread; default 4096-px chunk
let pq = vec![[0.58_f32, 0.58, 0.58]; 1024];      // PQ-encoded BT.2020 RGB
let mut srgb_out = vec![[0u8; 3]; 1024];
tonemap_pq_to_srgb8_row_simd(&mut scratch, &pq, &mut srgb_out, &tm);
```

`TonemapScratch` owns the per-chunk intermediates, caps working-set memory at `chunk_size` pixels regardless of strip length, and makes the pipelines allocation-free per call. One scratch per worker thread or video stream.

ISO 21496-1 / Ultra HDR gain-map splitter — round-trippable HDR ↔ (SDR, log2 gain):

```rust
use zentone::{HableFilmic, LumaGainMapSplitter, SplitConfig, SplitStats};

let splitter = LumaGainMapSplitter::new(HableFilmic::new(), SplitConfig::default());
let hdr = vec![0.5_f32, 1.2, 0.3]; // one interleaved RGB pixel, linear light
let mut sdr_out = vec![0.0_f32; 3];
let mut gain_out = vec![0.0_f32; 1];
let mut stats = SplitStats::default();
splitter.split_row(&hdr, &mut sdr_out, &mut gain_out, 3, &mut stats);
```

The splitter emits raw f32 log2 gain; u8 quantization and gamma encoding are the encoder's job. See the `gainmap` module docs for the full contract.

## API tiers

- **Hot path — strip / row SIMD.** `pipeline::tonemap_pq_*_row_simd`, `pipeline::tonemap_hlg_*_row_simd`, `gamut::apply_matrix_row_simd`, `gamut::soft_clip_row_simd`, `hlg::hlg_ootf_row_simd`, and the `ToneMap::map_strip_simd` trait method (with SIMD overrides on `Bt2408Tonemapper`, `Bt2446A/B/C`, and `CompiledFilmicSpline`). Use these for any non-trivial workload.
- **Per-pixel reference.** `ToneMap::map_rgb`, the named-curve scalar functions in `curves` (`reinhard_simple`, `bt2390_tonemap`, `narkowicz_aces`, …), `gamut::apply_matrix`, `gamut::soft_clip`. Suitable for one-off use, doctests, and cross-checks against external implementations. Don't put these in inner loops.
- **Stateful tonemappers.** `Bt2408Tonemapper`, `Bt2446A`, `Bt2446B`, `Bt2446C`, `CompiledFilmicSpline`. Constructed once with `(content_peak_nits, display_peak_nits)` (or a `FilmicSplineConfig`); apply via the `ToneMap` trait.
- **Gain map splitter.** `LumaGainMapSplitter`, `LumaToneMap`, `SplitConfig`, `SplitStats`, plus the curve adapters `Bt2408Yrgb`, `ExtendedReinhardLuma`, `HableFilmic`, and the `LumaFn` closure wrapper.
- **Experimental.** `experimental::AdaptiveTonemapper` (LUT fitter from an HDR/SDR pair), `experimental::StreamingTonemapper` (single-pass spatially-local tonemap), `experimental::ProfileToneCurve` (DNG camera-profile tone curve), `experimental::detect::detect_standard`. Feature-gated, semver-unstable.

## Curves

| Family | Members |
|---|---|
| Stateless `ToneMapCurve` | `Reinhard`, `ExtendedReinhard`, `ReinhardJodie`, `TunedReinhard`, `Narkowicz`, `HableFilmic`, `AcesAp1`, `Agx(AgxLook::{Default, Punchy, Golden})`, `Bt2390`, `Clamp` |
| ITU broadcast standards | `Bt2408Tonemapper` (BT.2408 Annex 5 PQ-domain Hermite, YRGB or MaxRGB), `Bt2446A` / `Bt2446B` / `Bt2446C` (BT.2446 Methods A, B, C) |
| Filmic spline | `CompiledFilmicSpline` + `FilmicSplineConfig` (darktable/Blender rational spline with toe/linear/shoulder regions and per-pixel highlight desaturation) |
| Luma curves for the splitter | `Bt2408Yrgb`, `ExtendedReinhardLuma`, `HableFilmic` (also re-exported as a stateless `ToneMapCurve` variant), and any `Bt2446{A,B,C}` / `CompiledFilmicSpline` (they implement `LumaToneMap` directly) |

Curves that need RGB→Y weights take them at construction. Use `LUMA_BT709`, `LUMA_BT2020`, or `LUMA_P3` from the crate root, picking the constant that matches the input primaries.

## Utility modules

| Module | Contents |
|---|---|
| `gamut` | Six gamut conversion matrices (BT.709 ↔ BT.2020 ↔ Display P3), hue-preserving `soft_clip`, SIMD strip forms |
| `hlg` | HLG system gamma, OOTF and inverse OOTF (spec-correct and libultrahdr-compat variants), SIMD strip forms |
| `sdr_hdr` | Reference-white scaling (100 ↔ 203 nits), OOTF gamma adjustments per BT.2408 §5.1 |
| `pipeline` | Fused PQ/HLG → tone-map → BT.709 → soft-clip strip kernels, with optional sRGB-u8 output |
| `gainmap` | `LumaGainMapSplitter`, `LumaToneMap`, `SplitConfig`, `SplitStats`, plus adapters and PQ/HLG row helpers |

## Architecture

SIMD dispatch goes through [`archmage`](https://docs.rs/archmage) and [`magetypes`](https://docs.rs/magetypes). The `#[archmage::magetypes(...)]` macro generates per-tier kernels (AVX-512 → AVX2 → SSE4.2 → NEON → WASM-SIMD → scalar) from a single source body and dispatches at runtime via CPU capability tokens. Coverage varies per kernel — the simpler curves cover all six tiers; transcendental-using kernels (AgX log2/pow, BT.2390 Hermite) ship V3+NEON+WASM128+scalar.

`#![forbid(unsafe_code)]`. `no_std + alloc` is the default-supported configuration; `std` is opt-in for ergonomics in downstream consumers. Tested on `thumbv7em-none-eabihf` for `no_std` integrity.

## Features

- `std` (default) — passes through to `linear-srgb`, `archmage`, and `magetypes`.
- `avx512` (default) — gates the AVX-512 (`v4`) magetypes tier in `archmage` and `magetypes`. Disable to fall back to AVX2 as the top tier.
- `experimental` — opt-in. Adds `AdaptiveTonemapper`, `StreamingTonemapper`, `ProfileToneCurve`, and `detect_standard`. Light test coverage; APIs may change without semver bumps until stabilized.

## Compatibility

- **MSRV:** Rust 1.89, 2024 edition.
- **CI:** Linux x86_64, Windows ARM64 (`windows-11-arm`), macOS Intel + Apple Silicon, i686-unknown-linux-gnu (via `cross`), `wasm32-wasip1` (unit tests under wasmtime), `wasm32-unknown-unknown` (build check), `thumbv7em-none-eabihf` (no_std build check).

## Reference parity

Curves that claim a standard name are validated against their reference implementation. Golden CSVs from standalone C++ extractions live under `reference-checks/golden/`; property tests in `tests/exhaustive_properties.rs` verify monotonicity, finite output, alpha preservation, and channel-count consistency across a 14×14×14 grid for all stateless curve configurations.

## Limitations

- **No transfer-function support beyond what the pipelines need.** sRGB / PQ / HLG decode and encode live in [`linear-srgb`](https://lib.rs/crates/linear-srgb). zentone's `pipeline` module composes them for the PQ→sRGB and HLG→sRGB cases; for other combinations, do the linearization yourself and feed linear-light f32 in.
- **No perceptual gamut mapping.** The pipeline applies a hue-preserving `soft_clip` after the BT.2020→BT.709 matrix, which preserves channel ratios for out-of-gamut highlights. Hellwig 2022 JMh / ACES 2.0 perceptual compression is not implemented ([#14](https://github.com/imazen/zentone/issues/14)).
- **Gain map encode/decode container handling.** zentone produces and consumes raw f32 log2 gain; ISO 21496-1 / Ultra HDR container math (MPF, XMP, gamma encoding, u8 quantization) lives in [`ultrahdr-core`](https://lib.rs/crates/ultrahdr-core).
- **No pixel-format conversion.** Inputs are `&mut [f32]` or packed `&[[f32; 3]]` / `&[[f32; 4]]`. For u8/u16/planar buffers, convert first via `zenpixels-convert` or your own pipeline.

## Links

- Documentation: <https://docs.rs/zentone>
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Repository: <https://github.com/imazen/zentone>

## License

`AGPL-3.0-only OR LicenseRef-Imazen-Commercial`. Use under AGPL-3.0 (see [`LICENSE-AGPL3`](LICENSE-AGPL3)) or a commercial license from Imazen (see [`LICENSE-COMMERCIAL`](LICENSE-COMMERCIAL)).
