<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zentone

HDR to SDR tone mapping curves in safe Rust. Classical curves (Hable, Narkowicz, ACES, AgX, Möbius), ITU-R BT.2408 / BT.2446 Methods B and C, the darktable/Blender filmic spline, and an ISO 21496-1 / Apple Ultra HDR gain-map splitter. The production-best HDR→SDR curve from the 2026-06-22 audited shootout — BT.2446 Method A — graduated to [`zenpixels-convert`](https://lib.rs/crates/zenpixels-convert), where it composes into a one-call `HdrToSdr` pipeline with primary conversion and OKLch soft compression; the deprecated Reinhard family in `curves` is queued for removal in the next breaking release.

zentone is a **curve library**, not a color pipeline. It expects linear-light f32 input and returns linear-light f32 output. Transfer-function decode/encode (PQ, HLG, sRGB), primary conversion, and ICC handling live elsewhere — `linear-srgb` for the math, `zenpixels-convert` for format negotiation. The `pipeline` module composes a few of those pieces into fused PQ/HLG → sRGB strip kernels for the common case. There's no codec dependency.

> **Active development as of June 2026.** Public APIs may rename or reorganize through the next few minor releases; anything under the `experimental` feature is explicitly unstable and may change without semver bumps. Pin minor versions and read `CHANGELOG.md` before upgrading. The 0.2.0 release relocated `Bt2446A` (the production-best HDR→SDR curve per the 2026-06-22 shootout) to [`zenpixels-convert`](https://lib.rs/crates/zenpixels-convert), removed the bundled `HdrToSdr` wrapper in favor of the convert crate's richer pipeline (primary conversion + OKLch soft compression), and deprecated the channel-independent Reinhard family.

## Quick start

```toml
[dependencies]
zentone = "0.2"
```

Or `cargo add zentone`. `std` and `avx512` are on by default; build with `--no-default-features` for `no_std`, and add `experimental` for the adaptive/streaming tonemappers.

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

`map_row` rewrites interleaved **linear-light** f32 in place to linear SDR. `channels` is a `u8` that must be 3 or 4 (any other value panics), and the row length must be a multiple of it. Output is mostly `[0, 1]`, though some unclamped curves slightly exceed it — the sRGB OETF and 8-bit quantization live in [`linear-srgb`](https://lib.rs/crates/linear-srgb), or use the fused `pipeline` kernels below for direct sRGB-`u8` output.

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

- **Hot path — strip / row SIMD.** `pipeline::tonemap_pq_*_row_simd`, `pipeline::tonemap_hlg_*_row_simd`, `gamut::apply_matrix_row_simd`, `gamut::soft_clip_row_simd`, `hlg::hlg_ootf_row_simd`, and the `ToneMap::map_strip_simd` trait method (with SIMD overrides on `Bt2408Tonemapper`, `Bt2446B`, `Bt2446C`, and `CompiledFilmicSpline`). Use these for any non-trivial workload.
- **Per-pixel reference.** `ToneMap::map_rgb`, the named-curve scalar functions in `curves` (`filmic_narkowicz`, `hable_filmic`, `aces_ap1`, `agx_tonemap`, `bt2390_tonemap`, …), `gamut::apply_matrix`, `gamut::soft_clip`. Suitable for one-off use, doctests, and cross-checks against external implementations. Don't put these in inner loops.
- **Stateful tonemappers.** `Bt2408Tonemapper`, `Bt2446B`, `Bt2446C`, `CompiledFilmicSpline`. Constructed once with `(content_peak_nits, display_peak_nits)` (or a `FilmicSplineConfig`); apply via the `ToneMap` trait. For BT.2446 Method A, use [`zenpixels_convert::hdr::Bt2446A`](https://lib.rs/crates/zenpixels-convert) — it graduated to the convert crate where it composes with CLL measurement and primary conversion.
- **Gain map splitter.** `LumaGainMapSplitter`, `LumaToneMap`, `SplitConfig`, `SplitStats`, plus the curve adapters `Bt2408Yrgb`, `ExtendedReinhardLuma`, `HableFilmic`, and the `LumaFn` closure wrapper.
- **Experimental.** `experimental::AdaptiveTonemapper` (LUT fitter from an HDR/SDR pair), `experimental::StreamingTonemapper` (single-pass spatially-local tonemap), `experimental::ProfileToneCurve` (DNG camera-profile tone curve), `experimental::detect::detect_standard`. Feature-gated, semver-unstable.

## Curves

| Family | Members |
|---|---|
| Stateless `ToneMapCurve` | `Narkowicz`, `HableFilmic`, `AcesAp1`, `Agx(AgxLook::{Default, Punchy, Golden})`, `Bt2390 { source_peak, target_peak }`, `Mobius { source_peak, knee }` (libplacebo's `mobius()`), `Clamp`. (The channel-independent `Reinhard`, `ExtendedReinhard`, `ReinhardJodie`, and `TunedReinhard` variants stay public for one more release but are queued for removal — they dispatch through the `#[deprecated]` + `#[doc(hidden)]` free functions `curves::reinhard_simple` / `reinhard_extended` / `reinhard_jodie`, superseded by `zenpixels_convert::hdr::Bt2446A` per the 2026-06-22 shootout.) |
| ITU broadcast standards | `Bt2408Tonemapper` (BT.2408 Annex 5 PQ-domain Hermite, YRGB or MaxRGB), `Bt2446B` / `Bt2446C` (BT.2446 Methods B and C). Method A lives in [`zenpixels_convert::hdr::Bt2446A`](https://lib.rs/crates/zenpixels-convert) — the canonical home for the production-best HDR→SDR curve. |
| Filmic spline | `CompiledFilmicSpline` + `FilmicSplineConfig` (darktable/Blender rational spline with toe/linear/shoulder regions and per-pixel highlight desaturation) |
| Luma curves for the splitter | `Bt2408Yrgb`, `ExtendedReinhardLuma`, `HableFilmic` (also re-exported as a stateless `ToneMapCurve` variant), and any `Bt2446{B,C}` / `CompiledFilmicSpline` (they implement `LumaToneMap` directly) |

Curves that need RGB→Y weights take them at construction. Use `LUMA_BT709`, `LUMA_BT2020`, or `LUMA_P3` from the crate root, picking the constant that matches the input primaries.

## Utility modules

| Module | Contents |
|---|---|
| `gamut` | Six gamut conversion matrices (BT.709 ↔ BT.2020 ↔ Display P3), hue-preserving `soft_clip`, SIMD strip forms |
| `hlg` | HLG system gamma, OOTF and inverse OOTF (spec-correct and libultrahdr-compat variants), SIMD strip forms |
| `sdr_hdr` | Reference-white scaling (100 ↔ 203 nits), OOTF gamma adjustments per BT.2408 §5.1 |
| `pipeline` | Fused PQ/HLG → tone-map → BT.709 → soft-clip strip kernels, with optional sRGB-u8 output |
| `gainmap` | `LumaGainMapSplitter`, `LumaToneMap`, `SplitConfig`, `SplitStats`, plus adapters and PQ/HLG row helpers |

The six `gamut` matrices are public consts in the `gamut` module — `gamut::BT2020_TO_BT709`, `BT709_TO_BT2020`, `P3_TO_BT709`, `BT709_TO_P3`, `BT2020_TO_P3`, `P3_TO_BT2020` — applied with `gamut::apply_matrix` (one `[f32; 3]`), `gamut::apply_matrix_row` (interleaved, `channels: u8`), or `gamut::apply_matrix_row_simd` / `apply_matrix_row_simd_rgba` (a `&mut [[f32; 3]]` / `&mut [[f32; 4]]` strip). The common HDR step is `BT2020_TO_BT709`.

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
- Changelog: [`CHANGELOG.md`](https://github.com/imazen/zentone/blob/main/CHANGELOG.md)
- Repository: <https://github.com/imazen/zentone>


## License

`AGPL-3.0-only OR LicenseRef-Imazen-Commercial`. Use under AGPL-3.0 (see [`LICENSE-AGPL3`](https://github.com/imazen/zentone/blob/main/LICENSE-AGPL3)) or a commercial license from Imazen (see [`LICENSE-COMMERCIAL`](https://github.com/imazen/zentone/blob/main/LICENSE-COMMERCIAL)).

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · **zentone** |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
