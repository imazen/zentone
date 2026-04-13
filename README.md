# zentone ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zentone/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zentone?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/zentone?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/zentone?style=flat-square) ![MSRV](https://img.shields.io/crates/msrv/zentone?style=flat-square) ![license](https://img.shields.io/crates/l/zentone?style=flat-square)

HDR to SDR tone mapping in safe Rust. Classical curves, ITU-R BT.2408/BT.2446 standards, darktable filmic spline, and experimental adaptive/streaming tonemappers.

`no_std + alloc`. `#![forbid(unsafe_code)]`. Zero allocation in hot paths. SIMD-accelerated on x86-64 (AVX2+FMA) with scalar fallback everywhere else.

## Quick start

```rust
use zentone::{ToneMap, ToneMapCurve, AgxLook};

// Stateless — pick a curve, call map_row on interleaved f32 pixels.
let mut row = vec![2.5_f32, 1.8, 0.4, 0.8, 2.0, 0.1];
ToneMapCurve::Narkowicz.map_row(&mut row, 3); // channels: 3=RGB, 4=RGBA (alpha preserved)

// Per-pixel is also available.
let out = ToneMapCurve::Agx(AgxLook::Default).map_rgb([2.5, 1.8, 0.4]);
```

Stateful tonemappers construct once and apply to many rows:

```rust
use zentone::{Bt2408Tonemapper, ToneMap};

let tm = Bt2408Tonemapper::new(4000.0, 1000.0); // content peak, display peak (nits)
let mut row = vec![0.3_f32, 0.5, 0.2, 0.7, 0.1, 0.9];
tm.map_row(&mut row, 3);
```

Copy path for separate source and destination buffers:

```rust
use zentone::{ToneMap, ToneMapCurve};

let src = [2.5_f32, 1.8, 0.4, 0.8, 2.0, 0.1];
let mut dst = [0.0_f32; 6];
ToneMapCurve::HableFilmic.map_into(&src, &mut dst, 3);
```

## Tonemappers

### Classical curves (`ToneMapCurve`)

Stateless enum with unified `ToneMap` dispatch. Luma-carrying variants bake coefficients at construction time.

| Variant | Algorithm | Notes |
|---------|-----------|-------|
| `Reinhard` | `x / (1 + x)` per channel | Simplest; tends toward gray at high input |
| `ExtendedReinhard` | Luminance-preserving with white point | `l_max` controls highlight rolloff |
| `ReinhardJodie` | Per-channel + luminance blend | Better color retention than simple Reinhard |
| `TunedReinhard` | Display-aware with content/display peak | Derived from nit values with 203-nit reference white |
| `Narkowicz` | ACES-inspired rational polynomial | Fast S-curve; good default for games/realtime |
| `HableFilmic` | John Hable (GDC 2010) | Exposure bias 2.0, white point 11.2 |
| `AcesAp1` | ACES AP1 RRT+ODT (Narkowicz fit) | Cross-channel matrix; slight negative at near-black |
| `Agx(look)` | Blender AgX with log2 encoding | Default, Punchy (1.4x sat), Golden (warm) |
| `Bt2390` | ITU-R BT.2390 Hermite EETF | Scene-linear in, scene-linear out; needs peak params |
| `Clamp` | `min(x, 1.0)` | Baseline for comparison |

### ITU-R standards

| Type | Standard | Description |
|------|----------|-------------|
| `Bt2408Tonemapper` | BT.2408 Annex 5 | PQ-domain Hermite spline EETF with YRGB or MaxRGB application space |
| `Bt2446A` | BT.2446 Method A | Perceptual linearization with psychophysically-verified knee |
| `Bt2446B` | BT.2446 Method B | Broadcast-oriented; 291 cd/m² ceiling |
| `Bt2446C` | BT.2446 Method C | Parametric piecewise with exact algebraic inverse |

### Filmic spline

`CompiledFilmicSpline` implements the darktable/Blender-style filmic pipeline: rational spline with configurable latitude, contrast, balance, saturation, and output power. Validated against darktable `filmicrgb.c` V3 (0 eval error, 1 ULP in RGB).

### Experimental (`experimental` feature)

| Type | Description |
|------|-------------|
| `AdaptiveTonemapper` | Fits a luminance or per-channel LUT from an HDR/SDR reference pair |
| `StreamingTonemapper` | Spatially-local, single-pass, bounded-memory pull API |
| `ProfileToneCurve` | DNG camera-profile tone curve with per-channel and luminance-preserving views |
| `detect_standard()` | Identifies which standard curve was applied to a fitted LUT |

Lightly tested; API may change without semver bumps until stabilized.

## Utility modules

| Module | Contents |
|--------|----------|
| `gamut` | 6 gamut conversion matrices (BT.709, BT.2020, Display P3) with `apply_matrix` / `apply_matrix_row` |
| `hlg` | HLG system gamma, OOTF, inverse OOTF, `hlg_to_display` (raw HLG OETF/EOTF in `linear-srgb`) |
| `sdr_hdr` | Reference-white scaling (100↔203 nits), OOTF gamma adjustment |
| `pipeline` | One-call PQ→linear-sRGB and HLG→linear-sRGB with pluggable `&dyn ToneMap` |

## Performance

Single-threaded, 3840-pixel RGB row, AMD Ryzen 9 7950X. No `-C target-cpu=native`; runtime SIMD dispatch via `archmage::incant!`. Full results in `benchmarks/`.

| Curve | map_row (linear f32) | Method |
|-------|---------------------|--------|
| Reinhard | 0.8 µs | AVX2+FMA f32x8 |
| Narkowicz | 1.1 µs | AVX2+FMA rational polynomial |
| HableFilmic | 1.4 µs | AVX2+FMA rational polynomial |
| TunedReinhard | 2.6 µs | SOA 8-pixel luma dot product |
| ExtendedReinhard | 3.0 µs | SOA 8-pixel luma |
| ReinhardJodie | 3.2 µs | SOA 8-pixel luma + blend |
| AcesAp1 | 3.8 µs | Per-pixel under `#[arcane]` (cross-channel matrix) |
| Agx(Default) | 10.2 µs | SOA 8-pixel: vectorized `log2_midp` + polynomial + matrices |
| Agx(Punchy) | 12.5 µs | SOA 8-pixel: saturation blend (pow skipped) |
| Agx(Golden) | 21.4 µs | SOA 8-pixel: `pow_midp(0.8)` + saturation blend |

For a **full 8K frame** (7680x4320 = 33.2 Mpx), Reinhard takes 26 ms, AgX Default takes 93 ms; see `benchmarks/` for full results.

### Comparison with gainforge

Full sRGB→tonemap→sRGB pipeline (including gamma decode/encode), 3840 pixels:

| Curve | gainforge 0.4.1 | zentone | Speedup |
|-------|----------------|---------|---------|
| Reinhard | 13.0 µs | 11.4 µs | 1.1x |
| Hable | 16.1 µs | 11.8 µs | 1.4x |
| ACES | 17.9 µs | 14.0 µs | 1.3x |
| AgX Default | 117.4 µs | 20.9 µs | 5.6x |

Gainforge uses LUT-based gamma; zentone uses SIMD batch `srgb_to_linear_slice`/`linear_to_srgb_slice` from `linear-srgb`. Both measured on the same hardware with `cargo bench`, no target-cpu flags. Benchmark code in `benches/tonemap_bench.rs`.

## Reference parity

Every curve that claims a standard name has been numerically validated against its reference implementation. Golden CSV files from standalone C++ extractions are committed under `reference-checks/golden/`.

| Curve | Reference | Max error | Test |
|-------|-----------|-----------|------|
| Reinhard | libultrahdr `ReinhardMap` | 0 (exact) | `reference_parity.rs` |
| Bt2408 | libplacebo `bt2390()` PQ domain | 4.5e-5 relative | `reference_parity.rs` |
| BT.2390 | libplacebo scene-linear | 0 (exact) | `reference_parity.rs` |
| FilmicSpline | darktable `filmicrgb.c` V3 | 0 eval, 1 ULP RGB | `reference_parity.rs` |
| Reinhard / Hable / Narkowicz | gainforge formula extraction | 1e-6 to 1e-7 | `exhaustive_properties.rs` |
| ACES AP1 | gainforge matrix extraction | 1e-5 on 2744-pixel grid | `exhaustive_properties.rs` |
| AgX contrast | gainforge / Blender polynomial | Coefficient-identical | `cross_reference.rs` |
| PQ OETF/EOTF | ST.2084 formula (f64 reference) | < 1e-5 | `transfer_functions.rs` |
| HLG OETF/EOTF | BT.2100 formula (f64 reference) | < 1e-5 | `transfer_functions.rs` |

Property tests in `exhaustive_properties.rs` verify monotonicity, finite output, SDR-in→SDR-out, alpha preservation, and channel-count consistency across a 14x14x14 = 2744-pixel grid for all 17 tonemapper configurations.

## Limitations

zentone operates on linear-light f32 pixel data. It does not handle:

- **Gamma decode/encode.** Use [`linear-srgb`](https://lib.rs/crates/linear-srgb) for sRGB, or `zentone::pipeline` for one-call PQ/HLG→sRGB conversion.
- **Gamut mapping.** The `gamut` module provides conversion matrices but no perceptual gamut clipping. Use a CMS for out-of-gamut handling.
- **Gain map application.** Gain map math (ISO 21496-1 / Ultra HDR) lives in [`ultrahdr-core`](https://lib.rs/crates/ultrahdr-core), not here.
- **Pixel format conversion.** zentone expects `&mut [f32]`; for u8/u16/planar buffers, convert first via `zenpixels-convert` or your own pipeline.
- **NEON/WASM SIMD.** Current SIMD kernels target x86-64 AVX2+FMA only. NEON and WASM128 dispatch exists but falls through to scalar. Real NEON/WASM kernels are planned.

The experimental API (`AdaptiveTonemapper`, `StreamingTonemapper`, `ProfileToneCurve`) has lighter test coverage and may change without semver bumps.

## Prior art

[gainforge](https://github.com/awxkee/gainforge) (Radzivon Bartoshyk, BSD-3-Clause) covers similar classical-curve territory with multi-bit-depth entry points and a moxcms-based color pipeline. Both crates implement tone curves from the same public specifications. zentone adds BT.2408/BT.2446 EETF standards, AVX2 SIMD kernels, adaptive/streaming primitives, and a golden-file validation suite against C++ reference extractions.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See `LICENSE-AGPL3` and `LICENSE-COMMERCIAL`.
