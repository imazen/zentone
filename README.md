# zentone

HDR to SDR tone mapping. Classical curves, BT.2408 PQ-domain EETF, darktable/Blender filmic spline, and experimental adaptive and streaming tonemappers.

`no_std + alloc`. Zero unsafe. Zero allocation in hot paths.

Every curve that claims a standard name has been numerically validated against its reference implementation.

## Usage

All tonemappers implement the `ToneMap` trait:

```rust
use zentone::{ToneMap, ToneMapCurve, AgxLook, LUMA_BT709};

// Pick a curve — luma-carrying variants bake their coefficients at construction.
let curve = ToneMapCurve::Agx(AgxLook::Punchy);

// In-place on a row of interleaved f32 pixels.
// channels = 3 (RGB) or 4 (RGBA, alpha preserved).
let mut row = vec![2.5_f32, 1.8, 0.4, 0.8, 2.0, 0.1];
curve.map_row(&mut row, 3);
```

Stateful tonemappers (BT.2408, filmic spline) construct once and apply many times:

```rust
use zentone::{Bt2408Tonemapper, ToneMap};

let tm = Bt2408Tonemapper::new(4000.0, 1000.0); // content peak, display peak (nits)
let mut row = vec![0.3_f32, 0.5, 0.2, 0.7, 0.1, 0.9];
tm.map_row(&mut row, 3);
```

## Reference parity

| Curve | Reference | Max error |
|---|---|---|
| `reinhard_extended` | libultrahdr `ReinhardMap` | 0 (exact) |
| `Bt2408Tonemapper` | libplacebo `bt2390()` (PQ domain) | 4.5e-5 relative |
| `bt2390_tonemap` | libplacebo scene-linear | 0 (exact) |
| `CompiledFilmicSpline` | darktable `filmicrgb.c` V3 | 0 eval, 1 ULP RGB |
| `agx_tonemap` | gainforge / Blender polynomial | Endpoint-verified |
| PQ OETF/EOTF | ST.2084 formula (f64) | < 1e-5 |
| HLG OETF/EOTF | BT.2100 formula (f64) | < 1e-5 |

Golden CSV files from standalone C++ extractions of each reference are committed under `reference-checks/golden/`.

## Curves

**Stable (default feature):**
- `ToneMapCurve` enum: Reinhard (simple / extended / Jodie), Narkowicz filmic, Hable filmic (Hable), ACES AP1, AgX (Default / Punchy / Golden), BT.2390 EETF, Clamp
- `Bt2408Tonemapper` — ITU-R BT.2408 Annex 5 EETF (PQ-domain Hermite, YRGB application space)
- `CompiledFilmicSpline` — darktable-style filmic (rational spline with latitude / balance / saturation)

**Experimental (`experimental` feature):**
- `AdaptiveTonemapper` — fits a luminance or per-channel LUT from an HDR/SDR pair
- `StreamingTonemapper` — spatially-local, single-pass, bounded-memory, pull API
- `ProfileToneCurve` — DNG camera profile tone curve with per-channel and luminance-preserving views

## Prior art

[gainforge](https://github.com/awxkee/gainforge) (Radzivon Bartoshyk, BSD-3-Clause) independently covers similar classical-curve territory with multi-bit-depth entry points and a moxcms-based color pipeline. Our implementations were written from the same public specs in parallel. zentone differs in its BT.2408 EETF, adaptive/streaming primitives, zen-ecosystem integration, and reference-validated golden-file test suite.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See `LICENSE-AGPL3` and `LICENSE-COMMERCIAL`.
