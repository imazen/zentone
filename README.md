# zentone

HDR → SDR tone mapping: classical curves, BT.2408 PQ-domain Hermite spline, darktable/Blender filmic spline, and experimental adaptive and streaming tonemappers.

`no_std + alloc` compatible. Zero unsafe.

## Scope

**Stable (default feature):**
- `ToneMapCurve` enum dispatch over classical curves:
  - Reinhard (simple / extended / Jodie / tuned)
  - Uncharted 2 / Hable filmic
  - Narkowicz filmic (ACES-inspired S-curve)
  - ACES AP1 (RRT+ODT fit)
  - AgX (Blender) with Default / Punchy / Golden looks
  - BT.2390 EETF
  - Clamp
- `Bt2408Tonemapper` — ITU-R BT.2408 PQ-domain Hermite spline (content/display peak nits)
- `CompiledFilmicSpline` — darktable/Blender Filmic with latitude / balance / saturation parameters
- Row-oriented API: `tonemap_row(&curve, row, channels, luma_coeffs)`

**Experimental (`experimental` feature, lightly tested):**
- `AdaptiveTonemapper` — fits a luminance or per-channel LUT from an HDR/SDR pair, preserving the original artistic intent for re-encodes
- `StreamingTonemapper` — single-pass spatially-local tonemapper with a lookahead row buffer and grid-based adaptation (~6 MB for 4K vs ~130 MB full frame)
- `ProfileToneCurve` — DNG camera profile tone curve from 257 (x,y) control points → 4096-entry LUT, per-channel or luminance-preserving

These APIs have light test coverage and may change without semver bumps until stabilized.

## Example

```rust,ignore
use zentone::{ToneMapCurve, AgxLook, tonemap_row};

let curve = ToneMapCurve::Agx(AgxLook::Punchy);
let luma_coeffs = [0.2126, 0.7152, 0.0722]; // BT.709

let mut row = vec![2.5_f32, 1.8, 0.4, /* ...more HDR pixels... */];
tonemap_row(&curve, &mut row, 3, luma_coeffs);
// `row` now holds SDR-range linear RGB, ready for an OETF (sRGB, BT.709, ...)
```

## Prior art

`gainforge` (Radzivon Bartoshyk, BSD-3-Clause) independently covers similar classical-curve territory with multi-bit-depth entry points and a moxcms-based color pipeline. Our implementations were written from the same public specs (Hable GDC talk, ITU-R BT.2408, Blender AgX source) in parallel; the shared code is small and derives from identical references. zentone differs in its unique adaptive / streaming / DNG-profile primitives, zen-ecosystem integration points, and future archmage SIMD dispatch.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial — dual-licensed. See `LICENSE-AGPL3` and `LICENSE-COMMERCIAL`.
