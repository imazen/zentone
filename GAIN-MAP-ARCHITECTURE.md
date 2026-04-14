# Gain Map Architecture: What Goes Where

Status after zentone commits `4b2b8c3` and `3205c61`. Covers the three-crate
layering, what's done, what remains, and what ultrahdr-core should delegate.

## The three layers

| Layer | Crate | Owns | Does NOT own |
|---|---|---|---|
| **Wire format** | zencodec | `GainMapParams` (ISO 21496-1 fields), XMP/binary serialization | Tone curves, pixel math, transfer functions, u8 quantization |
| **Pixel math** | zentone | `LumaGainMapSplitter`, tone curves, `LumaToneMap` trait, PQ/HLG linearization helpers, pre-desaturation, gamut matrices | Wire-format quantization, container framing, downsampling/upsampling |
| **Resampling** | zenresize | Gain map downsampling (encode) and upsampling (decode) with proper filter kernels | Gain math, wire format |
| **Codec integration** | ultrahdr-core | `RawImage` pixel-format linearization, JPEG/MPF/XMP container, streaming ring-buffer encoder/decoder, `display_boost` weight, u8 gain quantization (`GainMapLut`, `compute_and_encode_gain`) | Generic tone curves, ISO 21496-1 field semantics |

## What zentone provides today

### `experimental::gain_map` module

**Splitter core:**
- `LumaToneMap` trait — scalar Y curve contract (monotonic, near-zero at black)
- `LumaGainMapSplitter<T>` — stateless, row-at-a-time, zero allocation in hot path
  - `split_row(hdr, sdr_out, gain_out, channels, stats)` — linear f32 → (linear SDR f32, log2 gain f32)
  - `apply_row(sdr, gain, hdr_out, channels)` — inverse
- `SplitConfig` — luma weights, offsets (1/64 default), log2 clamp range (-4..6), pre-desaturation alpha
- `SplitStats` — observed min/max log2 gain, clipped pixel count

**Transfer function helpers (convenience; hot paths should pre-linearize with reused buffers):**
- `split_pq_row` / `apply_pq_row` — PQ wire → linearize → split → (SDR, gain)
- `split_hlg_row` / `apply_hlg_row` — HLG wire → OOTF → linearize → split → (SDR, gain)
- `pq_to_normalized_linear_row` / `normalized_linear_to_pq_row`
- `hlg_to_normalized_linear_row` / `normalized_linear_to_hlg_row`

**Pre-desaturation:**
- `SplitConfig::pre_desaturate` — BT.2446-C–style crosstalk matrix (0.0–0.33)
- Applied before luma rescale, inverse after gain computation
- Reduces `clipped_sdr_pixels` on saturated highlights
- Transparent to grayscale round-trip

**Qualifying `LumaToneMap` impls:**

| Type | Notes |
|---|---|
| `Bt2446C` | Exact algebraic inverse |
| `Bt2446A` | Shadow-lifter; BT.2020 luma |
| `Bt2446B` | Simplified HLG; broadcast |
| `Bt2408Yrgb` | Newtype enforcing Yrgb mode; rejects MaxRgb at compile time |
| `ExtendedReinhardLuma` | Luminance-preserving Reinhard with white point |
| `CompiledFilmicSpline` | darktable filmic; grayscale-invariant (ratios=[1,1,1] neutralizes desaturation) |
| `LumaFn<F>` | Ad-hoc closure wrapper |

**Not qualifying** (per-channel or matrix-based — must never impl `LumaToneMap`):
Reinhard, ReinhardJodie, Narkowicz, HableFilmic, AcesAp1, Agx(*), Clamp, Bt2408::MaxRgb.

**Wire-format alignment:**
- Field names match `zencodec::GainMapParams`: `base_offset`, `alternate_offset`
- Decode formula: `HDR_i = (SDR_i + base_offset) · 2^g − alternate_offset`
- Splitter emits raw f32 log2 gain; u8 quantization is ultrahdr-core's job
- `SplitConfig` + `SplitStats` → `GainMapParams` is a 10-line downstream mapping

**Constants:**
- `LUMA_BT709`, `LUMA_BT2020`, `LUMA_P3`

**Test coverage (19 tests):**
Well-behavedness per curve, exact grayscale round-trip (all 6 curves), isolated
chromatic clipping, RGBA passthrough, extreme highlight clamping, PQ round-trip,
HLG round-trip, PQ/HLG helper symmetry, pre-desaturation clip reduction,
pre-desaturation grayscale exactness.

## What zencodec provides (and its scope boundary)

zencodec is metadata-types-only. It owns `GainMapParams`, `GainMapChannel`,
`Fraction`/`UFraction`, and the ISO 21496-1 binary + XMP serialization. It does
NOT take on computational code (quantization, LUTs, pixel math).

### `build_gainmap_params` helper (optional, small)

```rust
pub fn build_gainmap_params(
    base_offset: f64,
    alternate_offset: f64,
    observed_min_log2: f64,
    observed_max_log2: f64,
    gamma: f64,
    base_hdr_headroom: f64,
    alternate_hdr_headroom: f64,
) -> GainMapParams;
```

Convenience for zentone callers who have `SplitConfig` + `SplitStats` and want
to produce metadata without constructing `GainMapParams` field by field. This is
pure struct construction — no math — so it fits zencodec's scope.

### u8 gain quantization stays in ultrahdr-core

`compute_and_encode_gain` (encode) and `GainMapLut` (decode LUT) are codec
integration code — they bridge zentone's f32 log2 gains with the 8-bit JPEG
gain map image. They're parameterized by `GainMapParams` but depend on
ultrahdr-core's `GainMapConfig` (min/max boost, gamma) and `display_boost`
weight calculation. They stay where they are.

## What ultrahdr-core should delegate

### Encode path

**Current** (`compute_gainmap`, ~200 LOC + streaming encoder ~500 LOC):
```
RawImage → get_linear_rgb() → rgb_to_luminance() → compute_and_encode_gain() → GainMap u8
```

**After delegation:**
```
RawImage → [linearize per PixelFormat]
         → zentone::pq_to_normalized_linear_row (or hlg_, or already linear)
         → zentone::LumaGainMapSplitter::split_row → (SDR f32, gain f32)
         → zenresize::downsample(gain, scale_factor)     ← gain map shrink
         → ultrahdr_core::compute_and_encode_gain()       ← u8 quantize (stays here)
         → GainMap { data: Vec<u8>, width, height, channels: 1 }
```

**ultrahdr-core keeps:**
- `RawImage` pixel-format dispatch (`get_linear_rgb` for Rgba8/Rgba32F/etc.)
- `GainMapConfig` (scale_factor, min/max boost, gamma — user-facing knobs)
- `compute_gainmap()` as the orchestrator — calls into zentone + zenresize
- `compute_and_encode_gain()` — u8 quantization from f32 log2 gain
- `StreamEncoder` ring-buffer machinery — feeds rows to the splitter

**ultrahdr-core deletes:**
- `compute_luminance_gainmap()` inner loop (~50 LOC) → zentone split_row
- `compute_multichannel_gainmap()` inner loop (~50 LOC) → stays or becomes zentone multi-channel later
- `rgb_to_luminance()` for gain computation → zentone does this internally via `SplitConfig::luma_weights`
- Center-pixel block sampling (~10 LOC) → zenresize handles downsampling properly

### Decode path

**Current** (`apply_gainmap`, ~250 LOC + streaming decoder ~400 LOC):
```
GainMap u8 → GainMapLut::new() → sample_gainmap_row_lut() → apply_gain_row → RawImage
```

**After delegation:**
```
GainMap u8 → ultrahdr_core::GainMapLut::lookup()          ← u8 → f32 gain (stays here)
           → zenresize::upsample(gain, scale_factor)       ← gain map expand
           → zentone::LumaGainMapSplitter::apply_row        ← (SDR + gain → HDR)
           → [encode to output PixelFormat]                  ← RawImage
```

**ultrahdr-core keeps:**
- `apply_gainmap()` as the orchestrator
- `GainMapLut` — u8 → f32 decode LUT (stays here, not zencodec)
- `HdrOutputFormat` enum + output pixel-format encoding
- `read_sdr_row_linear()` — linearization from u8/u16/f32 pixel formats
- `StreamDecoder` / `RowDecoder` ring-buffer machinery
- `calculate_weight()` — display_boost → weight mapping
- `apply_simd.rs` SIMD gain row kernel

**ultrahdr-core deletes (eventually):**
- `apply_gain_row_presampled` scalar loop → zentone `apply_row` (once zentone has SIMD)
- `sample_gainmap_row_lut()` bilinear sampling → zenresize upsample

### Estimated impact

| File | Current LOC | After delegation | Reduction |
|---|---|---|---|
| `compute.rs` | 824 | ~400 (orchestration + multi-channel + tests) | ~50% |
| `apply.rs` | 1072 | ~700 (orchestration + pixel-format adapters + tests) | ~35% |
| `apply_simd.rs` | 404 | 404 (keep until zentone SIMD) | 0% |
| `streaming.rs` | 1682 | ~1400 (ring-buffer stays, inner loops call zentone) | ~17% |
| **Total** | 3996 | ~2900 | ~27% |

The remaining ~2900 LOC is genuinely Ultra HDR–specific: pixel-format dispatch,
JPEG/MPF/XMP container, streaming orchestration, display_boost weight, and the
multi-channel gain map path.

## What zentone does NOT own

These stay out of zentone:

- **u8 quantization** → ultrahdr-core (codec integration, parameterized by `GainMapParams` + `GainMapConfig`)
- **Gain map downsampling/upsampling** → zenresize (proper filter kernels)
- **Metadata POD** → zencodec's `GainMapParams` is the single source of truth
- **Container framing** (JPEG, MPF, XMP) → ultrahdr-core
- **Pixel-format linearization** (u8/u16/f32/1010102 → linear f32) → ultrahdr-core + zenpixels-convert
- **display_boost weight** → ultrahdr-core (viewer concern)
- **Perceptual gamut mapping** → zentone issue #14, but not part of the gain map splitter

## Remaining zentone work (lower priority)

| Item | Description | Effort |
|---|---|---|
| SIMD split/apply | SOA log2/exp2 + crosstalk under `#[arcane]` | Medium |
| Benchmarks | `benches/gain_map_bench.rs`, 3840-pixel row throughput | Small |
| Multi-channel RGB gain map | Per-channel gain instead of scalar luma | Large |
| Boost estimation | `recommend_config(hdr_rows) → SplitConfig` from luminance histogram | Small |
| Golden-file parity test | Feed identical HDR/SDR to zentone + ultrahdr-core, compare gain maps | Medium (lives in ultrahdr-core tests) |

## Recommended next steps

1. **ultrahdr-core**: Wire `compute_luminance_gainmap` to zentone's splitter. Keep orchestration + `compute_and_encode_gain` for u8 packing. Delete inner-loop gain math.
2. **ultrahdr-core**: Wire `apply_gainmap` decode path to zentone's `apply_row`. Keep `GainMapLut` + pixel-format adapters.
3. **zencodec** (optional): Add `build_gainmap_params` convenience helper for zentone callers.
4. **zentone**: SIMD kernels for split/apply once the delegation is working.
