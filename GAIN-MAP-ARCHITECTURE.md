# Gain Map Architecture: What Goes Where

Cross-crate feedback from the ultrahdr-core 0.5 slim work. Covers zentone's
gap list for "excellent luma gain map support" and where each piece should live.

## The three layers

| Layer | Crate | Owns | Does NOT own |
|---|---|---|---|
| **Wire format** | zencodec | `GainMapParams` (ISO 21496-1 fields), u8 quantization/dequantization, serialization to XMP/binary | Tone curves, pixel math |
| **Pixel math** | zentone | Tone curves, `LumaGainMapSplitter` (f32 log2 gain), gamut matrices, transfer functions | Wire-format quantization, container framing |
| **Codec integration** | ultrahdr-core | RawImage pixel-format matrix, JPEG/MPF/XMP container, streaming ring buffers | Generic tone-mapping curves, ISO 21496-1 field semantics |

This layering is already mostly right. The feedback below is about the gaps.

## Per-item placement

### 1. `pack_gain_u8` / `unpack_gain_u8`

**Proposed home**: zentone.
**My recommendation**: **zencodec**, not zentone.

The u8 gain byte IS the wire format — it's what gets stored in the JPEG gain map
image. The encoding is parameterized by `GainMapParams` fields (min, max, gamma)
which zencodec owns. The decode side (u8 → f32 linear gain) is exactly what
ultrahdr-core's `GainMapLut` does today — precompute a 256-entry table from
`GainMapParams`, look up a byte.

ultrahdr-core already has `compute_and_encode_gain()` (encode) and `GainMapLut`
(decode). These should move DOWN to zencodec, not sideways to zentone. Then both
zentone callers and ultrahdr-core callers use the same zencodec primitive.

Zentone's `LumaGainMapSplitter` should continue to emit raw f32 log2 gain. The
quantization boundary is explicitly at "hand off to the codec" — which is zencodec's job.

If zentone adds `pack_gain_u8`, anyone who imports both zentone and zencodec now
has two implementations of the same quantization. One will drift.

### 2. `LumaGainMapMetadata` POD

**Proposed home**: zentone (without zencodec dep).
**My recommendation**: **skip entirely**.

`zencodec::GainMapParams` already IS this type. Creating a second one in zentone
means two POD types for the same ISO 21496-1 fields. They will drift.

Three alternatives, pick one:
1. **zentone depends on zencodec for `GainMapParams`**. Clean. zencodec is tiny
   (no_std, no alloc needed for the POD). ultrahdr-core already depends on both.
2. **zentone stays metadata-agnostic**. `SplitConfig` already has the fields
   zentone needs at runtime (luma_weights, base_offset, alternate_offset, min/max
   log2). `SplitStats` reports observed ranges. A downstream consumer (zencodec or
   ultrahdr-core) maps `(SplitConfig, SplitStats) → GainMapParams` in 10 lines:
   ```rust
   fn build_params(cfg: &SplitConfig, stats: &SplitStats) -> GainMapParams {
       let mut p = GainMapParams::default();
       for ch in &mut p.channels {
           ch.base_offset = cfg.base_offset as f64;
           ch.alternate_offset = cfg.alternate_offset as f64;
           ch.min = stats.observed_min_log2 as f64;
           ch.max = stats.observed_max_log2 as f64;
           ch.gamma = 1.0; // or from a QuantizeConfig
       }
       p
   }
   ```
3. **zencodec re-exports a tiny `GainMapFields` trait** that both zentone and
   ultrahdr-core can impl. Over-engineered for the current use case; skip.

I'd pick option 2. Zentone's `SplitConfig` + `SplitStats` are already a complete
surface — anyone who wants GainMapParams can build it from those. No new type, no
new dep, no drift risk.

### 3. Gain map downsampling

**zentone** ✓

This is the single highest-value item. The entire point of the ISO 21496-1 wire
format is that the gain map is smaller than the SDR base. Without downsampling,
every caller reimplements block averaging. Without matching upsampling in
`apply_row`, the decode path doesn't interop with the encode path.

Add `scale_factor: u8` to `SplitConfig`. In `split_row`, block-average the gain
values for each `scale_factor × scale_factor` cell (or emit at the downsampled
grid). In `apply_row`, bilinear-upsample the gain grid to full resolution before
applying.

ultrahdr-core's `compute_luminance_gainmap` does this today (samples center pixel
per block, lines 159-162). zentone should own the primitive.

### 4. HLG support

**zentone** ✓

Mirror of the existing PQ path. HLG needs the OOTF applied (display-peak-dependent
system gamma), which zentone already has in `hlg::hlg_system_gamma` +
`hlg::hlg_ootf`.

Add `split_hlg_row` / `apply_hlg_row` that linearize via OOTF before/after
the splitter.

### 5. Gamut conversion during split

**zentone** ✓

Most HDR is BT.2020; most SDR is BT.709. zentone already has
`gamut::apply_matrix` + `gamut::soft_clip` + the 6 direct matrices.

Add optional `target_primaries: Option<...>` to `SplitConfig`. When set,
apply the gamut matrix + soft-clip after the luma tone map, before computing
the gain. This keeps the SDR in-gamut and reduces `clipped_sdr_pixels`.

### 6. sRGB-encoded SDR output

**zenpixels-convert** long-term. **zentone convenience OK** short-term.

linear f32 → u8 sRGB is exactly zenpixels-convert's job (`RowConverter` with
transfer function). But shipping a one-liner
`split_row_to_srgb8(hdr, sdr_u8_out, gain_out, ...)` in zentone is fine as
a convenience if zenpixels-convert isn't wired in yet.

### 7. Golden-file parity test

**ultrahdr-core tests** or **codec-eval**, not zentone.

This is a test, not library code. The reference implementation is ultrahdr-core.
The test feeds identical HDR+SDR to both zentone's splitter and ultrahdr-core's
`compute_gainmap`, compares gain maps byte-for-byte. Lives with the code being
validated against.

### 8. Pre-desaturation hook

**zentone** ✓

BT.2446-C's alpha parameter applied before the luma curve to pre-squash
saturation, preventing out-of-gamut after chromaticity-preserving RGB rescale.
This mitigates `clipped_sdr_pixels` — currently the stat reports the problem
but nothing prevents it.

### 9. Filmic spline `LumaToneMap` impl

**zentone** ✓. Trivial: `CompiledFilmicSpline` in luminance-preserving mode
via `self.luminance(luma)`.

### 10. End-to-end wire test

**zencodec tests** (if pack/unpack lives there) or a shared integration test.

### 11–15. SIMD, benchmarks, multi-channel, boost estimation, streaming splitter

All **zentone** ✓. Lower priority, larger work, all obviously zentone's domain.

### 16. Perceptual gamut mapping (Hellwig 2022)

**zentone** ✓. Already tracked (zentone issue #14).

### 17. Adobe Gain Map XMP round-trip

**zencodec** (metadata dialect serialization) + **ultrahdr-core** (JPEG container).
The math is identical across ISO 21496-1 / Adobe / Apple variants; the container
metadata wrapping differs. Probably a zencodec feature flag.

## Recommended next commit for zentone

The proposed slice (#1 + #2 + #4 + #5) has the right spirit but wrong placement
for #1 and #2. My counter-proposal:

**zentone**: #4 (HLG) + #5 (gamut conversion) + #3 (downsampling) + #9 (filmic impl)

These four close the gap between "produces correct f32 pairs" and "ready for
zencodec to quantize and pack." They're coherent, they don't step on zencodec's
wire-format responsibilities, and #3 (downsampling) is the highest-value single
item on the whole list.

**zencodec** (separate commit): #1 (pack/unpack u8 gain). Take over the math from
ultrahdr-core's `compute_and_encode_gain` + `GainMapLut`. One encode fn, one decode
fn (or a LUT builder), parameterized by `GainMapParams`.

**ultrahdr-core** (after both land): delete `compute_and_encode_gain`, shrink
`GainMapLut` to re-export zencodec's, delegate `compute_luminance_gainmap` to
zentone's downsampling splitter + zencodec's u8 packer. Estimated ~200 LOC cut.

## What ultrahdr-core could eventually delegate

Once zentone + zencodec fill the gaps, ultrahdr-core's gain map code path becomes:

```
RawImage → [linearize per PixelFormat] → zentone::LumaGainMapSplitter
  → zencodec::pack_gain_u8 → GainMap (u8 grid)
  → zencodec::build_gainmap_params(SplitConfig, SplitStats) → GainMapMetadata
```

And decode:
```
GainMap (u8 grid) + GainMapMetadata → zencodec::GainMapLut
  → zentone apply_row (with bilinear upsample from downsampled grid)
  → [delinearize to output format] → RawImage
```

ultrahdr-core's remaining job: the `[linearize]` and `[delinearize]` pixel-format
adapters, JPEG/MPF/XMP container glue, and the streaming ring-buffer orchestration.
That's ~1,200 non-test LOC that's genuinely Ultra HDR-specific — no further
trimming needed.
