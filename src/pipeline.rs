//! HDR → SDR pipeline helpers.
//!
//! Convenience functions that compose transfer-function decoding,
//! tone mapping, BT.2020 → BT.709 gamut conversion, and (optionally) sRGB
//! u8 encoding into single calls. The tone curve is plugged in as
//! `&dyn ToneMap` so any zentone tonemapper works.
//!
//! All entry points apply the tone curve in RGB (per-channel) space and
//! handle any out-of-gamut pixels after the matrix with a hue-preserving
//! soft clip. That combination is the right "just works" default; more
//! control (luma-preserving application space, different clip policy) is
//! an internal concern today and not yet part of the public surface. If
//! you need it, open an issue.
//!
//! # Hot path — SIMD strip form
//!
//! Use these for any non-trivial workload (a row, a strip, a whole image).
//! Inputs are packed `&[[f32; 3]]` / `&[[f32; 4]]` / `&[[u8; 3]]` slices.
//!
//! - [`tonemap_pq_row_simd`]          — PQ → linear sRGB f32 RGB
//! - [`tonemap_pq_rgba_row_simd`]     — PQ → linear sRGB f32 RGBA (alpha preserved)
//! - [`tonemap_pq_to_srgb8_row_simd`] — PQ → sRGB-encoded `u8` RGB
//! - [`tonemap_pq_to_srgb8_rgba_row_simd`] — PQ → sRGB-encoded `u8` RGBA
//! - [`tonemap_hlg_row_simd`]         — HLG → linear sRGB f32 RGB
//! - [`tonemap_hlg_rgba_row_simd`]    — HLG → linear sRGB f32 RGBA
//!
//! ```
//! use zentone::{Bt2408Tonemapper, pipeline::tonemap_pq_to_srgb8_row_simd};
//!
//! // PQ-encoded BT.2020 RGB strip, 4000 cd/m² master, target sRGB display.
//! let pq = vec![[0.58_f32, 0.58, 0.58]; 1024]; // ~203 nits, mid-gray
//! let curve = Bt2408Tonemapper::new(4000.0, 1000.0);
//! let mut srgb_out = vec![[0u8; 3]; 1024];
//! tonemap_pq_to_srgb8_row_simd(&pq, &mut srgb_out, &curve);
//! ```
//!
//! # Deprecated scalar entry points
//!
//! The original `&[f32]` + `channels: u8` forms are kept for back-compat
//! but produce scalar code:
//!
//! - `tonemap_pq_to_linear_srgb`  — superseded by [`tonemap_pq_row_simd`]
//! - `tonemap_pq_to_srgb8`        — superseded by [`tonemap_pq_to_srgb8_row_simd`]
//! - `tonemap_hlg_to_linear_srgb` — superseded by [`tonemap_hlg_row_simd`]

use crate::ToneMap;
use crate::gamut::{
    BT2020_TO_BT709, ToneMapSpace, apply_matrix, apply_matrix_row_simd, is_out_of_gamut,
    is_out_of_gamut_mask_simd, soft_clip, soft_clip_row_simd, tonemap_luma_preserving,
};
use alloc::vec;
use alloc::vec::Vec;

/// Selects which HLG OOTF formula the HLG-input pipelines apply.
///
/// `Exact` (default) follows ITU-R BT.2100-2 Table 5 — luminance-preserving,
/// matches the spec.
///
/// `LibultrahdrCompat` mirrors libultrahdr's `hlgOotfApprox`: per-channel
/// `pow(c, γ)`. Bends chromaticity in saturated highlights but matches
/// libultrahdr's encoder output bit-for-bit at the OOTF stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HlgOotfMode {
    /// Spec-correct, luminance-preserving HLG OOTF (default).
    #[default]
    Exact,
    /// Per-channel `pow(c, γ)` approximation — for bit-parity with libultrahdr.
    LibultrahdrCompat,
}

/// Pipeline configuration for HDR→SDR conversion (internal).
///
/// Controls which color space the tone curve is applied in.
/// Out-of-gamut colors after gamut matrix conversion are always
/// handled with hue-preserving soft-clipping.
///
/// For principled perceptual gamut compression (rather than post-matrix
/// clipping), use ACES 2.0 ([issue #14](https://github.com/imazen/zentone/issues/14))
/// which compresses in Hellwig 2022 JMh space.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub(crate) struct PipelineConfig {
    /// Color space for tone curve application.
    pub tone_map_space: ToneMapSpace,
    /// HLG OOTF formula for HLG-input pipelines.
    pub hlg_ootf_mode: HlgOotfMode,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tone_map_space: ToneMapSpace::Rgb,
            hlg_ootf_mode: HlgOotfMode::Exact,
        }
    }
}

/// Apply a tone curve to a linear RGB triple respecting the configured space.
#[inline]
fn apply_tonemap(rgb: [f32; 3], tm: &dyn ToneMap, space: ToneMapSpace) -> [f32; 3] {
    match space {
        ToneMapSpace::Rgb => tm.map_rgb(rgb),
        ToneMapSpace::LumaPreserving { luma } => tonemap_luma_preserving(rgb, luma, tm),
    }
}

/// Tonemap a PQ-encoded BT.2020 RGB row to linear sRGB.
///
/// Pipeline: PQ EOTF → linear BT.2020 → tone map (per-channel) →
/// BT.2020→BT.709 matrix → hue-preserving soft-clip on out-of-gamut
/// pixels.
#[deprecated(
    since = "0.2.0",
    note = "use `tonemap_pq_row_simd` for AoS [[f32; 3]] strips with SIMD throughput. \
            The `&[f32]` + `channels: u8` form is kept for back-compat but is scalar."
)]
pub fn tonemap_pq_to_linear_srgb(pq_row: &[f32], out: &mut [f32], tm: &dyn ToneMap, channels: u8) {
    tonemap_pq_to_linear_srgb_config(pq_row, out, tm, channels, &PipelineConfig::default());
}

/// Tonemap a PQ-encoded BT.2020 RGB row to linear sRGB with explicit config.
///
/// Pipeline: PQ EOTF → linear BT.2020 → tonemap (in configured space) →
/// BT.2020→BT.709 → gamut clip.
pub(crate) fn tonemap_pq_to_linear_srgb_config(
    pq_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    channels: u8,
    cfg: &PipelineConfig,
) {
    debug_assert_eq!(pq_row.len(), out.len());
    let ch = channels as usize;

    for (src, dst) in pq_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];

        let tonemapped = apply_tonemap(linear_2020, tm, cfg.tone_map_space);
        let bt709 = {
            let rgb = apply_matrix(&BT2020_TO_BT709, tonemapped);
            if is_out_of_gamut(rgb) {
                soft_clip(rgb)
            } else {
                rgb
            }
        };

        dst[0] = bt709[0];
        dst[1] = bt709[1];
        dst[2] = bt709[2];
        if ch == 4 {
            dst[3] = src[3];
        }
    }
}

/// Tonemap a PQ-encoded BT.2020 RGB row to sRGB-encoded u8.
///
/// Same pipeline as [`tonemap_pq_to_linear_srgb`], with a final sRGB
/// OETF + `u8` quantization on each output byte.
#[deprecated(
    since = "0.2.0",
    note = "use `tonemap_pq_to_srgb8_row_simd` for AoS [[f32; 3]] → [[u8; 3]] strips \
            with SIMD throughput. The `&[f32]` + `channels: u8` form is kept for \
            back-compat but is scalar."
)]
pub fn tonemap_pq_to_srgb8(pq_row: &[f32], out: &mut [u8], tm: &dyn ToneMap, channels: u8) {
    tonemap_pq_to_srgb8_config(pq_row, out, tm, channels, &PipelineConfig::default());
}

/// Tonemap a PQ-encoded BT.2020 RGB row to sRGB u8 with explicit config.
pub(crate) fn tonemap_pq_to_srgb8_config(
    pq_row: &[f32],
    out: &mut [u8],
    tm: &dyn ToneMap,
    channels: u8,
    cfg: &PipelineConfig,
) {
    debug_assert_eq!(pq_row.len(), out.len());
    let ch = channels as usize;

    for (src, dst) in pq_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];

        let tonemapped = apply_tonemap(linear_2020, tm, cfg.tone_map_space);
        let bt709 = {
            let rgb = apply_matrix(&BT2020_TO_BT709, tonemapped);
            if is_out_of_gamut(rgb) {
                soft_clip(rgb)
            } else {
                rgb
            }
        };

        dst[0] = linear_to_srgb_u8(bt709[0]);
        dst[1] = linear_to_srgb_u8(bt709[1]);
        dst[2] = linear_to_srgb_u8(bt709[2]);
        if ch == 4 {
            dst[3] = (src[3] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Tonemap an HLG-encoded BT.2020 RGB row to linear sRGB.
///
/// Pipeline: HLG EOTF + OOTF (for `display_peak_nits`) → linear display
/// BT.2020 → tone map (per-channel) → BT.2020→BT.709 matrix → soft-clip.
#[deprecated(
    since = "0.2.0",
    note = "use `tonemap_hlg_row_simd` for AoS [[f32; 3]] strips with SIMD throughput. \
            The `&[f32]` + `channels: u8` form is kept for back-compat but is scalar."
)]
pub fn tonemap_hlg_to_linear_srgb(
    hlg_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
    channels: u8,
) {
    tonemap_hlg_to_linear_srgb_config(
        hlg_row,
        out,
        tm,
        display_peak_nits,
        channels,
        &PipelineConfig::default(),
    );
}

/// Tonemap an HLG-encoded BT.2020 RGB row to linear sRGB with explicit config.
pub(crate) fn tonemap_hlg_to_linear_srgb_config(
    hlg_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
    channels: u8,
    cfg: &PipelineConfig,
) {
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);
    let ch = channels as usize;

    for (src, dst) in hlg_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let scene = [
            linear_srgb::tf::hlg_to_linear(src[0]),
            linear_srgb::tf::hlg_to_linear(src[1]),
            linear_srgb::tf::hlg_to_linear(src[2]),
        ];

        let display = match cfg.hlg_ootf_mode {
            HlgOotfMode::Exact => crate::hlg::hlg_ootf(scene, gamma),
            HlgOotfMode::LibultrahdrCompat => crate::hlg::hlg_ootf_approx(scene, gamma),
        };
        let tonemapped = apply_tonemap(display, tm, cfg.tone_map_space);
        let bt709 = {
            let rgb = apply_matrix(&BT2020_TO_BT709, tonemapped);
            if is_out_of_gamut(rgb) {
                soft_clip(rgb)
            } else {
                rgb
            }
        };

        dst[0] = bt709[0];
        dst[1] = bt709[1];
        dst[2] = bt709[2];
        if ch == 4 {
            dst[3] = src[3];
        }
    }
}

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = linear_srgb::tf::linear_to_srgb(v);
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

// ============================================================================
// SIMD strip-form fused pipelines (PR3).
//
// Compose PR2's building blocks (`apply_matrix_row_simd`, `soft_clip_row_simd`,
// `is_out_of_gamut_mask_simd`, `hlg_ootf_row_simd`) plus `linear-srgb`'s
// `pq_to_linear_slice` / `hlg_to_linear_slice` / `linear_to_srgb_u8_slice`
// SIMD slice forms with `ToneMap::map_strip_simd` for the curve step.
//
// Each `_row_simd` function is a strip-form sibling to one of the existing
// scalar functions above. Inputs and outputs are `&[[f32; 3]]` / `&mut [[f32; 3]]`
// (or RGBA / u8 variants); the scalar functions remain the parity surface and
// the fallback for shapes that don't fit the strip APIs.
// ============================================================================

/// Tonemap a PQ-encoded BT.2020 RGB strip to linear sRGB f32 (SIMD).
///
/// Strip-form sibling of [`tonemap_pq_to_linear_srgb`] for callers that
/// already have packed `[[f32; 3]]` buffers. Pipeline:
/// PQ EOTF (via `linear-srgb`) → `tm.map_strip_simd` → BT.2020→BT.709 matrix
/// → out-of-gamut mask + hue-preserving soft clip on flagged pixels.
///
/// The OOG branch is masked rather than per-pixel: a tiny mask buffer
/// (one f32 per pixel) is filled by [`is_out_of_gamut_mask_simd`], pixels
/// with `mask >= 0.5` are gathered into a side buffer, soft-clipped, and
/// scattered back. This avoids running soft clip on the in-gamut majority.
///
/// `pq_row` and `out` must have equal length.
///
/// # Examples
///
/// ```
/// use zentone::{Bt2408Tonemapper, pipeline::tonemap_pq_row_simd};
/// let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
/// let pq = [[0.58_f32, 0.58, 0.58]; 4];
/// let mut out = [[0.0_f32; 3]; 4];
/// tonemap_pq_row_simd(&pq, &mut out, &tm);
/// ```
#[inline]
pub fn tonemap_pq_row_simd(pq_row: &[[f32; 3]], out: &mut [[f32; 3]], tm: &dyn ToneMap) {
    assert_eq!(
        pq_row.len(),
        out.len(),
        "tonemap_pq_row_simd: input and output strip must have equal length"
    );
    if pq_row.is_empty() {
        return;
    }
    // Stage 1: PQ EOTF — linear-srgb's `pq_to_linear_slice` operates on
    // `&mut [f32]`, so we copy into `out` as a flat buffer first.
    out.copy_from_slice(pq_row);
    linear_srgb::default::pq_to_linear_slice(out.as_flattened_mut());

    // Stage 2: tone curve (applies through the trait's `map_strip_simd`).
    tm.map_strip_simd(out);

    // Stage 3: gamut convert + soft-clip OOG pixels.
    apply_matrix_and_soft_clip(out);
}

/// Tonemap a PQ-encoded BT.2020 RGBA strip to linear sRGB f32 (SIMD).
///
/// Same pipeline as [`tonemap_pq_row_simd`]; alpha is preserved.
#[inline]
pub fn tonemap_pq_rgba_row_simd(pq_row: &[[f32; 4]], out: &mut [[f32; 4]], tm: &dyn ToneMap) {
    assert_eq!(
        pq_row.len(),
        out.len(),
        "tonemap_pq_rgba_row_simd: input and output strip must have equal length"
    );
    if pq_row.is_empty() {
        return;
    }
    // Stage 1: copy through PQ EOTF using the alpha-preserving slice form.
    out.copy_from_slice(pq_row);
    linear_srgb::default::pq_to_linear_rgba_slice(out.as_flattened_mut());

    // Stage 2: tone-map RGB only (gather/scatter through a small buffer).
    let mut rgb: Vec<[f32; 3]> = out.iter().map(|p| [p[0], p[1], p[2]]).collect();
    tm.map_strip_simd(&mut rgb);

    // Stage 3: gamut convert + soft clip on the RGB buffer.
    apply_matrix_and_soft_clip(&mut rgb);

    // Scatter back, preserving alpha.
    for (dst, mapped) in out.iter_mut().zip(rgb.iter()) {
        dst[0] = mapped[0];
        dst[1] = mapped[1];
        dst[2] = mapped[2];
    }
}

/// Tonemap an HLG-encoded BT.2020 RGB strip to linear sRGB f32 (SIMD).
///
/// Strip-form sibling of [`tonemap_hlg_to_linear_srgb`]. Pipeline:
/// HLG EOTF → spec-correct `hlg_ootf_row_simd` (computed for the system gamma
/// of `display_peak_nits`) → `tm.map_strip_simd` → BT.2020→BT.709 →
/// soft-clip OOG.
///
/// `display_peak_nits` is the target display peak; system gamma is derived via
/// [`crate::hlg::hlg_system_gamma`]. The OOTF formula is the spec-correct
/// chromaticity-preserving variant (matches [`HlgOotfMode::Exact`]).
#[inline]
pub fn tonemap_hlg_row_simd(
    hlg_row: &[[f32; 3]],
    out: &mut [[f32; 3]],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
) {
    assert_eq!(
        hlg_row.len(),
        out.len(),
        "tonemap_hlg_row_simd: input and output strip must have equal length"
    );
    if hlg_row.is_empty() {
        return;
    }
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);

    // Stage 1: HLG EOTF (scene-linear).
    out.copy_from_slice(hlg_row);
    linear_srgb::default::hlg_to_linear_slice(out.as_flattened_mut());

    // Stage 2: HLG OOTF (display-linear) with spec-correct chromaticity-
    // preserving formula.
    crate::hlg::hlg_ootf_row_simd(out, gamma);

    // Stage 3: tone curve.
    tm.map_strip_simd(out);

    // Stage 4: gamut convert + soft-clip OOG.
    apply_matrix_and_soft_clip(out);
}

/// Tonemap an HLG-encoded BT.2020 RGBA strip to linear sRGB f32 (SIMD).
///
/// Same pipeline as [`tonemap_hlg_row_simd`]; alpha is preserved.
#[inline]
pub fn tonemap_hlg_rgba_row_simd(
    hlg_row: &[[f32; 4]],
    out: &mut [[f32; 4]],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
) {
    assert_eq!(
        hlg_row.len(),
        out.len(),
        "tonemap_hlg_rgba_row_simd: input and output strip must have equal length"
    );
    if hlg_row.is_empty() {
        return;
    }
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);

    out.copy_from_slice(hlg_row);
    linear_srgb::default::hlg_to_linear_rgba_slice(out.as_flattened_mut());

    let mut rgb: Vec<[f32; 3]> = out.iter().map(|p| [p[0], p[1], p[2]]).collect();
    crate::hlg::hlg_ootf_row_simd(&mut rgb, gamma);
    tm.map_strip_simd(&mut rgb);
    apply_matrix_and_soft_clip(&mut rgb);

    for (dst, mapped) in out.iter_mut().zip(rgb.iter()) {
        dst[0] = mapped[0];
        dst[1] = mapped[1];
        dst[2] = mapped[2];
    }
}

/// Tonemap a PQ-encoded BT.2020 RGB strip to sRGB-encoded `u8` (SIMD).
///
/// Strip-form sibling of [`tonemap_pq_to_srgb8`]. Same pipeline as
/// [`tonemap_pq_row_simd`] up through the soft clip, then a final
/// `linear_to_srgb_u8_slice` (`linear-srgb`) emits sRGB-encoded bytes.
///
/// `pq_row` and `out` must have equal length.
#[inline]
pub fn tonemap_pq_to_srgb8_row_simd(pq_row: &[[f32; 3]], out: &mut [[u8; 3]], tm: &dyn ToneMap) {
    assert_eq!(
        pq_row.len(),
        out.len(),
        "tonemap_pq_to_srgb8_row_simd: input and output strip must have equal length"
    );
    if pq_row.is_empty() {
        return;
    }
    let mut linear: Vec<[f32; 3]> = pq_row.to_vec();
    linear_srgb::default::pq_to_linear_slice(linear.as_flattened_mut());
    tm.map_strip_simd(&mut linear);
    apply_matrix_and_soft_clip(&mut linear);

    // Final OETF + u8 quantization. After soft clip every channel is in [0, 1].
    linear_srgb::default::linear_to_srgb_u8_slice(linear.as_flattened(), out.as_flattened_mut());
}

/// Tonemap a PQ-encoded BT.2020 RGBA strip to sRGB-encoded `u8` (SIMD).
///
/// Same pipeline as [`tonemap_pq_to_srgb8_row_simd`]; alpha is preserved as
/// a straight `(a * 255).round().clamp(0, 255)` quantization.
#[inline]
pub fn tonemap_pq_to_srgb8_rgba_row_simd(
    pq_row: &[[f32; 4]],
    out: &mut [[u8; 4]],
    tm: &dyn ToneMap,
) {
    assert_eq!(
        pq_row.len(),
        out.len(),
        "tonemap_pq_to_srgb8_rgba_row_simd: input and output strip must have equal length"
    );
    if pq_row.is_empty() {
        return;
    }
    // Linearize RGB+alpha straight, then split alpha out for quantization.
    let mut linear_rgba: Vec<[f32; 4]> = pq_row.to_vec();
    linear_srgb::default::pq_to_linear_rgba_slice(linear_rgba.as_flattened_mut());

    let mut rgb: Vec<[f32; 3]> = linear_rgba.iter().map(|p| [p[0], p[1], p[2]]).collect();
    tm.map_strip_simd(&mut rgb);
    apply_matrix_and_soft_clip(&mut rgb);

    // Bulk OETF on RGB.
    let mut rgb_u8 = vec![[0u8; 3]; rgb.len()];
    linear_srgb::default::linear_to_srgb_u8_slice(rgb.as_flattened(), rgb_u8.as_flattened_mut());

    for ((dst, rgb_byte), src_alpha) in out.iter_mut().zip(rgb_u8.iter()).zip(linear_rgba.iter()) {
        dst[0] = rgb_byte[0];
        dst[1] = rgb_byte[1];
        dst[2] = rgb_byte[2];
        // Alpha pre-EOTF (i.e. "linear-srgb", but for an alpha channel that
        // never went through PQ in the first place we just want the raw value
        // back). pq_to_linear_rgba_slice already left index 3 untouched.
        dst[3] = (src_alpha[3] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Apply BT.2020 → BT.709 matrix to a strip and soft-clip out-of-gamut pixels.
///
/// Two-pass approach for cache friendliness on long strips:
/// 1. Vectorized matrix apply across the whole strip.
/// 2. Vectorized OOG mask scan; if any pixels are flagged, gather them into
///    a small contiguous buffer, soft-clip in SIMD, scatter back.
///
/// Doing the soft clip on the gathered subset keeps the common case (in-gamut
/// pixels) on the fast path — most video content is mostly in gamut after
/// tone-mapping, so paying the gather/scatter only on flagged pixels wins
/// over running `soft_clip_row_simd` on every pixel.
fn apply_matrix_and_soft_clip(strip: &mut [[f32; 3]]) {
    apply_matrix_row_simd(&BT2020_TO_BT709, strip);

    // OOG mask. Stack-allocate the mask via a pre-sized Vec; for typical
    // strip widths (a few hundred pixels) this is one alloc and lives only
    // for the duration of this call.
    let mut mask = vec![0.0_f32; strip.len()];
    is_out_of_gamut_mask_simd(strip, &mut mask);

    // Count how many pixels need clipping. Fast majority-in-gamut path:
    // if zero, return.
    let oog_count = mask.iter().filter(|&&m| m >= 0.5).count();
    if oog_count == 0 {
        return;
    }

    // If most pixels are out of gamut (>50%), clip everything in-place.
    // The threshold is a heuristic — gather/scatter is only worth it when
    // the OOG fraction is small enough that the bookkeeping pays off.
    if oog_count * 2 >= strip.len() {
        soft_clip_row_simd(strip);
        return;
    }

    // Otherwise gather into a side buffer, clip, scatter back.
    let mut gathered: Vec<[f32; 3]> = Vec::with_capacity(oog_count);
    let mut indices: Vec<usize> = Vec::with_capacity(oog_count);
    for (i, (&m, px)) in mask.iter().zip(strip.iter()).enumerate() {
        if m >= 0.5 {
            gathered.push(*px);
            indices.push(i);
        }
    }
    soft_clip_row_simd(&mut gathered);
    for (&i, px) in indices.iter().zip(gathered.iter()) {
        strip[i] = *px;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bt2408Tonemapper;

    #[test]
    #[allow(deprecated)] // scalar fallback path, deprecation expected.
    fn pq_to_linear_srgb_black() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.0_f32; 3];
        let mut out = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb(&pq, &mut out, &tm, 3);
        for c in out {
            assert!(c.abs() < 1e-3, "black should stay black: {c}");
        }
    }

    #[test]
    #[allow(deprecated)] // scalar fallback path, deprecation expected.
    fn pq_to_srgb8_produces_valid_bytes() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        // SDR reference white in PQ: ~0.58
        let pq = [0.58_f32, 0.58, 0.58];
        let mut out = [0u8; 3];
        tonemap_pq_to_srgb8(&pq, &mut out, &tm, 3);
        // 203 nits in a 4000-nit content range → modest gray after EETF
        assert!(out[0] > 30 && out[0] < 255, "SDR white byte: {}", out[0]);
    }

    #[test]
    #[allow(deprecated)] // scalar fallback path, deprecation expected.
    fn hlg_to_linear_srgb_at_reference_white() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        // HLG 75% = reference white
        let hlg = [0.75_f32, 0.75, 0.75];
        let mut out = [0.0_f32; 3];
        tonemap_hlg_to_linear_srgb(&hlg, &mut out, &tm, 1000.0, 3);
        for c in out {
            assert!(c.is_finite() && c > 0.0, "HLG ref white: {c}");
        }
    }

    #[test]
    #[allow(deprecated)] // scalar fallback path, deprecation expected.
    fn rgba_alpha_preserved() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.5_f32, 0.5, 0.5, 0.42];
        let mut out = [0.0_f32; 4];
        tonemap_pq_to_linear_srgb(&pq, &mut out, &tm, 4);
        assert!((out[3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn luma_preserving_config_produces_valid_output() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let cfg = PipelineConfig {
            tone_map_space: ToneMapSpace::LumaPreserving {
                luma: crate::LUMA_BT709,
            },
            ..PipelineConfig::default()
        };
        // Bright saturated PQ red
        let pq = [0.7_f32, 0.3, 0.2];
        let mut out = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq, &mut out, &tm, 3, &cfg);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.01).contains(&v),
                "luma-preserving cfg: ch {i} = {v}"
            );
        }
        // Channel ordering should be preserved (R > G > B since input R > G > B)
        assert!(
            out[0] >= out[1] && out[1] >= out[2],
            "luma-preserving should preserve channel ordering: {out:?}"
        );
    }

    #[test]
    fn luma_preserving_less_oog_than_rgb() {
        let tm = crate::ToneMapCurve::Reinhard;
        let pq_green = [0.2_f32, 0.7, 0.1];

        let cfg_rgb = PipelineConfig {
            tone_map_space: ToneMapSpace::Rgb,
            ..PipelineConfig::default()
        };
        let mut out_rgb = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq_green, &mut out_rgb, &tm, 3, &cfg_rgb);

        let cfg_luma = PipelineConfig {
            tone_map_space: ToneMapSpace::LumaPreserving {
                luma: crate::LUMA_BT709,
            },
            ..PipelineConfig::default()
        };
        let mut out_luma = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq_green, &mut out_luma, &tm, 3, &cfg_luma);

        for &v in out_rgb.iter().chain(out_luma.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    #[allow(deprecated)] // scalar fallback path, deprecation expected.
    fn pipeline_config_default_matches_simple_function() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.5_f32, 0.4, 0.3];

        let mut out_simple = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb(&pq, &mut out_simple, &tm, 3);

        let mut out_config = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq, &mut out_config, &tm, 3, &PipelineConfig::default());

        for i in 0..3 {
            assert!(
                (out_simple[i] - out_config[i]).abs() < 1e-7,
                "default config should match simple fn: ch {i}: {} vs {}",
                out_simple[i],
                out_config[i]
            );
        }
    }

    #[test]
    fn soft_clip_pipeline_output_in_range() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let cfg = PipelineConfig {
            tone_map_space: ToneMapSpace::Rgb,
            ..PipelineConfig::default()
        };
        let pq = [0.6_f32, 0.6, 0.6];
        let mut out = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq, &mut out, &tm, 3, &cfg);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "soft clip: ch {i} = {v} out of [0,1]"
            );
        }
    }
}
