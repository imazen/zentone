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
//! # SIMD strip form
//!
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

use crate::ToneMap;
use crate::gamut::{BT2020_TO_BT709, apply_matrix_row_simd, soft_clip_row_simd};
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

// ============================================================================
// SIMD strip-form fused pipelines.
//
// Compose the building blocks (`apply_matrix_row_simd`, `soft_clip_row_simd`,
// `hlg_ootf_row_simd`) plus `linear-srgb`'s `pq_to_linear_slice` /
// `hlg_to_linear_slice` / `linear_to_srgb_u8_slice` SIMD slice forms with
// `ToneMap::map_strip_simd` for the curve step.
// ============================================================================

/// Tonemap a PQ-encoded BT.2020 RGB strip to linear sRGB f32 (SIMD).
///
/// Pipeline:
/// PQ EOTF (via `linear-srgb`) → `tm.map_strip_simd` → BT.2020→BT.709 matrix
/// → hue-preserving soft clip applied unconditionally (identity on in-gamut
/// pixels, so the in-gamut majority sees no behavioral change).
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
/// Pipeline:
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
/// Same pipeline as [`tonemap_pq_row_simd`] up through the soft clip, then a
/// final `linear_to_srgb_u8_slice` (`linear-srgb`) emits sRGB-encoded bytes.
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
/// `soft_clip_row_simd` is identity-equivalent on in-gamut pixels (verified by
/// the dense `[0, 1]^3` sweep in `tests/gamut_hardening.rs`), so applying it
/// unconditionally avoids the per-call mask buffer + gather/scatter dance and
/// keeps the kernel branch-free on the hot path.
fn apply_matrix_and_soft_clip(strip: &mut [[f32; 3]]) {
    apply_matrix_row_simd(&BT2020_TO_BT709, strip);
    soft_clip_row_simd(strip);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bt2408Tonemapper;

    #[test]
    fn pq_to_linear_srgb_black() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [[0.0_f32; 3]];
        let mut out = [[0.0_f32; 3]];
        tonemap_pq_row_simd(&pq, &mut out, &tm);
        for c in out[0] {
            assert!(c.abs() < 1e-3, "black should stay black: {c}");
        }
    }

    #[test]
    fn pq_to_srgb8_produces_valid_bytes() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        // SDR reference white in PQ: ~0.58
        let pq = [[0.58_f32, 0.58, 0.58]];
        let mut out = [[0u8; 3]];
        tonemap_pq_to_srgb8_row_simd(&pq, &mut out, &tm);
        // 203 nits in a 4000-nit content range → modest gray after EETF
        assert!(
            out[0][0] > 30 && out[0][0] < 255,
            "SDR white byte: {}",
            out[0][0]
        );
    }

    #[test]
    fn hlg_to_linear_srgb_at_reference_white() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        // HLG 75% = reference white
        let hlg = [[0.75_f32, 0.75, 0.75]];
        let mut out = [[0.0_f32; 3]];
        tonemap_hlg_row_simd(&hlg, &mut out, &tm, 1000.0);
        for c in out[0] {
            assert!(c.is_finite() && c > 0.0, "HLG ref white: {c}");
        }
    }

    #[test]
    fn rgba_alpha_preserved() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [[0.5_f32, 0.5, 0.5, 0.42]];
        let mut out = [[0.0_f32; 4]];
        tonemap_pq_rgba_row_simd(&pq, &mut out, &tm);
        assert!((out[0][3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn soft_clip_pipeline_output_in_range() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [[0.6_f32, 0.6, 0.6]];
        let mut out = [[0.0_f32; 3]];
        tonemap_pq_row_simd(&pq, &mut out, &tm);
        for (i, &v) in out[0].iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "soft clip: ch {i} = {v} out of [0,1]"
            );
        }
    }
}
