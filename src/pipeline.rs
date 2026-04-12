//! HDR→SDR pipeline helpers.
//!
//! Convenience functions that compose transfer-function decoding,
//! gamut conversion, tone mapping, and sRGB encoding into single calls.
//! These are the "I have PQ/HLG content, give me sRGB output" functions.

use crate::ToneMap;
use crate::gamut::{BT2020_TO_BT709, apply_matrix};

/// Tonemap a PQ-encoded BT.2020 RGB row to linear sRGB.
///
/// Pipeline: PQ EOTF → linear BT.2020 → tonemap → gamut convert to
/// BT.709. The output is linear sRGB ready for an OETF (sRGB gamma).
///
/// `pq_row`: interleaved PQ-encoded RGB in [0, 1] (1.0 = 10000 nits).
/// `out`: linear sRGB output, same length.
/// `tm`: any tonemapper (Bt2408, Bt2446A, etc.).
/// `channels`: 3 (RGB) or 4 (RGBA, alpha passed through).
pub fn tonemap_pq_to_linear_srgb(pq_row: &[f32], out: &mut [f32], tm: &dyn ToneMap, channels: u8) {
    debug_assert_eq!(pq_row.len(), out.len());
    let ch = channels as usize;

    for (src, dst) in pq_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        // Step 1: PQ EOTF → linear BT.2020
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];

        // Step 2: Tonemap
        let tonemapped = tm.map_rgb(linear_2020);

        // Step 3: BT.2020 → BT.709
        let bt709 = apply_matrix(&BT2020_TO_BT709, tonemapped);

        dst[0] = bt709[0].clamp(0.0, 1.0);
        dst[1] = bt709[1].clamp(0.0, 1.0);
        dst[2] = bt709[2].clamp(0.0, 1.0);
        if ch == 4 {
            dst[3] = src[3];
        }
    }
}

/// Tonemap a PQ-encoded BT.2020 RGB row to sRGB-encoded u8.
///
/// Full pipeline: PQ EOTF → linear BT.2020 → tonemap → BT.2020→BT.709
/// → sRGB OETF → quantize to u8.
pub fn tonemap_pq_to_srgb8(pq_row: &[f32], out: &mut [u8], tm: &dyn ToneMap, channels: u8) {
    debug_assert_eq!(pq_row.len(), out.len());
    let ch = channels as usize;

    for (src, dst) in pq_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];

        let tonemapped = tm.map_rgb(linear_2020);
        let bt709 = apply_matrix(&BT2020_TO_BT709, tonemapped);

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
/// Pipeline: HLG inverse OETF → OOTF (system gamma) → tonemap →
/// BT.2020→BT.709.
pub fn tonemap_hlg_to_linear_srgb(
    hlg_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
    channels: u8,
) {
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);
    let ch = channels as usize;

    for (src, dst) in hlg_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        // HLG inverse OETF → scene-linear
        let scene = [
            linear_srgb::tf::hlg_to_linear(src[0]),
            linear_srgb::tf::hlg_to_linear(src[1]),
            linear_srgb::tf::hlg_to_linear(src[2]),
        ];

        // OOTF → display-linear
        let display = crate::hlg::hlg_ootf(scene, gamma);

        // Tonemap
        let tonemapped = tm.map_rgb(display);

        // BT.2020 → BT.709
        let bt709 = apply_matrix(&BT2020_TO_BT709, tonemapped);

        dst[0] = bt709[0].clamp(0.0, 1.0);
        dst[1] = bt709[1].clamp(0.0, 1.0);
        dst[2] = bt709[2].clamp(0.0, 1.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bt2408Tonemapper;

    #[test]
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
    fn rgba_alpha_preserved() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.5_f32, 0.5, 0.5, 0.42];
        let mut out = [0.0_f32; 4];
        tonemap_pq_to_linear_srgb(&pq, &mut out, &tm, 4);
        assert!((out[3] - 0.42).abs() < 1e-6);
    }
}
