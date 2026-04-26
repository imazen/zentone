//! HLG system gamma, OOTF, and display adaptation.
//!
//! The raw HLG OETF/EOTF are in `linear-srgb` (SIMD-accelerated).
//! This module provides the composition layer on top:
//!
//! - [`hlg_system_gamma`] — BT.2390 §6.2 formula
//! - [`hlg_ootf`] — luminance-preserving display-referred OOTF
//! - [`hlg_inverse_ootf`] — scene-referred inverse
//! - [`hlg_to_display`] — full HLG signal → display nits
//!
//! Reference: ITU-R BT.2100-2, BT.2390-11 §6.2, BT.2408-8 §3.2.

use crate::math::powf;

/// BT.709 luminance coefficients used in the HLG OOTF.
const LR: f32 = 0.2627;
const LG: f32 = 0.6780;
const LB: f32 = 0.0593;

/// Compute the HLG system gamma for a given display peak luminance.
///
/// Formula from BT.2390-11 §6.2:
/// ```text
/// gamma = 1.2 + 0.42 * log10(Lw / 1000)
/// ```
///
/// At `Lw = 1000` nits: gamma = 1.2 (the reference).
/// At `Lw = 4000` nits: gamma ≈ 1.453.
/// At `Lw = 400` nits: gamma ≈ 1.033.
#[inline]
pub fn hlg_system_gamma(display_peak_nits: f32) -> f32 {
    1.2 + 0.42 * libm::log10f(display_peak_nits / 1000.0)
}

/// HLG OOTF: scene-linear RGB → display-linear RGB.
///
/// Applies the system gamma to convert from scene-referred light to
/// display-referred light. The BT.2100 OOTF is:
/// ```text
/// Ys = LR*Rs + LG*Gs + LB*Bs         (scene luminance, BT.2020 weights)
/// Rd = alpha * Ys^(gamma-1) * Rs      (display red)
/// Gd = alpha * Ys^(gamma-1) * Gs      (display green)
/// Bd = alpha * Ys^(gamma-1) * Bs      (display blue)
/// ```
///
/// `alpha` scales the output to the display peak: for a 1000-nit display
/// and a nominal system gamma of 1.2, `alpha = display_peak_nits`.
/// For simplicity, this function returns values normalized to [0, 1]
/// where 1.0 = `display_peak_nits`.
///
/// `rgb`: scene-linear HLG-decoded RGB in [0, 1].
/// `gamma`: from [`hlg_system_gamma`].
pub fn hlg_ootf(rgb: [f32; 3], gamma: f32) -> [f32; 3] {
    let ys = LR * rgb[0] + LG * rgb[1] + LB * rgb[2];
    if ys <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let scale = powf(ys, gamma - 1.0);
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// HLG inverse OOTF: display-linear RGB → scene-linear RGB.
///
/// Inverse of [`hlg_ootf`].
pub fn hlg_inverse_ootf(rgb: [f32; 3], gamma: f32) -> [f32; 3] {
    let yd = LR * rgb[0] + LG * rgb[1] + LB * rgb[2];
    if yd <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let inv_gamma = 1.0 / gamma;
    let scale = powf(yd, (1.0 - gamma) * inv_gamma);
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Per-channel HLG OOTF approximation: `[pow(r, γ), pow(g, γ), pow(b, γ)]`.
///
/// **Not chromaticity-preserving.** Use only when bit-matching libultrahdr's
/// encode output (`hlgOotfApprox` in `gainmapmath.cpp`); the exact OOTF is
/// [`hlg_ootf`] and is the spec-correct default.
pub fn hlg_ootf_approx(rgb: [f32; 3], gamma: f32) -> [f32; 3] {
    [
        powf(rgb[0], gamma),
        powf(rgb[1], gamma),
        powf(rgb[2], gamma),
    ]
}

/// Per-channel inverse of [`hlg_ootf_approx`].
///
/// **Not chromaticity-preserving.** Use only for libultrahdr-compat round-trips;
/// the spec-correct inverse is [`hlg_inverse_ootf`].
pub fn hlg_inverse_ootf_approx(rgb: [f32; 3], gamma: f32) -> [f32; 3] {
    let inv = 1.0 / gamma;
    [powf(rgb[0], inv), powf(rgb[1], inv), powf(rgb[2], inv)]
}

/// Full HLG EOTF with display adaptation: HLG signal → display-linear.
///
/// Composes: HLG inverse OETF (from `linear-srgb`) → OOTF → result.
///
/// `hlg_signal`: HLG-encoded value per channel in [0, 1].
/// `display_peak_nits`: peak luminance of the target display.
///
/// Returns display-linear RGB normalized to [0, 1] where 1.0 = display peak.
pub fn hlg_to_display(hlg_signal: [f32; 3], display_peak_nits: f32) -> [f32; 3] {
    let gamma = hlg_system_gamma(display_peak_nits);
    // Step 1: HLG inverse OETF → scene-linear
    let scene = [
        linear_srgb::tf::hlg_to_linear(hlg_signal[0]),
        linear_srgb::tf::hlg_to_linear(hlg_signal[1]),
        linear_srgb::tf::hlg_to_linear(hlg_signal[2]),
    ];
    // Step 2: OOTF → display-linear
    hlg_ootf(scene, gamma)
}

// ============================================================================
// SIMD strip-form siblings — building blocks for fused HLG pipelines.
// 8-pixel SOA dispatch via `archmage::incant!`. Per-pixel reference
// functions above stay the parity surface and the scalar tail.
// ============================================================================

/// Chromaticity-preserving HLG OOTF applied to an RGB strip in place (SIMD).
///
/// SIMD-equivalent to calling [`hlg_ootf`] per pixel: compute BT.2100
/// luminance once, raise to `pow(Y, gamma - 1)`, multiply each channel.
/// Pixels with `Y <= 0` are forced to `[0, 0, 0]` to match the per-pixel
/// reference. Tail pixels fall through to scalar `pow`.
///
/// `gamma` is the HLG system gamma from [`hlg_system_gamma`].
///
/// # Examples
///
/// ```
/// use zentone::hlg::{hlg_ootf_row_simd, hlg_system_gamma};
/// let mut row = [[0.5_f32, 0.3, 0.8]];
/// hlg_ootf_row_simd(&mut row, hlg_system_gamma(1000.0));
/// ```
#[inline]
pub fn hlg_ootf_row_simd(row: &mut [[f32; 3]], gamma: f32) {
    let k = gamma - 1.0;
    archmage::incant!(
        crate::simd::blocks::hlg_ootf_exact_tier(row, k),
        [v3, neon, wasm128, scalar]
    );
}

/// Inverse of [`hlg_ootf_row_simd`]: display-linear → scene-linear (SIMD).
///
/// SIMD-equivalent to [`hlg_inverse_ootf`] per pixel. Same `Y <= 0 → 0`
/// behavior as the forward direction.
#[inline]
pub fn hlg_inverse_ootf_row_simd(row: &mut [[f32; 3]], gamma: f32) {
    // Per-pixel `hlg_inverse_ootf` uses `pow(Y, (1 - gamma) / gamma)`.
    let k = (1.0 - gamma) / gamma;
    archmage::incant!(
        crate::simd::blocks::hlg_ootf_exact_tier(row, k),
        [v3, neon, wasm128, scalar]
    );
}

/// Per-channel HLG OOTF approx applied to an RGB strip in place (SIMD).
///
/// SIMD-equivalent to [`hlg_ootf_approx`]: three independent `pow(c, gamma)`
/// calls per pixel, no luminance coupling.
///
/// **Not chromaticity-preserving.** Use only when bit-matching libultrahdr's
/// encode path; the spec-correct default is [`hlg_ootf_row_simd`].
#[inline]
pub fn hlg_ootf_approx_row_simd(row: &mut [[f32; 3]], gamma: f32) {
    archmage::incant!(
        crate::simd::blocks::hlg_ootf_approx_tier(row, gamma),
        [v3, neon, wasm128, scalar]
    );
}

/// Inverse of [`hlg_ootf_approx_row_simd`] (SIMD).
///
/// **Not chromaticity-preserving.** Use only for libultrahdr-compat
/// round-trips; the spec-correct inverse is [`hlg_inverse_ootf_row_simd`].
#[inline]
pub fn hlg_inverse_ootf_approx_row_simd(row: &mut [[f32; 3]], gamma: f32) {
    let inv = 1.0 / gamma;
    archmage::incant!(
        crate::simd::blocks::hlg_ootf_approx_tier(row, inv),
        [v3, neon, wasm128, scalar]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_gamma_reference_1000_nits() {
        let g = hlg_system_gamma(1000.0);
        assert!(
            (g - 1.2).abs() < 1e-5,
            "1000 nits should give gamma=1.2, got {g}"
        );
    }

    #[test]
    fn system_gamma_4000_nits() {
        let g = hlg_system_gamma(4000.0);
        // 1.2 + 0.42 * log10(4) = 1.2 + 0.42 * 0.602 = 1.453
        assert!((g - 1.453).abs() < 0.01, "4000 nits gamma ≈ 1.453, got {g}");
    }

    #[test]
    fn ootf_preserves_black() {
        let out = hlg_ootf([0.0, 0.0, 0.0], 1.2);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn ootf_adds_contrast_for_gamma_gt_1() {
        // OOTF with gamma > 1 makes mid-tones darker relative to peak
        // (the "system gamma" adds contrast). At peak (1.0), output = 1.0.
        // At mid-gray (0.5), output < 0.5 because Ys^(gamma-1) < 1 for Ys < 1.
        let mid = hlg_ootf([0.5, 0.5, 0.5], 1.2);
        let mid_lum = LR * mid[0] + LG * mid[1] + LB * mid[2];
        assert!(
            mid_lum < 0.5 && mid_lum > 0.3,
            "OOTF should darken mid-gray: got {mid_lum}"
        );

        let peak = hlg_ootf([1.0, 1.0, 1.0], 1.2);
        let peak_lum = LR * peak[0] + LG * peak[1] + LB * peak[2];
        assert!(
            (peak_lum - 1.0).abs() < 1e-5,
            "OOTF should preserve peak: got {peak_lum}"
        );
    }

    #[test]
    fn ootf_roundtrip() {
        let rgb = [0.3, 0.6, 0.1];
        let gamma = 1.2;
        let display = hlg_ootf(rgb, gamma);
        let back = hlg_inverse_ootf(display, gamma);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-5,
                "OOTF roundtrip[{i}]: {:.6} vs {:.6}",
                back[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn ootf_approx_roundtrip() {
        // Per-channel approx round-trips exactly (no luminance coupling).
        for &gamma in &[1.0_f32, 1.033, 1.2, 1.453, 1.5] {
            for &rgb in &[[0.3_f32, 0.6, 0.1], [0.001, 0.5, 0.999], [0.18, 0.18, 0.18]] {
                let display = hlg_ootf_approx(rgb, gamma);
                let back = hlg_inverse_ootf_approx(display, gamma);
                for i in 0..3 {
                    assert!(
                        (back[i] - rgb[i]).abs() < 5e-6,
                        "approx OOTF roundtrip[{i}] gamma={gamma} rgb={rgb:?}: \
                         {:.6} vs {:.6}",
                        back[i],
                        rgb[i]
                    );
                }
            }
        }
    }

    #[test]
    fn ootf_approx_diverges_from_exact_on_saturated_color() {
        // The whole point of having both: per-channel approx bends chromaticity
        // for non-grey RGB, but agrees with the exact OOTF on greys.
        let gamma = 1.2;
        let grey = [0.5_f32, 0.5, 0.5];
        let exact_grey = hlg_ootf(grey, gamma);
        let approx_grey = hlg_ootf_approx(grey, gamma);
        for i in 0..3 {
            assert!(
                (exact_grey[i] - approx_grey[i]).abs() < 1e-5,
                "exact and approx must agree on greys: {exact_grey:?} vs {approx_grey:?}"
            );
        }

        let red = [0.9_f32, 0.1, 0.1];
        let exact_red = hlg_ootf(red, gamma);
        let approx_red = hlg_ootf_approx(red, gamma);
        let max_diff = (0..3)
            .map(|i| (exact_red[i] - approx_red[i]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "exact and approx should differ on saturated red: {exact_red:?} vs {approx_red:?}"
        );
    }

    #[test]
    fn hlg_to_display_at_reference_white() {
        // HLG 75% signal → 203 nits on a 1000-nit display (reference white).
        // After inverse OETF + OOTF, the output should be in a reasonable range.
        let out = hlg_to_display([0.75, 0.75, 0.75], 1000.0);
        for c in out {
            assert!(c.is_finite() && c > 0.0 && c < 1.0, "ref white: {c}");
        }
    }
}
