//! BT.2446 Method A tone mapper — the ITU's reference for graded
//! HDR↔SDR conversion with psychophysically verified round-trip fidelity.
//!
//! ITU-R BT.2446-1 §4. TMO: 1000 cd/m² HDR → 100 cd/m² SDR.
//!
//! Psychophysical evaluation (Annex 1): 12 participants, 115 images,
//! 2AFC on Sony BVM-X300 — imperceptible degradation after full
//! round-trip (p = 0.167 HDR, p = 0.196 SDR).
//!
//! The method operates in a perceptually linearized log-luminance domain
//! with a piecewise polynomial knee, Hunt-effect color correction in
//! Y'Cb'Cr', and BT.2020 luma weights.
//!
//! Reference: ITU-R BT.2446-1 (03/2021) §4 + Annex 1.

use crate::ToneMap;
use crate::math::powf;

/// BT.2020 luma weights (BT.2446 uses BT.2020, not BT.709).
const LR: f32 = 0.2627;
const LG: f32 = 0.6780;
const LB: f32 = 0.0593;

/// BT.2446 Method A tonemapper.
///
/// Construct with `new()`, then apply via the [`ToneMap`] trait. Input
/// is linear-light BT.2020 RGB normalized so `1.0 = hdr_peak_nits`.
/// Output is gamma-domain BT.2020 RGB normalized so `1.0 = sdr_peak_nits`.
///
/// # When to pick this
///
/// The most rigorously validated HDR → SDR curve in zentone — the only
/// one with a published psychophysical study showing imperceptible
/// degradation after a full HDR → SDR → HDR round-trip on graded content.
/// Pick when broadcast-grade fidelity matters. Heavier than
/// [`Bt2408Tonemapper`](crate::Bt2408Tonemapper) (extra perceptual-log domain conversion + Hunt
/// color correction), so reserve for offline transcodes or when subjective
/// tests confirm the difference is worth it.
///
/// Reference: ITU-R BT.2446-1 (03/2021) §4 + Annex 1 (12 participants,
/// 115 images, Sony BVM-X300, 2AFC, p ≈ 0.17 indistinguishability).
///
/// # Examples
///
/// ```
/// use zentone::{Bt2446A, ToneMap};
///
/// let curve = Bt2446A::new(1000.0, 100.0);
/// let sdr = curve.map_rgb([0.6, 0.4, 0.2]);
/// assert!(sdr.iter().all(|&c| (0.0..=1.0).contains(&c)));
/// ```
pub struct Bt2446A {
    rho_hdr: f32,
    inv_log_rho_hdr: f32,
    rho_sdr: f32,
    inv_rho_sdr_minus_1: f32,
}

impl Bt2446A {
    /// Create a new BT.2446 Method A tonemapper.
    ///
    /// `hdr_peak_nits`: peak luminance of HDR content (typically 1000).
    /// `sdr_peak_nits`: peak luminance of SDR target (typically 100).
    pub fn new(hdr_peak_nits: f32, sdr_peak_nits: f32) -> Self {
        let rho_hdr = 1.0 + 32.0 * powf(hdr_peak_nits / 10000.0, 2.4);
        let log_rho_hdr = libm::logf(rho_hdr);
        let rho_sdr = 1.0 + 32.0 * powf(sdr_peak_nits / 10000.0, 2.4);
        Self {
            rho_hdr,
            inv_log_rho_hdr: 1.0 / log_rho_hdr,
            rho_sdr,
            inv_rho_sdr_minus_1: 1.0 / (rho_sdr - 1.0),
        }
    }

    /// Perceptual linearization: Y' → Y'_p (log domain).
    #[inline]
    fn perceptual_linearize(&self, y_prime: f32) -> f32 {
        libm::logf(1.0 + (self.rho_hdr - 1.0) * y_prime) * self.inv_log_rho_hdr
    }

    /// BT.2446 Method A piecewise tone curve.
    #[inline]
    fn tone_curve(y_p: f32) -> f32 {
        if y_p <= 0.7399 {
            1.0770 * y_p
        } else if y_p < 0.9909 {
            -1.1510 * y_p * y_p + 2.7811 * y_p - 0.6302
        } else {
            0.5000 * y_p + 0.5000
        }
    }

    /// Inverse perceptual linearization: Y'_c → Y'_SDR.
    #[inline]
    fn perceptual_delinearize(&self, y_c: f32) -> f32 {
        (powf(self.rho_sdr, y_c) - 1.0) * self.inv_rho_sdr_minus_1
    }
}

impl ToneMap for Bt2446A {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Step 1: nonlinear transfer (gamma 1/2.4)
        let r_prime = powf(rgb[0].max(0.0), 1.0 / 2.4);
        let g_prime = powf(rgb[1].max(0.0), 1.0 / 2.4);
        let b_prime = powf(rgb[2].max(0.0), 1.0 / 2.4);

        // Luma in gamma domain
        let y_prime = LR * r_prime + LG * g_prime + LB * b_prime;
        if y_prime <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Perceptual linearization
        let y_p = self.perceptual_linearize(y_prime);

        // Piecewise tone curve
        let y_c = Self::tone_curve(y_p);

        // Convert back from perceptual to gamma domain
        let y_sdr = self.perceptual_delinearize(y_c);

        // Hunt-effect color correction in Y'Cb'Cr' (Table 3)
        let f = y_sdr / (1.1 * y_prime);
        let cb = f * (b_prime - y_prime) / 1.8814;
        let cr = f * (r_prime - y_prime) / 1.4746;

        // Adjusted luma
        let y_tmo = y_sdr - 0.1_f32.max(0.0) * cr.max(0.0); // max(0.1*Cr, 0) subtracted

        // Y'Cb'Cr' → R'G'B'
        let r_out = (y_tmo + 1.4746 * cr).clamp(0.0, 1.0);
        let g_out = (y_tmo - (0.16455 / 0.6780) * cb - (0.57135 / 0.6780) * cr).clamp(0.0, 1.0);
        let b_out = (y_tmo + 1.8814 * cb).clamp(0.0, 1.0);

        [r_out, g_out, b_out]
    }

    fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        archmage::incant!(
            crate::simd::curves::bt2446a_tier(
                strip,
                self.rho_hdr,
                self.inv_log_rho_hdr,
                self.rho_sdr,
                self.inv_rho_sdr_minus_1,
            ),
            [v3, neon, wasm128, scalar]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_to_black() {
        let tm = Bt2446A::new(1000.0, 100.0);
        let out = tm.map_rgb([0.0, 0.0, 0.0]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn neutral_gray_passes_through_approximately() {
        let tm = Bt2446A::new(1000.0, 100.0);
        // Mid-gray at 1000 cd/m²: ~18% linear = 0.18
        let out = tm.map_rgb([0.18, 0.18, 0.18]);
        // All channels should be equal (neutral) and in a reasonable range
        assert!(
            (out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5,
            "neutral gray should stay neutral: {out:?}"
        );
        assert!(
            out[0] > 0.1 && out[0] < 0.8,
            "mid-gray should map to reasonable SDR level: {}",
            out[0]
        );
    }

    #[test]
    fn peak_maps_to_sdr_range() {
        let tm = Bt2446A::new(1000.0, 100.0);
        // HDR peak (1.0 = 1000 cd/m²) should map near SDR peak
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in out {
            assert!(
                c > 0.8 && c <= 1.0,
                "peak should map to near-SDR-white: {c}"
            );
        }
    }

    #[test]
    fn monotonic_on_neutral_ramp() {
        let tm = Bt2446A::new(1000.0, 100.0);
        let mut last = -1.0_f32;
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let out = tm.map_rgb([v, v, v]);
            let lum = out[0]; // neutral → all channels equal
            assert!(
                lum >= last - 1e-5,
                "monotonicity violated at {v}: {lum} < {last}"
            );
            last = lum;
        }
    }

    #[test]
    fn colored_input_stays_finite_and_bounded() {
        let tm = Bt2446A::new(1000.0, 100.0);
        let cases = [
            [0.8, 0.2, 0.05],
            [0.1, 0.9, 0.05],
            [0.3, 0.3, 0.8],
            [0.5, 0.5, 0.5],
        ];
        for rgb in cases {
            let out = tm.map_rgb(rgb);
            for (i, c) in out.iter().enumerate() {
                assert!(
                    c.is_finite() && *c >= 0.0 && *c <= 1.001,
                    "Bt2446A({rgb:?})[{i}] = {c}"
                );
            }
        }
    }
}
