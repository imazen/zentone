//! BT.2446 Method C — parametric tone mapper with exact mathematical inverse.
//!
//! ITU-R BT.2446-1 §6. Designed for skin-tone preservation with a
//! piecewise linear+log curve adjustable by 4 parameters (k1-k4).
//!
//! Key feature: the ITMO is the exact algebraic inverse of the TMO
//! (when the optional chroma correction is excluded), giving perfect
//! round-trip. This makes it a strong detection target for
//! `AdaptiveTonemapper::detect_standard`.
//!
//! The default parameters (k1=0.83802, k2=15.09968, k3=0.74204,
//! k4=78.99439) are derived from BT.2408 Annex 4 skin tone analysis
//! with HDR Reference White mapping to 96% SDR.
//!
//! Reference: ITU-R BT.2446-1 (03/2021) §6.

use crate::ToneMap;

/// PQ peak luminance (10 000 nits) — used as the input ceiling so that
/// `f32::MAX` inputs cannot drive the log branch to infinity.
const NITS_CEIL: f32 = 10_000.0;

/// BT.2446 Method C tonemapper.
///
/// Operates on linear-light RGB. Input: `1.0 = hdr_peak_nits`.
/// Output: `1.0 = sdr_peak_nits` (allows super-whites up to 1.09).
///
/// The tone curve (per ITU-R BT.2446-1 §6.1.4, equation 5), with
/// luminance in nits, is:
///
/// ```text
/// Y_SDR = k1 * Y_HDR                                       for Y_HDR <  Y_ip
/// Y_SDR = k2 * ln(Y_HDR / Y_ip - k3) + k4                  for Y_HDR >= Y_ip
/// ```
///
/// where the inflection point `Y_ip = 58.5 / k1` (eq. 6, knee at 80 % SDR =
/// 58.5 nits on a 100-nit display), `k2 = k1 * (1 - k3) * Y_ip` (eq. 7), and
/// `k4 = k1 * Y_ip - k2 * ln(1 - k3)` (eq. 8). The default constants
/// (k1=0.83802, k2=15.09968, k3=0.74204, k4=78.99439) come from §6.1.4
/// eq. (10) — derived for the BT.2408 Annex-4 skin-tone relationship
/// with 75 %-HLG HDR Reference White mapping to 96 % SDR, assuming a
/// 1000-nit HDR / 100-nit SDR system.
///
/// # When to pick this
///
/// Pick when you need an **invertible** tone curve — the inverse is exact
/// modulo the optional crosstalk, so HDR → SDR → HDR round-trips with no
/// quantization error. This makes Method C the natural target for
/// gain-map workflows ([`LumaGainMapSplitter`](crate::LumaGainMapSplitter))
/// and for curve detection (`experimental::detect::detect_standard`).
/// Default parameters are tuned for skin-tone preservation per BT.2408
/// Annex 4.
///
/// Reference: ITU-R BT.2446-1 §6 (03/2021).
///
/// # Examples
///
/// ```
/// use zentone::{Bt2446C, ToneMap};
///
/// // 1000 cd/m² HDR content → 100 cd/m² SDR.
/// let curve = Bt2446C::new(1000.0, 100.0);
/// let sdr = curve.map_rgb([2.0, 1.0, 0.5]);
/// assert!(sdr.iter().all(|&c| c.is_finite() && c >= 0.0));
///
/// // Round-trip on luminance is exact (luminance argument is in nits).
/// let y_hdr_nits = 600.0;
/// let y_sdr_nits = curve.tone_curve(y_hdr_nits);
/// let back_nits = curve.inverse_tone_curve(y_sdr_nits);
/// assert!((back_nits - y_hdr_nits).abs() < 1e-2);
/// ```
pub struct Bt2446C {
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    /// Inflection point in nits — `Y_HDR,ip = 58.5 / k1` (eq. 6).
    y_ip: f32,
    /// Scaling factor that takes normalized-to-hdr-peak input into nits.
    hdr_peak_nits: f32,
    /// Scaling factor that takes tone-curve output (nits) into
    /// normalized-to-sdr-peak output.
    sdr_peak_nits: f32,
    alpha: f32, // crosstalk parameter
}

impl Bt2446C {
    /// Create with the default BT.2446 §6 parameters.
    ///
    /// `hdr_peak_nits`: typically 1000.
    /// `sdr_peak_nits`: typically 100 (or 120 for Method C's super-whites).
    pub fn new(hdr_peak_nits: f32, sdr_peak_nits: f32) -> Self {
        Self::with_params(
            hdr_peak_nits,
            sdr_peak_nits,
            0.83802,
            15.09968,
            0.74204,
            78.99439,
            0.0,
        )
    }

    /// Create with custom k1-k4 parameters and crosstalk alpha.
    ///
    /// `alpha`: crosstalk parameter in [0, 0.33]. Controls pre-desaturation
    /// to prevent hue shifts from clipping. 0 = no desaturation.
    pub fn with_params(
        hdr_peak_nits: f32,
        sdr_peak_nits: f32,
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        alpha: f32,
    ) -> Self {
        // Per ITU-R BT.2446-1 §6.1.4 eq. (6): Y_HDR,ip = 58.5 / k1.
        // (80 % SDR = 58.5 nits on a 100-nit SDR display, mapped back
        // through the linear-branch gain k1.)
        let y_ip = 58.5 / k1;
        Self {
            k1,
            k2,
            k3,
            k4,
            y_ip,
            hdr_peak_nits,
            sdr_peak_nits,
            alpha,
        }
    }

    /// Apply the crosstalk matrix (pre-desaturation).
    #[inline]
    fn crosstalk(&self, rgb: [f32; 3]) -> [f32; 3] {
        if self.alpha <= 0.0 {
            return rgb;
        }
        let a = self.alpha;
        let d = 1.0 - 2.0 * a;
        [
            d * rgb[0] + a * rgb[1] + a * rgb[2],
            a * rgb[0] + d * rgb[1] + a * rgb[2],
            a * rgb[0] + a * rgb[1] + d * rgb[2],
        ]
    }

    /// Inverse crosstalk matrix.
    #[inline]
    fn inv_crosstalk(&self, rgb: [f32; 3]) -> [f32; 3] {
        if self.alpha <= 0.0 {
            return rgb;
        }
        // The 3×3 crosstalk matrix with parameter α has inverse with
        // parameter −α/(1−3α) when 3α ≠ 1. For small α this is accurate.
        let a = self.alpha;
        let inv_a = -a / (1.0 - 3.0 * a);
        let d = 1.0 - 2.0 * inv_a;
        [
            d * rgb[0] + inv_a * rgb[1] + inv_a * rgb[2],
            inv_a * rgb[0] + d * rgb[1] + inv_a * rgb[2],
            inv_a * rgb[0] + inv_a * rgb[1] + d * rgb[2],
        ]
    }

    /// The piecewise tone curve from ITU-R BT.2446-1 §6.1.4 eq. (5).
    ///
    /// Input and output are luminance in **nits** (cd/m²). The default
    /// constants are calibrated for `Y_HDR ∈ [0, 1000]` → `Y_SDR ∈ [0, ≈118]`,
    /// with `Y_HDR,ip ≈ 69.81 nits` (eq. 6). C0-continuous (and C1-continuous
    /// per eq. (7)) at `Y_HDR,ip`.
    #[inline]
    pub fn tone_curve(&self, y_hdr_nits: f32) -> f32 {
        if y_hdr_nits < self.y_ip {
            self.k1 * y_hdr_nits
        } else {
            // Spec eq. (5): k2 * ln(Y_HDR/Y_ip - k3) + k4.
            // Guard the log argument: at Y = Y_ip we have (1 - k3) ≈ 0.258 > 0,
            // so the guard only fires under pathological inputs (e.g. inverse
            // round-tripping outside the SDR codomain).
            let arg = (y_hdr_nits / self.y_ip - self.k3).max(f32::MIN_POSITIVE);
            self.k2 * libm::logf(arg) + self.k4
        }
    }

    /// Exact algebraic inverse of [`tone_curve`](Self::tone_curve).
    ///
    /// Argument and return value are luminance in nits.
    #[inline]
    pub fn inverse_tone_curve(&self, y_sdr_nits: f32) -> f32 {
        // The linear branch covers Y_HDR ∈ [0, Y_ip), Y_SDR ∈ [0, k1 * Y_ip).
        let threshold = self.k1 * self.y_ip;
        if y_sdr_nits < threshold {
            y_sdr_nits / self.k1
        } else {
            // Inverse of eq. (5) for the log branch:
            //   Y_SDR = k2 ln(Y_HDR/Y_ip - k3) + k4
            //     ⇒  Y_HDR = Y_ip * (exp((Y_SDR - k4)/k2) + k3)
            self.y_ip * (libm::expf((y_sdr_nits - self.k4) / self.k2) + self.k3)
        }
    }
}

impl ToneMap for Bt2446C {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Per spec: take normalized-to-hdr-peak input into absolute nits,
        // run the curve, then back to normalized-to-sdr-peak output.
        // Clamp absolute luminance to NITS_CEIL (10 000 — the PQ peak)
        // so brute-force `f32::MAX` inputs can't take the log argument
        // to infinity and produce NaN through the inf/inf ratio.
        let to_nits = self.hdr_peak_nits;
        let inv_sdr = 1.0 / self.sdr_peak_nits;
        let rgb_nits = [
            (rgb[0].max(0.0) * to_nits).min(NITS_CEIL),
            (rgb[1].max(0.0) * to_nits).min(NITS_CEIL),
            (rgb[2].max(0.0) * to_nits).min(NITS_CEIL),
        ];

        // §6.1.2: Crosstalk (pre-desaturation).
        let ct = self.crosstalk(rgb_nits);

        // §6.1.3: BT.2020 → Y. The luminance weights match the spec's
        // BT.2020→XYZ row 2.
        let y = 0.2627 * ct[0] + 0.6780 * ct[1] + 0.0593 * ct[2];
        if y <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // §6.1.4: Tone map Y in nits.
        let y_sdr = self.tone_curve(y);

        // §6.1.5: Scaling RGB by Y_SDR/Y_HDR is algebraically equivalent
        // to scaling Y in xyY while holding (x, y) fixed (both R, G, B and
        // therefore X, Y, Z scale by the same factor, leaving the
        // chromaticity coordinates invariant). This saves an explicit
        // BT.2020↔XYZ round-trip without changing the result.
        let ratio = y_sdr / y;
        let sdr_nits = [ct[0] * ratio, ct[1] * ratio, ct[2] * ratio];

        // §6.1.6: Inverse crosstalk.
        let sdr = self.inv_crosstalk(sdr_nits);

        // §6.1.4 caps Y_SDR at 100 nits when clamping; equivalently
        // post-mapping RGB is clamped to 109 % (118.4 → 109 by the
        // spec's allowed super-white range). Negative output is clipped
        // to 0 per the spec.
        [
            (sdr[0] * inv_sdr).clamp(0.0, 1.09),
            (sdr[1] * inv_sdr).clamp(0.0, 1.09),
            (sdr[2] * inv_sdr).clamp(0.0, 1.09),
        ]
    }

    fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        archmage::incant!(
            crate::simd::curves::bt2446c_tier(
                strip,
                self.k1,
                self.k2,
                self.k3,
                self.k4,
                self.y_ip,
                self.alpha,
                self.hdr_peak_nits,
                self.sdr_peak_nits,
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
        let tm = Bt2446C::new(1000.0, 100.0);
        assert_eq!(tm.map_rgb([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn monotonic_neutral_ramp() {
        let tm = Bt2446C::new(1000.0, 100.0);
        let mut last = -1.0_f32;
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let out = tm.map_rgb([v, v, v]);
            assert!(out[0] >= last - 1e-5, "mono at {v}: {} < {last}", out[0]);
            last = out[0];
        }
    }

    #[test]
    fn tone_curve_inverse_roundtrip() {
        let tm = Bt2446C::new(1000.0, 100.0);
        for &nits in &[0.0_f32, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0] {
            let sdr = tm.tone_curve(nits);
            let back = tm.inverse_tone_curve(sdr);
            assert!((back - nits).abs() < 0.1, "roundtrip at {nits}: got {back}");
        }
    }

    /// Spec sanity: 1000-nit HDR white maps to ≈118.39 nits SDR (spec
    /// §6.1.4 — "Map [0,1000] to [0,118.3868]"). This is the load-bearing
    /// reference data point and is documented in every conformant
    /// implementation.
    #[test]
    fn spec_reference_white_maps_to_118_nits() {
        let tm = Bt2446C::new(1000.0, 100.0);
        let sdr = tm.tone_curve(1000.0);
        assert!(
            (sdr - 118.3868).abs() < 1e-1,
            "Y_SDR(1000) = {sdr} (expected ≈118.3868 per BT.2446-1 §6.1.4)"
        );
    }

    /// Spec sanity: the inflection point is at 58.5/k1 ≈ 69.81 nits
    /// (BT.2446-1 §6.1.4 eq. 6), not at k4/k1 ≈ 94.26.
    #[test]
    fn inflection_point_matches_spec() {
        let tm = Bt2446C::new(1000.0, 100.0);
        let expected = 58.5_f32 / 0.83802;
        assert!(
            (tm.y_ip - expected).abs() < 1e-3,
            "y_ip = {} (expected {expected} per eq. 6)",
            tm.y_ip
        );
    }

    /// Round-trip a full RGB pixel through `map_rgb` and the documented
    /// linear-branch inverse. Method C is selling itself on exactly this
    /// property — verify it survives the scaling and crosstalk chain. In
    /// the linear branch the curve is `Y_SDR = k1·Y_HDR`, so the inverse
    /// for a peak-1000-nit / peak-100-nit caller is `hdr = sdr * (100 /
    /// (k1 * 1000)) = sdr / (10 * k1)`.
    #[test]
    fn rgb_roundtrip_via_inverse() {
        let tm = Bt2446C::new(1000.0, 100.0);
        // Pick low values that stay in the linear branch of the spec.
        // Y_HDR = LR·R+LG·G+LB·B in nits. For (0.03,0.02,0.005)·1000:
        // Y = 7.88+13.56+0.2965 = 21.7 nits < Y_ip (≈69.8). Linear branch.
        let hdr_in = [0.03_f32, 0.02, 0.005]; // 1.0 = 1000 nits
        let sdr = tm.map_rgb(hdr_in);
        // Inverse: sdr_pct → sdr_nits → Y_HDR_nits → hdr_norm.
        let inv_scale = (1.0 / 0.83802_f32) * (100.0 / 1000.0);
        let hdr_back = [sdr[0] * inv_scale, sdr[1] * inv_scale, sdr[2] * inv_scale];
        for i in 0..3 {
            assert!(
                (hdr_back[i] - hdr_in[i]).abs() < 1e-4,
                "channel {i}: {} vs {}",
                hdr_back[i],
                hdr_in[i]
            );
        }
    }

    #[test]
    fn colored_bounded() {
        let tm = Bt2446C::new(1000.0, 100.0);
        for rgb in [[0.8, 0.2, 0.05], [0.1, 0.9, 0.05], [0.3, 0.3, 0.8]] {
            let out = tm.map_rgb(rgb);
            for (i, c) in out.iter().enumerate() {
                assert!(
                    c.is_finite() && *c >= 0.0 && *c <= 1.2,
                    "Bt2446C({rgb:?})[{i}] = {c}"
                );
            }
        }
    }

    #[test]
    fn crosstalk_roundtrip() {
        let tm = Bt2446C::with_params(1000.0, 100.0, 0.83802, 15.09968, 0.74204, 78.99439, 0.1);
        let rgb = [0.5_f32, 0.3, 0.8];
        let ct = tm.crosstalk(rgb);
        let back = tm.inv_crosstalk(ct);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-5,
                "crosstalk roundtrip[{i}]: {:.5} vs {:.5}",
                back[i],
                rgb[i]
            );
        }
    }
}
