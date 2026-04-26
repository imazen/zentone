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

/// BT.2446 Method C tonemapper.
///
/// Operates on linear-light RGB. Input: `1.0 = hdr_peak_nits`.
/// Output: `1.0 = sdr_peak_nits` (allows super-whites up to 1.09).
///
/// The tone curve is:
/// ```text
/// Y_SDR = k1 * Y_HDR                                 for Y_HDR < Y_ip
/// Y_SDR = k2 * ln(Y_HDR / Y_ip) + k4                for Y_HDR >= Y_ip
/// ```
///
/// Where `Y_ip = k4 / k1` (the inflection point, derived from C0
/// continuity).
///
/// # When to pick this
///
/// Pick when you need an **invertible** tone curve — the algebraic inverse
/// is exact (modulo optional crosstalk), so HDR → SDR → HDR round-trips
/// with no quantization error. This makes Method C the natural target for
/// gain-map workflows (`experimental::LumaGainMapSplitter`) and for curve
/// detection (`experimental::detect::detect_standard`). Default parameters
/// are tuned for skin-tone preservation per BT.2408 Annex 4.
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
/// // Round-trip on luminance is exact.
/// let y_hdr_pct = 50.0;
/// let y_sdr_pct = 0.83802 * y_hdr_pct; // below inflection: linear
/// let back_pct = curve.inverse_tone_curve(y_sdr_pct);
/// assert!((back_pct - y_hdr_pct).abs() < 1e-3);
/// ```
pub struct Bt2446C {
    k1: f32,
    k2: f32,
    k4: f32,
    y_ip: f32,  // inflection point in percentage (k3 * 100)
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
        _k3: f32,
        k4: f32,
        alpha: f32,
    ) -> Self {
        // C0 continuity: k1 * y_ip = k2 * ln(y_ip/y_ip) + k4 = k4
        // Therefore: y_ip = k4 / k1
        // (k3 is the HDR reference white fraction, NOT the inflection point)
        let y_ip = k4 / k1;
        let _ = (hdr_peak_nits, sdr_peak_nits); // reserved for future ITMO
        Self {
            k1,
            k2,
            k4,
            y_ip,
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

    /// The piecewise tone curve (operates on percentage luminance 0-100).
    ///
    /// C0-continuous at `y_ip = k4 / k1`. The inflection is near HDR peak
    /// (94.3% with default params) so the C1 discontinuity is rarely visible.
    #[inline]
    fn tone_curve(&self, y_hdr_pct: f32) -> f32 {
        if y_hdr_pct < self.y_ip {
            self.k1 * y_hdr_pct
        } else {
            self.k2 * libm::logf(y_hdr_pct / self.y_ip) + self.k4
        }
    }

    /// Inverse tone curve (percentage → percentage).
    #[inline]
    pub fn inverse_tone_curve(&self, y_sdr_pct: f32) -> f32 {
        let threshold = self.k1 * self.y_ip;
        if y_sdr_pct < threshold {
            y_sdr_pct / self.k1
        } else {
            self.y_ip * libm::expf((y_sdr_pct - self.k4) / self.k2)
        }
    }
}

impl ToneMap for Bt2446C {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Convert to percentage of HDR peak (0-100). Clamp to avoid
        // overflow in the log branch at extreme input values.
        let rgb_pct = [
            rgb[0].clamp(0.0, 100.0) * 100.0,
            rgb[1].clamp(0.0, 100.0) * 100.0,
            rgb[2].clamp(0.0, 100.0) * 100.0,
        ];

        // Crosstalk (pre-desaturation)
        let ct = self.crosstalk(rgb_pct);

        // Luminance (BT.2020 weights)
        let y = 0.2627 * ct[0] + 0.6780 * ct[1] + 0.0593 * ct[2];
        if y <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Tone map luminance (percentage domain)
        let y_sdr = self.tone_curve(y);

        // Scale channels by luminance ratio (preserves chromaticity)
        let ratio = y_sdr / y;
        let sdr_pct = [ct[0] * ratio, ct[1] * ratio, ct[2] * ratio];

        // Inverse crosstalk
        let sdr = self.inv_crosstalk(sdr_pct);

        // Normalize from percentage (0-100+) to [0, 1.09+]
        // Method C allows SDR super-whites to 109%
        [
            (sdr[0] / 100.0).clamp(0.0, 1.09),
            (sdr[1] / 100.0).clamp(0.0, 1.09),
            (sdr[2] / 100.0).clamp(0.0, 1.09),
        ]
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
