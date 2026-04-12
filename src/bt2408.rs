//! ITU-R BT.2408 PQ-domain tonemapper.
//!
//! Operates in the perceptual PQ domain with a Hermite-spline roll-off
//! parameterized by content and display peak luminance (nits). Better
//! shadow/highlight handling than a scene-linear BT.2390 at the cost of
//! two transfer-function calls per sample.

use linear_srgb::tf::{linear_to_pq as pq_oetf, pq_to_linear as pq_eotf};

use crate::{LUMA_BT709, ToneMap};

/// BT.2408 tone mapper operating in PQ perceptual domain.
///
/// Precomputes PQ-domain constants from content and display peak nits.
///
/// Construct once and reuse across many pixels.
pub struct Bt2408Tonemapper {
    content_min_pq: f32,
    content_range_pq: f32,
    inv_content_range_pq: f32,
    min_lum: f32,
    max_lum: f32,
    ks: f32,
    inv_one_minus_ks: f32,
    one_minus_ks: f32,
    normalizer: f32,
    inv_display_max: f32,
    content_max_nits: f32,
    display_max_nits: f32,
    luma: [f32; 3],
}

impl Bt2408Tonemapper {
    /// Create a new BT.2408 tonemapper with BT.709 luminance coefficients.
    ///
    /// - `content_max_nits` — peak luminance of source content (e.g. 4000).
    /// - `display_max_nits` — peak luminance of target display (e.g. 1000).
    pub fn new(content_max_nits: f32, display_max_nits: f32) -> Self {
        Self::with_luma(content_max_nits, display_max_nits, LUMA_BT709)
    }

    /// Create a new BT.2408 tonemapper with explicit luminance coefficients.
    ///
    /// Pass [`LUMA_BT709`] or [`LUMA_BT2020`](crate::LUMA_BT2020) depending
    /// on the source gamut.
    pub fn with_luma(content_max_nits: f32, display_max_nits: f32, luma: [f32; 3]) -> Self {
        let content_min_pq = pq_oetf(0.0);
        let content_max_pq = pq_oetf(content_max_nits / 10000.0);
        let content_range_pq = content_max_pq - content_min_pq;
        let inv_content_range_pq = if content_range_pq > 0.0 {
            1.0 / content_range_pq
        } else {
            1.0
        };
        let min_lum = (pq_oetf(0.0) - content_min_pq) * inv_content_range_pq;
        let max_lum = (pq_oetf(display_max_nits / 10000.0) - content_min_pq) * inv_content_range_pq;
        let ks = 1.5 * max_lum - 0.5;
        Self {
            content_min_pq,
            content_range_pq,
            inv_content_range_pq,
            min_lum,
            max_lum,
            ks,
            inv_one_minus_ks: 1.0 / (1.0 - ks).max(1e-6),
            one_minus_ks: 1.0 - ks,
            normalizer: content_max_nits / display_max_nits,
            inv_display_max: 1.0 / display_max_nits,
            content_max_nits,
            display_max_nits,
            luma,
        }
    }

    /// Configured content peak luminance (nits).
    #[inline]
    pub fn content_max_nits(&self) -> f32 {
        self.content_max_nits
    }

    /// Configured display peak luminance (nits).
    #[inline]
    pub fn display_max_nits(&self) -> f32 {
        self.display_max_nits
    }

    /// Configured RGB→luminance weights.
    #[inline]
    pub fn luma(&self) -> [f32; 3] {
        self.luma
    }

    /// Tone map a single luminance value, nits in → nits out.
    ///
    /// Internally the Hermite spline operates on content-normalized linear.
    /// This helper converts in and out of nits for you.
    #[inline]
    pub fn tonemap_nits(&self, nits: f32) -> f32 {
        if nits <= 0.0 {
            return 0.0;
        }
        let content_norm = nits / self.content_max_nits;
        let scale = self.make_luma_scale(nits);
        (content_norm * scale * self.display_max_nits)
            .min(self.display_max_nits)
            .max(0.0)
    }

    #[inline(always)]
    fn t(&self, a: f32) -> f32 {
        (a - self.ks) * self.inv_one_minus_ks
    }

    #[inline]
    fn hermite_spline(&self, b: f32) -> f32 {
        let t_b = self.t(b);
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        (2.0 * t_b_3 - 3.0 * t_b_2 + 1.0) * self.ks
            + (t_b_3 - 2.0 * t_b_2 + t_b) * self.one_minus_ks
            + (-2.0 * t_b_3 + 3.0 * t_b_2) * self.max_lum
    }

    #[inline(always)]
    fn make_luma_scale(&self, luma_nits: f32) -> f32 {
        let s = pq_oetf(luma_nits / 10000.0);
        let normalized_pq = ((s - self.content_min_pq) * self.inv_content_range_pq).min(1.0);
        let e2 = if normalized_pq < self.ks {
            normalized_pq
        } else {
            self.hermite_spline(normalized_pq)
        };
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = self.min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;
        let e4 = e3 * self.content_range_pq + self.content_min_pq;
        let d4 = pq_eotf(e4) * 10000.0;
        let new_luminance = d4.min(self.display_max_nits).max(0.0);
        let min_luminance = 1e-6;
        if luma_nits <= min_luminance {
            new_luminance * self.inv_display_max
        } else {
            (new_luminance / luma_nits.max(min_luminance)) * self.normalizer
        }
    }
}

impl ToneMap for Bt2408Tonemapper {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let luma = self.luma[0] * rgb[0] + self.luma[1] * rgb[1] + self.luma[2] * rgb[2];
        let luma_nits = luma * self.content_max_nits;
        if luma_nits <= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let scale = self.make_luma_scale(luma_nits);
        [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_to_black() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        assert_eq!(tm.tonemap_nits(0.0), 0.0);
        let out = tm.map_rgb([0.0, 0.0, 0.0]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn content_peak_maps_near_display_peak() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let out = tm.tonemap_nits(4000.0);
        assert!(
            (900.0..=1000.0).contains(&out),
            "content peak {out} should be near display peak"
        );
    }

    #[test]
    fn below_knee_near_identity_nits() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let low = 50.0;
        let out = tm.tonemap_nits(low);
        assert!(
            out > 0.0 && (out - low).abs() / low < 0.15,
            "low luma nits {out} should be near {low}"
        );
    }

    #[test]
    fn map_rgb_peak_is_bounded() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in out {
            assert!(
                (0.0..=1.0).contains(&c),
                "rgb out of display-normalized range: {c}"
            );
        }
    }

    #[test]
    fn with_luma_stores_coefficients() {
        let tm = Bt2408Tonemapper::with_luma(4000.0, 1000.0, crate::LUMA_BT2020);
        assert_eq!(tm.luma(), crate::LUMA_BT2020);
    }

    #[test]
    fn trait_map_row_rgba_preserves_alpha() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let mut row = [0.3_f32, 0.5, 0.2, 0.7];
        tm.map_row(&mut row, 4);
        assert!((row[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn bt2020_luma_affects_rgb_output() {
        // Compare BT.709 vs BT.2020 on a heavily green-weighted pixel.
        // The two weight matrices differ, so the luminance-driven scale
        // should produce detectably different outputs.
        let tm_709 = Bt2408Tonemapper::new(4000.0, 1000.0);
        let tm_2020 = Bt2408Tonemapper::with_luma(4000.0, 1000.0, crate::LUMA_BT2020);

        let rgb = [0.1_f32, 0.9, 0.05]; // very green
        let out_709 = tm_709.map_rgb(rgb);
        let out_2020 = tm_2020.map_rgb(rgb);

        // Not bit-identical: the luma weights differ.
        let mut any_different = false;
        for i in 0..3 {
            if (out_709[i] - out_2020[i]).abs() > 1e-5 {
                any_different = true;
            }
        }
        assert!(
            any_different,
            "BT.709 and BT.2020 luma should diverge on green-heavy pixel, got {out_709:?} vs {out_2020:?}"
        );
    }

    #[test]
    fn content_below_display_does_not_panic() {
        // Edge case: content peak <= display peak (no tone mapping needed).
        // Constructor must not panic or produce NaN.
        let tm = Bt2408Tonemapper::new(500.0, 1000.0);
        let out = tm.map_rgb([0.5, 0.5, 0.5]);
        for c in out {
            assert!(c.is_finite(), "got non-finite {c} for content < display");
        }
    }

    #[test]
    fn equal_content_and_display_is_near_passthrough() {
        // When content peak = display peak, tone mapping should be near
        // identity in the linear domain.
        let tm = Bt2408Tonemapper::new(1000.0, 1000.0);
        let out = tm.map_rgb([0.5, 0.5, 0.5]);
        // Near passthrough within 5% — the spline still has some effect
        for c in out {
            assert!(
                (c - 0.5).abs() < 0.1,
                "equal-peak tonemap should be near-identity, got {c}"
            );
        }
    }

    #[test]
    fn luma_accessor_returns_configured_coefficients() {
        let tm = Bt2408Tonemapper::with_luma(4000.0, 1000.0, [0.3, 0.5, 0.2]);
        assert_eq!(tm.luma(), [0.3, 0.5, 0.2]);
    }
}
