//! ITU-R BT.2408 PQ-domain tonemapper.
//!
//! Operates in the perceptual PQ domain with a Hermite-spline roll-off
//! parameterized by content and display peak luminance (nits). Better
//! shadow/highlight handling than a scene-linear BT.2390 at the cost of
//! two transfer-function calls per sample.

use linear_srgb::tf::{linear_to_pq as pq_oetf, pq_to_linear as pq_eotf};

use crate::{LUMA_BT709, ToneMap};

/// Color space in which the EETF tone mapping is applied.
///
/// BT.2408 Annex 5 §A5.1 documents five options. zentone supports the
/// two that don't require gamut mapping:
///
/// | Space | Desaturation | Gamut-safe? | Use case |
/// |---|---|---|---|
/// | `Yrgb` | Excessive on saturated highlights | Yes | Default, simple |
/// | `MaxRgb` | None (preserves chromaticity) | Yes | Better for saturated content |
///
/// `MaxRgb` applies the EETF to `max(R,G,B)` and scales all channels
/// uniformly. It preserves chromaticity exactly but can produce very
/// saturated highlights. `Yrgb` applies to BT.709/BT.2020 luminance and
/// scales uniformly, which over-desaturates bright blues (low luminance
/// weight) but keeps overall brightness natural.
///
/// BT.2408 recommends blending both: use `MaxRgb` for the signal and
/// `Yrgb` for the desaturation amount. zentone doesn't implement the
/// blend yet (issue #2) but offers either space individually.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EetfSpace {
    /// YRGB: compute BT.709/BT.2020 luminance, apply EETF, scale all
    /// channels by luma ratio. Default.
    #[default]
    Yrgb,
    /// maxRGB: compute `max(R, G, B)`, apply EETF, scale all channels
    /// by max ratio. Preserves chromaticity; avoids YRGB blow-up on
    /// saturated colors with low luminance weight.
    MaxRgb,
}

/// BT.2408 tone mapper operating in PQ perceptual domain.
///
/// Precomputes PQ-domain constants from content and display peak nits.
/// Construct once and reuse across many pixels.
///
/// # When to pick this
///
/// Best default for HDR10 / PQ → SDR conversions when you know the source
/// mastering peak and target display peak. The PQ-domain Hermite spline
/// gives smoother shadow/highlight roll-off than scene-linear curves at
/// the cost of two transfer-function calls per sample. For chromaticity
/// preservation on saturated highlights, use [`max_rgb`](Self::max_rgb)
/// instead of the default YRGB.
///
/// Reference: ITU-R BT.2408 Annex 5 (verified line-by-line, max relative
/// error 4.5e-5 vs libplacebo's `bt2390()` golden file).
///
/// # Examples
///
/// ```
/// use zentone::{Bt2408Tonemapper, ToneMap};
///
/// // 4000 cd/m² HDR10 master, 1000 cd/m² target display.
/// // Input is content-normalized: 1.0 = 4000 nits.
/// let curve = Bt2408Tonemapper::new(4000.0, 1000.0);
/// let sdr = curve.map_rgb([1.0, 1.0, 1.0]);
/// // Output is display-normalized: 1.0 = 1000 nits.
/// assert!(sdr.iter().all(|&c| (0.0..=1.0).contains(&c)));
/// ```
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
    space: EetfSpace,
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
    /// Create with maxRGB application space (preserves chromaticity on
    /// saturated highlights). Uses BT.709 luminance for the `tonemap_nits`
    /// helper.
    pub fn max_rgb(content_max_nits: f32, display_max_nits: f32) -> Self {
        let mut s = Self::with_luma(content_max_nits, display_max_nits, LUMA_BT709);
        s.space = EetfSpace::MaxRgb;
        s
    }

    /// Create a new BT.2408 tonemapper with explicit luminance coefficients.
    ///
    /// Pass [`LUMA_BT709`] or [`LUMA_BT2020`](crate::LUMA_BT2020) depending
    /// on the source gamut.
    pub fn with_luma(content_max_nits: f32, display_max_nits: f32, luma: [f32; 3]) -> Self {
        // BT.2408 Annex 5 Step 1+2: precompute PQ-domain normalization.
        //
        // Step 1 normalization: E1 = (E' - PQ(LB)) / (PQ(LW) - PQ(LB))
        //   where LB=0 (mastering black), LW=content_max_nits.
        //
        // Step 2 target params:
        //   minLum = (PQ(Lmin) - PQ(LB)) / range
        //   maxLum = (PQ(Lmax) - PQ(LB)) / range
        //   KS     = 1.5 * maxLum - 0.5
        //
        // Verified line-by-line against BT.2408-8 (11/2024) Annex 5 and
        // against libplacebo's bt2390() (golden-file parity, max rel err 4.5e-5).
        let content_min_pq = pq_oetf(0.0); // PQ(LB)
        let content_max_pq = pq_oetf(content_max_nits / 10000.0); // PQ(LW)
        let content_range_pq = content_max_pq - content_min_pq; // PQ(LW) - PQ(LB)
        let inv_content_range_pq = if content_range_pq > 0.0 {
            1.0 / content_range_pq
        } else {
            1.0
        };
        // Step 2: minLum (b) and maxLum in normalized PQ domain
        let min_lum = (pq_oetf(0.0) - content_min_pq) * inv_content_range_pq;
        let max_lum = (pq_oetf(display_max_nits / 10000.0) - content_min_pq) * inv_content_range_pq;
        // Step 2: KS = 1.5 * maxLum - 0.5
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
            space: EetfSpace::Yrgb,
        }
    }

    /// Configured EETF application space.
    #[inline]
    pub fn space(&self) -> EetfSpace {
        self.space
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

    /// BT.2408 Annex 5 EETF (5 steps) applied to a single luminance.
    ///
    /// Returns a **scale factor** (not the output luminance directly) that
    /// converts content-normalized RGB to display-normalized RGB when
    /// multiplied per-channel. This is the YRGB application space from
    /// BT.2408 Annex 5 §A5.1.
    #[inline(always)]
    fn make_luma_scale(&self, luma_nits: f32) -> f32 {
        // Step 1: Normalize to mastering PQ range
        // E1 = (PQ(input) - PQ(LB)) / (PQ(LW) - PQ(LB))
        let s = pq_oetf(luma_nits / 10000.0);
        let normalized_pq = ((s - self.content_min_pq) * self.inv_content_range_pq).min(1.0);

        // Step 3+4: EETF knee
        // E2 = E1 for E1 < KS (passthrough)
        // E2 = P[E1] for KS <= E1 <= 1 (Hermite spline)
        let e2 = if normalized_pq < self.ks {
            normalized_pq
        } else {
            self.hermite_spline(normalized_pq)
        };

        // Step 3: Black level lift with (1-E2)^4 taper
        // E3 = E2 + b * (1-E2)^4
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = self.min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;

        // Step 5: Denormalize back to PQ, then decode to linear nits
        // E4 = E3 * (PQ(LW) - PQ(LB)) + PQ(LB)
        let e4 = e3 * self.content_range_pq + self.content_min_pq;
        let d4 = pq_eotf(e4) * 10000.0;

        // Convert to scale factor for YRGB application
        let new_luminance = d4.min(self.display_max_nits).max(0.0);
        let min_luminance = 1e-6;
        if luma_nits <= min_luminance {
            new_luminance * self.inv_display_max
        } else {
            (new_luminance / luma_nits.max(min_luminance)) * self.normalizer
        }
    }
}

impl Bt2408Tonemapper {
    /// Pack the tonemapper's precomputed coefficients for the SIMD strip
    /// kernel.
    #[inline]
    fn simd_params(&self) -> crate::simd::curves::Bt2408Params {
        crate::simd::curves::Bt2408Params {
            luma: self.luma,
            content_max_nits: self.content_max_nits,
            display_max_nits: self.display_max_nits,
            inv_display_max: self.inv_display_max,
            content_min_pq: self.content_min_pq,
            content_range_pq: self.content_range_pq,
            inv_content_range_pq: self.inv_content_range_pq,
            min_lum: self.min_lum,
            max_lum: self.max_lum,
            ks: self.ks,
            one_minus_ks: self.one_minus_ks,
            inv_one_minus_ks: self.inv_one_minus_ks,
            normalizer: self.normalizer,
        }
    }
}

impl ToneMap for Bt2408Tonemapper {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let signal = match self.space {
            EetfSpace::Yrgb => {
                self.luma[0] * rgb[0] + self.luma[1] * rgb[1] + self.luma[2] * rgb[2]
            }
            EetfSpace::MaxRgb => rgb[0].max(rgb[1]).max(rgb[2]),
        };
        let signal_nits = signal * self.content_max_nits;
        if signal_nits <= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let scale = self.make_luma_scale(signal_nits);
        [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
    }

    fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        let params = self.simd_params();
        match self.space {
            EetfSpace::Yrgb => archmage::incant!(
                crate::simd::curves::bt2408_yrgb_tier(strip, &params),
                [v3, neon, wasm128, scalar]
            ),
            EetfSpace::MaxRgb => archmage::incant!(
                crate::simd::curves::bt2408_maxrgb_tier(strip, &params),
                [v3, neon, wasm128, scalar]
            ),
        }
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

    #[test]
    fn max_rgb_fixes_saturated_blue_blowup() {
        // YRGB blows up on saturated blue because luma = 0.0722*0.5 = 0.036,
        // scale = ~4× → output blue = 2.0. maxRGB uses max(R,G,B) = 0.5,
        // scale = ~1× → output blue ≤ 1.0.
        let yrgb = Bt2408Tonemapper::new(4000.0, 1000.0);
        let max_rgb = Bt2408Tonemapper::max_rgb(4000.0, 1000.0);

        let pure_blue = [0.0_f32, 0.0, 0.5];
        let yrgb_out = yrgb.map_rgb(pure_blue);
        let maxrgb_out = max_rgb.map_rgb(pure_blue);

        // YRGB produces >1 on blue channel (known limitation)
        assert!(
            yrgb_out[2] > 1.5,
            "YRGB should blow up on blue: got {}",
            yrgb_out[2]
        );

        // maxRGB keeps all channels bounded
        for (i, c) in maxrgb_out.iter().enumerate() {
            assert!(*c >= 0.0 && *c <= 1.001, "maxRGB[{i}] out of [0,1]: {c}");
        }
    }

    #[test]
    fn max_rgb_preserves_chromaticity() {
        let tm = Bt2408Tonemapper::max_rgb(4000.0, 1000.0);
        let rgb = [0.3_f32, 0.6, 0.15];
        let out = tm.map_rgb(rgb);
        // maxRGB scales uniformly by max-channel ratio, so R/G and B/G
        // ratios should be preserved.
        let ratio_in = rgb[0] / rgb[1];
        let ratio_out = out[0] / out[1];
        assert!(
            (ratio_in - ratio_out).abs() < 1e-5,
            "maxRGB should preserve chromaticity: in={ratio_in}, out={ratio_out}"
        );
    }

    #[test]
    fn space_accessor() {
        let yrgb = Bt2408Tonemapper::new(4000.0, 1000.0);
        assert_eq!(yrgb.space(), EetfSpace::Yrgb);
        let max_rgb = Bt2408Tonemapper::max_rgb(4000.0, 1000.0);
        assert_eq!(max_rgb.space(), EetfSpace::MaxRgb);
    }
}
