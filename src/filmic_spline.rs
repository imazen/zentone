//! darktable / Blender-style Filmic spline tone mapping.
//!
//! Parameterized curve with independent toe, linear, and shoulder regions.
//! A log-encoded shaper brings scene-linear input into `[0, 1]`, a rational
//! spline rolls off the toe and shoulder, and a per-pixel desaturation term
//! pulls highlights toward white.

use crate::math::{expf, log2f, powf, sqrtf};
use crate::{LUMA_BT709, ToneMap};

/// Filmic spline configuration parameters.
///
/// The rational spline math follows darktable's `filmicrgb.c` (V3 path,
/// GPL-3.0, verified against commit `a193e27`). However, the **default
/// parameter values differ** from darktable's defaults — zentone's
/// defaults are tuned for linear-light output (no display gamma) and a
/// wider linear latitude, while darktable defaults target a display
/// transfer with `output_power = 4.0`.
///
/// | Parameter | darktable | zentone |
/// |---|---|---|
/// | `output_power` | 4.0 | 1.0 (linear) |
/// | `white_point_source` | 4.0 EV | 3.0 EV |
/// | `latitude` | 0.01% | 33.0% |
/// | `contrast` | 1.0 | 1.18 |
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmicSplineConfig {
    /// Output power (gamma). Default: 1.0.
    pub output_power: f32,
    /// Latitude percentage (0–100). Default: 33.0.
    pub latitude: f32,
    /// White point in EV (stops above middle gray). Default: 3.0.
    pub white_point_source: f32,
    /// Black point in EV (stops below middle gray). Default: -8.0.
    pub black_point_source: f32,
    /// Contrast at middle gray. Default: 1.18.
    pub contrast: f32,
    /// Target black luminance (%). Default: 0.01517634.
    pub black_point_target: f32,
    /// Target middle gray (%). Default: 18.45.
    pub grey_point_target: f32,
    /// Target white luminance (%). Default: 100.0.
    pub white_point_target: f32,
    /// Balance (-50 to 50). Default: 0.0.
    pub balance: f32,
    /// Extreme luminance saturation. Default: 0.0.
    pub saturation: f32,
}

impl Default for FilmicSplineConfig {
    fn default() -> Self {
        Self {
            output_power: 1.0,
            latitude: 33.0,
            white_point_source: 3.0,
            black_point_source: -8.0,
            contrast: 1.18,
            black_point_target: 0.01517634,
            grey_point_target: 18.45,
            white_point_target: 100.0,
            balance: 0.0,
            saturation: 0.0,
        }
    }
}

/// Compiled filmic spline (precomputed from [`FilmicSplineConfig`]).
#[derive(Debug, Clone)]
pub struct CompiledFilmicSpline {
    m1: [f32; 3],
    m2: [f32; 3],
    m3: [f32; 3],
    m4: [f32; 3],
    latitude_min: f32,
    latitude_max: f32,
    grey_source: f32,
    black_source: f32,
    dynamic_range: f32,
    sigma_toe: f32,
    sigma_shoulder: f32,
    saturation: f32,
    luma: [f32; 3],
}

impl CompiledFilmicSpline {
    /// Build a compiled spline from parameters with BT.709 luminance weights.
    pub fn new(p: &FilmicSplineConfig) -> Self {
        Self::with_luma(p, LUMA_BT709)
    }

    /// Build a compiled spline with explicit luminance weights.
    pub fn with_luma(p: &FilmicSplineConfig, luma: [f32; 3]) -> Self {
        let hardness = p.output_power;
        let grey_display = powf(0.1845, 1.0 / hardness);
        let latitude = p.latitude.clamp(0.0, 100.0) / 100.0;
        let white_source = p.white_point_source;
        let black_source = p.black_point_source;
        let dynamic_range = white_source - black_source;
        let grey_log = black_source.abs() / dynamic_range;
        let white_log = 1.0_f32;
        let black_log = 0.0_f32;
        let black_display = powf(
            p.black_point_target.clamp(0.0, p.grey_point_target) / 100.0,
            1.0 / hardness,
        );
        let white_display = powf(
            p.white_point_target.max(p.grey_point_target) / 100.0,
            1.0 / hardness,
        );
        let balance = p.balance.clamp(-50.0, 50.0) / 100.0;
        let slope = p.contrast * dynamic_range / 8.0;
        let mut min_contrast = 1.0_f32;
        // Geometric constraint: the linear segment's slope must be steep
        // enough to reach both the white and black display levels from
        // grey. darktable V3 checks both directions; an earlier zentone
        // version only checked white-to-grey.
        let mc_white = (white_display - grey_display) / (white_log - grey_log);
        if mc_white.is_finite() {
            min_contrast = min_contrast.max(mc_white);
        }
        let mc_black = (grey_display - black_display) / (grey_log - black_log);
        if mc_black.is_finite() {
            min_contrast = min_contrast.max(mc_black);
        }
        const SAFETY_MARGIN: f32 = 0.01;
        min_contrast += SAFETY_MARGIN;
        let mut contrast = slope / (hardness * powf(grey_display, hardness - 1.0));
        contrast = contrast.clamp(min_contrast, 100.0);
        let linear_intercept = grey_display - contrast * grey_log;
        let xmin = (black_display + SAFETY_MARGIN * (white_display - black_display)
            - linear_intercept)
            / contrast;
        let xmax =
            (white_display - SAFETY_MARGIN * (white_display - black_display) - linear_intercept)
                / contrast;
        let mut toe_log = (1.0 - latitude) * grey_log + latitude * xmin;
        let mut shoulder_log = (1.0 - latitude) * grey_log + latitude * xmax;
        let balance_correction = if balance > 0.0 {
            2.0 * balance * (shoulder_log - grey_log)
        } else {
            2.0 * balance * (grey_log - toe_log)
        };
        toe_log -= balance_correction;
        shoulder_log -= balance_correction;
        toe_log = toe_log.max(xmin);
        shoulder_log = shoulder_log.min(xmax);
        let toe_display = toe_log * contrast + linear_intercept;
        let shoulder_display = shoulder_log * contrast + linear_intercept;
        let latitude_min = toe_log;
        let latitude_max = shoulder_log;
        let saturation = 2.0 * p.saturation / 100.0 + 1.0;
        let sigma_toe = {
            let v = latitude_min / 3.0;
            v * v
        };
        let sigma_shoulder = {
            let v = (1.0 - latitude_max) / 3.0;
            v * v
        };
        let m2_2 = contrast;
        let m1_2 = toe_display - m2_2 * toe_log;
        let (m1_0, m2_0, m3_0, m4_0) =
            Self::compute_rational([black_log, black_display], [toe_log, toe_display], contrast);
        let (m1_1, m2_1, m3_1, m4_1) = Self::compute_rational(
            [white_log, white_display],
            [shoulder_log, shoulder_display],
            contrast,
        );
        Self {
            m1: [m1_0, m1_1, m1_2],
            m2: [m2_0, m2_1, m2_2],
            m3: [m3_0, m3_1, 0.0],
            m4: [m4_0, m4_1, 0.0],
            latitude_min,
            latitude_max,
            grey_source: 0.1845,
            black_source,
            dynamic_range,
            sigma_toe,
            sigma_shoulder,
            saturation,
            luma,
        }
    }

    /// Configured RGB→luminance weights.
    #[inline]
    pub fn luma(&self) -> [f32; 3] {
        self.luma
    }

    fn compute_rational(p1: [f32; 2], p0: [f32; 2], g: f32) -> (f32, f32, f32, f32) {
        // darktable uses different subtraction orders for the toe
        // (x = P0-P1) vs shoulder (x = P1-P0) to keep x and y positive.
        // We use abs() so argument order doesn't matter — both the
        // toe and shoulder produce the same positive geometric distances.
        let x = (p0[0] - p1[0]).abs();
        let y = (p0[1] - p1[1]).abs();
        let jx_pre = x * g / y + 1.0;
        let jx = (jx_pre * jx_pre).max(4.0);
        let b = g / (2.0 * y) + (sqrtf(jx - 4.0) - 1.0) / (2.0 * x);
        let c = y / g * (b * x * x + x) / (b * x * x + x - y / g);
        let a = c * g;
        (a, b, c, p0[1])
    }

    /// Apply the spline to a single value in the log-encoded domain.
    #[inline]
    pub fn apply_spline(&self, x: f32) -> f32 {
        if x < self.latitude_min {
            let xi = self.latitude_min - x;
            let rat = xi * (xi * self.m2[0] + 1.0);
            self.m4[0] - self.m1[0] * rat / (rat + self.m3[0])
        } else if x > self.latitude_max {
            let xi = x - self.latitude_max;
            let rat = xi * (xi * self.m2[1] + 1.0);
            self.m4[1] + self.m1[1] * rat / (rat + self.m3[1])
        } else {
            self.m1[2] + x * self.m2[2]
        }
    }

    #[inline]
    fn shaper(&self, x: f32) -> f32 {
        ((log2f(x.max(1.525879e-05) / self.grey_source) - self.black_source) / self.dynamic_range)
            .clamp(0.0, 1.0)
    }

    #[inline]
    fn desaturate(&self, x: f32) -> f32 {
        let radius_toe = x;
        let radius_shoulder = 1.0 - x;
        let sat2 = 0.5 / sqrtf(self.saturation);
        let key_toe = expf(-radius_toe * radius_toe / self.sigma_toe * sat2);
        let key_shoulder = expf(-radius_shoulder * radius_shoulder / self.sigma_shoulder * sat2);
        self.saturation - (key_toe + key_shoulder) * self.saturation
    }
}

impl ToneMap for CompiledFilmicSpline {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut norm = (rgb[0] * self.luma[0] + rgb[1] * self.luma[1] + rgb[2] * self.luma[2])
            .max(1.525879e-05);
        let mut ratios = [rgb[0] / norm, rgb[1] / norm, rgb[2] / norm];
        let min_ratio = ratios[0].min(ratios[1]).min(ratios[2]);
        if min_ratio < 0.0 {
            ratios[0] -= min_ratio;
            ratios[1] -= min_ratio;
            ratios[2] -= min_ratio;
        }
        norm = self.shaper(norm);
        let desat = self.desaturate(norm);
        let mapped = self.apply_spline(norm).clamp(0.0, 1.0);
        [
            ((ratios[0] + (1.0 - ratios[0]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0),
            ((ratios[1] + (1.0 - ratios[1]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0),
            ((ratios[2] + (1.0 - ratios[2]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_compiles() {
        let _ = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
    }

    #[test]
    fn default_maps_black_to_near_black() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let out = spline.map_rgb([0.0, 0.0, 0.0]);
        for c in out {
            assert!((0.0..0.05).contains(&c), "black should stay dark: {c}");
        }
    }

    #[test]
    fn default_maps_unit_to_unit_range() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let out = spline.map_rgb([1.0, 1.0, 1.0]);
        for c in out {
            assert!((0.0..=1.0).contains(&c), "out of range: {c}");
        }
    }

    #[test]
    fn hdr_input_clamps_in_range() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let out = spline.map_rgb([8.0, 4.0, 2.0]);
        for c in out {
            assert!((0.0..=1.0).contains(&c), "HDR out of range: {c}");
        }
    }

    #[test]
    fn row_api_preserves_alpha() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let mut row = [0.5_f32, 0.5, 0.5, 0.42];
        spline.map_row(&mut row, 4);
        assert!((row[3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn output_power_changes_midtone_output() {
        // `output_power` sets `hardness`, which changes `grey_display =
        // 0.1845^(1/hardness)`. This is a free parameter (not clamped
        // by the geometric constraints that clamp `contrast` in most
        // reasonable ranges), so two values should always diverge.
        let a = CompiledFilmicSpline::new(&FilmicSplineConfig {
            output_power: 1.0,
            ..Default::default()
        });
        let b = CompiledFilmicSpline::new(&FilmicSplineConfig {
            output_power: 2.0,
            ..Default::default()
        });

        let mid = [0.3_f32, 0.3, 0.3];
        let out_a = a.map_rgb(mid);
        let out_b = b.map_rgb(mid);

        assert!(
            (out_a[0] - out_b[0]).abs() > 1e-3,
            "output_power 1.0 vs 2.0 produced indistinguishable outputs: {out_a:?} vs {out_b:?}"
        );
    }

    #[test]
    fn saturation_changes_colorful_output() {
        // `saturation` modulates the per-pixel desaturation term in
        // shadow/highlight regions. A non-neutral RGB input should
        // diverge between saturation = 0 and saturation = 50.
        let neutral = CompiledFilmicSpline::new(&FilmicSplineConfig {
            saturation: 0.0,
            ..Default::default()
        });
        let punchy = CompiledFilmicSpline::new(&FilmicSplineConfig {
            saturation: 50.0,
            ..Default::default()
        });

        // A color with obvious chroma — red-ish HDR pixel.
        let color = [2.5_f32, 0.6, 0.3];
        let out_n = neutral.map_rgb(color);
        let out_p = punchy.map_rgb(color);

        let mut any_different = false;
        for i in 0..3 {
            if (out_n[i] - out_p[i]).abs() > 1e-4 {
                any_different = true;
            }
        }
        assert!(
            any_different,
            "saturation 0 vs 50 produced identical outputs: {out_n:?} vs {out_p:?}"
        );
    }

    #[test]
    fn contrast_clamped_by_geometric_floor() {
        // Smoke test documenting a quirk of the darktable-style filmic
        // spline: `contrast` values below the geometric `min_contrast`
        // (derived from the other parameters) get clamped, so two
        // "low" contrast values can produce bit-identical output. This
        // is intentional spline behavior, not a bug. Calibration work
        // that tunes contrast needs to use values above the floor.
        let low = CompiledFilmicSpline::new(&FilmicSplineConfig {
            contrast: 0.9,
            ..Default::default()
        });
        let also_low = CompiledFilmicSpline::new(&FilmicSplineConfig {
            contrast: 1.5,
            ..Default::default()
        });
        let mid = [0.3_f32, 0.3, 0.3];
        assert_eq!(
            low.map_rgb(mid),
            also_low.map_rgb(mid),
            "both should hit the same min_contrast clamp"
        );
    }

    #[test]
    fn shoulder_hits_white_point() {
        // Regression test for the shoulder rational spline sign bug.
        // With default params, apply_spline(1.0) should produce a value
        // near white_display (= 1.0 for output_power=1.0). The old buggy
        // code missed by ~4e-4 at default params; with darktable's
        // defaults (output_power=4.0) the error was up to 0.18.
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let at_white = spline.apply_spline(1.0);
        assert!(
            (at_white - 1.0).abs() < 1e-3,
            "spline at x=1.0 should be near white_display (1.0), got {at_white}"
        );
    }

    #[test]
    fn shoulder_with_darktable_defaults() {
        // Uses darktable's actual defaults (output_power=4.0, latitude=0.01,
        // white_point_source=4.0, contrast=1.0) — this is where the
        // shoulder sign bug was most visible (~0.18 error at white point).
        let cfg = FilmicSplineConfig {
            output_power: 4.0,
            latitude: 0.01,
            white_point_source: 4.0,
            contrast: 1.0,
            ..Default::default()
        };
        let spline = CompiledFilmicSpline::new(&cfg);
        let at_white = spline.apply_spline(1.0);

        // white_display = (100/100)^(1/4) = 1.0
        // The spline at x=1.0 should land near white_display.
        assert!(
            (at_white - 1.0).abs() < 0.01,
            "darktable-default spline at x=1.0 should be near 1.0, got {at_white}"
        );
    }

    #[test]
    fn with_luma_stores_bt2020_coefficients() {
        let spline =
            CompiledFilmicSpline::with_luma(&FilmicSplineConfig::default(), crate::LUMA_BT2020);
        assert_eq!(spline.luma(), crate::LUMA_BT2020);
    }

    #[test]
    fn bt2020_luma_diverges_from_bt709_on_green_heavy() {
        let s_709 = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let s_2020 =
            CompiledFilmicSpline::with_luma(&FilmicSplineConfig::default(), crate::LUMA_BT2020);
        let rgb = [0.1_f32, 0.9, 0.05];
        let a = s_709.map_rgb(rgb);
        let b = s_2020.map_rgb(rgb);
        let mut any_different = false;
        for i in 0..3 {
            if (a[i] - b[i]).abs() > 1e-5 {
                any_different = true;
            }
        }
        assert!(any_different, "709 vs 2020 matched on green-heavy pixel");
    }
}
