//! darktable / Blender-style Filmic spline tone mapping.
//!
//! Parameterized curve with independent toe, linear, and shoulder regions.
//! A log-encoded shaper brings scene-linear input into `[0, 1]`, a rational
//! spline rolls off the toe and shoulder, and a per-pixel desaturation term
//! pulls highlights toward white.

use crate::math::{expf, log2f, powf, sqrtf};

/// Filmic spline configuration parameters.
///
/// Defaults match the darktable/Ansel v2 Filmic module.
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
}

impl CompiledFilmicSpline {
    /// Build a compiled spline from parameters.
    pub fn new(p: &FilmicSplineConfig) -> Self {
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
        let mc2 = (white_display - grey_display) / (white_log - grey_log);
        if mc2.is_finite() {
            min_contrast = min_contrast.max(mc2);
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
        }
    }

    fn compute_rational(p1: [f32; 2], p0: [f32; 2], g: f32) -> (f32, f32, f32, f32) {
        let x = p0[0] - p1[0];
        let y = p0[1] - p1[1];
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

    /// Tone map an RGB triple through the filmic spline.
    ///
    /// `luma_coeffs` are the RGB→luminance weights
    /// (typically [`crate::LUMA_BT709`]).
    pub fn tonemap_rgb(&self, rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
        let mut norm =
            (rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2])
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
    use crate::LUMA_BT709;

    #[test]
    fn default_compiles() {
        let _ = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
    }

    #[test]
    fn default_maps_black_to_near_black() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let out = spline.tonemap_rgb([0.0, 0.0, 0.0], LUMA_BT709);
        for c in out {
            assert!((0.0..0.05).contains(&c), "black should stay dark: {c}");
        }
    }

    #[test]
    fn default_maps_unit_to_unit_range() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let out = spline.tonemap_rgb([1.0, 1.0, 1.0], LUMA_BT709);
        for c in out {
            assert!((0.0..=1.0).contains(&c), "out of range: {c}");
        }
    }

    #[test]
    fn hdr_input_clamps_in_range() {
        let spline = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        let out = spline.tonemap_rgb([8.0, 4.0, 2.0], LUMA_BT709);
        for c in out {
            assert!((0.0..=1.0).contains(&c), "HDR out of range: {c}");
        }
    }
}
