//! DNG camera profile tone curve.
//!
//! DNG profiles carry a `ProfileToneCurve` as 257 (x, y) control points that
//! define an S-curve mapping linear [0, 1] input to linear [0, 1] output.
//! This module expands those points into a 4097-entry LUT and evaluates it
//! with linear interpolation.

use alloc::vec::Vec;

/// Size of the expanded LUT (plus one sentinel for linear interpolation).
const LUT_SIZE: usize = 4096;

/// DNG ProfileToneCurve — a precomputed LUT-based tone curve.
///
/// Built from 257 (x, y) control points (typical DNG camera profile size) or
/// from a pre-built LUT. Evaluation uses 4096-entry interpolation.
#[derive(Clone, Debug)]
pub struct ProfileToneCurve {
    /// 4097 entries (4096 + 1 sentinel) mapping `[0, 1]` → `[0, 1]`.
    lut: Vec<f32>,
}

impl ProfileToneCurve {
    /// Build from raw DNG tone curve data (`n_points` × 2 floats, `[x0, y0, x1, y1, …]`).
    ///
    /// Returns `None` if there are fewer than 2 points.
    pub fn from_xy_pairs(tc_data: &[f32]) -> Option<Self> {
        let n_points = tc_data.len() / 2;
        if n_points < 2 {
            return None;
        }
        let points: Vec<(f32, f32)> = (0..n_points)
            .map(|i| (tc_data[i * 2], tc_data[i * 2 + 1]))
            .collect();
        let lut: Vec<f32> = (0..=LUT_SIZE)
            .map(|i| {
                let x = i as f32 / LUT_SIZE as f32;
                interpolate_curve(&points, x)
            })
            .collect();
        Some(Self { lut })
    }

    /// Build from a pre-built LUT (must have exactly `LUT_SIZE + 1` entries).
    pub fn from_lut(lut: Vec<f32>) -> Option<Self> {
        if lut.len() != LUT_SIZE + 1 {
            return None;
        }
        Some(Self { lut })
    }

    /// Linear identity curve (passthrough).
    pub fn identity() -> Self {
        let lut: Vec<f32> = (0..=LUT_SIZE).map(|i| i as f32 / LUT_SIZE as f32).collect();
        Self { lut }
    }

    /// Evaluate the curve at a single value in `[0, 1]`.
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let idx_f = x * LUT_SIZE as f32;
        let idx = (idx_f as usize).min(LUT_SIZE - 1);
        let frac = idx_f - idx as f32;
        self.lut[idx] * (1.0 - frac) + self.lut[idx + 1] * frac
    }

    /// Apply per-channel to an RGB triple.
    #[inline]
    pub fn apply_per_channel(&self, rgb: [f32; 3]) -> [f32; 3] {
        [self.eval(rgb[0]), self.eval(rgb[1]), self.eval(rgb[2])]
    }

    /// Apply luminance-preserving to an RGB triple.
    ///
    /// Maps the luminance through the curve, then scales all channels by the
    /// same ratio to preserve hue.
    #[inline]
    pub fn apply_lum_preserving(&self, rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
        let lum = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
        if lum <= 1e-10 {
            return [0.0, 0.0, 0.0];
        }
        let mapped = self.eval(lum.min(1.0));
        let ratio = mapped / lum;
        [
            (rgb[0] * ratio).min(1.0),
            (rgb[1] * ratio).min(1.0),
            (rgb[2] * ratio).min(1.0),
        ]
    }

    /// Apply to a full row of interleaved pixel data (per-channel mode).
    ///
    /// Alpha (channel index 3) is passed through unchanged when `channels == 4`.
    pub fn apply_row_per_channel(&self, row: &mut [f32], channels: usize) {
        debug_assert!(channels == 3 || channels == 4);
        for chunk in row.chunks_exact_mut(channels) {
            chunk[0] = self.eval(chunk[0]);
            chunk[1] = self.eval(chunk[1]);
            chunk[2] = self.eval(chunk[2]);
        }
    }

    /// Apply to a full row of interleaved pixel data (luminance-preserving mode).
    pub fn apply_row_lum_preserving(
        &self,
        row: &mut [f32],
        channels: usize,
        luma_coeffs: [f32; 3],
    ) {
        debug_assert!(channels == 3 || channels == 4);
        for chunk in row.chunks_exact_mut(channels) {
            let lum =
                chunk[0] * luma_coeffs[0] + chunk[1] * luma_coeffs[1] + chunk[2] * luma_coeffs[2];
            if lum > 1e-10 {
                let mapped = self.eval(lum.min(1.0));
                let ratio = mapped / lum;
                chunk[0] = (chunk[0] * ratio).min(1.0);
                chunk[1] = (chunk[1] * ratio).min(1.0);
                chunk[2] = (chunk[2] * ratio).min(1.0);
            } else {
                chunk[0] = 0.0;
                chunk[1] = 0.0;
                chunk[2] = 0.0;
            }
        }
    }
}

/// Linear interpolation in a sorted list of (x, y) control points.
fn interpolate_curve(points: &[(f32, f32)], x: f32) -> f32 {
    if points.is_empty() {
        return x;
    }
    if x <= points[0].0 {
        return points[0].1;
    }
    if x >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }
    let mut lo = 0;
    let mut hi = points.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if points[mid].0 <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = points[hi].0 - points[lo].0;
    if dx <= 0.0 {
        return points[lo].1;
    }
    let t = (x - points[lo].0) / dx;
    points[lo].1 * (1.0 - t) + points[hi].1 * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn identity_passes_through() {
        let curve = ProfileToneCurve::identity();
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert!(
                (curve.eval(x) - x).abs() < 1e-3,
                "identity eval({x}) = {}",
                curve.eval(x)
            );
        }
    }

    #[test]
    fn from_pairs_matches_controls_roughly() {
        let pts = vec![0.0_f32, 0.0, 0.5, 0.25, 1.0, 1.0];
        let curve = ProfileToneCurve::from_xy_pairs(&pts).unwrap();
        assert!(curve.eval(0.0) < 1e-3);
        assert!((curve.eval(0.5) - 0.25).abs() < 1e-2);
        assert!((curve.eval(1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn from_lut_wrong_length_rejected() {
        let short = vec![0.0_f32; 100];
        assert!(ProfileToneCurve::from_lut(short).is_none());
    }

    #[test]
    fn apply_row_rgba_alpha_untouched() {
        let curve = ProfileToneCurve::identity();
        let mut row = [0.5_f32, 0.5, 0.5, 0.42];
        curve.apply_row_per_channel(&mut row, 4);
        assert!((row[3] - 0.42).abs() < 1e-6);
    }
}
