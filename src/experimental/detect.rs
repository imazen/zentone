//! Detect which standard tone mapping curve was used from a fitted
//! AdaptiveTonemapper's LUT.
//!
//! After `AdaptiveTonemapper::fit_luminance` produces a `LuminanceCurve`
//! with a 4096-entry LUT, this module matches it against reference LUTs
//! from known algorithms to identify the original curve and its
//! parameters.
//!
//! # Status
//!
//! Experimental. Behind the `experimental` feature flag.

use crate::{Bt2408Tonemapper, filmic_narkowicz, hable_filmic, reinhard_extended, reinhard_simple};

/// Result of curve detection.
#[derive(Debug, Clone)]
pub struct DetectedCurve {
    /// Name of the detected curve.
    pub name: &'static str,
    /// RMS error between the fitted LUT and the reference curve.
    pub rms_error: f32,
    /// Parameters (curve-specific).
    pub params: DetectedParams,
}

/// Curve-specific parameters extracted from detection.
#[derive(Debug, Clone)]
pub enum DetectedParams {
    /// No parameters (Reinhard simple, Narkowicz, Hable, etc.)
    None,
    /// Extended Reinhard: detected l_max.
    ExtendedReinhard {
        /// Detected maximum luminance parameter.
        l_max: f32,
    },
    /// BT.2408: detected content and display nits.
    Bt2408 {
        /// Detected content peak nits.
        content_nits: f32,
        /// Detected display peak nits.
        display_nits: f32,
    },
}

/// Detect the standard curve from a LUT.
///
/// `lut`: a 4096-entry LUT mapping normalized HDR luminance [0, max_hdr]
/// to SDR luminance [0, ~1]. This is the internal LUT from
/// `AdaptiveTonemapper`'s `LuminanceCurve`.
///
/// `max_hdr`: the maximum HDR luminance the LUT covers.
///
/// Returns the best match if the RMS error is below the threshold, or
/// `None` if no curve matches well enough.
pub fn detect_standard(lut: &[f32], max_hdr: f32, threshold: f32) -> Option<DetectedCurve> {
    let n = lut.len();
    if n < 256 {
        return None;
    }

    let mut best: Option<DetectedCurve> = None;

    // Parameterless curves
    type ScalarCurve = fn(f32) -> f32;
    let parameterless: &[(&str, ScalarCurve)] = &[
        ("Reinhard", reinhard_simple as fn(f32) -> f32),
        ("Narkowicz", filmic_narkowicz),
        ("HableFilmic", hable_filmic),
    ];

    for (name, curve_fn) in parameterless {
        let rms = compute_rms(lut, max_hdr, |x| curve_fn(x).min(1.0));
        if rms < threshold {
            let candidate = DetectedCurve {
                name,
                rms_error: rms,
                params: DetectedParams::None,
            };
            if best.as_ref().is_none_or(|b| rms < b.rms_error) {
                best = Some(candidate);
            }
        }
    }

    // Extended Reinhard: search l_max in [1, max_hdr*2]
    let l_max_candidates = [1.0, 2.0, 4.0, 8.0, 16.0, max_hdr, max_hdr * 2.0];
    for &l_max in &l_max_candidates {
        let rms = compute_rms(lut, max_hdr, |x| reinhard_extended(x, l_max).min(1.0));
        if rms < threshold {
            let candidate = DetectedCurve {
                name: "ExtendedReinhard",
                rms_error: rms,
                params: DetectedParams::ExtendedReinhard { l_max },
            };
            if best.as_ref().is_none_or(|b| rms < b.rms_error) {
                best = Some(candidate);
            }
        }
    }

    // BT.2408: search content/display nits
    let content_nits = [1000.0, 2000.0, 4000.0, 10000.0];
    let display_nits = [100.0, 203.0, 400.0, 1000.0];
    for &cn in &content_nits {
        for &dn in &display_nits {
            if cn <= dn {
                continue;
            }
            let tm = Bt2408Tonemapper::new(cn, dn);
            let rms = compute_rms(lut, max_hdr, |x| {
                // Bt2408 operates on content-normalized linear; scale x by content_nits
                let nits = x * cn;
                tm.tonemap_nits(nits) / dn
            });
            if rms < threshold {
                let candidate = DetectedCurve {
                    name: "Bt2408",
                    rms_error: rms,
                    params: DetectedParams::Bt2408 {
                        content_nits: cn,
                        display_nits: dn,
                    },
                };
                if best.as_ref().is_none_or(|b| rms < b.rms_error) {
                    best = Some(candidate);
                }
            }
        }
    }

    best
}

/// Compute RMS error between a LUT and a reference curve.
fn compute_rms(lut: &[f32], max_hdr: f32, curve: impl Fn(f32) -> f32) -> f32 {
    let n = lut.len();
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;

    for (i, &lut_val) in lut.iter().enumerate() {
        let x = (i as f32 / (n - 1) as f32) * max_hdr;
        let ref_val = curve(x);
        let diff = (lut_val - ref_val) as f64;
        sum_sq += diff * diff;
        count += 1;
    }

    if count == 0 {
        return f32::MAX;
    }
    libm::sqrt(sum_sq / count as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Generate a LUT by sampling a known curve.
    fn generate_lut(max_hdr: f32, curve: impl Fn(f32) -> f32) -> Vec<f32> {
        let n = 4096;
        (0..n)
            .map(|i| {
                let x = (i as f32 / (n - 1) as f32) * max_hdr;
                curve(x)
            })
            .collect()
    }

    #[test]
    fn detect_reinhard() {
        let lut = generate_lut(4.0, reinhard_simple);
        let result = detect_standard(&lut, 4.0, 0.01);
        assert!(result.is_some(), "should detect reinhard");
        let r = result.unwrap();
        assert_eq!(r.name, "Reinhard");
        assert!(r.rms_error < 1e-5, "rms: {}", r.rms_error);
    }

    #[test]
    fn detect_narkowicz() {
        let lut = generate_lut(4.0, filmic_narkowicz);
        let result = detect_standard(&lut, 4.0, 0.01);
        assert!(result.is_some(), "should detect narkowicz");
        assert_eq!(result.unwrap().name, "Narkowicz");
    }

    #[test]
    fn detect_hable() {
        let lut = generate_lut(4.0, |x| hable_filmic(x).min(1.0));
        let result = detect_standard(&lut, 4.0, 0.01);
        assert!(result.is_some(), "should detect hable");
        assert_eq!(result.unwrap().name, "HableFilmic");
    }

    #[test]
    fn detect_extended_reinhard() {
        let l_max = 4.0;
        let lut = generate_lut(4.0, |x| reinhard_extended(x, l_max));
        let result = detect_standard(&lut, 4.0, 0.01);
        assert!(result.is_some(), "should detect extended reinhard");
        let r = result.unwrap();
        assert_eq!(r.name, "ExtendedReinhard");
        if let DetectedParams::ExtendedReinhard { l_max: detected } = r.params {
            assert!(
                (detected - l_max).abs() < 0.1,
                "l_max: expected {l_max}, got {detected}"
            );
        }
    }

    #[test]
    fn no_match_on_random_lut() {
        let lut: Vec<f32> = (0..4096)
            .map(|i| (i as f32 / 4095.0) * 0.3 + 0.2) // linear, doesn't match any curve
            .collect();
        let result = detect_standard(&lut, 4.0, 0.01);
        assert!(result.is_none(), "random LUT should not match");
    }
}
