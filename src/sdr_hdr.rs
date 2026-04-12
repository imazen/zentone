//! Display-referred SDR↔HDR signal-level conversion utilities.
//!
//! These are NOT tone mappers — they're linear scaling and gamma
//! corrections for placing SDR content into an HDR container (or
//! vice versa) while preserving display appearance.
//!
//! Reference: BT.2408-8 §5.1 (display-referred SDR→HDR mapping).
//!
//! The three key conversions:
//!
//! 1. **100→203 cd/m² linear scaling** (§5.1.1): multiply by 2.03.
//!    Maps SDR peak white (BT.1886 100 cd/m²) to HDR Reference White
//!    (BT.2408 203 cd/m²).
//!
//! 2. **OOTF gamma adjustment** (§5.1.3.2): `L_out = L_in^1.15`.
//!    Compensates for the perceptual difference between viewing
//!    SDR at 100 cd/m² vs 203 cd/m². BBC + ARIB subjective tests
//!    confirmed 1.15–1.16 independently.
//!
//! 3. **203↔100 gamma correction** (BT.2408 Annex 11): `L_out = L_in^(1/1.08)`.
//!    Preserves shadow detail at the perceivable black threshold
//!    (0.02 cd/m²) when converting between the two SDR reference
//!    white standards.

use crate::math::powf;

/// Scale SDR linear light from 100 cd/m² peak to 203 cd/m² peak.
///
/// This is a pure linear scaling: `output = input * 2.03`.
/// BT.2408 §5.1.1.
#[inline]
pub fn scale_100_to_203(linear: f32) -> f32 {
    linear * 2.03
}

/// Scale HDR linear light from 203 cd/m² peak to 100 cd/m² peak.
#[inline]
pub fn scale_203_to_100(linear: f32) -> f32 {
    linear / 2.03
}

/// Apply the OOTF gamma adjustment for SDR→HDR appearance matching.
///
/// `L_out = L_in^gamma` where `gamma` is typically 1.15 (BBC) or
/// 1.16 (ARIB). BT.2408 §5.1.3.2.
///
/// Use after [`scale_100_to_203`] to produce an HDR signal that
/// subjectively matches the SDR appearance on a 203 cd/m² display.
#[inline]
pub fn ootf_gamma_adjust(linear: f32, gamma: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }
    powf(linear, gamma)
}

/// Apply the 203↔100 cd/m² gamma correction (BT.2408 Annex 11).
///
/// `gamma = 1/1.08 ≈ 0.926`. Preserves shadow detail at the
/// perceivable black threshold (0.02 cd/m²).
///
/// Use when the SDR base targets 100 cd/m² but the gain map system
/// assumes 203 cd/m² (or vice versa).
#[inline]
pub fn gamma_203_to_100(linear: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }
    powf(linear, 1.0 / 1.08)
}

/// Inverse of [`gamma_203_to_100`].
#[inline]
pub fn gamma_100_to_203(linear: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }
    powf(linear, 1.08)
}

/// BT.2408 recommended OOTF gamma for BBC viewing conditions.
pub const OOTF_GAMMA_BBC: f32 = 1.15;

/// BT.2408 recommended OOTF gamma for ARIB viewing conditions.
pub const OOTF_GAMMA_ARIB: f32 = 1.16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_roundtrip() {
        let v = 0.5_f32;
        let up = scale_100_to_203(v);
        let back = scale_203_to_100(up);
        assert!((back - v).abs() < 1e-6);
    }

    #[test]
    fn scale_203_at_peak() {
        let out = scale_100_to_203(1.0);
        assert!((out - 2.03).abs() < 1e-5);
    }

    #[test]
    fn ootf_at_unity() {
        // gamma^1.0 = 1.0 for any gamma
        let out = ootf_gamma_adjust(1.0, OOTF_GAMMA_BBC);
        assert!((out - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ootf_darkens_midtones() {
        // gamma > 1 → mid-gray gets darker
        let out = ootf_gamma_adjust(0.5, OOTF_GAMMA_BBC);
        assert!(out < 0.5, "OOTF should darken mid-gray: {out}");
    }

    #[test]
    fn gamma_203_100_roundtrip() {
        let v = 0.3_f32;
        let down = gamma_203_to_100(v);
        let back = gamma_100_to_203(down);
        assert!((back - v).abs() < 1e-5, "roundtrip: {back} vs {v}");
    }

    #[test]
    fn gamma_preserves_black() {
        assert_eq!(gamma_203_to_100(0.0), 0.0);
        assert_eq!(gamma_100_to_203(0.0), 0.0);
    }
}
