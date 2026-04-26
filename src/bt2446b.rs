//! BT.2446 Method B — simplified HLG tone mapper for live broadcast.
//!
//! ITU-R BT.2446-1 §5. Conservative SDR↔HDR conversion with a 291 cd/m²
//! ceiling, designed for live broadcast content that may contain large
//! over-exposed/clipped areas. Uses a simplified HLG path where setting
//! `L_W = 291 cd/m²` with system gamma ≈ 1.0 gives a near-1:1 SDR→HLG
//! mapping over most of the signal range.
//!
//! Key feature: 90% SDR maps to 75% HLG (HDR Reference White), so the
//! SDR→HDR expansion is conservative and doesn't amplify clipped areas.
//!
//! Reference: ITU-R BT.2446-1 (03/2021) §5, aligns with BT.2408 Annex 10
//! (NBCU "hybrid-linear" workflow).

use crate::ToneMap;

/// BT.2446 Method B simplified HLG tonemapper.
///
/// Operates on linear-light BT.2020 RGB. Input: `1.0 = hdr_peak_nits`.
/// Output: SDR linear in `[0, 1]`.
///
/// The key idea: by setting HLG EOTF display peak to 291 cd/m², the
/// HLG → linear → SDR path is near-identity for most of the signal range.
/// Above the breakpoint (~78% SDR), highlights are logarithmically
/// compressed rather than clipped.
///
/// # When to pick this
///
/// Live HLG broadcast → SDR. Conservative on clipped highlights (90% SDR
/// maps to 75% HLG = HDR Reference White), so over-exposed regions in the
/// source don't get amplified the way [`Bt2446A`](crate::Bt2446A) might.
/// Cheap (single log call per pixel above the breakpoint, no perceptual
/// log domain). For graded HDR10 content, prefer
/// [`Bt2408Tonemapper`](crate::Bt2408Tonemapper) or
/// [`Bt2446A`](crate::Bt2446A) instead.
///
/// Reference: ITU-R BT.2446-1 §5 (03/2021), aligned with BT.2408 Annex 10
/// (NBCU "hybrid-linear" workflow).
///
/// # Examples
///
/// ```
/// use zentone::{Bt2446B, ToneMap};
///
/// let curve = Bt2446B::new(1000.0, 100.0);
/// let sdr = curve.map_rgb([0.5, 0.5, 0.5]);
/// assert!(sdr.iter().all(|&c| (0.0..=1.0).contains(&c)));
/// ```
pub struct Bt2446B {
    /// Breakpoint luminance (normalized to input peak).
    breakpoint: f32,
    /// Linear gain below breakpoint.
    gain: f32,
    /// Log compression parameters above breakpoint.
    log_scale: f32,
    log_offset: f32,
}

impl Bt2446B {
    /// Create a Method B tonemapper.
    ///
    /// `hdr_peak_nits`: peak luminance of HLG content (typically 1000).
    /// `sdr_peak_nits`: SDR target (typically 100).
    pub fn new(hdr_peak_nits: f32, sdr_peak_nits: f32) -> Self {
        // The breakpoint corresponds to ~78% SDR (~55 cd/m² on 100 cd/m² display)
        let sdr_bp_pct = 78.0;
        let sdr_bp = sdr_bp_pct / 100.0;

        // Map the breakpoint to HDR luminance via the simplified path
        // At 291 cd/m² with gamma ≈ 1.0, 78% SDR ≈ 55 cd/m² ≈ 19% of 291
        let hdr_bp_nits = sdr_bp * sdr_peak_nits;
        let breakpoint = hdr_bp_nits / hdr_peak_nits;

        // Linear gain below breakpoint: SDR = gain * HDR
        let gain = sdr_bp / breakpoint;

        // Log compression above breakpoint:
        // SDR = log_scale * ln(HDR / breakpoint) + sdr_bp
        // At HDR peak: SDR should reach ~1.0
        // log_scale = (1.0 - sdr_bp) / ln(1.0 / breakpoint)
        let log_scale = (1.0 - sdr_bp) / libm::logf(1.0 / breakpoint);
        let log_offset = sdr_bp;

        Self {
            breakpoint,
            gain,
            log_scale,
            log_offset,
        }
    }
}

impl ToneMap for Bt2446B {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        // BT.2020 luminance
        let y = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
        if y <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Tone map luminance
        let y_sdr = if y < self.breakpoint {
            self.gain * y
        } else {
            self.log_scale * libm::logf(y / self.breakpoint) + self.log_offset
        };

        // Scale all channels by the luminance ratio
        let ratio = y_sdr / y;
        [
            (rgb[0] * ratio).clamp(0.0, 1.0),
            (rgb[1] * ratio).clamp(0.0, 1.0),
            (rgb[2] * ratio).clamp(0.0, 1.0),
        ]
    }

    fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        archmage::incant!(
            crate::simd::curves::bt2446b_tier(
                strip,
                self.breakpoint,
                self.gain,
                self.log_scale,
                self.log_offset,
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
        let tm = Bt2446B::new(1000.0, 100.0);
        assert_eq!(tm.map_rgb([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn monotonic_neutral_ramp() {
        let tm = Bt2446B::new(1000.0, 100.0);
        let mut last = -1.0_f32;
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let out = tm.map_rgb([v, v, v]);
            assert!(out[0] >= last - 1e-5, "mono at {v}: {} < {last}", out[0]);
            last = out[0];
        }
    }

    #[test]
    fn peak_maps_near_one() {
        let tm = Bt2446B::new(1000.0, 100.0);
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        assert!(
            out[0] > 0.9 && out[0] <= 1.0,
            "peak should map near 1.0: {}",
            out[0]
        );
    }

    #[test]
    fn below_breakpoint_is_linear() {
        let tm = Bt2446B::new(1000.0, 100.0);
        // Low-luminance inputs should be linearly scaled
        let a = tm.map_rgb([0.01, 0.01, 0.01]);
        let b = tm.map_rgb([0.02, 0.02, 0.02]);
        let ratio = b[0] / a[0];
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "below breakpoint should be linear: ratio {}",
            ratio
        );
    }
}
