//! One-call HDR → SDR conversion: tone-map → soft-clip-knee.
//!
//! See [`HdrToSdr`] for the full pipeline and defaults.

use crate::ToneMap;
use crate::curves::ToneMapCurve;
use crate::gamut::{soft_clip_knee, soft_clip_knee_strip};

/// One-call HDR-to-SDR conversion: tone-map (Möbius) → optional gamut
/// matrix → soft-clip-knee.
///
/// **Source peak is caller-provided** — `zentone` stays independent of
/// `zenpixels-convert`. Use
/// `zenpixels_convert::hdr::CllMeasure::measure_robust` (gated on
/// `hdr-experimental`) to derive `source_peak_nits` from input pixels,
/// or read it from container metadata.
///
/// Defaults match production:
/// - `target_peak_nits = 100.0` (SDR reference white).
/// - `knee_tone = 0.30` (libplacebo `linear_knee` default).
/// - `knee_gamut = 0.95` (gamut soft-clip kicks in 5 % below boundary).
///
/// # Pipeline
///
/// Per pixel (in both [`apply_strip`](Self::apply_strip) and
/// [`apply_rgb`](Self::apply_rgb)):
///
/// 1. Compute `peak_normalized = source_peak_nits / target_peak_nits`.
///    If `<= 1.0`, the input already fits the target — tone mapping is
///    skipped (only the gamut soft-clip runs).
/// 2. Apply [`ToneMapCurve::Mobius { source_peak: peak_normalized, knee:
///    knee_tone }`](crate::ToneMapCurve::Mobius) per channel.
/// 3. Apply [`gamut::soft_clip_knee`](crate::gamut::soft_clip_knee) with
///    `knee_gamut`.
///
/// Gamut conversion (e.g. BT.2020 → BT.709 via
/// [`gamut::apply_matrix`](crate::gamut::apply_matrix)) is the **caller's**
/// responsibility before [`apply_strip`](Self::apply_strip); this struct
/// only handles luminance compression + gamut-edge soft-clip in the
/// target gamut. The result is normalized so `1.0 = target_peak_nits`.
///
/// **RGB-only:** the strip variant is `&mut [[f32; 3]]`. For RGBA, drop
/// alpha to a separate channel before calling; alpha is not luminance-
/// bearing and tone-mapping it is meaningless.
///
/// # Example
///
/// ```
/// use zentone::HdrToSdr;
///
/// // 1000-nit HDR source → 100-nit SDR target (default).
/// let converter = HdrToSdr::new(1000.0);
/// let mut pixels = vec![[5.0_f32, 3.0, 1.5], [0.5, 0.5, 0.5]];
/// converter.apply_strip(&mut pixels);
/// for px in &pixels {
///     for &c in px {
///         assert!((0.0..=1.0).contains(&c));
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HdrToSdr {
    /// Source peak luminance in nits (cd/m²). Input pixels are assumed
    /// normalized so `1.0 = source_peak_nits`.
    pub source_peak_nits: f32,
    /// Target display peak luminance in nits. Output pixels are
    /// normalized so `1.0 = target_peak_nits`.
    pub target_peak_nits: f32,
    /// Möbius knee point (libplacebo default `0.30`, in the normalized
    /// output range).
    pub knee_tone: f32,
    /// Gamut soft-clip knee (default `0.95` — rolloff kicks in 5 %
    /// below the gamut boundary).
    pub knee_gamut: f32,
}

impl HdrToSdr {
    /// Construct an HDR→SDR converter for a given source peak with
    /// production defaults: `target_peak_nits = 100`, `knee_tone = 0.30`,
    /// `knee_gamut = 0.95`.
    #[must_use]
    pub fn new(source_peak_nits: f32) -> Self {
        Self {
            source_peak_nits,
            target_peak_nits: 100.0,
            knee_tone: 0.30,
            knee_gamut: 0.95,
        }
    }

    /// Compute the normalized peak (`source_peak_nits / target_peak_nits`).
    /// Used internally; exposed so callers that already have it can
    /// short-circuit.
    #[inline]
    fn peak_normalized(&self) -> f32 {
        if self.target_peak_nits > 0.0 {
            self.source_peak_nits / self.target_peak_nits
        } else {
            1.0 // identity guard against /0
        }
    }

    /// Apply the pipeline to a strip of linear `RGB f32` pixels in place.
    ///
    /// Input is assumed normalized so `1.0 = source_peak_nits`
    /// (relative-linear, the standard zenpixels convention). Output is
    /// normalized so `1.0 = target_peak_nits`. Caller handles container
    /// conversion (CICP / transfer / gamut matrix to BT.709 if needed)
    /// separately.
    pub fn apply_strip(&self, rgb: &mut [[f32; 3]]) {
        let peak = self.peak_normalized();
        if peak > 1.0 {
            let curve = ToneMapCurve::Mobius {
                source_peak: peak,
                knee: self.knee_tone,
            };
            curve.map_strip_simd(rgb);
        }
        soft_clip_knee_strip(rgb, self.knee_gamut);
    }

    /// Per-pixel variant of [`apply_strip`](Self::apply_strip).
    #[must_use]
    pub fn apply_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let peak = self.peak_normalized();
        let after_tone = if peak > 1.0 {
            let curve = ToneMapCurve::Mobius {
                source_peak: peak,
                knee: self.knee_tone,
            };
            curve.map_rgb(rgb)
        } else {
            rgb
        };
        soft_clip_knee(after_tone, self.knee_gamut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn defaults_match_spec() {
        let c = HdrToSdr::new(1000.0);
        assert_eq!(c.source_peak_nits, 1000.0);
        assert_eq!(c.target_peak_nits, 100.0);
        assert!((c.knee_tone - 0.30).abs() < 1e-7);
        assert!((c.knee_gamut - 0.95).abs() < 1e-7);
    }

    #[test]
    fn identity_when_source_equals_target() {
        // source_peak == target_peak → no tone mapping, only soft-clip
        // (which is identity for in-gamut content).
        let c = HdrToSdr {
            source_peak_nits: 100.0,
            target_peak_nits: 100.0,
            knee_tone: 0.30,
            knee_gamut: 0.95,
        };
        let rgb = [0.4_f32, 0.7, 0.2];
        let out = c.apply_rgb(rgb);
        for i in 0..3 {
            assert!(
                (out[i] - rgb[i]).abs() < 1e-7,
                "identity failed [{i}]: {out:?}"
            );
        }
    }

    #[test]
    fn mid_bright_compresses_toward_target() {
        // HDR mid-tones above target peak should be compressed below 1.0.
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([5.0, 5.0, 5.0]);
        for v in out {
            assert!(
                v.is_finite() && v < 1.0,
                "expected compression below 1, got {v}"
            );
        }
    }

    #[test]
    fn output_in_unit_range() {
        // Stress: a strip of HDR pixels, output must end up in [0, 1].
        let c = HdrToSdr::new(4000.0);
        let mut pixels = vec![
            [0.0_f32, 0.0, 0.0],
            [0.1, 0.5, 0.3],
            [1.0, 1.0, 1.0],
            [10.0, 5.0, 2.0],
            [40.0, 0.0, 0.0],  // peak red @ source peak
            [30.0, 30.0, 0.0], // peak yellow
        ];
        c.apply_strip(&mut pixels);
        for (i, px) in pixels.iter().enumerate() {
            for (k, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (0.0..=1.0).contains(&v),
                    "pixel {i} ch{k} out of [0,1]: {v} (full: {px:?})"
                );
            }
        }
    }

    #[test]
    fn hue_preserved_on_saturated_color() {
        // A pure-red HDR pixel — channel ratios after the pipeline should
        // keep red as the dominant channel, with green/blue still small.
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([10.0_f32, 0.5, 0.5]);
        assert!(out[0] > out[1], "red should remain dominant");
        assert!(out[0] > out[2]);
        for v in out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn apply_rgb_matches_apply_strip() {
        let c = HdrToSdr::new(2000.0);
        let pixels = [
            [0.0_f32, 0.0, 0.0],
            [0.3, 0.5, 0.1],
            [5.0, 2.0, 1.0],
            [20.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ];
        let mut strip = pixels.to_vec();
        c.apply_strip(&mut strip);
        for (i, &p) in pixels.iter().enumerate() {
            let expected = c.apply_rgb(p);
            for (k, (a, e)) in strip[i].iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-6,
                    "strip vs per-pixel diverge at px[{i}]ch[{k}]: {a} vs {e}"
                );
            }
        }
    }

    #[test]
    fn empty_strip_is_noop() {
        let c = HdrToSdr::new(1000.0);
        let mut strip: alloc::vec::Vec<[f32; 3]> = alloc::vec::Vec::new();
        c.apply_strip(&mut strip);
        assert!(strip.is_empty());
    }

    #[test]
    fn source_below_target_skips_tone_map() {
        // source_peak < target_peak → only soft-clip runs. In-gamut
        // values pass through; out-of-gamut roll off.
        let c = HdrToSdr {
            source_peak_nits: 80.0,
            target_peak_nits: 100.0,
            knee_tone: 0.30,
            knee_gamut: 0.95,
        };
        let out = c.apply_rgb([0.5_f32, 0.5, 0.5]);
        for v in out {
            assert!((v - 0.5).abs() < 1e-7);
        }
    }

    #[test]
    fn pipeline_finite_on_extreme_inputs() {
        // Defensive: very high HDR + very low target. Output must stay
        // finite and in [0, 1].
        let c = HdrToSdr {
            source_peak_nits: 10_000.0,
            target_peak_nits: 100.0,
            knee_tone: 0.30,
            knee_gamut: 0.95,
        };
        let mut strip = vec![[100.0_f32, 50.0, 1.0], [0.0, 0.0, 0.0]];
        c.apply_strip(&mut strip);
        for px in &strip {
            for &v in px {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn custom_knees_change_output() {
        // Sanity-check the configuration knobs actually take effect:
        // a different knee should yield a measurably different pixel.
        let rgb = [3.0_f32, 1.5, 0.8];
        let default = HdrToSdr::new(1000.0).apply_rgb(rgb);
        let custom = HdrToSdr {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            knee_tone: 0.05,
            knee_gamut: 0.70,
        }
        .apply_rgb(rgb);
        let mut differs = false;
        for i in 0..3 {
            if (default[i] - custom[i]).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "expected different output with different knees: default={default:?} custom={custom:?}"
        );
    }
}
