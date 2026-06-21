//! One-call HDR → SDR conversion: tone-map → soft-clip-knee.
//!
//! See [`HdrToSdr`] for the full pipeline and defaults.

use crate::Bt2446A;
use crate::ToneMap;
use crate::gamut::{soft_clip_knee, soft_clip_knee_strip};

/// One-call HDR-to-SDR conversion: tone-map (BT.2446 Method A) → optional
/// gamut matrix → soft-clip-knee.
///
/// **Source peak is caller-provided** — `zentone` stays independent of
/// `zenpixels-convert`. Use
/// `zenpixels_convert::hdr::CllMeasure::measure_robust` (gated on
/// `hdr-experimental`) to derive `source_peak_nits` from input pixels,
/// or read it from container metadata.
///
/// # Rationale: why BT.2446 Method A
///
/// The 76-sample HDR shootout
/// ([`benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`](../../benchmarks/hdr_tone_map_shootout_full_2026-06-20.md))
/// measured 20 curve cells against producer-graded SDR on both UltraHDR
/// JPEG and iPhone HEIC sources under the production `measure_robust`
/// peak method. Median ΔE2000 verdict:
///
/// - BT.2446 Method A — **3.17**
/// - BT.2390 — 6.09
/// - Möbius (previous default) — 16.65
///
/// Method A wins on both device classes — a ~5× improvement over Möbius
/// at the median. Visual review confirmed Bt2446A is the closest match to
/// producer-graded SDR; see CHANGELOG for the swap entry.
///
/// Callers who specifically need libplacebo-compatible HDR-playback
/// behavior (mpv / VLC / FFmpeg parity) can still construct
/// [`ToneMapCurve::Mobius`](crate::ToneMapCurve::Mobius) directly —
/// `HdrToSdr` targets the distinct "match what users see in their phone
/// or camera gallery after the device ISP grades the SDR" intent.
///
/// Defaults:
/// - `target_peak_nits = 100.0` (SDR reference white).
/// - `knee_gamut = 0.95` (gamut soft-clip kicks in 5 % below boundary).
///
/// # Pipeline
///
/// Per pixel (in both [`apply_strip`](Self::apply_strip) and
/// [`apply_rgb`](Self::apply_rgb)):
///
/// 1. Apply [`Bt2446A`](crate::Bt2446A) with `(source_peak_nits,
///    target_peak_nits)`. The EETF natively absorbs the source-norm input
///    contract (`1.0 = source_peak_nits`) and emits target-norm output
///    in `[0, 1]` (`1.0 = target_peak_nits`).
/// 2. Apply [`gamut::soft_clip_knee`](crate::gamut::soft_clip_knee) with
///    `knee_gamut` as defense against rare out-of-gamut excursions (NaN
///    inputs, over-saturated content). Bt2446A clamps to `[0, 1]`
///    internally so this is usually a no-op.
///
/// Gamut conversion (e.g. BT.2020 → BT.709 via
/// [`gamut::apply_matrix`](crate::gamut::apply_matrix)) is the **caller's**
/// responsibility before [`apply_strip`](Self::apply_strip); this struct
/// only handles luminance compression + gamut-edge soft-clip in the
/// target gamut.
///
/// **RGB-only:** the strip variant is `&mut [[f32; 3]]`. For RGBA, drop
/// alpha to a separate channel before calling; alpha is not luminance-
/// bearing and tone-mapping it is meaningless.
///
/// # Precision
///
/// [`apply_strip`](Self::apply_strip) routes through
/// [`Bt2446A::map_strip_simd_for_u8`], the 8-bit-display-targeted SIMD
/// fast path. Output is byte-identical to the spec-strict
/// [`Bt2446A::map_strip_simd`] path after sRGB-u8 encoding for ~all
/// pixels. Callers writing 16-bit PNG / EXR / half-float output should
/// invoke [`Bt2446A`] directly instead — the 5.21e-4 linear-light error
/// of the fast path is visible at 16-bit precision.
///
/// # Example
///
/// ```
/// use zentone::HdrToSdr;
///
/// // 1000-nit HDR source → 100-nit SDR target (default).
/// // Input is source-normalized: 1.0 = source_peak_nits = 1000 nits.
/// let converter = HdrToSdr::new(1000.0);
/// let mut pixels = vec![
///     [1.0_f32, 0.6, 0.3],     // bright HDR pixel near source peak
///     [0.1, 0.1, 0.1],         // dim shadow region
/// ];
/// converter.apply_strip(&mut pixels);
/// // Output is target-normalized: 1.0 = target_peak_nits = 100 nits.
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
    /// Gamut soft-clip knee (default `0.95` — rolloff kicks in 5 %
    /// below the gamut boundary).
    pub knee_gamut: f32,
}

impl HdrToSdr {
    /// Construct an HDR→SDR converter for a given source peak with
    /// production defaults: `target_peak_nits = 100`, `knee_gamut = 0.95`.
    #[must_use]
    pub fn new(source_peak_nits: f32) -> Self {
        Self {
            source_peak_nits,
            target_peak_nits: 100.0,
            knee_gamut: 0.95,
        }
    }

    /// Apply the pipeline to a strip of linear `RGB f32` pixels in place.
    ///
    /// Input is assumed normalized so `1.0 = source_peak_nits` (the
    /// standard zenpixels source-normalized convention — a saturated HDR
    /// pixel reads near `1.0` in input). Output is normalized so
    /// `1.0 = target_peak_nits`. Caller handles container conversion
    /// (CICP / transfer / gamut matrix to BT.709 if needed) separately.
    ///
    /// Uses the 8-bit-display-targeted
    /// [`Bt2446A::map_strip_simd_for_u8`] SIMD fast path. Output is
    /// byte-identical to the spec-strict
    /// [`Bt2446A::map_strip_simd`] path after sRGB-u8 encoding for ~all
    /// pixels — see that method's docstring for the precision tradeoff
    /// and when to drop down to the spec-strict path directly.
    pub fn apply_strip(&self, rgb: &mut [[f32; 3]]) {
        // Bt2446A's input contract is source-normalized
        // (1.0 = source_peak_nits) and output is target-normalized
        // (1.0 = sdr_peak_nits = target_peak_nits) — no rescale needed.
        // When source_peak <= target_peak the EETF curve still runs but
        // the rho_hdr / rho_sdr ratio collapses toward identity; we don't
        // short-circuit because the small 1.0770 boost in the low segment
        // is intentional BT.2446 behavior.
        let tm = Bt2446A::new(self.source_peak_nits, self.target_peak_nits);
        tm.map_strip_simd_for_u8(rgb);
        soft_clip_knee_strip(rgb, self.knee_gamut);
    }

    /// Per-pixel variant of [`apply_strip`](Self::apply_strip).
    ///
    /// Same input/output normalization contract — input is source-
    /// normalized, output is target-normalized.
    #[must_use]
    pub fn apply_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let tm = Bt2446A::new(self.source_peak_nits, self.target_peak_nits);
        let after_tone = tm.map_rgb(rgb);
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
        assert!((c.knee_gamut - 0.95).abs() < 1e-7);
    }

    #[test]
    fn source_equals_target_stays_in_unit_range() {
        // source_peak == target_peak: Bt2446A is NOT exact identity here
        // — its low-segment scale factor (1.0770 · y_p) intentionally
        // boosts midtones by ~7.7% in the perceptual-log domain. So we
        // pin the looser invariant: a mid-gray input lands well within
        // [0, 1], roughly preserves neutrality (channels stay equal),
        // and doesn't catastrophically diverge from input.
        let c = HdrToSdr {
            source_peak_nits: 100.0,
            target_peak_nits: 100.0,
            knee_gamut: 0.95,
        };
        let rgb = [0.4_f32, 0.4, 0.4];
        let out = c.apply_rgb(rgb);
        for v in out {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
        // Neutral input stays neutral.
        assert!((out[0] - out[1]).abs() < 1e-4);
        assert!((out[1] - out[2]).abs() < 1e-4);
        // Output is in the same general region as input (not far off the
        // diagonal — BT.2446-A at source==target gives a mild boost, not
        // a wild excursion).
        assert!(
            (out[0] - rgb[0]).abs() < 0.25,
            "source==target output strayed far from input: out={out:?} rgb={rgb:?}"
        );
    }

    #[test]
    fn mid_bright_via_bt2446a_compresses_toward_target() {
        // Source-normalized input: 0.5 = half of source_peak_nits =
        // 500 nits HDR midtone for a 1000-nit source. Output must be
        // finite and at or below 1.0 (= target peak).
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([0.5, 0.5, 0.5]);
        for v in out {
            assert!(
                v.is_finite() && v < 1.0,
                "expected compression below 1, got {v}"
            );
        }
    }

    #[test]
    fn bt2446a_emits_in_unit_range() {
        // Saturated HDR pixel (= 1.0 source-norm = source_peak_nits)
        // must map to roughly target-peak-level output ([0.8, 1.0]) per
        // BT.2446-A's EETF, where ρ_H ≈ 33 at 10 000 nits drives the
        // log-domain compression hard enough that the inverse delog
        // lands near SDR white. Pre-Bt2446A this was a regression test
        // for a Möbius input-normalization bug; now it's a generic pin
        // on the new curve's well-defined peak behavior.
        let c = HdrToSdr::new(4000.0);
        let out = c.apply_rgb([1.0, 1.0, 1.0]);
        for v in out {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "saturated source pixel must land in [0, 1] target-norm: got {v}"
            );
        }
        // And smaller midtones (0.1 source-norm = 400 nits) also stay
        // in range.
        let small = c.apply_rgb([0.1, 0.1, 0.1]);
        for v in small {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn output_in_unit_range() {
        // Stress: a strip of HDR pixels (source-norm). Output must end
        // up in [0, 1] target-norm.
        let c = HdrToSdr::new(4000.0);
        let mut pixels = vec![
            [0.0_f32, 0.0, 0.0],
            [0.025, 0.125, 0.075], // ≈ [100, 500, 300] nits midtone
            [0.25, 0.25, 0.25],    // ≈ 1000 nits each
            [1.0, 1.0, 1.0],       // peak white
            [1.0, 0.0, 0.0],       // peak red
            [0.75, 0.75, 0.0],     // peak yellow
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
        // A pure-red HDR pixel (1.0 red = source peak nits, faint
        // green/blue from noise) — channel ratios after the pipeline
        // should keep red as the dominant channel.
        let c = HdrToSdr::new(1000.0);
        let out = c.apply_rgb([1.0_f32, 0.05, 0.05]);
        assert!(out[0] > out[1], "red should remain dominant");
        assert!(out[0] > out[2]);
        for v in out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn apply_rgb_matches_apply_strip() {
        // `apply_strip` uses the 8-bit-display-targeted SIMD fast path
        // (`Bt2446A::map_strip_simd_for_u8` — 2-piece sqrt-substituted
        // polynomial for `x^(1/2.4)` + Estrin EOTF), while `apply_rgb`
        // uses scalar `libm::powf` end-to-end. The two are byte-identical
        // after sRGB-u8 encoding, but the linear-light intermediate can
        // diverge by up to ~5.21e-4 (the documented polynomial-amplified
        // kernel error). Tolerance reflects that — if the divergence
        // grows beyond it, either the polynomial constants regressed or
        // the SIMD body's math drifted.
        let c = HdrToSdr::new(2000.0);
        let pixels = [
            [0.0_f32, 0.0, 0.0],
            [0.15, 0.25, 0.05], // ≈ [300, 500, 100] nits
            [0.5, 0.2, 0.1],
            [1.0, 0.0, 0.0], // peak red
            [0.25, 0.25, 0.25],
        ];
        let mut strip = pixels.to_vec();
        c.apply_strip(&mut strip);
        for (i, &p) in pixels.iter().enumerate() {
            let expected = c.apply_rgb(p);
            for (k, (a, e)) in strip[i].iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-3,
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
    fn source_below_target_stays_in_unit_range() {
        // source_peak < target_peak: Bt2446A still runs (no internal
        // short-circuit), but the rho_hdr ≈ rho_sdr ratio means the
        // compression is mild — we just pin that output stays in [0, 1]
        // and remains finite. (Pre-Bt2446A this branch was a hard
        // identity pass-through; we no longer make that promise because
        // the tonemap-as-a-function-of-(peak_hdr, peak_sdr) contract is
        // cleaner this way and downstream consumers don't depend on
        // identity at this boundary.)
        let c = HdrToSdr {
            source_peak_nits: 80.0,
            target_peak_nits: 100.0,
            knee_gamut: 0.95,
        };
        let out = c.apply_rgb([0.5_f32, 0.5, 0.5]);
        for v in out {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn pipeline_finite_on_extreme_inputs() {
        // Defensive: very high HDR (source = 10 000 nits) + over-peak
        // input (1.0 source-norm is the documented max but a defect
        // pixel can land slightly above). Output must stay finite and
        // in [0, 1].
        let c = HdrToSdr {
            source_peak_nits: 10_000.0,
            target_peak_nits: 100.0,
            knee_gamut: 0.95,
        };
        let mut strip = vec![
            [1.0_f32, 0.5, 0.01], // peak red ≈ 10k nits
            [1.5, 1.0, 0.0],      // slight defect above source peak
            [0.0, 0.0, 0.0],
        ];
        c.apply_strip(&mut strip);
        for px in &strip {
            for &v in px {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn custom_gamut_knee_changes_output() {
        // Sanity-check the configuration knobs actually take effect:
        // a very aggressive gamut knee should change a near-gamut-edge
        // pixel measurably. (`knee_tone` is gone — Bt2446A is fully
        // parameterized by source/target peaks.)
        let rgb = [0.95_f32, 0.05, 0.05]; // near-gamut-edge red
        let default = HdrToSdr::new(1000.0).apply_rgb(rgb);
        let custom = HdrToSdr {
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            knee_gamut: 0.30, // very aggressive — rolloff starts at 30 %
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
            "expected different output with different knee_gamut: default={default:?} custom={custom:?}"
        );
    }
}
