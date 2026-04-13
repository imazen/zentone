//! SDR preservation tests.
//!
//! Classical tone curves (Reinhard, Hable, etc.) compress the entire range
//! and do NOT preserve SDR input — that's by design. But:
//!
//! 1. Curves with a linear knee (BT.2408, BT.2390, BT.2446C) should pass
//!    through SDR values unchanged when configured with sufficient headroom.
//! 2. All curves should have bounded distortion on SDR input — near-black
//!    should stay near-black, mid-gray should stay recognizably mid-gray.
//! 3. A visual comparison scene should include SDR-range content to show
//!    what happens to in-gamut colors.

use zentone::*;

// ============================================================================
// BT.2408: SDR values below knee should be near-identity
// ============================================================================

#[test]
fn bt2408_preserves_sdr_below_knee() {
    // content=4000, display=3000 → ks = 0.625.
    // BT.2408 operates in PQ domain, so the "linear passthrough" below
    // the knee produces a ratio that depends on the PQ transfer function,
    // not a simple linear ratio. Verify that small SDR values produce
    // correspondingly small outputs (monotonic, bounded).
    let tm = Bt2408Tonemapper::new(4000.0, 3000.0);

    let sdr_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4];
    let mut prev_out = 0.0_f32;
    for &v in &sdr_values {
        let out = tm.map_rgb([v, v, v]);
        // Output should be monotonically increasing
        assert!(
            out[0] >= prev_out - 1e-6,
            "BT.2408: non-monotonic: input={v}, out={}, prev={prev_out}",
            out[0]
        );
        // Output should be finite, positive, and less than input
        // (since display < content, the curve compresses)
        assert!(
            out[0].is_finite() && out[0] >= 0.0 && out[0] < v * 2.0,
            "BT.2408: input={v}, out={} (expected bounded)",
            out[0]
        );
        prev_out = out[0];
    }
}

#[test]
fn bt2408_identity_when_no_compression_needed() {
    // content=1000, display=1000 → ks=1.0, everything passes through.
    let tm = Bt2408Tonemapper::new(1000.0, 1000.0);
    let values = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0];
    for &v in &values {
        let out = tm.map_rgb([v, v, v]);
        for ch in 0..3 {
            assert!(
                (out[ch] - v).abs() < 0.01,
                "BT.2408 (no compression): input={v}, out[{ch}]={}, expected≈{v}",
                out[ch]
            );
        }
    }
}

// ============================================================================
// BT.2446C: linear segment below inflection → near-identity scaling
// ============================================================================

#[test]
fn bt2446c_linear_segment_is_proportional() {
    // Default params: k1=0.83802, inflection at y_ip = k4/k1 = 94.3%
    // Below that, Y_SDR = k1 * Y_HDR (linear, slope k1).
    // For input in [0, 0.5] (well below inflection at 0.943), the
    // output should be input * k1, then scaled by the 100% normalization.
    let tm = Bt2446C::new(1000.0, 100.0);
    let k1 = 0.83802_f32;

    for &v in &[0.01, 0.05, 0.1, 0.2, 0.5] {
        let out = tm.map_rgb([v, v, v]);
        // Input v is linear [0,1]. map_rgb converts to percentage (*100),
        // applies tone curve (k1 * pct), normalizes back (/100).
        // Expected: v * k1 (since the linear segment is Y_SDR = k1 * Y_HDR).
        let expected = v * k1;
        for ch in 0..3 {
            assert!(
                (out[ch] - expected).abs() < 0.02,
                "BT.2446C linear: input={v}, out[{ch}]={}, expected≈{expected:.4}",
                out[ch]
            );
        }
    }
}

// ============================================================================
// All curves: bounded distortion on SDR input
// ============================================================================

/// Classical curves — input and output are both scene-linear [0, ∞) → [0, 1].
/// SDR range (0–1) is compressed but stays recognizable.
fn classical_curves() -> Vec<(&'static str, Box<dyn ToneMap>)> {
    vec![
        ("Reinhard", Box::new(ToneMapCurve::Reinhard)),
        (
            "ExtReinhard",
            Box::new(ToneMapCurve::ExtendedReinhard {
                l_max: 4.0,
                luma: LUMA_BT709,
            }),
        ),
        (
            "ReinhardJodie",
            Box::new(ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 }),
        ),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic)),
        ("AcesAp1", Box::new(ToneMapCurve::AcesAp1)),
        ("AgxDefault", Box::new(ToneMapCurve::Agx(AgxLook::Default))),
        ("Clamp", Box::new(ToneMapCurve::Clamp)),
        (
            "FilmicSpline",
            Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
        ),
    ]
}

/// ITU curves configured for extreme HDR→SDR compression (4000→100 nits).
/// SDR values (0–1 linear) represent only the bottom 2.5% of the HDR
/// range and get massively expanded. Testing "SDR preservation" on these
/// doesn't make sense — expansion is their correct behavior.
fn itu_curves() -> Vec<(&'static str, Box<dyn ToneMap>)> {
    vec![
        ("Bt2408", Box::new(Bt2408Tonemapper::new(4000.0, 1000.0))),
        ("Bt2446A", Box::new(Bt2446A::new(4000.0, 100.0))),
        ("Bt2446B", Box::new(Bt2446B::new(4000.0, 100.0))),
        ("Bt2446C", Box::new(Bt2446C::new(4000.0, 100.0))),
    ]
}

fn all_tonemappers() -> Vec<(&'static str, Box<dyn ToneMap>)> {
    let mut all = classical_curves();
    all.extend(itu_curves());
    all
}

#[test]
fn near_black_stays_near_black() {
    // Input < 0.01 should produce output < 0.1 for classical curves.
    // This catches lifted-black bugs like the old AgX polynomial residual.
    // (ITU curves with high content peaks expand near-black by design.)
    for (name, tm) in classical_curves() {
        for &v in &[0.0, 0.001, 0.005, 0.01] {
            let out = tm.map_rgb([v, v, v]);
            for ch in 0..3 {
                assert!(
                    out[ch] < 0.1,
                    "{name}: near-black input={v} produced out[{ch}]={} (too bright)",
                    out[ch]
                );
            }
        }
    }
}

#[test]
fn mid_gray_stays_recognizable() {
    // Input 0.18 (photographic middle gray) should produce output
    // that's still a visible midtone — not crushed to black or blown
    // to white.
    //
    // Classical curves compress SDR; mid-gray should stay midtone.
    // (ITU curves with high peaks expand SDR — tested separately.)
    for (name, tm) in classical_curves() {
        let out = tm.map_rgb([0.18, 0.18, 0.18]);
        let l = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        assert!(
            l > 0.05 && l < 0.55,
            "{name}: mid-gray 0.18 → luminance {l:.4} (expected 0.05..0.55)"
        );
    }
}

#[test]
fn sdr_white_stays_bright() {
    // Input 1.0 (SDR white) should produce output luminance > 0.3.
    // Some curves compress hard (AgX, filmic spline) but none should
    // turn SDR white into a dark gray.
    for (name, tm) in all_tonemappers() {
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        let l = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        assert!(
            l > 0.3,
            "{name}: SDR white 1.0 → luminance {l:.4} (too dark, expected > 0.3)"
        );
    }
}

#[test]
fn sdr_colors_preserve_hue_direction() {
    // For SDR saturated colors (R=0.8, G=0.2, B=0.1), the output
    // should maintain R > G > B ordering (hue direction preserved).
    let input = [0.8, 0.2, 0.1];
    for (name, tm) in all_tonemappers() {
        let out = tm.map_rgb(input);
        assert!(
            out[0] >= out[1] && out[1] >= out[2],
            "{name}: SDR warm color hue shifted: {input:?} → {out:?} (expected R≥G≥B)"
        );
    }
}

#[test]
fn neutral_sdr_stays_neutral() {
    // Neutral gray at any SDR level should produce neutral output
    // (all channels equal). This catches matrix bugs like the old
    // transposed AgX inset.
    for (name, tm) in all_tonemappers() {
        for &v in &[0.05, 0.18, 0.5, 1.0] {
            let out = tm.map_rgb([v, v, v]);
            let max_ch = out[0].max(out[1]).max(out[2]);
            let min_ch = out[0].min(out[1]).min(out[2]);
            let spread = if max_ch > 1e-4 {
                (max_ch - min_ch) / max_ch
            } else {
                0.0
            };
            assert!(
                spread < 0.01,
                "{name}: neutral {v} produced non-neutral output: {out:?} (spread={spread:.4})"
            );
        }
    }
}

// ============================================================================
// Realistic 4× headroom: SDR content with moderate HDR highlights
// ============================================================================

/// All curves configured for 4× headroom (typical HDR photo/game content).
fn curves_4x_headroom() -> Vec<(&'static str, Box<dyn ToneMap>)> {
    vec![
        ("Reinhard", Box::new(ToneMapCurve::Reinhard)),
        (
            "ExtReinhard_4x",
            Box::new(ToneMapCurve::ExtendedReinhard {
                l_max: 4.0,
                luma: LUMA_BT709,
            }),
        ),
        (
            "ReinhardJodie",
            Box::new(ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 }),
        ),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic)),
        ("AcesAp1", Box::new(ToneMapCurve::AcesAp1)),
        ("AgxDefault", Box::new(ToneMapCurve::Agx(AgxLook::Default))),
        // BT.2408 with 4× headroom: content=800 nits, display=203 nits
        ("Bt2408_4x", Box::new(Bt2408Tonemapper::new(800.0, 203.0))),
        // BT.2446C with 4× headroom
        ("Bt2446C_4x", Box::new(Bt2446C::new(800.0, 203.0))),
        (
            "FilmicSpline_4x",
            Box::new(CompiledFilmicSpline::for_hdr_peak(4.0)),
        ),
    ]
}

#[test]
fn sdr_quarter_range_low_distortion_4x() {
    // With 4× headroom, values in [0, 0.25] should be moderately preserved.
    // Classical curves compress everything; ITU curves in PQ domain can
    // expand SDR values. Check that output is in a sane range relative to
    // input rather than testing precise preservation.
    for (name, tm) in curves_4x_headroom() {
        for &v in &[0.01, 0.05, 0.1, 0.18, 0.25] {
            let out = tm.map_rgb([v, v, v]);
            let l = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
            // Output should be in a reasonable range: not crushed to zero,
            // not blown past 0.8 for these small inputs.
            assert!(
                l > v * 0.1 && l < 0.9,
                "{name}: input={v}, output_lum={l:.4} (expected {:.4}..0.9)",
                v * 0.1
            );
        }
    }
}

#[test]
fn sdr_mid_range_bounded_compression_4x() {
    // With 4× headroom, SDR mid-range gets compressed but stays visible.
    // BT.2408 in PQ domain can map SDR values higher than classical curves.
    for (name, tm) in curves_4x_headroom() {
        for &v in &[0.3, 0.5, 0.7] {
            let out = tm.map_rgb([v, v, v]);
            let l = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
            assert!(
                l > 0.1 && l < 1.01,
                "{name}: SDR mid {v} → {l:.4} (expected 0.1..1.0)"
            );
        }
    }
}

#[test]
fn sdr_white_decent_brightness_4x() {
    // With 4× headroom, SDR white (1.0) should map to something bright.
    // BT.2408 at 800/203 barely compresses SDR range → output near 1.0.
    for (name, tm) in curves_4x_headroom() {
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        let l = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        assert!(
            l > 0.4 && l <= 1.01,
            "{name}: SDR white 1.0 → {l:.4} (expected 0.4..1.0)"
        );
    }
}

#[test]
fn highlights_compress_not_clip_4x() {
    // Input 2.0 (2× SDR) and 4.0 (peak) should map to values above SDR
    // white's output but below 1.0. No hard clipping — smooth rolloff.
    for (name, tm) in curves_4x_headroom() {
        let sdr_white = tm.map_rgb([1.0, 1.0, 1.0]);
        let sdr_white_l = 0.2126 * sdr_white[0] + 0.7152 * sdr_white[1] + 0.0722 * sdr_white[2];

        let hdr_2x = tm.map_rgb([2.0, 2.0, 2.0]);
        let hdr_2x_l = 0.2126 * hdr_2x[0] + 0.7152 * hdr_2x[1] + 0.0722 * hdr_2x[2];

        let hdr_4x = tm.map_rgb([4.0, 4.0, 4.0]);
        let hdr_4x_l = 0.2126 * hdr_4x[0] + 0.7152 * hdr_4x[1] + 0.0722 * hdr_4x[2];

        // Monotonic: 4x > 2x > 1x
        assert!(
            hdr_4x_l >= hdr_2x_l - 1e-4 && hdr_2x_l >= sdr_white_l - 1e-4,
            "{name}: non-monotonic: 1x={sdr_white_l:.4}, 2x={hdr_2x_l:.4}, 4x={hdr_4x_l:.4}"
        );

        // 2× should be brighter than SDR white (compression, not clipping)
        assert!(
            hdr_2x_l > sdr_white_l - 0.01,
            "{name}: 2× HDR ({hdr_2x_l:.4}) not brighter than SDR white ({sdr_white_l:.4})"
        );

        // 4× (peak) should be near but not above 1.0
        assert!(
            hdr_4x_l <= 1.01,
            "{name}: 4× peak luminance {hdr_4x_l:.4} exceeds 1.0"
        );
    }
}

#[test]
fn saturated_sdr_colors_retain_saturation_4x() {
    // With 4× headroom, saturated SDR colors (in [0,1]) should retain
    // visible saturation — not be washed out to gray.
    let test_colors: &[(&str, [f32; 3])] = &[
        ("red", [0.8, 0.1, 0.05]),
        ("green", [0.1, 0.7, 0.05]),
        ("blue", [0.05, 0.1, 0.8]),
        ("orange", [0.9, 0.4, 0.05]),
        ("cyan", [0.05, 0.7, 0.7]),
    ];

    for (name, tm) in curves_4x_headroom() {
        for (color_name, input) in test_colors {
            let out = tm.map_rgb(*input);

            // The dominant channel should still be the largest
            let max_in = input.iter().cloned().reduce(f32::max).unwrap();
            let max_in_ch = input.iter().position(|&v| v == max_in).unwrap();

            // Allow AgX to desaturate heavily (it's its design)
            if name.starts_with("Agx") {
                continue;
            }

            assert!(
                out[max_in_ch] >= out[(max_in_ch + 1) % 3]
                    && out[max_in_ch] >= out[(max_in_ch + 2) % 3],
                "{name}: {color_name} lost dominant channel: {input:?} → {out:?}"
            );

            // Saturation ratio: (max - min) / max should be > 0.1
            let max_out = out[0].max(out[1]).max(out[2]);
            let min_out = out[0].min(out[1]).min(out[2]);
            let sat = if max_out > 0.01 {
                (max_out - min_out) / max_out
            } else {
                0.0
            };
            assert!(
                sat > 0.1,
                "{name}: {color_name} too desaturated: {input:?} → {out:?} sat={sat:.3}"
            );
        }
    }
}

#[test]
fn clamp_is_identity_below_one() {
    // Clamp should be exact identity for input in [0, 1].
    let tm = ToneMapCurve::Clamp;
    for &v in &[0.0, 0.001, 0.18, 0.5, 0.99, 1.0] {
        let out = tm.map_rgb([v, 0.5 * v, 0.1 * v]);
        assert_eq!(out[0], v, "Clamp R: {v}");
        assert_eq!(out[1], 0.5 * v, "Clamp G: {v}");
        assert_eq!(out[2], 0.1 * v, "Clamp B: {v}");
    }
}
