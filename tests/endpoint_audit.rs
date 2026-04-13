//! Systematic endpoint and boundary tests for every tonemapper.
//!
//! Verifies: black→black, white→white-ish, no NaN/Inf, no negative output
//! (except documented ACES near-black), monotonicity at extremes.

use zentone::*;

// ============================================================================
// Test data
// ============================================================================

/// All tonemapper configurations to test.
fn all_tonemappers() -> Vec<(&'static str, Box<dyn ToneMap>)> {
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
        (
            "TunedReinhard",
            Box::new(ToneMapCurve::TunedReinhard {
                content_max_nits: 4000.0,
                display_max_nits: 250.0,
                luma: LUMA_BT709,
            }),
        ),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic)),
        ("AcesAp1", Box::new(ToneMapCurve::AcesAp1)),
        (
            "Bt2390",
            Box::new(ToneMapCurve::Bt2390 {
                source_peak: 4.0,
                target_peak: 1.0,
            }),
        ),
        ("AgxDefault", Box::new(ToneMapCurve::Agx(AgxLook::Default))),
        ("AgxPunchy", Box::new(ToneMapCurve::Agx(AgxLook::Punchy))),
        ("AgxGolden", Box::new(ToneMapCurve::Agx(AgxLook::Golden))),
        ("Clamp", Box::new(ToneMapCurve::Clamp)),
        ("Bt2408", Box::new(Bt2408Tonemapper::new(4000.0, 1000.0))),
        (
            "Bt2408_MaxRgb",
            Box::new(Bt2408Tonemapper::max_rgb(4000.0, 1000.0)),
        ),
        ("Bt2446A", Box::new(Bt2446A::new(4000.0, 100.0))),
        ("Bt2446B", Box::new(Bt2446B::new(4000.0, 100.0))),
        ("Bt2446C", Box::new(Bt2446C::new(4000.0, 100.0))),
        (
            "FilmicSpline",
            Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
        ),
    ]
}

// ============================================================================
// Black → near-black (no lifted blacks)
// ============================================================================

#[test]
fn black_maps_to_near_black() {
    let black = [0.0_f32, 0.0, 0.0];
    for (name, tm) in all_tonemappers() {
        let out = tm.map_rgb(black);
        for (ch, &v) in out.iter().enumerate() {
            // ACES AP1 has documented near-black negativity (-2.7e-4).
            // All others must be in [0, 0.005].
            if name == "AcesAp1" {
                assert!(
                    v > -0.001 && v < 0.005,
                    "{name}: black[{ch}] = {v} (expected near 0, ACES allows slight negative)"
                );
            } else {
                assert!(
                    v >= 0.0 && v < 0.005,
                    "{name}: black[{ch}] = {v} (expected [0, 0.005))"
                );
            }
        }
    }
}

// ============================================================================
// Very small input → finite, non-negative (except ACES)
// ============================================================================

#[test]
fn tiny_input_is_finite_and_nonnegative() {
    let tiny_values = [1e-10_f32, 1e-7, 1e-5, 1e-3, 0.001, 0.01];
    for (name, tm) in all_tonemappers() {
        for &v in &tiny_values {
            let out = tm.map_rgb([v, v, v]);
            for (ch, &o) in out.iter().enumerate() {
                assert!(
                    o.is_finite(),
                    "{name}: tiny input {v} → channel {ch} = {o} (not finite)"
                );
                // ACES has documented near-black negativity.
                if name != "AcesAp1" {
                    assert!(
                        o >= -1e-6,
                        "{name}: tiny input {v} → channel {ch} = {o} (negative)"
                    );
                }
            }
        }
    }
}

// ============================================================================
// SDR white (1.0, 1.0, 1.0) → bounded output
// ============================================================================

#[test]
fn sdr_white_maps_to_bounded_output() {
    let white = [1.0_f32, 1.0, 1.0];
    for (name, tm) in all_tonemappers() {
        let out = tm.map_rgb(white);
        for (ch, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 0.0 && v <= 1.5,
                "{name}: white[{ch}] = {v} (expected [0, 1.5])"
            );
        }
    }
}

// ============================================================================
// Extreme HDR → finite, bounded
// ============================================================================

#[test]
fn extreme_hdr_is_finite_and_bounded() {
    let extreme_values = [10.0_f32, 100.0, 1000.0, 10000.0];
    for (name, tm) in all_tonemappers() {
        for &v in &extreme_values {
            let out = tm.map_rgb([v, v, v]);
            for (ch, &o) in out.iter().enumerate() {
                assert!(
                    o.is_finite(),
                    "{name}: extreme input {v} → channel {ch} = {o} (not finite)"
                );
                assert!(
                    o <= 2.0,
                    "{name}: extreme input {v} → channel {ch} = {o} (unbounded)"
                );
            }
        }
    }
}

// ============================================================================
// map_row black row → all near-zero (exercises SIMD paths)
// ============================================================================

#[test]
fn map_row_black_row_stays_black() {
    for (name, tm) in all_tonemappers() {
        // RGB
        let mut row = vec![0.0_f32; 24]; // 8 pixels
        tm.map_row(&mut row, 3);
        for (i, &v) in row.iter().enumerate() {
            if name == "AcesAp1" {
                assert!(v.abs() < 0.005, "{name}: black row RGB[{i}] = {v}");
            } else {
                assert!(v >= 0.0 && v < 0.005, "{name}: black row RGB[{i}] = {v}");
            }
        }

        // RGBA — alpha should be 0.0 (passed through)
        let mut row = vec![0.0_f32; 32]; // 8 pixels
        tm.map_row(&mut row, 4);
        for i in 0..8 {
            let alpha = row[i * 4 + 3];
            assert!(
                alpha.abs() < 1e-6,
                "{name}: black row RGBA pixel {i} alpha = {alpha}"
            );
        }
    }
}

// ============================================================================
// Negative input → finite output (defensive)
// ============================================================================

#[test]
fn negative_input_produces_finite_output() {
    let neg = [-0.1_f32, -1.0, -0.5];
    for (name, tm) in all_tonemappers() {
        let out = tm.map_rgb(neg);
        for (ch, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{name}: negative input → channel {ch} = {v} (not finite)"
            );
        }
    }
}

// ============================================================================
// NaN/Inf input → output should not propagate silently
// ============================================================================

#[test]
fn nan_inf_input_does_not_crash() {
    let cases: &[(&str, [f32; 3])] = &[
        ("NaN", [f32::NAN, 0.5, 0.5]),
        ("Inf", [f32::INFINITY, 0.5, 0.5]),
        ("-Inf", [f32::NEG_INFINITY, 0.5, 0.5]),
        ("all-NaN", [f32::NAN, f32::NAN, f32::NAN]),
    ];
    for (name, tm) in all_tonemappers() {
        for (case_name, input) in cases {
            // Must not panic. Output may be NaN/Inf (we don't require
            // special handling), but the function must not crash.
            let _out = tm.map_rgb(*input);
            // If it returned, we're good — no panic.
            let _ = format!("{name}/{case_name}: {:?}", _out);
        }
    }
}

// ============================================================================
// Monotonicity: for neutral input, brighter → brighter
// ============================================================================

#[test]
fn monotonic_neutral_ramp() {
    // Ramp from 0 to 10 (most curves operate on [0, ~source_peak]).
    // BT.2390 normalizes internally in ToneMapCurve::map_rgb, so
    // it handles inputs > 1.0 correctly.
    let ramp: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect();
    for (name, tm) in all_tonemappers() {
        let mut prev_lum = -1.0_f32;
        for &v in &ramp {
            let out = tm.map_rgb([v, v, v]);
            let lum = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
            assert!(
                lum >= prev_lum - 1e-4,
                "{name}: monotonicity broken at input {v}: lum={lum:.6} < prev={prev_lum:.6}"
            );
            prev_lum = lum;
        }
    }
}
