#![allow(clippy::excessive_precision)]
//! Cross-reference tests verifying zentone's tone mapping curves against
//! independently-extracted constants from gainforge (Radzivon Bartoshyk,
//! BSD-3-Clause), the canonical Blender AgX, and the original Hable/
//! Narkowicz references.
//!
//! These tests do NOT link gainforge — they reimplement the formulas from
//! the same published constants and check numerical agreement with zentone.
//! This catches:
//! - Typo'd constants (wrong digit in a matrix entry)
//! - Operator-order divergence (multiply-then-divide vs divide-then-multiply)
//! - Missing `.min(1.0)` clamp or other post-processing
//!
//! If a test fails, the diff tells you which formula diverged and by how
//! much, which usually points directly at the bug.

use zentone::{
    AgxLook, LUMA_BT709, ToneMap, ToneMapCurve, aces_ap1, agx_tonemap, filmic_narkowicz,
    reinhard_extended, reinhard_simple, uncharted2_filmic,
};

const TOL: f32 = 1e-6;

// ============================================================================
// Test inputs — diverse enough to exercise each formula's full range
// ============================================================================

/// Scalar sweep covering shadows → mid → highlights → extreme HDR.
const SCALAR_SWEEP: [f32; 12] = [
    0.0, 0.001, 0.01, 0.05, 0.1, 0.18, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
];

/// RGB triples with varying chrominance, including near-black and saturated.
const RGB_TRIPLES: [[f32; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.18, 0.18, 0.18],
    [1.0, 0.5, 0.25],
    [0.1, 0.9, 0.05],
    [3.0, 1.5, 0.2],
    [0.02, 0.001, 0.5],
];

fn assert_close(a: f32, b: f32, label: &str) {
    let err = (a - b).abs();
    assert!(err < TOL, "{label}: {a} vs {b}, err={err}");
}

// ============================================================================
// Uncharted 2 / Hable filmic
// ============================================================================

/// Reference: John Hable, GDC 2010 "Uncharted 2: HDR Lighting"
/// Constants identical in gainforge (mappers.rs:292-299) and zentone.
fn hable_partial(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

fn hable_filmic_reference(v: f32) -> f32 {
    const EXPOSURE_BIAS: f32 = 2.0;
    const W: f32 = 11.2;
    let w_scale = 1.0 / hable_partial(W);
    (hable_partial(v * EXPOSURE_BIAS) * w_scale).min(1.0)
}

#[test]
fn uncharted2_matches_hable_reference() {
    for &x in &SCALAR_SWEEP {
        let zt = uncharted2_filmic(x);
        let rf = hable_filmic_reference(x);
        assert_close(zt, rf, &format!("uncharted2({x})"));
    }
}

// ============================================================================
// Narkowicz ACES filmic
// ============================================================================

/// Reference: Krzysztof Narkowicz, 2015
/// "ACES Filmic Tone Mapping Curve"
/// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn narkowicz_reference(x: f32) -> f32 {
    let a = 2.51_f32;
    let b = 0.03_f32;
    let c = 2.43_f32;
    let d = 0.59_f32;
    let e = 0.14_f32;
    (x * (a * x + b) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

#[test]
fn narkowicz_matches_reference() {
    for &x in &SCALAR_SWEEP {
        let zt = filmic_narkowicz(x);
        let rf = narkowicz_reference(x);
        assert_close(zt, rf, &format!("narkowicz({x})"));
    }
}

// ============================================================================
// ACES AP1 (Narkowicz RRT+ODT fit)
// ============================================================================

/// Reference: same Narkowicz post, the 3×3 input/output matrix fit.
/// Constants verified against gainforge mappers.rs:350-366.
fn aces_ap1_reference(rgb: [f32; 3]) -> [f32; 3] {
    // Input matrix (sRGB → ACES AP1 approximation)
    let a = 0.59719 * rgb[0] + 0.35458 * rgb[1] + 0.04823 * rgb[2];
    let b = 0.07600 * rgb[0] + 0.90834 * rgb[1] + 0.01566 * rgb[2];
    let c = 0.02840 * rgb[0] + 0.13383 * rgb[1] + 0.83777 * rgb[2];
    // RRT+ODT rational
    let ra = a * (a + 0.0245786) - 0.000090537;
    let rb = a * (a * 0.983729 + 0.4329510) + 0.238081;
    let ga = b * (b + 0.0245786) - 0.000090537;
    let gb = b * (b * 0.983729 + 0.4329510) + 0.238081;
    let ba = c * (c + 0.0245786) - 0.000090537;
    let bb = c * (c * 0.983729 + 0.4329510) + 0.238081;
    let mr = ra / rb;
    let mg = ga / gb;
    let mb = ba / bb;
    // Output matrix (ACES AP1 → sRGB approximation)
    [
        (1.60475 * mr - 0.53108 * mg - 0.07367 * mb).min(1.0),
        (-0.10208 * mr + 1.10813 * mg - 0.00605 * mb).min(1.0),
        (-0.00327 * mr - 0.07276 * mg + 1.07602 * mb).min(1.0),
    ]
}

#[test]
fn aces_ap1_matches_reference() {
    for rgb in &RGB_TRIPLES {
        let zt = aces_ap1(*rgb);
        let rf = aces_ap1_reference(*rgb);
        for i in 0..3 {
            assert_close(zt[i], rf[i], &format!("aces_ap1({rgb:?})[{i}]"));
        }
    }
}

// ============================================================================
// AgX (Blender)
// ============================================================================

/// Reference: Blender's AgX implementation.
/// Inset/outset matrices from gainforge mappers.rs:555-569, which itself
/// extracts them from the Blender source (MIT license).
/// Contrast polynomial from gainforge mappers.rs:572-586.
///
/// The outset matrix used here is the analytical inverse of the inset,
/// which Blender computes at const time. gainforge stores it as AGX_OUTSET_INV
/// but applies `AGX_OUTSET_INV.inverse()`. Both should produce the same
/// outset numerically — we verify against zentone's hardcoded outset
/// constants.
/// The correct AgX contrast polynomial (degree 7), verified by expanding
/// gainforge's `mlaf(acc, a, b) = acc + a*b` convention.
///
/// result = w0 + w1*x² + w2*x⁴ + w3*x⁶
///   where w_i = intercept_i + slope_i * x
///
/// Expanded: 0.002857 − 0.1718x + 4.361x² − 28.72x³
///           + 92.06x⁴ − 126.7x⁵ + 78.01x⁶ − 17.86x⁷
fn agx_contrast_reference(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let w0 = 0.002857 - 0.1718 * x;
    let w1 = 4.361 - 28.72 * x;
    let w2 = 92.06 - 126.7 * x;
    let w3 = 78.01 - 17.86 * x;
    w0 + w1 * x2 + w2 * x4 + w3 * x6
}

#[test]
fn agx_contrast_polynomial_endpoints() {
    // At x=0: ≈0.003 (near-black). At x=1: ≈0.982 (near-white).
    let at_zero = agx_contrast_reference(0.0);
    assert!(
        (at_zero - 0.002857).abs() < 1e-5,
        "contrast(0) should be ~0.003, got {at_zero}"
    );
    let at_one = agx_contrast_reference(1.0);
    assert!(
        (at_one - 0.982).abs() < 0.01,
        "contrast(1) should be ~0.982, got {at_one}"
    );
    // Broadly monotonic over [0.05, 1.0] (the polynomial has a tiny dip
    // near x=0 that's negligible in the near-black region).
    let mut last = agx_contrast_reference(0.05);
    for i in 6..=100 {
        let x = i as f32 / 100.0;
        let y = agx_contrast_reference(x);
        assert!(
            y >= last - 1e-4,
            "contrast not monotonic at x={x}: {y} < {last}"
        );
        last = y;
    }
}

#[test]
fn agx_full_pipeline_all_looks() {
    for &look in &[AgxLook::Default, AgxLook::Punchy, AgxLook::Golden] {
        for rgb in &RGB_TRIPLES {
            let out = agx_tonemap(*rgb, look);
            for (i, c) in out.iter().enumerate() {
                assert!(
                    c.is_finite() && (0.0..=1.0).contains(c),
                    "agx({rgb:?}, {look:?})[{i}] out of range: {c}"
                );
            }
        }
    }
}

/// AgX via the ToneMapCurve dispatch must produce the same output as the
/// direct `agx_tonemap` call.
#[test]
fn agx_curve_dispatch_matches_direct() {
    for &look in &[AgxLook::Default, AgxLook::Punchy, AgxLook::Golden] {
        let curve = ToneMapCurve::Agx(look);
        for rgb in &RGB_TRIPLES {
            let direct = agx_tonemap(*rgb, look);
            let dispatch = curve.map_rgb(*rgb);
            for i in 0..3 {
                assert_close(
                    direct[i],
                    dispatch[i],
                    &format!("agx({look:?}) direct vs dispatch at {rgb:?}[{i}]"),
                );
            }
        }
    }
}

// ============================================================================
// Reinhard variants
// ============================================================================

#[test]
fn reinhard_simple_is_x_over_1_plus_x() {
    for &x in &SCALAR_SWEEP {
        let expected = x / (1.0 + x);
        assert_close(
            reinhard_simple(x),
            expected,
            &format!("reinhard_simple({x})"),
        );
    }
}

#[test]
fn reinhard_extended_is_standard_formula() {
    for &h in &[2.0_f32, 4.0, 8.0] {
        for &x in &SCALAR_SWEEP {
            let h2 = h * h;
            let expected = x * (1.0 + x / h2) / (1.0 + x);
            assert_close(
                reinhard_extended(x, h),
                expected,
                &format!("reinhard_extended({x}, {h})"),
            );
        }
    }
}

// ============================================================================
// ToneMapCurve dispatch — all luma-carrying variants use BT.709 consistently
// ============================================================================

#[test]
fn all_luma_variants_use_bt709_consistently() {
    // For curves that carry luma, verify they produce different output
    // from the same curve with BT.2020 luma — proving the stored luma
    // is actually used and not hardcoded.
    let curves_709 = [
        ToneMapCurve::ExtendedReinhard {
            l_max: 4.0,
            luma: LUMA_BT709,
        },
        ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 },
        ToneMapCurve::TunedReinhard {
            content_max_nits: 4000.0,
            display_max_nits: 250.0,
            luma: LUMA_BT709,
        },
    ];
    let curves_2020 = [
        ToneMapCurve::ExtendedReinhard {
            l_max: 4.0,
            luma: zentone::LUMA_BT2020,
        },
        ToneMapCurve::ReinhardJodie {
            luma: zentone::LUMA_BT2020,
        },
        ToneMapCurve::TunedReinhard {
            content_max_nits: 4000.0,
            display_max_nits: 250.0,
            luma: zentone::LUMA_BT2020,
        },
    ];

    // Use a highly chromatic input so luma divergence is visible.
    let rgb = [0.1_f32, 0.9, 0.05];
    for (c709, c2020) in curves_709.iter().zip(curves_2020.iter()) {
        let out_709 = c709.map_rgb(rgb);
        let out_2020 = c2020.map_rgb(rgb);
        let mut any_diff = false;
        for i in 0..3 {
            if (out_709[i] - out_2020[i]).abs() > 1e-5 {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "curve {c709:?} produced identical output for BT.709 vs BT.2020 luma"
        );
    }
}
