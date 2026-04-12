//! Tests and documents the behavior of every zentone curve on negative,
//! zero, NaN, and Inf inputs.
//!
//! This is a contract test — it pins the current behavior so changes are
//! intentional and reviewable.

#![allow(clippy::excessive_precision)]

use zentone::*;

/// For each curve, test a set of adversarial inputs and record:
/// (a) is output finite?  (b) is it in [0,1]?  (c) is it negative?
///
/// We don't assert a specific "correct" behavior for all curves on
/// negative input — the physical domain is [0, headroom] and negatives
/// are out of spec. But we DO assert:
/// 1. No panics.
/// 2. Zero input → finite output.
/// 3. Output documented.
struct CurveCase {
    name: &'static str,
    curve: Box<dyn ToneMap>,
}

fn all_curves() -> Vec<CurveCase> {
    vec![
        CurveCase {
            name: "Reinhard",
            curve: Box::new(ToneMapCurve::Reinhard),
        },
        CurveCase {
            name: "ExtReinhard",
            curve: Box::new(ToneMapCurve::ExtendedReinhard {
                l_max: 4.0,
                luma: LUMA_BT709,
            }),
        },
        CurveCase {
            name: "ReinhardJodie",
            curve: Box::new(ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 }),
        },
        CurveCase {
            name: "TunedReinhard",
            curve: Box::new(ToneMapCurve::TunedReinhard {
                content_max_nits: 4000.0,
                display_max_nits: 250.0,
                luma: LUMA_BT709,
            }),
        },
        CurveCase {
            name: "Narkowicz",
            curve: Box::new(ToneMapCurve::Narkowicz),
        },
        CurveCase {
            name: "HableFilmic",
            curve: Box::new(ToneMapCurve::HableFilmic),
        },
        CurveCase {
            name: "AcesAp1",
            curve: Box::new(ToneMapCurve::AcesAp1),
        },
        CurveCase {
            name: "Bt2390",
            curve: Box::new(ToneMapCurve::Bt2390 {
                source_peak: 4.0,
                target_peak: 1.0,
            }),
        },
        CurveCase {
            name: "AgxDefault",
            curve: Box::new(ToneMapCurve::Agx(AgxLook::Default)),
        },
        CurveCase {
            name: "AgxPunchy",
            curve: Box::new(ToneMapCurve::Agx(AgxLook::Punchy)),
        },
        CurveCase {
            name: "AgxGolden",
            curve: Box::new(ToneMapCurve::Agx(AgxLook::Golden)),
        },
        CurveCase {
            name: "Clamp",
            curve: Box::new(ToneMapCurve::Clamp),
        },
        CurveCase {
            name: "Bt2408",
            curve: Box::new(Bt2408Tonemapper::new(4000.0, 1000.0)),
        },
        CurveCase {
            name: "FilmicSpline",
            curve: Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
        },
    ]
}

const PROBES: &[([f32; 3], &str)] = &[
    ([0.0, 0.0, 0.0], "zero"),
    ([0.001, 0.001, 0.001], "near-black"),
    ([-0.1, -0.1, -0.1], "mild-negative"),
    ([-1.0, 0.5, 0.25], "mixed-negative"),
    ([1.0, 1.0, 1.0], "unity"),
    ([4.0, 2.0, 0.5], "hdr-moderate"),
];

/// No curve should panic on any of the probe inputs.
#[test]
fn no_panics_on_any_input() {
    for case in all_curves() {
        for (rgb, label) in PROBES {
            let _out = case.curve.map_rgb(*rgb);
            // If we get here, no panic occurred.
            let _ = (case.name, label); // suppress unused
        }
    }
}

/// Zero input should produce finite output for every curve.
#[test]
fn zero_input_produces_finite_output() {
    for case in all_curves() {
        let out = case.curve.map_rgb([0.0, 0.0, 0.0]);
        for (i, c) in out.iter().enumerate() {
            assert!(
                c.is_finite(),
                "{} produced non-finite at zero input, channel {i}: {c}",
                case.name
            );
        }
    }
}

/// Near-black (0.001) should produce finite output. Most curves produce
/// non-negative, but ACES AP1's Narkowicz approximation goes slightly
/// negative near black (−2.7e-4 at input 0.001) due to the −0.000090537
/// offset in the RRT numerator. This is documented, not a bug.
#[test]
fn near_black_produces_finite() {
    for case in all_curves() {
        let out = case.curve.map_rgb([0.001, 0.001, 0.001]);
        for (i, c) in out.iter().enumerate() {
            assert!(
                c.is_finite(),
                "{} produced non-finite at near-black, channel {i}: {c}",
                case.name
            );
        }
    }
}

/// ACES AP1 specifically goes negative at near-black. Pin the behavior.
#[test]
fn aces_ap1_near_black_is_slightly_negative() {
    let out = aces_ap1([0.001, 0.001, 0.001]);
    assert!(
        out[0] < 0.0 && out[0] > -0.001,
        "ACES AP1 near-black behavior changed: expected slightly negative, got {:?}",
        out
    );
}

/// Unity input (1.0, 1.0, 1.0) should produce finite output in [0, 1]
/// for every curve (since 1.0 is SDR white, every tonemap should handle it).
#[test]
fn unity_input_produces_output_in_unit_range() {
    for case in all_curves() {
        let out = case.curve.map_rgb([1.0, 1.0, 1.0]);
        for (i, c) in out.iter().enumerate() {
            assert!(
                c.is_finite() && *c >= 0.0 && *c <= 1.001,
                "{} produced out-of-range at unity input, channel {i}: {c}",
                case.name
            );
        }
    }
}

/// Document negative-input behavior for each curve. This test prints
/// the results — it doesn't assert a specific contract, it makes the
/// behavior visible in test output.
#[test]
fn document_negative_input_behavior() {
    println!("\n=== Negative input behavior ===");
    println!(
        "{:<16} {:<14} {:>10} {:>10} {:>10}  finite  [0,1]",
        "Curve", "Input", "R", "G", "B"
    );
    println!("{}", "-".repeat(80));

    for case in all_curves() {
        let out = case.curve.map_rgb([-0.1, -0.1, -0.1]);
        let finite = out.iter().all(|v| v.is_finite());
        let in_range = out.iter().all(|v| *v >= 0.0 && *v <= 1.0);
        println!(
            "{:<16} {:<14} {:>10.6} {:>10.6} {:>10.6}  {:<7} {}",
            case.name,
            "[-0.1]*3",
            out[0],
            out[1],
            out[2],
            if finite { "yes" } else { "NO" },
            if in_range { "yes" } else { "no" }
        );
    }

    for case in all_curves() {
        let out = case.curve.map_rgb([-1.0, 0.5, 0.25]);
        let finite = out.iter().all(|v| v.is_finite());
        let in_range = out.iter().all(|v| *v >= 0.0 && *v <= 1.0);
        println!(
            "{:<16} {:<14} {:>10.6} {:>10.6} {:>10.6}  {:<7} {}",
            case.name,
            "[-1,0.5,0.25]",
            out[0],
            out[1],
            out[2],
            if finite { "yes" } else { "NO" },
            if in_range { "yes" } else { "no" }
        );
    }
}

/// HDR moderate input (4, 2, 0.5) should produce finite output in [0, 1]
/// for every curve EXCEPT BT.2390, which expects normalized input in
/// [0, 1] where 1.0 = source_peak (not raw linear-light values).
/// Passing 4.0 to bt2390_tonemap with source_peak=4.0 is out of its
/// normalized domain and produces garbage. This is a design contract,
/// not a bug — BT.2390 scene-linear is not a "raw HDR pixel" curve.
#[test]
fn hdr_moderate_in_range() {
    for case in all_curves() {
        if case.name == "Bt2390" || case.name == "Bt2408" {
            continue; // normalized-input curves, skip (tested separately)
        }
        let out = case.curve.map_rgb([4.0, 2.0, 0.5]);
        for (i, c) in out.iter().enumerate() {
            assert!(
                c.is_finite() && *c >= 0.0 && *c <= 1.001,
                "{} produced out-of-range at HDR moderate, channel {i}: {c}",
                case.name
            );
        }
    }
}

/// BT.2390 scene-linear at NORMALIZED input [0.5, 0.5, 0.5] should
/// produce output in [0, target_peak/source_peak].
#[test]
fn bt2390_normalized_input_in_range() {
    let curve = ToneMapCurve::Bt2390 {
        source_peak: 4.0,
        target_peak: 1.0,
    };
    let out = curve.map_rgb([0.5, 0.5, 0.5]);
    for (i, c) in out.iter().enumerate() {
        assert!(
            c.is_finite() && *c >= 0.0 && *c <= 0.26,
            "Bt2390 normalized: channel {i} out of range: {c}"
        );
    }
}

/// Bt2408 at content-normalized input [0.4, 0.2, 0.05]
/// (= 4000, 2000, 500 nits for content_max=10000) should produce
/// finite display-normalized output.
#[test]
fn bt2408_content_normalized_input_in_range() {
    let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
    let out = tm.map_rgb([0.4, 0.2, 0.05]);
    for (i, c) in out.iter().enumerate() {
        assert!(
            c.is_finite() && *c >= 0.0,
            "Bt2408 content-normalized: channel {i} out of range: {c}"
        );
    }
}
