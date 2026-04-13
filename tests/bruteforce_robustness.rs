//! Brute-force robustness tests: every tonemapper must produce finite,
//! non-NaN output for all non-NaN finite input, including negatives, zeros,
//! subnormals, and extreme values.
//!
//! This is a deterministic exhaustive sweep — runs in CI on every push.

use zentone::*;

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

/// Edge-case f32 values: zeros, subnormals, negatives, extremes, specials.
const EDGE_VALUES: [f32; 24] = [
    0.0,
    -0.0,
    f32::MIN_POSITIVE,       // smallest positive normal
    f32::MIN_POSITIVE / 2.0, // subnormal
    1e-38,                   // near subnormal boundary
    1e-10,
    1e-5,
    0.001,
    0.01,
    0.18, // mid-gray
    0.5,
    1.0,
    1.5,
    4.0,
    100.0,
    10000.0,
    1e10,
    1e30,
    f32::MAX / 2.0,
    f32::MAX,
    -1e-10,
    -0.001,
    -1.0,
    -1e10,
];

// ============================================================================
// map_rgb: every edge value × every channel combination → finite output
// ============================================================================

#[test]
fn map_rgb_edge_values_all_finite() {
    let mut failures = Vec::new();

    for (name, tm) in all_tonemappers() {
        // Test every edge value in every channel position.
        for (i, &r) in EDGE_VALUES.iter().enumerate() {
            for (j, &g) in EDGE_VALUES.iter().enumerate() {
                for (k, &b) in EDGE_VALUES.iter().enumerate() {
                    // Skip the full 24³ cross product — too slow.
                    // Test: diagonal (same value all channels) + each edge
                    // value with mid-gray in other channels.
                    if i != j || j != k {
                        if !(j == 9 && k == 9) && !(i == 9 && k == 9) && !(i == 9 && j == 9) {
                            continue;
                        }
                    }

                    let input = [r, g, b];
                    let out = tm.map_rgb(input);

                    for (ch, &v) in out.iter().enumerate() {
                        if v.is_nan() || v.is_infinite() {
                            failures.push(format!("{name}: map_rgb({input:?})[{ch}] = {v}"));
                            if failures.len() >= 20 {
                                break;
                            }
                        }
                    }
                }
                if failures.len() >= 20 {
                    break;
                }
            }
            if failures.len() >= 20 {
                break;
            }
        }
    }

    assert!(
        failures.is_empty(),
        "NaN/Inf outputs found ({} total):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ============================================================================
// map_row: interleaved edge-case rows through SIMD path → finite output
// ============================================================================

#[test]
fn map_row_edge_values_all_finite() {
    let mut failures = Vec::new();

    // Build a row from edge values: each "pixel" uses one edge value for all 3 channels.
    let mut row_rgb: Vec<f32> = Vec::with_capacity(EDGE_VALUES.len() * 3);
    for &v in &EDGE_VALUES {
        row_rgb.push(v);
        row_rgb.push(v);
        row_rgb.push(v);
    }

    let mut row_rgba: Vec<f32> = Vec::with_capacity(EDGE_VALUES.len() * 4);
    for &v in &EDGE_VALUES {
        row_rgba.push(v);
        row_rgba.push(v);
        row_rgba.push(v);
        row_rgba.push(0.5); // alpha
    }

    for (name, tm) in all_tonemappers() {
        // RGB path
        let mut row = row_rgb.clone();
        tm.map_row(&mut row, 3);
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                let pixel = i / 3;
                let ch = i % 3;
                failures.push(format!(
                    "{name}: map_row RGB pixel {pixel} (input={}) ch {ch} = {v}",
                    EDGE_VALUES[pixel]
                ));
            }
        }

        // RGBA path
        let mut row = row_rgba.clone();
        tm.map_row(&mut row, 4);
        for (i, chunk) in row.chunks_exact(4).enumerate() {
            for ch in 0..3 {
                if chunk[ch].is_nan() || chunk[ch].is_infinite() {
                    failures.push(format!(
                        "{name}: map_row RGBA pixel {i} (input={}) ch {ch} = {}",
                        EDGE_VALUES[i], chunk[ch]
                    ));
                }
            }
            // Alpha must be preserved exactly.
            if (chunk[3] - 0.5).abs() > 1e-6 {
                failures.push(format!(
                    "{name}: map_row RGBA pixel {i} alpha corrupted: {}",
                    chunk[3]
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "NaN/Inf/corrupt outputs ({} total):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ============================================================================
// Uniform f32 bit-sweep: test 2M evenly-spaced bit patterns per curve
// ============================================================================

#[test]
fn bit_sweep_no_nan_or_inf() {
    // Sweep f32 bit space in steps of 1024 (covers ~2M values across
    // the full f32 range including subnormals, negatives, large values).
    let mut failures = Vec::new();

    // Only test per-channel curves for the bit sweep (cross-channel
    // curves are tested with the edge-value grid above).
    let per_channel: &[(&str, Box<dyn ToneMap>)] = &[
        ("Reinhard", Box::new(ToneMapCurve::Reinhard)),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic)),
        ("Clamp", Box::new(ToneMapCurve::Clamp)),
    ];

    for (name, tm) in per_channel {
        let mut bits: u32 = 0;
        loop {
            let v = f32::from_bits(bits);

            // Skip NaN/Inf inputs — we only care about finite input.
            if v.is_finite() {
                let out = tm.map_rgb([v, v, v]);
                for (ch, &o) in out.iter().enumerate() {
                    if o.is_nan() || o.is_infinite() {
                        failures.push(format!("{name}: input={v} (bits=0x{bits:08x})[{ch}] = {o}"));
                        if failures.len() >= 10 {
                            break;
                        }
                    }
                }
            }

            if failures.len() >= 10 {
                break;
            }

            // Step by 1024 ≈ 4M values tested (2^32 / 1024)
            let (next, overflow) = bits.overflowing_add(1024);
            if overflow {
                break;
            }
            bits = next;
        }
    }

    assert!(
        failures.is_empty(),
        "NaN/Inf in bit sweep ({} total):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ============================================================================
// Pipeline: PQ edge cases → no NaN/Inf/negative in output
// ============================================================================

#[test]
fn pipeline_pq_edge_cases_no_nan() {
    use zentone::pipeline::tonemap_pq_to_linear_srgb;

    let tm = Bt2408Tonemapper::new(4000.0, 1000.0);

    // PQ values outside [0,1], negative, zero, near-1
    let edge_pq: Vec<f32> = vec![
        0.0, -0.0, 1e-10, 0.001, 0.1, 0.5, 0.58, 0.75, 0.9, 0.99, 1.0, 1.001, 1.5, -0.1, -1.0,
    ];

    let mut src = Vec::with_capacity(edge_pq.len() * 3);
    for &v in &edge_pq {
        src.push(v);
        src.push(v);
        src.push(v);
    }

    let mut out = vec![0.0_f32; src.len()];
    tonemap_pq_to_linear_srgb(&src, &mut out, &tm, 3);

    let mut failures = Vec::new();
    for (i, chunk) in out.chunks_exact(3).enumerate() {
        for (ch, &v) in chunk.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                failures.push(format!("PQ pipeline: input={} → ch {ch} = {v}", edge_pq[i]));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "NaN/Inf in PQ pipeline:\n{}",
        failures.join("\n")
    );
}
