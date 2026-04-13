//! Continuity tests: detect discontinuities in tone curves by sweeping
//! fine-grained input and checking that output changes smoothly.
//!
//! A discontinuity shows up as a large output jump between adjacent
//! input samples. These tests use 100k samples across [0, peak] and
//! flag any step > threshold as a failure.

use zentone::gamut::{BT2020_TO_BT709, apply_matrix, soft_clip};
use zentone::*;

fn all_configs() -> Vec<(&'static str, Box<dyn ToneMap>, f32)> {
    // (name, tonemapper, input_peak)
    vec![
        ("Reinhard", Box::new(ToneMapCurve::Reinhard), 10.0),
        (
            "ExtReinhard",
            Box::new(ToneMapCurve::ExtendedReinhard {
                l_max: 4.0,
                luma: LUMA_BT709,
            }),
            10.0,
        ),
        (
            "ReinhardJodie",
            Box::new(ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 }),
            10.0,
        ),
        (
            "TunedReinhard",
            Box::new(ToneMapCurve::TunedReinhard {
                content_max_nits: 4000.0,
                display_max_nits: 250.0,
                luma: LUMA_BT709,
            }),
            10.0,
        ),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz), 10.0),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic), 10.0),
        ("AcesAp1", Box::new(ToneMapCurve::AcesAp1), 10.0),
        (
            "Bt2390",
            Box::new(ToneMapCurve::Bt2390 {
                source_peak: 4.0,
                target_peak: 1.0,
            }),
            4.0,
        ),
        (
            "AgxDefault",
            Box::new(ToneMapCurve::Agx(AgxLook::Default)),
            10.0,
        ),
        (
            "AgxPunchy",
            Box::new(ToneMapCurve::Agx(AgxLook::Punchy)),
            10.0,
        ),
        (
            "AgxGolden",
            Box::new(ToneMapCurve::Agx(AgxLook::Golden)),
            10.0,
        ),
        ("Clamp", Box::new(ToneMapCurve::Clamp), 2.0),
        (
            "Bt2408",
            Box::new(Bt2408Tonemapper::new(4000.0, 1000.0)),
            1.0,
        ),
        (
            "Bt2408_MaxRgb",
            Box::new(Bt2408Tonemapper::max_rgb(4000.0, 1000.0)),
            1.0,
        ),
        ("Bt2446A", Box::new(Bt2446A::new(4000.0, 100.0)), 1.0),
        ("Bt2446B", Box::new(Bt2446B::new(4000.0, 100.0)), 1.0),
        ("Bt2446C", Box::new(Bt2446C::new(4000.0, 100.0)), 1.0),
        (
            "FilmicSpline",
            Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
            8.0,
        ),
    ]
}

// ============================================================================
// Neutral ramp continuity: sweep a single channel value, check smoothness
// ============================================================================

#[test]
fn neutral_ramp_no_discontinuity() {
    let n = 100_000;
    let mut failures = Vec::new();

    for (name, tm, peak) in all_configs() {
        let mut prev = tm.map_rgb([0.0, 0.0, 0.0]);
        let mut max_step: f32 = 0.0;
        let mut worst_input: f32 = 0.0;

        for i in 1..=n {
            let v = peak * (i as f32 / n as f32);
            let cur = tm.map_rgb([v, v, v]);

            for ch in 0..3 {
                let step = (cur[ch] - prev[ch]).abs();
                if step > max_step {
                    max_step = step;
                    worst_input = v;
                }
            }
            prev = cur;
        }

        // Threshold: the input step is peak/100k. For a well-behaved
        // curve mapping [0, peak] → [0, 1], the max output step should
        // be roughly 1/100k * max_slope.
        //
        // Known near-black issues (NOT code bugs):
        // - Bt2446A: Hunt-effect f = y_sdr/(1.1*y') → ∞ near y'=0
        // - AgX: log2(max(v, 1e-10)) floor transition
        // - Clamp: hard edge at 1.0
        let threshold = match name {
            "Clamp" => 0.02,
            "Bt2446A" => 0.03,                 // Hunt-effect near-black
            n if n.starts_with("Agx") => 0.02, // log2 floor
            _ => 0.01,
        };

        if max_step > threshold {
            failures.push(format!(
                "{name}: max_step={max_step:.6} at input={worst_input:.6} (threshold={threshold})"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Discontinuities detected:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// Per-channel ramp: sweep R with G=B=0.5 to catch cross-channel issues
// ============================================================================

#[test]
fn per_channel_ramp_no_discontinuity() {
    let n = 100_000;
    let mut failures = Vec::new();

    for (name, tm, peak) in all_configs() {
        for sweep_ch in 0..3 {
            let mut prev = {
                let mut rgb = [0.5, 0.5, 0.5];
                rgb[sweep_ch] = 0.0;
                tm.map_rgb(rgb)
            };
            let mut max_step: f32 = 0.0;
            let mut worst_input: f32 = 0.0;

            for i in 1..=n {
                let v = peak * (i as f32 / n as f32);
                let mut rgb = [0.5, 0.5, 0.5];
                rgb[sweep_ch] = v;
                let cur = tm.map_rgb(rgb);

                for ch in 0..3 {
                    let step = (cur[ch] - prev[ch]).abs();
                    if step > max_step {
                        max_step = step;
                        worst_input = v;
                    }
                }
                prev = cur;
            }

            let threshold = match name {
                "Clamp" => 0.02,
                "Bt2446A" => 0.03,   // Hunt-effect near-black instability
                "AgxGolden" => 0.04, // pow_midp(0.8) + log2 floor
                n if n.starts_with("Agx") => 0.02, // log2 floor
                _ => 0.01,
            };
            if max_step > threshold {
                failures.push(format!(
                    "{name} ch{sweep_ch}: max_step={max_step:.6} at input={worst_input:.6}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Per-channel discontinuities:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// Diagonal ramp: sweep all channels together with varying saturation
// ============================================================================

#[test]
fn saturated_ramp_no_discontinuity() {
    let n = 50_000;
    let mut failures = Vec::new();

    // Sweep a "warm" color: R=lum, G=0.6*lum, B=0.2*lum
    let ratios = [1.0_f32, 0.6, 0.2];

    for (name, tm, peak) in all_configs() {
        let mut prev = tm.map_rgb([0.0, 0.0, 0.0]);
        let mut max_step: f32 = 0.0;
        let mut worst_input: f32 = 0.0;

        for i in 1..=n {
            let v = peak * (i as f32 / n as f32);
            let rgb = [v * ratios[0], v * ratios[1], v * ratios[2]];
            let cur = tm.map_rgb(rgb);

            for ch in 0..3 {
                let step = (cur[ch] - prev[ch]).abs();
                if step > max_step {
                    max_step = step;
                    worst_input = v;
                }
            }
            prev = cur;
        }

        let threshold = match name {
            "Clamp" => 0.02,
            "Bt2446A" => 0.03,
            "AgxGolden" => 0.04,
            n if n.starts_with("Agx") => 0.02,
            _ => 0.01,
        };
        if max_step > threshold {
            failures.push(format!(
                "{name}: max_step={max_step:.6} at input={worst_input:.6}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Saturated ramp discontinuities:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// SIMD vs scalar continuity: map_row output should have no extra jumps
// ============================================================================

#[test]
fn map_row_no_discontinuity() {
    // Coarser than the scalar tests (10k vs 100k) since we process a full
    // row. Near-black steps are proportionally larger → higher thresholds.
    let n = 10_000;

    for (name, tm, peak) in all_configs() {
        // Build a row of n neutral pixels ramping from 0 to peak
        let mut row: Vec<f32> = (0..n)
            .flat_map(|i| {
                let v = peak * (i as f32 / n as f32);
                [v, v, v]
            })
            .collect();

        tm.map_row(&mut row, 3);

        let mut max_step: f32 = 0.0;
        for i in 1..n {
            for ch in 0..3 {
                let step = (row[i * 3 + ch] - row[(i - 1) * 3 + ch]).abs();
                max_step = max_step.max(step);
            }
        }

        // 10× coarser than 100k scalar sweep → proportionally higher thresholds
        let threshold = match name {
            "Clamp" => 0.1,
            "Bt2446A" => 0.06,
            "AgxGolden" => 0.05,
            n if n.starts_with("Agx") => 0.03,
            _ => 0.02,
        };
        assert!(
            max_step <= threshold,
            "{name}: map_row max_step={max_step:.6} (threshold={threshold})"
        );
    }
}

// ============================================================================
// Spike detection: output luminance should not spike above BOTH neighbors
// ============================================================================

/// Compute BT.709 luminance.
fn lum(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

#[test]
fn no_luminance_spikes_on_neutral_ramp() {
    let n = 50_000;
    let mut failures = Vec::new();

    for (name, tm, peak) in all_configs() {
        let vals: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let v = peak * (i as f32 / n as f32);
                tm.map_rgb([v, v, v])
            })
            .collect();

        for i in 1..n - 1 {
            let prev = lum(vals[i - 1]);
            let cur = lum(vals[i]);
            let next = lum(vals[i + 1]);

            // A spike: cur is higher than BOTH neighbors by more than noise.
            let spike = cur - prev.max(next);
            if spike > 0.005 {
                let input = peak * (i as f32 / n as f32);
                failures.push(format!(
                    "{name}: luminance spike at input={input:.6}: prev={prev:.4}, cur={cur:.4}, next={next:.4}, spike={spike:.4}"
                ));
                if failures.len() >= 5 {
                    break;
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Luminance spikes detected:\n{}",
        failures.join("\n")
    );
}

#[test]
fn no_luminance_spikes_on_hue_sweep() {
    let n = 10_000;
    let mut failures = Vec::new();

    for (name, tm, _peak) in all_configs() {
        // Sweep hue at luminance 2.0 (moderately HDR)
        let lum_level = 2.0_f32;
        let vals: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let hue = core::f32::consts::TAU * (i as f32 / n as f32);
                let r = lum_level * (0.5 + 0.5 * hue.cos()).max(0.0);
                let g = lum_level * (0.5 + 0.5 * (hue - 2.094).cos()).max(0.0);
                let b = lum_level * (0.5 + 0.5 * (hue + 2.094).cos()).max(0.0);
                tm.map_rgb([r, g, b])
            })
            .collect();

        for i in 1..n - 1 {
            let prev = lum(vals[i - 1]);
            let cur = lum(vals[i]);
            let next = lum(vals[i + 1]);

            let spike = cur - prev.max(next);
            if spike > 0.01 {
                let hue_deg = 360.0 * (i as f32 / n as f32);
                failures.push(format!(
                    "{name}: hue spike at {hue_deg:.1}°: prev={prev:.4}, cur={cur:.4}, next={next:.4}"
                ));
                if failures.len() >= 5 {
                    break;
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Hue sweep luminance spikes:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// Monotonicity: neutral ramp → output luminance must be non-decreasing
// ============================================================================

#[test]
fn neutral_ramp_luminance_monotonic() {
    let n = 50_000;
    let mut failures = Vec::new();

    for (name, tm, peak) in all_configs() {
        let mut prev_lum = -1.0_f32;
        let mut worst_violation: f32 = 0.0;
        let mut worst_input: f32 = 0.0;

        for i in 0..n {
            let v = peak * (i as f32 / n as f32);
            let out = tm.map_rgb([v, v, v]);
            let l = lum(out);

            let violation = prev_lum - l;
            if violation > worst_violation {
                worst_violation = violation;
                worst_input = v;
            }
            prev_lum = l;
        }

        // Allow tiny floating-point noise. AgX's normalized polynomial
        // has ~20 ULP jitter near its peak from the (raw - P0) * SCALE
        // normalization.
        if worst_violation > 5e-5 {
            failures.push(format!(
                "{name}: monotonicity violation={worst_violation:.6} at input={worst_input:.6}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Monotonicity violations:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// Per-channel monotonicity: sweeping one channel up must not decrease output
// ============================================================================

#[test]
fn per_channel_sweep_monotonic() {
    let n = 20_000;
    let mut failures = Vec::new();

    for (name, tm, peak) in all_configs() {
        for sweep_ch in 0..3 {
            let mut prev_out_ch = -1.0_f32;
            let mut worst: f32 = 0.0;
            let mut worst_input: f32 = 0.0;

            for i in 0..n {
                let v = peak * (i as f32 / n as f32);
                let mut rgb = [0.3, 0.3, 0.3]; // mid-gray base
                rgb[sweep_ch] = v;
                let out = tm.map_rgb(rgb);

                // The swept channel's output should be non-decreasing.
                let violation = prev_out_ch - out[sweep_ch];
                if violation > worst {
                    worst = violation;
                    worst_input = v;
                }
                prev_out_ch = out[sweep_ch];
            }

            // Cross-channel curves (ACES, AgX, BT.2408) can produce
            // non-monotonic per-channel output because the matrix mixes
            // channels. Only flag large violations (> 0.01).
            if worst > 0.01 {
                failures.push(format!(
                    "{name} ch{sweep_ch}: output decreased by {worst:.4} at input={worst_input:.4}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Per-channel monotonicity violations:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// Pipeline spike detection: PQ neutral ramp through full pipeline
// ============================================================================

#[test]
fn pq_pipeline_no_luminance_spikes() {
    use zentone::pipeline::tonemap_pq_to_linear_srgb;

    let tonemappers: Vec<(&str, Box<dyn ToneMap>)> = vec![
        ("Bt2408", Box::new(Bt2408Tonemapper::new(4000.0, 1000.0))),
        ("Reinhard", Box::new(ToneMapCurve::Reinhard)),
        ("AgxDefault", Box::new(ToneMapCurve::Agx(AgxLook::Default))),
    ];

    let n = 10_000;
    let mut failures = Vec::new();

    for (name, tm) in &tonemappers {
        // PQ neutral ramp
        let mut pq_row = Vec::with_capacity(n * 3);
        for i in 0..n {
            let nits = 4000.0 * (i as f32 / n as f32);
            let pq = linear_srgb::tf::linear_to_pq(nits / 10000.0);
            pq_row.push(pq);
            pq_row.push(pq);
            pq_row.push(pq);
        }

        let mut out = vec![0.0_f32; pq_row.len()];
        tonemap_pq_to_linear_srgb(&pq_row, &mut out, tm.as_ref(), 3);

        for i in 1..n - 1 {
            let prev = lum([out[(i - 1) * 3], out[(i - 1) * 3 + 1], out[(i - 1) * 3 + 2]]);
            let cur = lum([out[i * 3], out[i * 3 + 1], out[i * 3 + 2]]);
            let next = lum([out[(i + 1) * 3], out[(i + 1) * 3 + 1], out[(i + 1) * 3 + 2]]);

            let spike = cur - prev.max(next);
            if spike > 0.005 {
                failures.push(format!(
                    "{name}: PQ pipeline spike at pixel {i}: prev={prev:.4}, cur={cur:.4}, next={next:.4}"
                ));
                if failures.len() >= 5 {
                    break;
                }
            }

            // Also check monotonicity
            if cur < prev - 1e-5 {
                failures.push(format!(
                    "{name}: PQ pipeline non-monotonic at pixel {i}: {prev:.6} > {cur:.6}"
                ));
                if failures.len() >= 5 {
                    break;
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "PQ pipeline spikes/non-monotonic:\n{}",
        failures.join("\n")
    );
}

// ============================================================================
// Gamut clamp spike detection: soft_clip should not create brightness spikes
// ============================================================================

#[test]
fn soft_clip_no_luminance_spikes_on_gamut_sweep() {
    let n = 10_000;
    let mut failures = Vec::new();

    // Sweep saturated BT.2020 colors through gamut convert + soft_clip
    for sweep_ch in 0..3 {
        let mut prev_lum = -1.0_f32;
        let mut prev_prev_lum = -1.0_f32;

        for i in 0..n {
            let v = 2.0 * (i as f32 / n as f32); // 0 to 2 linear
            let mut bt2020 = [0.3, 0.3, 0.3];
            bt2020[sweep_ch] = v;

            let bt709 = apply_matrix(&BT2020_TO_BT709, bt2020);
            let clipped = soft_clip(bt709);
            let l = lum(clipped);

            if i >= 2 {
                let spike = prev_lum - prev_prev_lum.max(l);
                if spike > 0.005 {
                    failures.push(format!(
                        "soft_clip ch{sweep_ch}: spike at v={v:.4}: \
                         prev_prev={prev_prev_lum:.4}, prev={prev_lum:.4}, cur={l:.4}"
                    ));
                }
            }

            prev_prev_lum = prev_lum;
            prev_lum = l;
        }
    }

    assert!(
        failures.is_empty(),
        "Soft clip luminance spikes:\n{}",
        failures.join("\n")
    );
}
