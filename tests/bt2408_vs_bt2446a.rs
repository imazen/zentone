//! Numerical comparison of BT.2408 EETF vs BT.2446 Method A for
//! the same conversion: 1000 cd/m² HDR → SDR.
//!
//! This addresses issue #8: which algorithm produces better output for
//! gain map SDR base generation? We compare on a structured test set
//! and measure key perceptual metrics.

use zentone::*;

const WIDTH: usize = 256;

/// Generate a neutral luminance ramp from 0 to 1.0 (= hdr_peak).
fn neutral_ramp() -> Vec<f32> {
    let mut row = Vec::with_capacity(WIDTH * 3);
    for i in 0..WIDTH {
        let v = i as f32 / (WIDTH - 1) as f32;
        row.extend_from_slice(&[v, v, v]);
    }
    row
}

/// Generate a saturated color ramp (strong red, increasing luminance).
fn saturated_ramp() -> Vec<f32> {
    let mut row = Vec::with_capacity(WIDTH * 3);
    for i in 0..WIDTH {
        let v = i as f32 / (WIDTH - 1) as f32;
        row.extend_from_slice(&[v, v * 0.1, v * 0.05]);
    }
    row
}

fn lum(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

struct ComparisonResult {
    name: &'static str,
    /// Maximum output value across all channels.
    max_output: f32,
    /// Output luminance at HDR mid-gray (18% = 0.18).
    midgray_lum: f32,
    /// Output luminance at HDR peak (1.0).
    peak_lum: f32,
    /// Contrast ratio (peak_lum / shadow_lum) where shadow = 1%.
    contrast_ratio: f32,
    /// Whether output is bounded to [0, 1].
    all_bounded: bool,
    /// Number of monotonicity violations on the neutral ramp.
    mono_violations: u32,
}

fn analyze(name: &'static str, tm: &dyn ToneMap, ramp: &[f32]) -> ComparisonResult {
    let mut row = ramp.to_vec();
    tm.map_row(&mut row, 3);

    let mut max_output = 0.0_f32;
    let mut all_bounded = true;
    let mut mono_violations = 0_u32;
    let mut last_lum = -1.0_f32;
    let mut midgray_lum = 0.0_f32;
    let mut peak_lum = 0.0_f32;
    let mut shadow_lum = 0.0_f32;

    for (px, chunk) in row.chunks_exact(3).enumerate() {
        let ol = lum([chunk[0], chunk[1], chunk[2]]);
        max_output = max_output.max(chunk[0]).max(chunk[1]).max(chunk[2]);
        if chunk[0] > 1.001 || chunk[1] > 1.001 || chunk[2] > 1.001 {
            all_bounded = false;
        }
        if ol < last_lum - 1e-5 {
            mono_violations += 1;
        }
        last_lum = ol;

        let input_frac = px as f32 / (WIDTH - 1) as f32;
        if (input_frac - 0.18).abs() < 0.01 {
            midgray_lum = ol;
        }
        if (input_frac - 0.01).abs() < 0.005 {
            shadow_lum = ol;
        }
        if px == WIDTH - 1 {
            peak_lum = ol;
        }
    }

    let contrast_ratio = if shadow_lum > 1e-6 {
        peak_lum / shadow_lum
    } else {
        f32::INFINITY
    };

    ComparisonResult {
        name,
        max_output,
        midgray_lum,
        peak_lum,
        contrast_ratio,
        all_bounded,
        mono_violations,
    }
}

#[test]
fn compare_bt2408_vs_bt2446a_neutral() {
    let ramp = neutral_ramp();

    let bt2408_yrgb = Bt2408Tonemapper::new(1000.0, 100.0);
    let bt2408_maxrgb = Bt2408Tonemapper::max_rgb(1000.0, 100.0);
    let bt2446a = Bt2446A::new(1000.0, 100.0);
    let bt2446b = Bt2446B::new(1000.0, 100.0);
    let bt2446c = Bt2446C::new(1000.0, 100.0);

    let results = [
        analyze("BT.2408 YRGB", &bt2408_yrgb, &ramp),
        analyze("BT.2408 maxRGB", &bt2408_maxrgb, &ramp),
        analyze("BT.2446A", &bt2446a, &ramp),
        analyze("BT.2446B", &bt2446b, &ramp),
        analyze("BT.2446C", &bt2446c, &ramp),
    ];

    println!("\n=== BT.2408 vs BT.2446 Comparison (neutral ramp, 1000→100 cd/m²) ===");
    println!(
        "{:<18} {:>8} {:>8} {:>8} {:>10} {:>8} {:>6}",
        "Algorithm", "MidGray", "Peak", "MaxOut", "Contrast", "Bounded", "Mono"
    );
    println!("{}", "-".repeat(75));
    for r in &results {
        println!(
            "{:<18} {:>8.4} {:>8.4} {:>8.4} {:>10.1} {:>8} {:>6}",
            r.name,
            r.midgray_lum,
            r.peak_lum,
            r.max_output,
            r.contrast_ratio,
            if r.all_bounded { "yes" } else { "NO" },
            r.mono_violations
        );
    }

    // Assertions: all should be monotonic on neutral ramp
    for r in &results {
        assert_eq!(
            r.mono_violations, 0,
            "{}: {} monotonicity violations",
            r.name, r.mono_violations
        );
    }
}

#[test]
fn compare_bt2408_vs_bt2446a_saturated() {
    let ramp = saturated_ramp();

    let bt2408_yrgb = Bt2408Tonemapper::new(1000.0, 100.0);
    let bt2408_maxrgb = Bt2408Tonemapper::max_rgb(1000.0, 100.0);
    let bt2446a = Bt2446A::new(1000.0, 100.0);

    let results = [
        analyze("BT.2408 YRGB", &bt2408_yrgb, &ramp),
        analyze("BT.2408 maxRGB", &bt2408_maxrgb, &ramp),
        analyze("BT.2446A", &bt2446a, &ramp),
    ];

    println!("\n=== BT.2408 vs BT.2446A (saturated red ramp, 1000→100 cd/m²) ===");
    println!(
        "{:<18} {:>8} {:>8} {:>8} {:>8}",
        "Algorithm", "MidGray", "Peak", "MaxOut", "Bounded"
    );
    println!("{}", "-".repeat(50));
    for r in &results {
        println!(
            "{:<18} {:>8.4} {:>8.4} {:>8.4} {:>8}",
            r.name,
            r.midgray_lum,
            r.peak_lum,
            r.max_output,
            if r.all_bounded { "yes" } else { "NO" }
        );
    }

    // BT.2408 YRGB should blow up on saturated red (known limitation)
    assert!(
        !results[0].all_bounded,
        "BT.2408 YRGB should exceed [0,1] on saturated red"
    );

    // BT.2408 maxRGB should stay bounded
    assert!(
        results[1].all_bounded,
        "BT.2408 maxRGB should stay bounded on saturated red"
    );
}
