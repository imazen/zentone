//! Brightness trace at HDR diffuse-white (203 nits) under the post-fix
//! reshoot's calling convention.
//!
//! Pins the three curves' actual SDR-linear output for an input pixel
//! representing HDR diffuse-white (1.0 in the LinearFloat buffer = 203
//! nits = SDR white). Producer-graded SDR places diffuse white near 1.0
//! linear (SDR 100 nits, the "diffuse-white-equivalent" anchor); these
//! pins catch under/over-exposure regressions at a glance.
//!
//! Added 2026-06-20 during the post-fix visual-mismatch investigation
//! (Bt2446C looks too bright, Bt2390/Bt2446A look too dark vs producer
//! SDR). The math here is what the post-fix reshoot `apply_curve` does
//! for source_peak_nits=1000 (typical HEIC HDR peak):
//!
//!   max_pixel_value = source_peak / 203 = 4.93
//!   content_norm_scale = 1 / 4.93 = 0.2030
//!   scratch[i] = hdr[i] * 0.2030   (so 1.0=203 nits → 0.2030; 1.0=1000 nits)
//!
//! Then the curve runs in source-peak-normalized domain (1.0 =
//! source_peak_nits), and Bt2390 rescales the output up to target-norm
//! by multiplying by 1/target_peak_in_src.

use zentone::{Bt2446A, Bt2446C, ToneMap, ToneMapCurve};

const DIFFUSE_WHITE_NITS: f32 = 203.0;
const TARGET_PEAK_NITS: f32 = 100.0;

/// Apply the post-fix reshoot's Bt2390 pipeline to a single pixel.
fn apply_bt2390(rgb: [f32; 3], source_peak_nits: f32) -> [f32; 3] {
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;
    let scratch = [
        rgb[0] * content_norm_scale,
        rgb[1] * content_norm_scale,
        rgb[2] * content_norm_scale,
    ];
    let target_peak_in_src = TARGET_PEAK_NITS / source_peak_nits.max(1.0);
    let tone = ToneMapCurve::Bt2390 {
        source_peak: 1.0,
        target_peak: target_peak_in_src,
    };
    let mapped = tone.map_rgb(scratch);
    let inv = 1.0 / target_peak_in_src.max(1e-6);
    [
        (mapped[0] * inv).clamp(0.0, 1.0),
        (mapped[1] * inv).clamp(0.0, 1.0),
        (mapped[2] * inv).clamp(0.0, 1.0),
    ]
}

/// Apply the post-fix reshoot's Bt2446A pipeline to a single pixel.
fn apply_bt2446a(rgb: [f32; 3], source_peak_nits: f32) -> [f32; 3] {
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;
    let scratch = [
        rgb[0] * content_norm_scale,
        rgb[1] * content_norm_scale,
        rgb[2] * content_norm_scale,
    ];
    let tm = Bt2446A::new(source_peak_nits, TARGET_PEAK_NITS);
    let mapped = tm.map_rgb(scratch);
    [
        mapped[0].clamp(0.0, 1.0),
        mapped[1].clamp(0.0, 1.0),
        mapped[2].clamp(0.0, 1.0),
    ]
}

/// Apply the post-fix reshoot's Bt2446C pipeline to a single pixel.
fn apply_bt2446c(rgb: [f32; 3], source_peak_nits: f32) -> [f32; 3] {
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;
    let scratch = [
        rgb[0] * content_norm_scale,
        rgb[1] * content_norm_scale,
        rgb[2] * content_norm_scale,
    ];
    let tm = Bt2446C::new(source_peak_nits, TARGET_PEAK_NITS);
    let mapped = tm.map_rgb(scratch);
    [
        mapped[0].clamp(0.0, 1.0),
        mapped[1].clamp(0.0, 1.0),
        mapped[2].clamp(0.0, 1.0),
    ]
}

#[test]
fn print_brightness_trace_for_diffuse_white_at_1000_nits_source() {
    // The corpus's typical HEIC measure_robust peak is ~1000 nits.
    let source_peak = 1000.0_f32;
    let dw = [1.0_f32, 1.0, 1.0]; // HDR diffuse white (203 nits)

    let bt2390_out = apply_bt2390(dw, source_peak);
    let bt2446a_out = apply_bt2446a(dw, source_peak);
    let bt2446c_out = apply_bt2446c(dw, source_peak);

    println!(
        "HDR diffuse-white (1.0,1.0,1.0) at source_peak={source_peak} nits:\n  \
         Bt2390   → {:.4} {:.4} {:.4}\n  \
         Bt2446A  → {:.4} {:.4} {:.4}\n  \
         Bt2446C  → {:.4} {:.4} {:.4}\n  \
         (producer SDR target for diffuse-white is ~1.0 in SDR linear, i.e. \
         100-nit equivalent — values significantly below 1.0 are under-exposed; \
         values clamped at 1.0 lose the super-white headroom.)",
        bt2390_out[0],
        bt2390_out[1],
        bt2390_out[2],
        bt2446a_out[0],
        bt2446a_out[1],
        bt2446a_out[2],
        bt2446c_out[0],
        bt2446c_out[1],
        bt2446c_out[2],
    );

    // Hypothesis: Bt2390 + Bt2446A drive diffuse white way below 1.0
    // (under-exposed) while Bt2446C is closest to 1.0 — matching the
    // user's visual report.
}

#[test]
fn print_brightness_trace_for_specular_highlight_at_1000_nits_source() {
    // A bright specular at HDR peak (1000 nits = 1.0/content_norm_scale =
    // 4.93 in the LinearFloat buffer).
    let source_peak = 1000.0_f32;
    let speca = [4.93_f32, 4.93, 4.93]; // HDR peak

    let bt2390_out = apply_bt2390(speca, source_peak);
    let bt2446a_out = apply_bt2446a(speca, source_peak);
    let bt2446c_out = apply_bt2446c(speca, source_peak);

    println!(
        "HDR peak (~1000 nits, buffer val {speca:?}):\n  \
         Bt2390   → {:.4} {:.4} {:.4}\n  \
         Bt2446A  → {:.4} {:.4} {:.4}\n  \
         Bt2446C  → {:.4} {:.4} {:.4}\n  \
         (producer SDR maxes at 1.0; values at 1.0 means the curve was \
         clamped or saturated.)",
        bt2390_out[0],
        bt2390_out[1],
        bt2390_out[2],
        bt2446a_out[0],
        bt2446a_out[1],
        bt2446a_out[2],
        bt2446c_out[0],
        bt2446c_out[1],
        bt2446c_out[2],
    );
}

#[test]
fn print_brightness_trace_across_input_grid_at_1000_nits_source() {
    // Sweep the buffer-input range covering 0 .. 1000-nit equivalent.
    // For source_peak=1000 nits and DW=203 nits, the buffer value at
    // X nits is X/203. So:
    //   18 nits (low) = 0.0887
    //   50 nits (mid-low) = 0.2463
    //   100 nits (= SDR-100) = 0.4926
    //   203 nits (= diffuse white) = 1.0
    //   500 nits (specular) = 2.463
    //   1000 nits (peak) = 4.926
    let source_peak = 1000.0_f32;
    let sample_points: &[(f32, &str)] = &[
        (18.0 / 203.0, "18 nits (deep shadow)"),
        (50.0 / 203.0, "50 nits (low-mid)"),
        (100.0 / 203.0, "100 nits (SDR 100-nit equiv)"),
        (1.0, "203 nits (HDR diffuse-white = SDR white)"),
        (500.0 / 203.0, "500 nits (specular)"),
        (1000.0 / 203.0, "1000 nits (HDR peak)"),
    ];
    println!("\nBrightness trace at source_peak={source_peak} nits:\n");
    println!(
        "  {:>40}  {:>11}  {:>11}  {:>11}",
        "Input (HDR nits)", "Bt2390", "Bt2446A", "Bt2446C"
    );
    for &(v, label) in sample_points {
        let bt2390_out = apply_bt2390([v, v, v], source_peak)[0];
        let bt2446a_out = apply_bt2446a([v, v, v], source_peak)[0];
        let bt2446c_out = apply_bt2446c([v, v, v], source_peak)[0];
        println!(
            "  {:>40}  {:>11.4}  {:>11.4}  {:>11.4}",
            label, bt2390_out, bt2446a_out, bt2446c_out
        );
    }
    // SDR baseline: producer SDR for a 203-nit-HDR-diffuse-white pixel
    // should land near 1.0 linear (= 100-nit SDR white). For shadow
    // detail at 50 nits HDR, SDR linear of ~0.5 is typical (sRGB
    // perceived mid-grey).
}

#[test]
fn print_brightness_trace_across_source_peaks() {
    // Same input (HDR diffuse white), but vary the source_peak across
    // the post-fix reshoot's range. Show how each curve responds to
    // peak-meter variation.
    let dw = [1.0_f32, 1.0, 1.0];
    let peaks = [500.0, 750.0, 1000.0, 1500.0, 2000.0, 4000.0];
    println!("\nDiffuse-white SDR output vs. source_peak_nits:");
    println!(
        "  {:>10}  {:>11}  {:>11}  {:>11}",
        "Peak (nits)", "Bt2390", "Bt2446A", "Bt2446C"
    );
    for &p in &peaks {
        let bt2390_out = apply_bt2390(dw, p)[0];
        let bt2446a_out = apply_bt2446a(dw, p)[0];
        let bt2446c_out = apply_bt2446c(dw, p)[0];
        println!(
            "  {:>10.0}  {:>11.4}  {:>11.4}  {:>11.4}",
            p, bt2390_out, bt2446a_out, bt2446c_out
        );
    }
}

#[test]
fn brightness_pins_at_1000_nits_source() {
    // These are the values calculated by tracing the math by hand.
    // If any of these change, the brightness behavior of the
    // post-fix reshoot pipeline shifted — investigate.
    let source_peak = 1000.0_f32;
    let dw = [1.0_f32, 1.0, 1.0];

    let bt2390_dw = apply_bt2390(dw, source_peak)[0];
    let bt2446a_dw = apply_bt2446a(dw, source_peak)[0];
    let bt2446c_dw = apply_bt2446c(dw, source_peak)[0];

    // Bt2390: hand trace yields ~0.237; allow 5 % slack for f32 rounding.
    assert!(
        (bt2390_dw - 0.237).abs() < 0.05,
        "Bt2390 diffuse-white expected ~0.237, got {bt2390_dw}"
    );
    // Bt2446A: hand trace yields ~0.393 (gamma-decoded linear); allow 5 %.
    assert!(
        (bt2446a_dw - 0.393).abs() < 0.05,
        "Bt2446A diffuse-white expected ~0.393, got {bt2446a_dw}"
    );
    // Bt2446C: hand trace yields ~0.907 (very close to 1.0 = SDR white).
    assert!(
        (bt2446c_dw - 0.907).abs() < 0.05,
        "Bt2446C diffuse-white expected ~0.907, got {bt2446c_dw}"
    );

    // Order: Bt2390 < Bt2446A < Bt2446C. Bt2446C is closest to producer
    // SDR's expected ~1.0; the others are 2-4x dimmer.
    assert!(bt2390_dw < bt2446a_dw);
    assert!(bt2446a_dw < bt2446c_dw);
}
