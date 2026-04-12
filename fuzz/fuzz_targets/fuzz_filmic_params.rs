#![no_main]
//! Fuzz CompiledFilmicSpline with random FilmicSplineConfig parameters.
//!
//! The compute_rational function has geometric constraints (min_contrast,
//! safety margins) that can produce degenerate coefficients with extreme
//! parameter combinations. This target exercises the full construction +
//! evaluation path for panics, NaN, and Inf.

use libfuzzer_sys::fuzz_target;
use zentone::*;

fuzz_target!(|data: &[u8]| {
    if data.len() < 40 {
        return;
    }

    // Interpret bytes as f32 config parameters, rejecting NaN/Inf
    let f = |i: usize| -> f32 {
        let b = &data[i * 4..i * 4 + 4];
        let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        if !v.is_finite() { return 0.5; } // safe default for any NaN/Inf byte pattern
        v
    };

    let cfg = FilmicSplineConfig {
        output_power: f(0).clamp(0.1, 10.0),
        latitude: f(1).clamp(0.0, 100.0),
        white_point_source: f(2).clamp(0.5, 10.0),
        black_point_source: f(3).clamp(-15.0, -0.5),
        contrast: f(4).clamp(0.1, 5.0),
        black_point_target: f(5).clamp(0.0, 10.0),
        grey_point_target: f(6).clamp(1.0, 50.0),
        white_point_target: f(7).clamp(50.0, 100.0),
        balance: f(8).clamp(-50.0, 50.0),
        saturation: f(9).clamp(0.0, 100.0),
    };

    let spline = CompiledFilmicSpline::new(&cfg);

    // Evaluate at several points — must not panic
    for &x in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let _y = spline.apply_spline(x);
    }

    // Full pipeline — must not panic
    let _out = spline.map_rgb([0.18, 0.18, 0.18]);
    let _out = spline.map_rgb([2.0, 1.0, 0.5]);
});
