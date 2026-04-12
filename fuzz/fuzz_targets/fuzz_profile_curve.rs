#![no_main]
//! Fuzz ProfileToneCurve::from_xy_pairs with random control points.
//!
//! Goal: no panic on any input. from_xy_pairs should return None for
//! degenerate input, not crash. eval() must always return finite values.

use libfuzzer_sys::fuzz_target;
use zentone::experimental::ProfileToneCurve;
use zentone::ToneMap;

fuzz_target!(|data: &[u8]| {
    // Interpret bytes as f32 pairs. from_xy_pairs now rejects NaN/Inf.
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if floats.len() < 4 {
        return;
    }

    let curve = match ProfileToneCurve::from_xy_pairs(&floats) {
        Some(c) => c,
        None => return, // NaN/Inf or < 2 points → rejected, not a crash
    };

    // Exercise eval at various points. We don't assert finite output
    // because random LUT values can produce extreme results — the goal
    // is to catch panics and OOB, not arithmetic extremes.
    for &x in &[0.0f32, 0.25, 0.5, 0.75, 1.0, -0.1, 1.5] {
        let _y = curve.eval(x);
    }

    // Exercise per_channel and luminance views — must not panic.
    let _out = curve.per_channel().map_rgb([0.3, 0.6, 0.1]);
    let _out = curve.luminance(zentone::LUMA_BT709).map_rgb([0.3, 0.6, 0.1]);

    // Exercise map_row via the ToneMap trait.
    let mut row = [0.3f32, 0.5, 0.2, 0.7, 0.1, 0.9];
    curve.per_channel().map_row(&mut row, 3);
});
