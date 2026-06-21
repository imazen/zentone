//! Throughput benchmark for `Bt2446A::map_strip_simd` and
//! `Bt2446A::map_strip_simd_for_u8`.
//!
//! Usage: `cargo run --release --example bt2446a_throughput`
//!
//! Runs a deterministic ramp+sparkle HDR buffer through both strip
//! kernels at five sizes (256², 1024², 2048², 3840×2160, 7680×4320)
//! and reports Mpix/s per size for each path, plus their speedup ratio.
//!
//! - `map_strip_simd` is the spec-strict path (kernel output error 5.88e-5,
//!   uses `pow_midp_unchecked(1/2.4)` for the input gamma encode).
//! - `map_strip_simd_for_u8` is the 8-bit-display-targeted fast path
//!   (kernel output error 5.21e-4, uses a 2-piece sqrt-substituted
//!   polynomial for the input gamma encode — sub-half-LSB at 8-bit).
//!
//! The deterministic seed ensures rerun stability; each cell is warmed
//! up 3× and then run for ~1 second of wall time. The output table is
//! checked into `benchmarks/bt2446a_throughput_<DATE>.md` for diff-able
//! provenance.

use std::time::Instant;
use zentone::{Bt2446A, ToneMap};

/// Build a synthetic HDR buffer with a deterministic ramp + sparkle so the
/// kernel sees both smooth gradients and high-frequency content. Matches the
/// pattern used by `measure_max_rgb_tier`'s bench in zenpixels-convert.
fn synth_buffer(width: usize, height: usize) -> Vec<[f32; 3]> {
    let n = width * height;
    let mut buf = Vec::with_capacity(n);
    // Ramp 0.0..=4.0 (typical HDR overrange for 1000-nit content) with a
    // pseudo-random sparkle modulation that exercises the tone-curve branches.
    let inv_n = 1.0 / (n as f32);
    for i in 0..n {
        let t = (i as f32) * inv_n;
        // LCG-style sparkle, deterministic across runs.
        let s = (((i * 2654435761) & 0xFFFF) as f32) / 65535.0;
        let r = (t * 4.0) * (0.7 + 0.3 * s);
        let g = ((1.0 - t) * 3.0) * (0.7 + 0.3 * s);
        let b = (t * (1.0 - t) * 2.0 + 0.05) * (0.6 + 0.4 * (1.0 - s));
        buf.push([r, g, b]);
    }
    buf
}

/// Per-path throughput measurement. `kind`: "strict" → `map_strip_simd`,
/// "u8" → `map_strip_simd_for_u8`.
///
/// Identical structure to the prior bench: warmup 3×, calibrate iters
/// from a 1-iter timing, then run `iters` rounds, all on the same buffer.
/// After the first iter the buffer is filled with tone-mapped values, so
/// the kernel processes already-mapped data on subsequent passes — that
/// distribution is the same for both paths, so the relative comparison
/// is fair.
fn bench_one_path(
    tm: &Bt2446A,
    width: usize,
    height: usize,
    target_secs: f32,
    kind: &str,
) -> (f32, u32, f32) {
    let mut buf = synth_buffer(width, height);

    let run = |buf: &mut Vec<[f32; 3]>| match kind {
        "strict" => tm.map_strip_simd(buf),
        "u8" => tm.map_strip_simd_for_u8(buf),
        _ => unreachable!(),
    };

    for _ in 0..3 {
        run(&mut buf);
    }

    let t0 = Instant::now();
    run(&mut buf);
    let one_iter = t0.elapsed().as_secs_f32().max(1e-9);
    let iters = (target_secs / one_iter).clamp(1.0, 1_000.0) as u32;

    let t1 = Instant::now();
    for _ in 0..iters {
        run(&mut buf);
    }
    let elapsed = t1.elapsed().as_secs_f32();
    let per_iter = elapsed / iters as f32;
    let mpix = (buf.len() as f64) / 1.0e6;
    let throughput = (mpix as f32) / per_iter;

    let sink = buf[0][0] + buf[buf.len() / 2][1] + buf[buf.len() - 1][2];
    let _ = sink;

    (per_iter, iters, throughput)
}

fn bench_cell(label: &str, width: usize, height: usize, target_secs: f32) {
    let tm = Bt2446A::new(1000.0, 100.0);
    let mpix = (width as f64 * height as f64) / 1.0e6;

    let (s_per, s_iters, s_thru) = bench_one_path(&tm, width, height, target_secs, "strict");
    let (u_per, u_iters, u_thru) = bench_one_path(&tm, width, height, target_secs, "u8");
    let speedup = u_thru / s_thru;

    println!(
        "{:14}  {:5.2} MP  strict: iters={:4} per-iter={:>7.3} ms {:7.1} Mpix/s  | \
         u8: iters={:4} per-iter={:>7.3} ms {:7.1} Mpix/s  | speedup={:.2}x",
        label,
        mpix,
        s_iters,
        s_per * 1000.0,
        s_thru,
        u_iters,
        u_per * 1000.0,
        u_thru,
        speedup,
    );
}

fn main() {
    println!("Bt2446A strip throughput: spec-strict vs 8-bit-display fast path");
    println!("HDR peak: 1000 nits → SDR peak: 100 nits");
    println!("Synthetic ramp+sparkle buffer, 3× warmup, ~1s per cell, both paths");
    println!();

    bench_cell("256x256", 256, 256, 1.0);
    bench_cell("1024x1024", 1024, 1024, 1.0);
    bench_cell("2048x2048", 2048, 2048, 1.0);
    bench_cell("3840x2160", 3840, 2160, 1.0);
    bench_cell("7680x4320", 7680, 4320, 1.0);
}
