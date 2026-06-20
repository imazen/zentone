//! Throughput benchmark for `Bt2446A::map_strip_simd`.
//!
//! Usage: `cargo run --release --example bt2446a_throughput`
//!
//! Runs a deterministic ramp+sparkle HDR buffer through the BT.2446A
//! SIMD strip kernel at five sizes (256², 1024², 2048², 3840×2160, 7680×4320)
//! and reports Mpix/s per size. Used to measure the impact of the
//! coefficient-hoist + scalar fast-path optimization in commit
//! `perf(bt2446a): SIMD + branch-free EETF for HdrToSdr default`.
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

fn bench_cell(label: &str, width: usize, height: usize, target_secs: f32) {
    let tm = Bt2446A::new(1000.0, 100.0);
    let mut buf = synth_buffer(width, height);
    let mpix = (width as f64 * height as f64) / 1.0e6;

    // Warmup 3× to stabilize cache + branch predictor.
    for _ in 0..3 {
        tm.map_strip_simd(&mut buf);
    }

    // Calibrate iters from a 1-iter timing.
    let t0 = Instant::now();
    tm.map_strip_simd(&mut buf);
    let one_iter = t0.elapsed().as_secs_f32().max(1e-9);
    let iters = (target_secs / one_iter).clamp(1.0, 1_000.0) as u32;

    let t1 = Instant::now();
    for _ in 0..iters {
        tm.map_strip_simd(&mut buf);
    }
    let elapsed = t1.elapsed().as_secs_f32();
    let per_iter = elapsed / iters as f32;
    let throughput = (mpix as f32) / per_iter;

    // Read a couple of pixels back so the optimizer can't eliminate the work.
    let sink = buf[0][0] + buf[buf.len() / 2][1] + buf[buf.len() - 1][2];
    println!(
        "{:14}  {:5.2} MP  iters={:4}  per-iter={:>8.3} ms  {:7.1} Mpix/s  sink={:.6}",
        label,
        mpix,
        iters,
        per_iter * 1000.0,
        throughput,
        sink
    );
}

fn main() {
    println!("Bt2446A::map_strip_simd throughput");
    println!("HDR peak: 1000 nits → SDR peak: 100 nits");
    println!("Synthetic ramp+sparkle buffer, 3× warmup, ~1s per cell");
    println!();
    println!(
        "{:14}  {:>7}  {:>10}  {:>13}  {:>11}",
        "size", "pixels", "iters", "per-iter", "throughput"
    );

    bench_cell("256x256", 256, 256, 1.0);
    bench_cell("1024x1024", 1024, 1024, 1.0);
    bench_cell("2048x2048", 2048, 2048, 1.0);
    bench_cell("3840x2160", 3840, 2160, 1.0);
    bench_cell("7680x4320", 7680, 4320, 1.0);
}
