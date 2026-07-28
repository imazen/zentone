//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! `tonemap_bench.rs` reports per-curve throughput, which shows which curves are
//! cheap but not whether the SIMD dispatch is earning its keep — a curve slower
//! than its own scalar fallback is invisible there. This bench runs the
//! identical `map_row` calls with the native SIMD token disabled. (The same gap
//! in linear-srgb was hiding a real regression.)
//!
//! Curves are chosen to span the cost range: a cheap rational (Reinhard), a
//! polynomial (Narkowicz), a matrix+curve (ACES AP1), and a log-space one with
//! transcendentals (AgX). If SIMD only pays off on some of those, this shows
//! which.
//!
//! Run: `cargo bench --bench tier_isolation --features _dev`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

use zentone::*;

const WIDTH: usize = 3840;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn synth_row_rgb() -> Vec<f32> {
    let mut row = Vec::with_capacity(WIDTH * 3);
    for i in 0..WIDTH {
        let t = i as f32 / WIDTH as f32;
        row.push(t * 4.0);
        row.push((1.0 - t) * 3.5);
        row.push(t * t * 2.0);
    }
    row
}

fn bench_one(c: &mut Criterion, name: &str, curve: &dyn ToneMap) {
    let src = synth_row_rgb();
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(WIDTH as u64));
    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
        group.bench_function(arm, |b| {
            set_simd(simd);
            let mut row = src.clone();
            b.iter(|| {
                // Reset the first pixel so the compiler can't elide the loop.
                row[0] = src[0];
                row[1] = src[1];
                row[2] = src[2];
                curve.map_row(black_box(&mut row), 3);
                black_box(&row);
            });
        });
    }
    set_simd(true);
    group.finish();
}

fn bench_tiers(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, build with --features _dev). \
             Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    let reinhard = ToneMapCurve::Reinhard;
    let narkowicz = ToneMapCurve::Narkowicz;
    let hable = ToneMapCurve::HableFilmic;
    let aces = ToneMapCurve::AcesAp1;
    let bt2390 = ToneMapCurve::Bt2390 {
        source_peak: 4.0,
        target_peak: 1.0,
    };
    let agx = ToneMapCurve::Agx(AgxLook::Default);

    bench_one(c, "reinhard", &reinhard);
    bench_one(c, "narkowicz", &narkowicz);
    bench_one(c, "hable_filmic", &hable);
    bench_one(c, "aces_ap1", &aces);
    bench_one(c, "bt2390", &bt2390);
    bench_one(c, "agx_default", &agx);

    set_simd(true);
}

criterion_group!(benches, bench_tiers);
criterion_main!(benches);
