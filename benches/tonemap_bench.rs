//! Paired benchmarks for every tone mapping curve in zentone.
//!
//! Each benchmark processes a 4K-width row (3840 pixels) through `map_row`
//! to measure throughput under realistic conditions. Both 3-channel (RGB)
//! and 4-channel (RGBA) variants are tested.
//!
//! Run:
//! ```
//! cargo bench -- --nocapture
//! ```

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

use zentone::*;

// ---------------------------------------------------------------------------
// Synthetic row generation
// ---------------------------------------------------------------------------

const WIDTH: usize = 3840;

fn synth_row_rgb() -> Vec<f32> {
    let mut row = Vec::with_capacity(WIDTH * 3);
    for i in 0..WIDTH {
        let t = i as f32 / WIDTH as f32;
        // Mild HDR: linear ramp 0..4 with some channel variation
        row.push(t * 4.0);
        row.push((1.0 - t) * 3.5);
        row.push(t * t * 2.0);
    }
    row
}

fn synth_row_rgba() -> Vec<f32> {
    let mut row = Vec::with_capacity(WIDTH * 4);
    for i in 0..WIDTH {
        let t = i as f32 / WIDTH as f32;
        row.push(t * 4.0);
        row.push((1.0 - t) * 3.5);
        row.push(t * t * 2.0);
        row.push(1.0); // alpha
    }
    row
}

// ---------------------------------------------------------------------------
// Curve benchmarks
// ---------------------------------------------------------------------------

fn bench_curve(c: &mut Criterion, name: &str, curve: &dyn ToneMap) {
    let src_rgb = synth_row_rgb();
    let src_rgba = synth_row_rgba();

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(WIDTH as u64));

    group.bench_function("rgb", |b| {
        let mut row = src_rgb.clone();
        b.iter(|| {
            // Reset the first pixel so the compiler can't elide the loop
            row[0] = src_rgb[0];
            row[1] = src_rgb[1];
            row[2] = src_rgb[2];
            curve.map_row(black_box(&mut row), 3);
            black_box(&row);
        });
    });

    group.bench_function("rgba", |b| {
        let mut row = src_rgba.clone();
        b.iter(|| {
            row[0] = src_rgba[0];
            row[1] = src_rgba[1];
            row[2] = src_rgba[2];
            curve.map_row(black_box(&mut row), 4);
            black_box(&row);
        });
    });

    group.finish();
}

fn curves_benchmark(c: &mut Criterion) {
    // Stateless curves via ToneMapCurve
    let reinhard = ToneMapCurve::Reinhard;
    let ext_reinhard = ToneMapCurve::ExtendedReinhard {
        l_max: 4.0,
        luma: LUMA_BT709,
    };
    let jodie = ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 };
    let tuned = ToneMapCurve::TunedReinhard {
        content_max_nits: 4000.0,
        display_max_nits: 250.0,
        luma: LUMA_BT709,
    };
    let narkowicz = ToneMapCurve::Narkowicz;
    let uncharted2 = ToneMapCurve::Uncharted2;
    let aces = ToneMapCurve::AcesAp1;
    let bt2390 = ToneMapCurve::Bt2390 {
        source_peak: 4.0,
        target_peak: 1.0,
    };
    let agx_default = ToneMapCurve::Agx(AgxLook::Default);
    let agx_punchy = ToneMapCurve::Agx(AgxLook::Punchy);
    let agx_golden = ToneMapCurve::Agx(AgxLook::Golden);
    let clamp = ToneMapCurve::Clamp;

    bench_curve(c, "reinhard", &reinhard);
    bench_curve(c, "ext_reinhard", &ext_reinhard);
    bench_curve(c, "reinhard_jodie", &jodie);
    bench_curve(c, "tuned_reinhard", &tuned);
    bench_curve(c, "narkowicz", &narkowicz);
    bench_curve(c, "uncharted2", &uncharted2);
    bench_curve(c, "aces_ap1", &aces);
    bench_curve(c, "bt2390", &bt2390);
    bench_curve(c, "agx_default", &agx_default);
    bench_curve(c, "agx_punchy", &agx_punchy);
    bench_curve(c, "agx_golden", &agx_golden);
    bench_curve(c, "clamp", &clamp);

    // Stateful curves
    let bt2408 = Bt2408Tonemapper::new(4000.0, 1000.0);
    bench_curve(c, "bt2408", &bt2408);

    let filmic = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
    bench_curve(c, "filmic_spline", &filmic);
}

// ---------------------------------------------------------------------------
// map_into benchmark (copy path)
// ---------------------------------------------------------------------------

fn map_into_benchmark(c: &mut Criterion) {
    let src = synth_row_rgba();
    let mut dst = vec![0.0f32; src.len()];
    let curve = ToneMapCurve::AcesAp1;

    let mut group = c.benchmark_group("map_into");
    group.throughput(Throughput::Elements(WIDTH as u64));

    group.bench_function("aces_rgba", |b| {
        b.iter(|| {
            curve.map_into(black_box(&src), black_box(&mut dst), 4);
            black_box(&dst);
        });
    });

    group.finish();
}

criterion_group!(benches, curves_benchmark, map_into_benchmark);
criterion_main!(benches);
