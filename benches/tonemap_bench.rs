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
    let hable_filmic = ToneMapCurve::HableFilmic;
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
    bench_curve(c, "hable_filmic", &hable_filmic);
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

// ---------------------------------------------------------------------------
// Gainforge comparison: full pipeline (sRGB u8 → tonemap → sRGB u8)
// ---------------------------------------------------------------------------

fn gainforge_comparison(c: &mut Criterion) {
    use ::moxcms::ColorProfile;
    use gainforge::*;

    // Synthetic sRGB u8 row — bright HDR-ish values encoded as sRGB
    let mut src_u8 = vec![0u8; WIDTH * 3];
    for i in 0..WIDTH {
        let t = i as f32 / WIDTH as f32;
        src_u8[i * 3] = (t * 255.0) as u8;
        src_u8[i * 3 + 1] = ((1.0 - t) * 220.0) as u8;
        src_u8[i * 3 + 2] = (t * t * 180.0) as u8;
    }
    let mut dst_u8 = vec![0u8; src_u8.len()];

    let bt2020_pq = ColorProfile::new_bt2020_pq();
    let srgb = ColorProfile::new_srgb();

    let methods: &[(&str, ToneMappingMethod)] = &[
        ("reinhard", ToneMappingMethod::Reinhard),
        ("hable", ToneMappingMethod::Filmic),
        ("aces", ToneMappingMethod::Aces),
        (
            "agx_default",
            ToneMappingMethod::Agx(gainforge::AgxLook::Agx),
        ),
    ];

    let mut group = c.benchmark_group("gainforge");
    group.throughput(Throughput::Elements(WIDTH as u64));

    for (name, method) in methods {
        let mapper = create_tone_mapper_rgb(
            &bt2020_pq,
            &srgb,
            method.clone(),
            MappingColorSpace::Rgb(RgbToneMapperParameters {
                exposure: 1.0,
                gamut_clipping: GamutClipping::NoClip,
            }),
        )
        .unwrap();

        group.bench_function(*name, |b| {
            b.iter(|| {
                mapper
                    .tonemap_lane(black_box(&src_u8), black_box(&mut dst_u8))
                    .unwrap();
                black_box(&dst_u8);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// zentone full pipeline: sRGB decode → tonemap → sRGB encode (u8 → u8)
// ---------------------------------------------------------------------------

fn zentone_full_pipeline(c: &mut Criterion) {
    use linear_srgb::default::{linear_to_srgb_slice, srgb_to_linear_slice};

    // Same u8 source as gainforge bench, pre-converted to f32 [0,1]
    let src_f32: Vec<f32> = (0..WIDTH)
        .flat_map(|i| {
            let t = i as f32 / WIDTH as f32;
            [t, 1.0 - t * 0.86, t * t * 0.71]
        })
        .collect();
    let mut linear = vec![0.0_f32; src_f32.len()];
    let mut dst_u8 = vec![0u8; src_f32.len()];

    let curves: &[(&str, ToneMapCurve)] = &[
        ("reinhard", ToneMapCurve::Reinhard),
        ("hable", ToneMapCurve::HableFilmic),
        ("aces", ToneMapCurve::AcesAp1),
        ("agx_default", ToneMapCurve::Agx(AgxLook::Default)),
    ];

    let mut group = c.benchmark_group("zentone_full");
    group.throughput(Throughput::Elements(WIDTH as u64));

    for (name, curve) in curves {
        group.bench_function(*name, |b| {
            b.iter(|| {
                // sRGB f32 → linear f32 (SIMD batch)
                linear.copy_from_slice(&src_f32);
                srgb_to_linear_slice(black_box(&mut linear));
                // Tonemap in linear space (SIMD)
                curve.map_row(black_box(&mut linear), 3);
                // Linear f32 → sRGB f32 (SIMD batch)
                linear_to_srgb_slice(black_box(&mut linear));
                // Quantize to u8
                for (dst, &src) in dst_u8.iter_mut().zip(linear.iter()) {
                    *dst = (src * 255.0 + 0.5).min(255.0) as u8;
                }
                black_box(&dst_u8);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    curves_benchmark,
    gainforge_comparison,
    zentone_full_pipeline,
    map_into_benchmark
);
criterion_main!(benches);
