//! Scalar-vs-SIMD benchmarks for PR1-PR4 kernels.
//!
//! Three sections:
//!
//! 1. **Curve `map_strip_simd` overrides** (PR4) — compares the per-pixel
//!    `for px { *px = self.map_rgb(*px); }` default against each curve's
//!    SIMD override, both routed through `pipeline::tonemap_pq_row_simd` so
//!    only the curve step differs.
//! 2. **Building blocks** (PR2) — `apply_matrix_row_simd`, `soft_clip_row_simd`,
//!    `hlg_ootf_row_simd`, `hlg_ootf_approx_row_simd` each vs a per-pixel
//!    reference loop.
//! 3. **PR1 curve regression gate** — checks `simd::*_row` curve kernels
//!    (Reinhard simple, Hable, Narkowicz, ACES AP1, AgX) didn't regress
//!    after the `#[magetypes]` rewrite. Compared against per-pixel scalar.
//!
//! Run: `cargo bench --bench pipeline_bench --features experimental`

use zenbench::prelude::*;

use zentone::gamut::{
    BT2020_TO_BT709, apply_matrix, apply_matrix_row_simd, soft_clip, soft_clip_row_simd,
};
use zentone::hlg::{
    hlg_ootf, hlg_ootf_approx, hlg_ootf_approx_row_simd, hlg_ootf_row_simd, hlg_system_gamma,
};
use zentone::pipeline::{tonemap_pq_row_simd, tonemap_pq_to_srgb8_row_simd};
use zentone::{
    AgxLook, Bt2408Tonemapper, Bt2446A, Bt2446B, Bt2446C, CompiledFilmicSpline, ToneMap,
    ToneMapCurve, TonemapScratch,
};

// ---------------------------------------------------------------------------
// Strip widths used across benches. 256 = cache-warm, 1024 = mid, 4096 =
// pulls from L2.
// ---------------------------------------------------------------------------

const WIDTHS: [usize; 3] = [256, 1024, 4096];

// ---------------------------------------------------------------------------
// Synthetic input generation. Produces a deterministic pseudo-HDR ramp; the
// same generator is used for every bench so cache state and branch profiles
// are comparable.
// ---------------------------------------------------------------------------

/// PQ-encoded ramp roughly covering [black, ~10× SDR diffuse white].
fn synth_pq_strip(width: usize) -> Vec<[f32; 3]> {
    (0..width)
        .map(|i| {
            let t = i as f32 / width as f32;
            // PQ values typically [0, 1]; spread across the curve.
            [0.05 + t * 0.85, 0.1 + (1.0 - t) * 0.7, 0.2 + t * t * 0.6]
        })
        .collect()
}

/// Linear HDR strip with mild over-1 highlights — what the curve sees after
/// PQ EOTF on real content.
fn synth_linear_hdr_strip(width: usize) -> Vec<[f32; 3]> {
    (0..width)
        .map(|i| {
            let t = i as f32 / width as f32;
            [t * 4.0, (1.0 - t) * 3.5, t * t * 2.0]
        })
        .collect()
}

/// Linear strip with ~30% pixels out-of-gamut (one channel > 1.0). Used for
/// soft_clip.
fn synth_oog_strip(width: usize) -> Vec<[f32; 3]> {
    (0..width)
        .map(|i| {
            let t = i as f32 / width as f32;
            // Every third pixel pushes red out of gamut.
            if i % 3 == 0 {
                [1.5 + t, 0.7 - t * 0.3, 0.4]
            } else {
                [0.6 * t, 0.5, 0.3 + t * 0.2]
            }
        })
        .collect()
}

/// HLG-decoded scene-linear strip (post-EOTF, pre-OOTF).
fn synth_hlg_scene_strip(width: usize) -> Vec<[f32; 3]> {
    (0..width)
        .map(|i| {
            let t = i as f32 / width as f32;
            // Range plausible for HLG-decoded scene-linear values.
            [0.05 + t * 0.9, 0.1 + (1.0 - t) * 0.85, 0.15 + t * t * 0.75]
        })
        .collect()
}

/// Flat row buffer (interleaved f32) for `simd::*_row` curve kernels.
fn synth_linear_flat(width: usize) -> Vec<f32> {
    let mut row = Vec::with_capacity(width * 3);
    for i in 0..width {
        let t = i as f32 / width as f32;
        row.push(t * 4.0);
        row.push((1.0 - t) * 3.5);
        row.push(t * t * 2.0);
    }
    row
}

// ---------------------------------------------------------------------------
// Bench 1 — Curve `map_strip_simd` overrides.
//
// `ScalarFallback` re-exposes only `map_rgb`, so the trait's default per-pixel
// `map_strip_simd` runs. `tonemap_pq_row_simd(&strip, &mut out, &fallback)`
// uses the fused outer kernel + scalar curve; passing the curve directly uses
// fused outer + SIMD curve. Apples-to-apples curve-only delta.
// ---------------------------------------------------------------------------

struct ScalarFallback<T: ToneMap>(T);
impl<T: ToneMap> ToneMap for ScalarFallback<T> {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.0.map_rgb(rgb)
    }
    // Don't override map_strip_simd — get the trait's default per-pixel loop.
}

fn bench_curve_map_strip_pair<T, F>(g: &mut BenchGroup, name: &str, mk: F, width: usize)
where
    T: ToneMap + Send + Sync + 'static,
    F: Fn() -> T,
{
    let pq_a = synth_pq_strip(width);
    let pq_b = pq_a.clone();
    let scalar = ScalarFallback(mk());
    let mut out_a = vec![[0.0_f32; 3]; width];
    let mut scratch_a = TonemapScratch::new();
    g.bench(format!("{name}_w{width}_scalar"), move |b| {
        b.iter(|| {
            tonemap_pq_row_simd(
                &mut scratch_a,
                black_box(&pq_a),
                black_box(&mut out_a),
                &scalar,
            );
            black_box(&out_a);
        });
    });

    let curve = mk();
    let mut out_b = vec![[0.0_f32; 3]; width];
    let mut scratch_b = TonemapScratch::new();
    g.bench(format!("{name}_w{width}_simd"), move |b| {
        b.iter(|| {
            tonemap_pq_row_simd(
                &mut scratch_b,
                black_box(&pq_b),
                black_box(&mut out_b),
                &curve,
            );
            black_box(&out_b);
        });
    });
}

fn bench_pr4_curves(suite: &mut Suite) {
    suite.group("pr4_bt2408", |g| {
        for &w in &WIDTHS {
            g.throughput(Throughput::Elements(w as u64));
            bench_curve_map_strip_pair(g, "bt2408", || Bt2408Tonemapper::new(4000.0, 1000.0), w);
        }
    });
    suite.group("pr4_bt2446a", |g| {
        for &w in &WIDTHS {
            g.throughput(Throughput::Elements(w as u64));
            bench_curve_map_strip_pair(g, "bt2446a", || Bt2446A::new(1000.0, 203.0), w);
        }
    });
    suite.group("pr4_bt2446b", |g| {
        for &w in &WIDTHS {
            g.throughput(Throughput::Elements(w as u64));
            bench_curve_map_strip_pair(g, "bt2446b", || Bt2446B::new(1000.0, 203.0), w);
        }
    });
    suite.group("pr4_bt2446c", |g| {
        for &w in &WIDTHS {
            g.throughput(Throughput::Elements(w as u64));
            bench_curve_map_strip_pair(g, "bt2446c", || Bt2446C::new(1000.0, 203.0), w);
        }
    });
    suite.group("pr4_filmic", |g| {
        for &w in &WIDTHS {
            g.throughput(Throughput::Elements(w as u64));
            bench_curve_map_strip_pair(
                g,
                "filmic",
                || CompiledFilmicSpline::for_hdr_peak(1000.0),
                w,
            );
        }
    });
}

// End-to-end pipeline gate: PQ-encoded BT.2020 → sRGB u8 with the BT.2408
// curve (one width). This is the closest the bench gets to a "real" client
// call: PQ EOTF + SIMD tone curve + gamut + soft clip + sRGB OETF + u8
// quantize. Scalar comparison routes the same fused kernel through
// `ScalarFallback` to isolate the curve's contribution to a fully-fused
// pipeline.
fn bench_e2e_pq_to_srgb8(suite: &mut Suite) {
    suite.group("e2e_pq_to_srgb8", |g| {
        let w = 1024;
        g.throughput(Throughput::Elements(w as u64));

        let pq = synth_pq_strip(w);
        let scalar = ScalarFallback(Bt2408Tonemapper::new(4000.0, 1000.0));
        let curve = Bt2408Tonemapper::new(4000.0, 1000.0);

        let pq_a = pq.clone();
        let mut out_a = vec![[0u8; 3]; w];
        let mut scratch_a = TonemapScratch::new();
        g.bench("pq_to_srgb8_w1024_scalar", move |b| {
            b.iter(|| {
                tonemap_pq_to_srgb8_row_simd(
                    &mut scratch_a,
                    black_box(&pq_a),
                    black_box(&mut out_a),
                    &scalar,
                );
                black_box(&out_a);
            });
        });

        let pq_b = pq;
        let mut out_b = vec![[0u8; 3]; w];
        let mut scratch_b = TonemapScratch::new();
        g.bench("pq_to_srgb8_w1024_simd", move |b| {
            b.iter(|| {
                tonemap_pq_to_srgb8_row_simd(
                    &mut scratch_b,
                    black_box(&pq_b),
                    black_box(&mut out_b),
                    &curve,
                );
                black_box(&out_b);
            });
        });
    });
}

// ---------------------------------------------------------------------------
// Bench 2 — Building blocks.
//
// Direct calls into the public `*_row_simd` SIMD entry points vs scalar
// reference loops over the per-pixel public functions.
// ---------------------------------------------------------------------------

fn bench_apply_matrix(suite: &mut Suite) {
    suite.group("blocks_apply_matrix", |g| {
        let w = 1024;
        g.throughput(Throughput::Elements(w as u64));

        let src = synth_linear_hdr_strip(w);

        let src_s = src.clone();
        g.bench("apply_matrix_w1024_scalar", move |b| {
            let src_inner = src_s.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                for px in row.iter_mut() {
                    *px = apply_matrix(&BT2020_TO_BT709, *px);
                }
                row
            });
        });

        let src_v = src;
        g.bench("apply_matrix_w1024_simd", move |b| {
            let src_inner = src_v.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                apply_matrix_row_simd(&BT2020_TO_BT709, &mut row);
                row
            });
        });
    });
}

fn bench_soft_clip(suite: &mut Suite) {
    suite.group("blocks_soft_clip", |g| {
        let w = 1024;
        g.throughput(Throughput::Elements(w as u64));

        let src = synth_oog_strip(w);

        let src_s = src.clone();
        g.bench("soft_clip_w1024_scalar", move |b| {
            let src_inner = src_s.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                for px in row.iter_mut() {
                    *px = soft_clip(*px);
                }
                row
            });
        });

        let src_v = src;
        g.bench("soft_clip_w1024_simd", move |b| {
            let src_inner = src_v.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                soft_clip_row_simd(&mut row);
                row
            });
        });
    });
}

fn bench_hlg_ootf_exact(suite: &mut Suite) {
    suite.group("blocks_hlg_ootf_exact", |g| {
        let w = 1024;
        g.throughput(Throughput::Elements(w as u64));

        let gamma = hlg_system_gamma(1000.0); // 1.2 system gamma
        let src = synth_hlg_scene_strip(w);

        let src_s = src.clone();
        g.bench("hlg_ootf_w1024_scalar", move |b| {
            let src_inner = src_s.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                for px in row.iter_mut() {
                    *px = hlg_ootf(*px, gamma);
                }
                row
            });
        });

        let src_v = src;
        g.bench("hlg_ootf_w1024_simd", move |b| {
            let src_inner = src_v.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                hlg_ootf_row_simd(&mut row, gamma);
                row
            });
        });
    });
}

fn bench_hlg_ootf_approx(suite: &mut Suite) {
    suite.group("blocks_hlg_ootf_approx", |g| {
        let w = 1024;
        g.throughput(Throughput::Elements(w as u64));

        let gamma = hlg_system_gamma(1000.0);
        let src = synth_hlg_scene_strip(w);

        let src_s = src.clone();
        g.bench("hlg_ootf_approx_w1024_scalar", move |b| {
            let src_inner = src_s.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                for px in row.iter_mut() {
                    *px = hlg_ootf_approx(*px, gamma);
                }
                row
            });
        });

        let src_v = src;
        g.bench("hlg_ootf_approx_w1024_simd", move |b| {
            let src_inner = src_v.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                hlg_ootf_approx_row_simd(&mut row, gamma);
                row
            });
        });
    });
}

// ---------------------------------------------------------------------------
// Bench 3 — PR1 curve regression gate.
//
// `simd::reinhard_simple_row` etc. are `pub(crate)`. We exercise them via the
// public `ToneMapCurve::map_row` dispatch, which routes per-channel curves
// (Reinhard, Narkowicz, Hable, ACES AP1, AgX) into the SIMD path. The scalar
// reference is a manual per-pixel loop calling the per-pixel scalar curves
// from `zentone::curves`.
// ---------------------------------------------------------------------------

fn pr1_scalar_loop(name: &str, row: &mut [f32]) {
    match name {
        "reinhard_simple" => {
            for v in row.iter_mut() {
                *v = zentone::curves::reinhard_simple(*v);
            }
        }
        "narkowicz" => {
            for v in row.iter_mut() {
                *v = zentone::curves::filmic_narkowicz(*v);
            }
        }
        "hable" => {
            for v in row.iter_mut() {
                *v = zentone::curves::hable_filmic(*v);
            }
        }
        "aces_ap1" => {
            for c in row.chunks_exact_mut(3) {
                let o = zentone::curves::aces_ap1([c[0], c[1], c[2]]);
                c[0] = o[0];
                c[1] = o[1];
                c[2] = o[2];
            }
        }
        "agx" => {
            for c in row.chunks_exact_mut(3) {
                let o = zentone::curves::agx_tonemap([c[0], c[1], c[2]], AgxLook::Default);
                c[0] = o[0];
                c[1] = o[1];
                c[2] = o[2];
            }
        }
        _ => unreachable!(),
    }
}

fn bench_pr1_curve(suite: &mut Suite, group: &str, name: &str, curve: ToneMapCurve) {
    suite.group(group, |g| {
        let w = 1024;
        g.throughput(Throughput::Elements(w as u64));

        let src = synth_linear_flat(w);
        let scalar_name = format!("{name}_w1024_scalar");
        let simd_name = format!("{name}_w1024_simd");

        let src_s = src.clone();
        let n_scalar = String::from(name);
        g.bench(scalar_name, move |b| {
            let src_inner = src_s.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                pr1_scalar_loop(&n_scalar, &mut row);
                row
            });
        });

        let src_v = src;
        g.bench(simd_name, move |b| {
            let src_inner = src_v.clone();
            b.with_input(move || src_inner.clone()).run(|mut row| {
                curve.map_row(&mut row, 3);
                row
            });
        });
    });
}

fn bench_pr1_curves(suite: &mut Suite) {
    bench_pr1_curve(
        suite,
        "pr1_reinhard_simple",
        "reinhard_simple",
        ToneMapCurve::Reinhard,
    );
    bench_pr1_curve(suite, "pr1_narkowicz", "narkowicz", ToneMapCurve::Narkowicz);
    bench_pr1_curve(suite, "pr1_hable", "hable", ToneMapCurve::HableFilmic);
    bench_pr1_curve(suite, "pr1_aces_ap1", "aces_ap1", ToneMapCurve::AcesAp1);
    bench_pr1_curve(suite, "pr1_agx", "agx", ToneMapCurve::Agx(AgxLook::Default));
}

zenbench::main!(
    bench_pr4_curves,
    bench_e2e_pq_to_srgb8,
    bench_apply_matrix,
    bench_soft_clip,
    bench_hlg_ootf_exact,
    bench_hlg_ootf_approx,
    bench_pr1_curves
);
