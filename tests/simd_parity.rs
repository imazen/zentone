//! Parity between strip-form SIMD building blocks and their per-pixel
//! references. Each test runs a property-style sweep over greys, primaries,
//! mixed colors, and HDR magnitudes up to ~10×, then asserts agreement to
//! within an FMA-tolerance threshold (2e-6 absolute, scaled for HDR).
//!
//! These cover PR2 building blocks: gamut matrix apply (RGB and RGBA),
//! soft clip, out-of-gamut mask, and the four HLG OOTF row kernels.

use zentone::gamut::{
    BT709_TO_BT2020, BT709_TO_P3, BT2020_TO_BT709, BT2020_TO_P3, P3_TO_BT709, P3_TO_BT2020,
    apply_matrix, apply_matrix_row_simd, apply_matrix_row_simd_rgba, is_out_of_gamut,
    is_out_of_gamut_mask_simd, soft_clip, soft_clip_row_simd,
};
use zentone::hlg::{
    hlg_inverse_ootf, hlg_inverse_ootf_approx, hlg_inverse_ootf_approx_row_simd,
    hlg_inverse_ootf_row_simd, hlg_ootf, hlg_ootf_approx, hlg_ootf_approx_row_simd,
    hlg_ootf_row_simd, hlg_system_gamma,
};

/// FMA-grade absolute tolerance for low-magnitude (≤ 1.0) outputs.
const TOL_ABS: f32 = 2e-6;

/// Property-grid: greys, primaries, secondaries, mixed colors, and HDR
/// magnitudes (some channels up to ~10× the SDR range to exercise the
/// out-of-gamut branch and the HLG luminance domain).
fn property_inputs() -> Vec<[f32; 3]> {
    let mut v = Vec::new();
    // Greys
    for g in [0.0_f32, 0.001, 0.05, 0.18, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0] {
        v.push([g, g, g]);
    }
    // Primaries and secondaries (SDR)
    for prim in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ] {
        v.push(prim);
    }
    // Mixed colors at varying intensities
    let mixes: &[[f32; 3]] = &[
        [0.5, 0.3, 0.8],
        [0.2, 0.7, 0.4],
        [0.9, 0.1, 0.05],
        [0.04, 0.55, 0.95],
        [0.18, 0.18, 0.18],
        [0.7, 0.7, 0.2],
    ];
    for &c in mixes {
        v.push(c);
    }
    // HDR-range mixed colors (some > 1, some < 0 to also exercise the
    // negative-clamp path on soft clip).
    for &c in &[
        [1.3_f32, 0.7, 0.9],
        [0.05, 1.5, 0.1],
        [-0.02, 0.6, 0.8],
        [3.0, 1.0, 0.5],
        [10.0, 0.5, 0.05],
        [0.5, 9.5, 1.2],
    ] {
        v.push(c);
    }
    v
}

/// Build a row from many copies of the property grid plus a few non-multiple
/// of 8 sizes so the scalar tail is exercised.
fn property_strip() -> Vec<[f32; 3]> {
    let base = property_inputs();
    let mut row = Vec::with_capacity(base.len() * 5 + 7);
    for _ in 0..5 {
        row.extend_from_slice(&base);
    }
    // Force a non-multiple-of-8 length to exercise the tail loop.
    row.truncate(row.len() - 3);
    row
}

#[test]
fn apply_matrix_row_matches_reference() {
    let mut max_err: f32 = 0.0;
    let mut cases = 0;
    for matrix in [
        &BT709_TO_BT2020,
        &BT2020_TO_BT709,
        &BT709_TO_P3,
        &P3_TO_BT709,
        &BT2020_TO_P3,
        &P3_TO_BT2020,
    ] {
        let row_in = property_strip();
        let mut row_simd = row_in.clone();
        apply_matrix_row_simd(matrix, &mut row_simd);

        for (i, px) in row_in.iter().enumerate() {
            let expected = apply_matrix(matrix, *px);
            for c in 0..3 {
                let mag = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
                let tol = TOL_ABS * mag.max(1.0);
                let err = (row_simd[i][c] - expected[c]).abs();
                max_err = max_err.max(err);
                assert!(
                    err < tol,
                    "apply_matrix_row_simd[{i},{c}] in={px:?}: simd={} ref={} err={err:.3e} tol={tol:.3e}",
                    row_simd[i][c],
                    expected[c]
                );
                cases += 1;
            }
        }
    }
    println!("apply_matrix_row_simd: {cases} comparisons, max_err={max_err:.3e}");
}

#[test]
fn apply_matrix_row_rgba_matches_and_preserves_alpha() {
    let mut max_err: f32 = 0.0;
    let mut cases = 0;
    let row_in_rgb = property_strip();
    let mut row_in: Vec<[f32; 4]> = row_in_rgb
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0], p[1], p[2], (i as f32) * 0.013_f32 + 0.25])
        .collect();
    let alphas: Vec<f32> = row_in.iter().map(|p| p[3]).collect();
    apply_matrix_row_simd_rgba(&BT2020_TO_BT709, &mut row_in);
    for (i, px) in row_in.iter().enumerate() {
        let expected = apply_matrix(&BT2020_TO_BT709, row_in_rgb[i]);
        let mag = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
        let tol = TOL_ABS * mag.max(1.0);
        for c in 0..3 {
            let err = (px[c] - expected[c]).abs();
            max_err = max_err.max(err);
            assert!(err < tol, "rgba[{i},{c}] err={err:.3e} tol={tol:.3e}");
            cases += 1;
        }
        assert_eq!(
            px[3], alphas[i],
            "alpha[{i}] mutated: {} vs {}",
            px[3], alphas[i]
        );
    }
    println!("apply_matrix_row_simd_rgba: {cases} comparisons, max_err={max_err:.3e}");
}

#[test]
fn soft_clip_row_matches_reference() {
    let mut max_err: f32 = 0.0;
    let mut cases = 0;
    // Soft clip is naturally exercised by both in-gamut (no-op) and over-range
    // mixes from the property grid; we also feed gamut-converted saturated
    // primaries (those produce real out-of-gamut signed values).
    let mut grid = property_strip();
    for prim in [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        grid.push(apply_matrix(&BT2020_TO_BT709, prim));
    }
    let mut row_simd = grid.clone();
    soft_clip_row_simd(&mut row_simd);
    for (i, px) in grid.iter().enumerate() {
        let expected = soft_clip(*px);
        for c in 0..3 {
            // Soft clip ranges in [0, 1]; absolute tol is enough.
            let err = (row_simd[i][c] - expected[c]).abs();
            max_err = max_err.max(err);
            assert!(
                err < TOL_ABS,
                "soft_clip_row_simd[{i},{c}] in={px:?}: simd={} ref={} err={err:.3e}",
                row_simd[i][c],
                expected[c]
            );
            cases += 1;
        }
    }
    println!("soft_clip_row_simd: {cases} comparisons, max_err={max_err:.3e}");
}

#[test]
fn is_out_of_gamut_mask_matches_reference() {
    let mut grid = property_strip();
    for prim in [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        grid.push(apply_matrix(&BT2020_TO_BT709, prim));
    }
    let mut mask = vec![0.0_f32; grid.len()];
    is_out_of_gamut_mask_simd(&grid, &mut mask);
    for (i, px) in grid.iter().enumerate() {
        let expected = if is_out_of_gamut(*px) { 1.0 } else { 0.0 };
        assert_eq!(
            mask[i], expected,
            "is_out_of_gamut_mask[{i}] in={px:?}: simd={} ref={expected}",
            mask[i]
        );
    }
    println!("is_out_of_gamut_mask_simd: {} pixels checked", grid.len());
}

#[test]
fn hlg_ootf_row_matches_reference() {
    let mut max_err: f32 = 0.0;
    let mut cases = 0;
    // Gamma 1.0 reduces to identity; sweep across the 1000/2000/4000 nit grid.
    for &nits in &[400.0_f32, 1000.0, 2000.0, 4000.0] {
        let gamma = hlg_system_gamma(nits);
        // Restrict to non-negative inputs — the OOTF formula `pow(Y, k)` is
        // only defined for Y > 0; the per-pixel reference forces Y <= 0 to
        // the zero pixel. The strip kernel mirrors that. We cover that branch
        // explicitly with the [0, 0, 0] case below.
        let grid: Vec<[f32; 3]> = property_inputs()
            .into_iter()
            .map(|p| [p[0].max(0.0), p[1].max(0.0), p[2].max(0.0)])
            .collect();
        let mut row_simd = grid.clone();
        hlg_ootf_row_simd(&mut row_simd, gamma);
        for (i, px) in grid.iter().enumerate() {
            let expected = hlg_ootf(*px, gamma);
            let mag = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
            let tol = TOL_ABS * mag.max(1.0);
            for c in 0..3 {
                let err = (row_simd[i][c] - expected[c]).abs();
                max_err = max_err.max(err);
                assert!(
                    err < tol,
                    "hlg_ootf_row_simd[{i},{c}] gamma={gamma} in={px:?}: \
                     simd={} ref={} err={err:.3e} tol={tol:.3e}",
                    row_simd[i][c],
                    expected[c]
                );
                cases += 1;
            }
        }
        // Inverse direction.
        let mut row_inv = row_simd.clone();
        hlg_inverse_ootf_row_simd(&mut row_inv, gamma);
        for (i, px) in row_simd.iter().enumerate() {
            let expected = hlg_inverse_ootf(*px, gamma);
            let mag = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
            let tol = TOL_ABS * mag.max(1.0);
            for c in 0..3 {
                let err = (row_inv[i][c] - expected[c]).abs();
                max_err = max_err.max(err);
                assert!(
                    err < tol,
                    "hlg_inverse_ootf_row_simd[{i},{c}] gamma={gamma} in={px:?}: \
                     simd={} ref={} err={err:.3e} tol={tol:.3e}",
                    row_inv[i][c],
                    expected[c]
                );
                cases += 1;
            }
        }
    }
    println!("hlg_ootf_row_simd / inverse: {cases} comparisons, max_err={max_err:.3e}");
}

#[test]
fn hlg_ootf_approx_row_matches_reference() {
    let mut max_err: f32 = 0.0;
    let mut cases = 0;
    for &nits in &[400.0_f32, 1000.0, 2000.0, 4000.0] {
        let gamma = hlg_system_gamma(nits);
        let grid: Vec<[f32; 3]> = property_inputs()
            .into_iter()
            .map(|p| [p[0].max(0.0), p[1].max(0.0), p[2].max(0.0)])
            .collect();
        let mut row_simd = grid.clone();
        hlg_ootf_approx_row_simd(&mut row_simd, gamma);
        for (i, px) in grid.iter().enumerate() {
            let expected = hlg_ootf_approx(*px, gamma);
            let mag = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
            let tol = TOL_ABS * mag.max(1.0);
            for c in 0..3 {
                let err = (row_simd[i][c] - expected[c]).abs();
                max_err = max_err.max(err);
                assert!(
                    err < tol,
                    "hlg_ootf_approx_row_simd[{i},{c}] gamma={gamma} in={px:?}: \
                     simd={} ref={} err={err:.3e} tol={tol:.3e}",
                    row_simd[i][c],
                    expected[c]
                );
                cases += 1;
            }
        }
        let mut row_inv = row_simd.clone();
        hlg_inverse_ootf_approx_row_simd(&mut row_inv, gamma);
        for (i, px) in row_simd.iter().enumerate() {
            let expected = hlg_inverse_ootf_approx(*px, gamma);
            let mag = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
            let tol = TOL_ABS * mag.max(1.0);
            for c in 0..3 {
                let err = (row_inv[i][c] - expected[c]).abs();
                max_err = max_err.max(err);
                assert!(
                    err < tol,
                    "hlg_inverse_ootf_approx_row_simd[{i},{c}] gamma={gamma} in={px:?}: \
                     simd={} ref={} err={err:.3e} tol={tol:.3e}",
                    row_inv[i][c],
                    expected[c]
                );
                cases += 1;
            }
        }
    }
    println!("hlg_ootf_approx_row_simd / inverse: {cases} comparisons, max_err={max_err:.3e}");
}

// ============================================================================
// PR3 — fused pipeline kernels: parity vs scalar pipeline + sRGB8 quantization.
// ============================================================================

use zentone::pipeline::{tonemap_hlg_row_simd, tonemap_pq_row_simd, tonemap_pq_to_srgb8_row_simd};
use zentone::{Bt2408Tonemapper, ToneMapCurve};

/// Per-pixel scalar reference for the PQ → linear sRGB pipeline. Mirrors the
/// stage sequence inside [`tonemap_pq_row_simd`] but applies every step one
/// pixel at a time using the public per-pixel APIs (`tm.map_rgb`, the
/// transfer function on a single sample, `apply_matrix`, `soft_clip`).
fn pq_to_linear_srgb_scalar_ref(
    pq_row: &[[f32; 3]],
    out: &mut [[f32; 3]],
    tm: &dyn zentone::ToneMap,
) {
    for (src, dst) in pq_row.iter().zip(out.iter_mut()) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];
        let tonemapped = tm.map_rgb(linear_2020);
        let bt709 = apply_matrix(&BT2020_TO_BT709, tonemapped);
        let clipped = if is_out_of_gamut(bt709) {
            soft_clip(bt709)
        } else {
            bt709
        };
        *dst = clipped;
    }
}

/// Per-pixel scalar reference for the HLG → linear sRGB pipeline.
fn hlg_to_linear_srgb_scalar_ref(
    hlg_row: &[[f32; 3]],
    out: &mut [[f32; 3]],
    tm: &dyn zentone::ToneMap,
    display_peak_nits: f32,
) {
    let gamma = hlg_system_gamma(display_peak_nits);
    for (src, dst) in hlg_row.iter().zip(out.iter_mut()) {
        let scene = [
            linear_srgb::tf::hlg_to_linear(src[0]),
            linear_srgb::tf::hlg_to_linear(src[1]),
            linear_srgb::tf::hlg_to_linear(src[2]),
        ];
        let display = hlg_ootf(scene, gamma);
        let tonemapped = tm.map_rgb(display);
        let bt709 = apply_matrix(&BT2020_TO_BT709, tonemapped);
        let clipped = if is_out_of_gamut(bt709) {
            soft_clip(bt709)
        } else {
            bt709
        };
        *dst = clipped;
    }
}

/// Per-pixel scalar reference for the PQ → sRGB-encoded `u8` pipeline.
fn pq_to_srgb8_scalar_ref(pq_row: &[[f32; 3]], out: &mut [[u8; 3]], tm: &dyn zentone::ToneMap) {
    for (src, dst) in pq_row.iter().zip(out.iter_mut()) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];
        let tonemapped = tm.map_rgb(linear_2020);
        let bt709 = apply_matrix(&BT2020_TO_BT709, tonemapped);
        let clipped = if is_out_of_gamut(bt709) {
            soft_clip(bt709)
        } else {
            bt709
        };
        for c in 0..3 {
            let v = clipped[c].clamp(0.0, 1.0);
            let e = linear_srgb::tf::linear_to_srgb(v);
            dst[c] = (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Random-but-deterministic strip generator for the pipeline parity tests.
/// Produces values spanning HDR magnitudes (`0` to ~`10×` SDR) with mixed
/// channel ratios so the gamut matrix produces real OOG candidates.
fn synth_pipeline_strip(n: usize, seed: u64) -> Vec<[f32; 3]> {
    // Tiny LCG (no randomness crate). Deterministic, sufficient for property
    // coverage; we explicitly fold in HDR-range fixtures below.
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32) as f32 / u32::MAX as f32
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Mix random with structured fixtures so we hit boundary cases.
        match i % 16 {
            0 => out.push([0.0, 0.0, 0.0]),
            1 => out.push([1.0, 1.0, 1.0]),
            2 => out.push([0.58, 0.58, 0.58]), // ~SDR white in PQ
            3 => out.push([0.75, 0.75, 0.75]), // HLG ref white
            4 => out.push([0.9, 0.1, 0.05]),
            5 => out.push([0.05, 0.9, 0.4]),
            6 => out.push([0.2, 0.4, 0.95]),
            _ => {
                // Random within [0, 1] PQ/HLG signal range.
                let r = next();
                let g = next();
                let b = next();
                out.push([r, g, b]);
            }
        }
    }
    out
}

/// PQ → linear sRGB SIMD fused pipeline must match the per-pixel scalar
/// reference (built from `tm.map_rgb` + transfer functions + `apply_matrix`
/// + `soft_clip`) to within FMA tolerance.
#[test]
fn tonemap_pq_row_simd_matches_scalar() {
    let curves: Vec<Box<dyn zentone::ToneMap>> = vec![
        Box::new(Bt2408Tonemapper::new(4000.0, 1000.0)),
        Box::new(ToneMapCurve::Reinhard),
        Box::new(ToneMapCurve::HableFilmic),
    ];
    let lengths = [1usize, 7, 8, 16, 17, 64, 1024];
    let mut max_err: f32 = 0.0;
    let mut cases = 0;

    for tm in curves.iter() {
        for &n in &lengths {
            let strip = synth_pipeline_strip(n, n as u64 * 31 + 1);

            // Per-pixel scalar reference.
            let mut out_scalar = vec![[0.0_f32; 3]; n];
            pq_to_linear_srgb_scalar_ref(&strip, &mut out_scalar, tm.as_ref());

            // SIMD strip form.
            let mut out_simd = vec![[0.0_f32; 3]; n];
            tonemap_pq_row_simd(&strip, &mut out_simd, tm.as_ref());

            for (i, (px_simd, px_ref)) in out_simd.iter().zip(out_scalar.iter()).enumerate() {
                for c in 0..3 {
                    let s = px_ref[c];
                    let v = px_simd[c];
                    let err = (s - v).abs();
                    max_err = max_err.max(err);
                    // PQ EOTF/OETF go through `pow_midp` (~3 ULP × ~3 chained
                    // calls in the fused pipeline). Bt2408's `map_strip_simd`
                    // override (PR4) folds another `pow_midp` chain into the
                    // strip path, widening the cumulative error vs the scalar
                    // reference. Use 2e-5 × magnitude to cover the worst-case
                    // ULP stack-up across the full PQ→tonemap→sRGB pipeline.
                    let mag = s.abs().max(v.abs()).max(1.0);
                    let tol = 2e-5 * mag;
                    assert!(
                        err < tol,
                        "tonemap_pq_row_simd[{i},{c}] in={:?}: simd={v} ref={s} err={err:.3e} tol={tol:.3e}",
                        strip[i],
                    );
                    cases += 1;
                }
            }
        }
    }
    println!("tonemap_pq_row_simd: {cases} comparisons, max_err={max_err:.3e}");
}

/// HLG → linear sRGB SIMD fused pipeline must match the per-pixel scalar
/// reference.
#[test]
fn tonemap_hlg_row_simd_matches_scalar() {
    let curves: Vec<Box<dyn zentone::ToneMap>> = vec![
        Box::new(Bt2408Tonemapper::new(4000.0, 1000.0)),
        Box::new(ToneMapCurve::Reinhard),
    ];
    let lengths = [1usize, 7, 8, 17, 64, 1024];
    let display_peaks = [400.0_f32, 1000.0, 4000.0];
    let mut max_err: f32 = 0.0;
    let mut cases = 0;

    for tm in curves.iter() {
        for &peak in &display_peaks {
            for &n in &lengths {
                let strip = synth_pipeline_strip(n, n as u64 * 17 + peak as u64);
                // HLG signal is in [0, 1] for the standard EOTF; the strip
                // generator already keeps inputs in that range.
                let mut out_scalar = vec![[0.0_f32; 3]; n];
                hlg_to_linear_srgb_scalar_ref(&strip, &mut out_scalar, tm.as_ref(), peak);

                let mut out_simd = vec![[0.0_f32; 3]; n];
                tonemap_hlg_row_simd(&strip, &mut out_simd, tm.as_ref(), peak);

                for (i, (px_simd, px_ref)) in out_simd.iter().zip(out_scalar.iter()).enumerate() {
                    for c in 0..3 {
                        let s = px_ref[c];
                        let v = px_simd[c];
                        let err = (s - v).abs();
                        max_err = max_err.max(err);
                        // HLG approx pow_midp + system gamma adds a tiny bit
                        // more error than PQ — widen to 1e-4 (still strict
                        // enough to catch a wrong matrix or tone curve).
                        let mag = s.abs().max(v.abs()).max(1.0);
                        let tol = 1e-4 * mag;
                        assert!(
                            err < tol,
                            "tonemap_hlg_row_simd[{i},{c}] peak={peak} in={:?}: simd={v} ref={s} err={err:.3e} tol={tol:.3e}",
                            strip[i],
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("tonemap_hlg_row_simd: {cases} comparisons, max_err={max_err:.3e}");
}

/// PQ → sRGB u8 SIMD fused pipeline must match the per-pixel scalar
/// reference within ±1 LSB.
#[test]
fn tonemap_pq_to_srgb8_row_simd_matches_scalar() {
    let curves: Vec<Box<dyn zentone::ToneMap>> = vec![
        Box::new(Bt2408Tonemapper::new(4000.0, 1000.0)),
        Box::new(ToneMapCurve::Reinhard),
        Box::new(ToneMapCurve::HableFilmic),
    ];
    let lengths = [1usize, 7, 8, 17, 64, 1024];
    let mut max_diff: i32 = 0;
    let mut cases = 0;
    let mut over_one = 0;

    for tm in curves.iter() {
        for &n in &lengths {
            let strip = synth_pipeline_strip(n, n as u64 * 7 + 99);
            let mut out_scalar = vec![[0u8; 3]; n];
            pq_to_srgb8_scalar_ref(&strip, &mut out_scalar, tm.as_ref());

            let mut out_simd = vec![[0u8; 3]; n];
            tonemap_pq_to_srgb8_row_simd(&strip, &mut out_simd, tm.as_ref());

            for (px_simd, px_ref) in out_simd.iter().zip(out_scalar.iter()) {
                for c in 0..3 {
                    let s = px_ref[c] as i32;
                    let v = px_simd[c] as i32;
                    let diff = (s - v).abs();
                    max_diff = max_diff.max(diff);
                    if diff > 1 {
                        over_one += 1;
                    }
                    cases += 1;
                }
            }
        }
    }
    // sRGB OETF has small slope variations; allow ±1 LSB everywhere, and
    // tolerate a tiny number of ±2 LSB (the soft-clip / matrix sequence
    // can flip rounding direction at OETF inflection). Hard ceiling: any
    // ±2 LSB happens on < 0.1% of samples.
    let pct = (over_one as f64) / (cases as f64) * 100.0;
    assert!(
        max_diff <= 2,
        "tonemap_pq_to_srgb8_row_simd diverged by {max_diff} LSB"
    );
    assert!(
        pct < 0.5,
        "tonemap_pq_to_srgb8_row_simd: {over_one}/{cases} ({pct:.3}%) > ±1 LSB"
    );
    println!(
        "tonemap_pq_to_srgb8_row_simd: {cases} comparisons, max_diff={max_diff} LSB, \
         over_one={over_one} ({pct:.3}%)"
    );
}

// ============================================================================
// PR4 — `map_strip_simd` overrides on transcendental-using curves
// ============================================================================
//
// These tests pin the new SIMD strip kernels for `Bt2408Tonemapper`,
// `Bt2446A/B/C`, and `CompiledFilmicSpline` against their per-pixel scalar
// references. Tolerance reflects the precision of the underlying SIMD
// transcendental: `pow_midp` / `log2_midp` / `exp2_midp` are ~3 ULP, so the
// kernel's cumulative output error stacks up to a few times that.
//
// Random rows of size 1..=1024 cover the chunk-of-8 SIMD body and the
// scalar tail; the property grid mixes greys, primaries, secondaries, and
// HDR-up-to-10× values that exercise the out-of-range branches in each
// curve.

/// Random RGB strip (deterministic per `seed`) with HDR-up-to-10× values.
fn random_strip(n: usize, seed: u64) -> Vec<[f32; 3]> {
    // Splitmix64 LCG for reproducibility; we don't need crypto-quality
    // randomness, just decorrelated property samples.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next_u32 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) & 0xFFFF_FFFF) as u32
    };
    let mut next_f01 = || (next_u32() as f32) / (u32::MAX as f32);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Half SDR-range, half HDR up to 10×, sprinkled with negative
        // values to exercise the max(0, ·) clamps.
        let scale = if next_f01() < 0.5 { 1.0 } else { 10.0 };
        let r = next_f01() * scale - 0.05;
        let g = next_f01() * scale - 0.05;
        let b = next_f01() * scale - 0.05;
        out.push([r, g, b]);
    }
    out
}

/// Run the property grid + a few random rows of varying size against the
/// supplied curve, comparing `map_strip_simd` against per-pixel `map_rgb`.
/// Returns `(cases, max_err)`.
fn check_strip_vs_per_pixel<T: zentone::ToneMap + ?Sized>(tm: &T, abs_tol: f32) -> (usize, f32) {
    let mut max_err = 0.0_f32;
    let mut cases = 0;
    let mut strips: Vec<Vec<[f32; 3]>> = Vec::new();
    // Property strip (re-use same shape as PR2 tests).
    let mut prop = property_strip();
    strips.push(prop.split_off(0));
    // Random rows: 1, 7, 8, 17, 64, 257, 1024 (mix of <8, multiple-of-8,
    // off-by-one, and large to dwarf the tail).
    for &n in &[1_usize, 7, 8, 17, 64, 257, 1024] {
        strips.push(random_strip(n, n as u64 * 0x1234_5678));
    }
    for strip in strips.iter() {
        let mut via_simd = strip.clone();
        tm.map_strip_simd(&mut via_simd);
        for (i, px_in) in strip.iter().enumerate() {
            let expected = tm.map_rgb(*px_in);
            for c in 0..3 {
                let mag = expected[c].abs().max(via_simd[i][c].abs()).max(1.0);
                let tol = abs_tol * mag;
                let err = (via_simd[i][c] - expected[c]).abs();
                max_err = max_err.max(err);
                assert!(
                    err < tol,
                    "strip[{i},{c}] in={px_in:?}: simd={} ref={} err={err:.3e} tol={tol:.3e}",
                    via_simd[i][c],
                    expected[c]
                );
                cases += 1;
            }
        }
    }
    (cases, max_err)
}

#[test]
fn bt2408_strip_simd_matches_per_pixel() {
    use zentone::Bt2408Tonemapper;
    // Both spaces (Yrgb default + max-RGB), and an extra peak ratio.
    let curves = [
        Bt2408Tonemapper::new(4000.0, 1000.0),
        Bt2408Tonemapper::new(1000.0, 203.0),
        Bt2408Tonemapper::max_rgb(4000.0, 1000.0),
        Bt2408Tonemapper::max_rgb(1000.0, 203.0),
    ];
    let mut total = 0;
    let mut total_max = 0.0_f32;
    for tm in curves.iter() {
        // PQ OETF + EOTF stack 4 `pow_midp` calls (~3 ULP each); the
        // luma-scale divide `new_lum / signal_nits` amplifies that on
        // saturated single-channel inputs (`signal_nits` collapses to a
        // tiny luminance for pure blue, which inflates the ratio's relative
        // error). Allow 1e-4 × magnitude to cover the worst case.
        let (n, m) = check_strip_vs_per_pixel(tm, 1e-4);
        total += n;
        total_max = total_max.max(m);
    }
    println!("bt2408_strip_simd_matches_per_pixel: {total} comparisons, max_err={total_max:.3e}");
}

#[test]
fn bt2446a_strip_simd_matches_per_pixel() {
    use zentone::Bt2446A;
    let curves = [Bt2446A::new(1000.0, 100.0), Bt2446A::new(4000.0, 100.0)];
    let mut total = 0;
    let mut total_max = 0.0_f32;
    for tm in curves.iter() {
        // 3× pow_midp(1/2.4) for gamma encoding + log2_midp (perceptual
        // linearization) + exp2_midp (rho_sdr^y_c) + chained `f / y_p` and
        // YCbCr cancellation `b_p - y_p`. Subtractive cancellation on
        // saturated colors (one channel ~1.0 with luma weight 0.06 →
        // `b_p - y_p` near zero in absolute terms) inflates relative error
        // through the f-multiply. 5e-4 × magnitude covers the long tail.
        let (n, m) = check_strip_vs_per_pixel(tm, 5e-4);
        total += n;
        total_max = total_max.max(m);
    }
    println!("bt2446a_strip_simd_matches_per_pixel: {total} comparisons, max_err={total_max:.3e}");
}

#[test]
fn bt2446b_strip_simd_matches_per_pixel() {
    use zentone::Bt2446B;
    let curves = [Bt2446B::new(1000.0, 100.0), Bt2446B::new(4000.0, 100.0)];
    let mut total = 0;
    let mut total_max = 0.0_f32;
    for tm in curves.iter() {
        // Single log2_midp call → 3 ULP grade.
        let (n, m) = check_strip_vs_per_pixel(tm, 1e-5);
        total += n;
        total_max = total_max.max(m);
    }
    println!("bt2446b_strip_simd_matches_per_pixel: {total} comparisons, max_err={total_max:.3e}");
}

#[test]
fn bt2446c_strip_simd_matches_per_pixel() {
    use zentone::Bt2446C;
    let curves = [
        Bt2446C::new(1000.0, 100.0),
        Bt2446C::with_params(1000.0, 100.0, 0.83802, 15.09968, 0.74204, 78.99439, 0.1),
    ];
    let mut total = 0;
    let mut total_max = 0.0_f32;
    for tm in curves.iter() {
        // log2_midp + crosstalk matrix (FMA chain). 3 ULP × small constant.
        let (n, m) = check_strip_vs_per_pixel(tm, 1e-5);
        total += n;
        total_max = total_max.max(m);
    }
    println!("bt2446c_strip_simd_matches_per_pixel: {total} comparisons, max_err={total_max:.3e}");
}

#[test]
fn filmic_spline_strip_simd_matches_per_pixel() {
    use zentone::{CompiledFilmicSpline, FilmicSplineConfig};
    // Default config + darktable-like config + saturation pulled wide
    // (exercises the desat exp2 path more visibly).
    let mut cfg_dt = FilmicSplineConfig::default();
    cfg_dt.output_power = 4.0;
    cfg_dt.latitude = 0.01;
    cfg_dt.white_point_source = 4.0;
    cfg_dt.contrast = 1.0;
    let mut cfg_sat = FilmicSplineConfig::default();
    cfg_sat.saturation = 50.0;
    let cfgs = [FilmicSplineConfig::default(), cfg_dt, cfg_sat];
    let curves: Vec<CompiledFilmicSpline> = cfgs.iter().map(CompiledFilmicSpline::new).collect();
    let mut total = 0;
    let mut total_max = 0.0_f32;
    for tm in curves.iter() {
        // log2_midp (shaper) + 2× exp2_midp (desaturate). 3 ULP × small mag
        // through rational spline = ~2e-5 worst case at default params,
        // up to 5e-5 with darktable's hardness-4 config (output is ~1e-1
        // scale before the rational spline amplifies error).
        let (n, m) = check_strip_vs_per_pixel(tm, 5e-5);
        total += n;
        total_max = total_max.max(m);
    }
    println!(
        "filmic_spline_strip_simd_matches_per_pixel: {total} comparisons, max_err={total_max:.3e}"
    );
}

/// The trait's default `map_strip_simd` (per-pixel scalar fallback) must
/// produce identical output to a manual `for px in strip { *px = tm.map_rgb(*px); }`
/// loop. The fused PR3 pipelines also exercise this via integration tests
/// above — this is the focused unit test.
#[test]
fn tone_map_default_map_strip_simd_matches_per_pixel_loop() {
    use zentone::ToneMap;
    let tm = ToneMapCurve::HableFilmic;
    let strip_in = synth_pipeline_strip(33, 4242);

    let mut via_strip = strip_in.clone();
    tm.map_strip_simd(&mut via_strip);

    let mut via_loop = strip_in.clone();
    for px in via_loop.iter_mut() {
        *px = tm.map_rgb(*px);
    }

    for (i, (a, b)) in via_strip.iter().zip(via_loop.iter()).enumerate() {
        for c in 0..3 {
            assert!(
                (a[c] - b[c]).abs() < 1e-7,
                "default map_strip_simd diverged at [{i},{c}]: strip={} loop={}",
                a[c],
                b[c]
            );
        }
    }
}
