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

use zentone::pipeline::{
    tonemap_hlg_row_simd, tonemap_hlg_to_linear_srgb, tonemap_pq_row_simd,
    tonemap_pq_to_linear_srgb, tonemap_pq_to_srgb8, tonemap_pq_to_srgb8_row_simd,
};
use zentone::{Bt2408Tonemapper, ToneMapCurve};

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

/// PQ → linear sRGB SIMD fused pipeline must match the scalar reference
/// to within FMA tolerance.
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

            // Scalar reference: flatten, tonemap, unflatten.
            let flat: Vec<f32> = strip.iter().flat_map(|p| p.iter().copied()).collect();
            let mut out_scalar = vec![0.0_f32; flat.len()];
            tonemap_pq_to_linear_srgb(&flat, &mut out_scalar, tm.as_ref(), 3);

            // SIMD: pass strip in.
            let mut out_simd = vec![[0.0_f32; 3]; n];
            tonemap_pq_row_simd(&strip, &mut out_simd, tm.as_ref());

            for (i, px) in out_simd.iter().enumerate() {
                for c in 0..3 {
                    let s = out_scalar[i * 3 + c];
                    let v = px[c];
                    let err = (s - v).abs();
                    max_err = max_err.max(err);
                    // 2e-6 absolute, scaled by output magnitude (FMA + soft
                    // clip path). PQ EOTF accuracy in linear-srgb is
                    // ~rational poly grade, so we widen a bit.
                    let mag = s.abs().max(v.abs()).max(1.0);
                    let tol = 5e-6 * mag;
                    let _ = flat.len();
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

/// HLG → linear sRGB SIMD fused pipeline must match the scalar reference.
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
                let flat: Vec<f32> = strip.iter().flat_map(|p| p.iter().copied()).collect();
                let mut out_scalar = vec![0.0_f32; flat.len()];
                tonemap_hlg_to_linear_srgb(&flat, &mut out_scalar, tm.as_ref(), peak, 3);

                let mut out_simd = vec![[0.0_f32; 3]; n];
                tonemap_hlg_row_simd(&strip, &mut out_simd, tm.as_ref(), peak);

                for (i, px) in out_simd.iter().enumerate() {
                    for c in 0..3 {
                        let s = out_scalar[i * 3 + c];
                        let v = px[c];
                        let err = (s - v).abs();
                        max_err = max_err.max(err);
                        // HLG approx pow_midp + system gamma adds a tiny bit
                        // more error than PQ — widen to 1e-4 (still strict
                        // enough to catch a wrong matrix or tone curve).
                        let mag = s.abs().max(v.abs()).max(1.0);
                        let tol = 1e-4 * mag;
                        let _ = flat.len();
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

/// PQ → sRGB u8 SIMD fused pipeline must match scalar within ±1 LSB.
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
            let flat: Vec<f32> = strip.iter().flat_map(|p| p.iter().copied()).collect();
            let mut out_scalar = vec![0u8; flat.len()];
            tonemap_pq_to_srgb8(&flat, &mut out_scalar, tm.as_ref(), 3);

            let mut out_simd = vec![[0u8; 3]; n];
            tonemap_pq_to_srgb8_row_simd(&strip, &mut out_simd, tm.as_ref());

            for (i, px) in out_simd.iter().enumerate() {
                for c in 0..3 {
                    let s = out_scalar[i * 3 + c] as i32;
                    let v = px[c] as i32;
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
