//! Brute-force hardening sweep for the public surface of `crate::gamut`.
//!
//! Each previous PR on `gamut.rs` surfaced a different latent bug in the
//! sorted-equal-channel branches of `soft_clip` (PR2: `r >= g == b` arm
//! forgot to clamp B; PR3: same arm with `r > 1` mapped pure red to white).
//! This file walks the input space densely so that future bugs of that
//! shape get caught instantly:
//!
//! 1. Sorted-equal channels (`[a, a, b]` permutations)
//! 2. Negative inputs
//! 3. NaN, ±∞, denormals
//! 4. HDR magnitudes 0..10× along primaries and greys
//! 5. Out-of-gamut highlights
//! 6. Boundary values (0, 1, ε, 1-ε, large finite)
//! 7. Ordering invariant (`is_out_of_gamut(soft_clip(x))` is always false
//!    for finite `x`)
//!
//! For each sweep, scalar and SIMD strip forms are compared on identical
//! inputs. Disagreement is a bug in one of them — neither is allowed to
//! drift. The bar is FMA-tolerance (2e-6 absolute, scaled by magnitude).

use zentone::gamut::{
    BT2020_TO_BT709, apply_matrix, apply_matrix_row, apply_matrix_row_simd,
    apply_matrix_row_simd_rgba, is_out_of_gamut, is_out_of_gamut_mask_simd, soft_clip,
    soft_clip_row_simd,
};

const TOL_ABS: f32 = 2e-6;

/// Compare two pixels with magnitude-scaled FMA tolerance.
fn assert_pixel_close(actual: [f32; 3], expected: [f32; 3], ctx: &str) {
    let mag = expected
        .iter()
        .chain(actual.iter())
        .map(|x| x.abs())
        .fold(1.0_f32, f32::max);
    let tol = TOL_ABS * mag.max(1.0);
    for c in 0..3 {
        let err = (actual[c] - expected[c]).abs();
        assert!(
            err < tol,
            "{ctx}[{c}]: actual={} expected={} err={err:.3e} tol={tol:.3e}",
            actual[c],
            expected[c]
        );
    }
}

/// Run scalar `soft_clip` per pixel, return the row.
fn soft_clip_scalar(row: &[[f32; 3]]) -> Vec<[f32; 3]> {
    row.iter().map(|p| soft_clip(*p)).collect()
}

/// Soft clip via the SIMD strip kernel.
fn soft_clip_simd(row: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut v = row.to_vec();
    soft_clip_row_simd(&mut v);
    v
}

/// Assert SIMD strip and scalar agree on every pixel of `row`.
fn assert_scalar_simd_agree_soft_clip(row: &[[f32; 3]], ctx: &str) {
    let scalar = soft_clip_scalar(row);
    let simd = soft_clip_simd(row);
    for (i, (s, v)) in scalar.iter().zip(simd.iter()).enumerate() {
        let mag = s
            .iter()
            .chain(v.iter())
            .map(|x| x.abs())
            .fold(1.0_f32, f32::max);
        let tol = TOL_ABS * mag.max(1.0);
        for c in 0..3 {
            // NaN-tolerant compare: if both are NaN, treat as equal.
            if s[c].is_nan() && v[c].is_nan() {
                continue;
            }
            let err = (s[c] - v[c]).abs();
            assert!(
                err < tol,
                "{ctx} pixel[{i}]={:?} chan[{c}]: scalar={} simd={} err={err:.3e}",
                row[i],
                s[c],
                v[c]
            );
        }
    }
}

// ============================================================================
// 1. Sorted-equal channels — every permutation of [a,a,b], [a,b,b], [a,a,a].
// ============================================================================

/// Every permutation of `[a, a, b]` and `[a, b, b]` and `[a, a, a]`,
/// for `a, b ∈ {0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0}`, plus signed variants.
/// soft_clip output must be in [0, 1] (after the negative clamp) and the
/// SIMD strip must agree with the scalar reference.
#[test]
fn sorted_equal_channels_exhaustive() {
    let values: Vec<f32> = vec![
        -1.0,
        -0.5,
        -0.1,
        0.0,
        0.001,
        0.5,
        1.0 - 1e-6,
        1.0,
        1.0 + 1e-6,
        1.2,
        1.5,
        2.0,
        5.0,
        10.0,
    ];
    let mut row = Vec::new();
    for &a in &values {
        for &b in &values {
            // [a,a,b] permutations
            row.push([a, a, b]);
            row.push([a, b, a]);
            row.push([b, a, a]);
            // [a,b,b]
            row.push([b, b, a]);
            row.push([b, a, b]);
            row.push([a, b, b]);
            // [a,a,a]
            row.push([a, a, a]);
            // [b,b,b]
            row.push([b, b, b]);
        }
    }
    // ~14 * 14 * 8 = 1568 pixels.
    let scalar = soft_clip_scalar(&row);
    for (i, &out) in scalar.iter().enumerate() {
        for (c, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "scalar soft_clip out of [0,1]: in={:?} out={:?} chan {c}={v}",
                row[i],
                out
            );
        }
    }
    assert_scalar_simd_agree_soft_clip(&row, "sorted_equal_channels");
}

// ============================================================================
// 2. Negative inputs — soft_clip clamps to 0; is_out_of_gamut flags them.
// ============================================================================

#[test]
fn negative_inputs_clamp_to_zero() {
    let cases: &[[f32; 3]] = &[
        [-0.1, 0.5, 0.5],
        [-1.0, 0.0, 0.0],
        [-2.0, -2.0, -2.0],
        [-100.0, 0.5, 0.5],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, -0.5],
        [-0.1, -0.1, -0.1],
        [-0.5, 0.5, 1.5], // mixed: under, in, over
        [-2.0, 2.0, -2.0],
        [-1e-30, 0.5, 0.5], // tiny negative
    ];
    for &c in cases {
        let out = soft_clip(c);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "soft_clip({c:?}) chan[{i}] = {v} out of [0,1]"
            );
        }
        // is_out_of_gamut must flag any input with a negative channel.
        if c.iter().any(|x| *x < 0.0) {
            assert!(is_out_of_gamut(c), "is_out_of_gamut missed {c:?}");
        }
    }
    assert_scalar_simd_agree_soft_clip(cases, "negative_inputs");
}

// ============================================================================
// 3. NaN, ±∞, denormals.
// ============================================================================

#[test]
fn nan_inf_denormal_inputs_dont_crash() {
    let cases: &[[f32; 3]] = &[
        [f32::NAN, 0.5, 0.5],
        [0.5, f32::NAN, 0.5],
        [0.5, 0.5, f32::NAN],
        [f32::NAN, f32::NAN, f32::NAN],
        [f32::INFINITY, 0.5, 0.5],
        [0.5, f32::INFINITY, 0.5],
        [f32::NEG_INFINITY, 0.5, 0.5],
        [f32::INFINITY, f32::INFINITY, f32::INFINITY],
        [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::MIN_POSITIVE, f32::MIN_POSITIVE, f32::MIN_POSITIVE],
        [1e-40, 0.0, 0.0], // subnormal
        [f32::INFINITY, 0.5, f32::NAN],
        [f32::INFINITY, f32::NEG_INFINITY, 0.5],
    ];
    for &c in cases {
        // Just shouldn't panic / loop forever. We don't assert specific output.
        let _ = soft_clip(c);
        let _ = is_out_of_gamut(c);
    }
    // Scalar and SIMD should agree on finite-and-not-NaN cases. For NaN/inf,
    // we only assert no panic from the SIMD path.
    let mut buf = cases.to_vec();
    soft_clip_row_simd(&mut buf);
    let mut mask = vec![0.0_f32; cases.len()];
    is_out_of_gamut_mask_simd(cases, &mut mask);
}

#[test]
fn finite_inputs_produce_finite_soft_clip() {
    // Sweep with only finite values — output must be finite and within [0,1].
    let mut row = Vec::new();
    for &a in &[-100.0_f32, -1.0, 0.0, 0.5, 1.0, 2.0, 1e10, 1e30] {
        for &b in &[-100.0_f32, -1.0, 0.0, 0.5, 1.0, 2.0, 1e10, 1e30] {
            for &c in &[-100.0_f32, -1.0, 0.0, 0.5, 1.0, 2.0, 1e10, 1e30] {
                row.push([a, b, c]);
            }
        }
    }
    let out = soft_clip_scalar(&row);
    for (i, &px) in out.iter().enumerate() {
        for (j, &v) in px.iter().enumerate() {
            assert!(
                v.is_finite(),
                "soft_clip({:?}) chan[{j}] = {v} not finite",
                row[i]
            );
            assert!(
                (0.0..=1.0).contains(&v),
                "soft_clip({:?}) chan[{j}] = {v} out of [0,1]",
                row[i]
            );
        }
    }
    assert_scalar_simd_agree_soft_clip(&row, "finite_extremes");
}

// ============================================================================
// 4. HDR magnitudes — 0..10× steps along primaries and greys.
// ============================================================================

#[test]
fn hdr_sweep_along_primaries_and_greys() {
    let mut row = Vec::new();
    let mut t = 0.0_f32;
    while t <= 10.0 + 1e-6 {
        // Greys
        row.push([t, t, t]);
        // Primaries
        row.push([t, 0.0, 0.0]);
        row.push([0.0, t, 0.0]);
        row.push([0.0, 0.0, t]);
        // Secondaries
        row.push([t, t, 0.0]);
        row.push([0.0, t, t]);
        row.push([t, 0.0, t]);
        // With small non-axis component (catches the equal-channel branches)
        row.push([t, 0.5, 0.5]);
        row.push([0.5, t, 0.5]);
        row.push([0.5, 0.5, t]);
        t += 0.1;
    }
    // Scalar must be in [0,1].
    let scalar = soft_clip_scalar(&row);
    for (i, &out) in scalar.iter().enumerate() {
        for (c, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "hdr_sweep: in={:?} chan[{c}]={v}",
                row[i]
            );
        }
    }
    assert_scalar_simd_agree_soft_clip(&row, "hdr_sweep");
}

// ============================================================================
// 5. Out-of-gamut highlights — soft_clip preserves hue, reduces magnitude.
// ============================================================================

#[test]
fn out_of_gamut_highlights_preserve_hue_ordering() {
    // For each (hi, mid, lo) triple with hi > 1 or mid > 1, after soft_clip
    // the channel ordering must be preserved.
    let levels = [0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0];
    let mut row = Vec::new();
    for &a in &levels {
        for &b in &levels {
            for &c in &levels {
                row.push([a, b, c]);
            }
        }
    }
    let scalar = soft_clip_scalar(&row);
    for (i, &out) in scalar.iter().enumerate() {
        let inp = row[i];
        let post_neg = [inp[0].max(0.0), inp[1].max(0.0), inp[2].max(0.0)];
        // After negative clamp, the ordering of post_neg must match out.
        for a in 0..3 {
            for b in 0..3 {
                if post_neg[a] >= post_neg[b] {
                    assert!(
                        out[a] >= out[b] - TOL_ABS,
                        "hue ordering violated: in={inp:?} out={out:?} a={a} b={b}"
                    );
                }
            }
        }
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "out_of_gamut_highlights: in={inp:?} out={out:?}"
            );
        }
    }
    assert_scalar_simd_agree_soft_clip(&row, "ordering_preserved");
}

#[test]
fn pure_primaries_clip_to_pure_primaries() {
    // [k, 0, 0] for k > 1 must clip to [1, 0, 0]. Same for the other
    // primary axes. This is the regression test for PR3's `r >= g == b`
    // bug that mapped pure red to white.
    for k in [1.001_f32, 1.5, 2.0, 5.0, 100.0, 1e6, 1e30] {
        let cases = [
            ([k, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, k, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, k], [0.0, 0.0, 1.0]),
            ([k, k, 0.0], [1.0, 1.0, 0.0]),
            ([0.0, k, k], [0.0, 1.0, 1.0]),
            ([k, 0.0, k], [1.0, 0.0, 1.0]),
        ];
        for (input, expected) in cases {
            let out = soft_clip(input);
            assert_pixel_close(out, expected, &format!("pure_primary k={k}"));
        }
    }
}

// ============================================================================
// 6. Boundary values — exactly 0.0, 1.0, ε, 1.0-ε, large finite.
// ============================================================================

#[test]
fn boundary_values() {
    let eps = f32::EPSILON;
    let cases: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [eps, eps, eps],
        [1.0 - eps, 1.0 - eps, 1.0 - eps],
        [1.0 + eps, 1.0, 1.0],
        [1.0, 1.0 + eps, 1.0],
        [1.0, 1.0, 1.0 + eps],
        [1e30, 0.5, 0.5],
        [1e30, 1e30, 0.5],
        [1e30, 1e30, 1e30],
        [-1e-30, 1.0, 0.5],
        [0.5, 0.5, 1.0], // boundary on max
        [0.5, 1.0, 0.5],
        [1.0, 0.5, 0.5],
    ];
    let scalar = soft_clip_scalar(cases);
    for (i, &out) in scalar.iter().enumerate() {
        for (c, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "boundary: in={:?} chan[{c}]={v}",
                cases[i]
            );
        }
    }
    assert_scalar_simd_agree_soft_clip(cases, "boundary");
}

// ============================================================================
// 7. Ordering invariant: is_out_of_gamut(soft_clip(x)) == false for finite x.
// ============================================================================

#[test]
fn soft_clip_then_is_out_of_gamut_is_false() {
    // Walk a dense grid of 21^3 = 9261 pixels covering [-2, 12] on each axis.
    let mut count = 0;
    for ri in 0..21 {
        for gi in 0..21 {
            for bi in 0..21 {
                let r = -2.0 + (ri as f32) * 14.0 / 20.0;
                let g = -2.0 + (gi as f32) * 14.0 / 20.0;
                let b = -2.0 + (bi as f32) * 14.0 / 20.0;
                let inp = [r, g, b];
                let clipped = soft_clip(inp);
                assert!(
                    !is_out_of_gamut(clipped),
                    "is_out_of_gamut(soft_clip({inp:?})) = true; clipped = {clipped:?}"
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, 9261);
}

// ============================================================================
// 8. apply_matrix scalar vs strip and SIMD agreement on dense grid.
// ============================================================================

#[test]
fn apply_matrix_scalar_vs_simd_dense() {
    let levels = [-2.0_f32, -0.5, 0.0, 0.18, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut row = Vec::new();
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                row.push([r, g, b]);
            }
        }
    }
    let mut simd = row.clone();
    apply_matrix_row_simd(&BT2020_TO_BT709, &mut simd);
    // apply_matrix_row in place — make a copy.
    let mut interleaved: Vec<f32> = row.iter().flat_map(|p| p.iter().copied()).collect();
    apply_matrix_row(&BT2020_TO_BT709, &mut interleaved, 3);

    for (i, px) in row.iter().enumerate() {
        let expected = apply_matrix(&BT2020_TO_BT709, *px);
        let row_form = [
            interleaved[i * 3],
            interleaved[i * 3 + 1],
            interleaved[i * 3 + 2],
        ];
        assert_pixel_close(row_form, expected, &format!("row pix={px:?}"));
        assert_pixel_close(simd[i], expected, &format!("simd pix={px:?}"));
    }
}

#[test]
fn apply_matrix_rgba_alpha_preserved() {
    let levels = [-2.0_f32, 0.0, 0.5, 1.0, 5.0];
    let mut row = Vec::new();
    let mut alphas = Vec::new();
    let mut counter = 0_u32;
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                let alpha = (counter as f32) * 0.0137 % 1.0;
                row.push([r, g, b, alpha]);
                alphas.push(alpha);
                counter += 1;
            }
        }
    }
    apply_matrix_row_simd_rgba(&BT2020_TO_BT709, &mut row);
    for (i, px) in row.iter().enumerate() {
        assert_eq!(
            px[3], alphas[i],
            "alpha mutated at index {i}: {} vs {}",
            px[3], alphas[i]
        );
    }
}

// ============================================================================
// 9. is_out_of_gamut scalar vs SIMD parity.
// ============================================================================

#[test]
fn is_out_of_gamut_scalar_vs_simd() {
    let levels = [
        -1.0_f32,
        -1e-6,
        0.0,
        1e-6,
        0.5,
        1.0 - 1e-6,
        1.0,
        1.0 + 1e-6,
        1.5,
        2.0,
    ];
    let mut row = Vec::new();
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                row.push([r, g, b]);
            }
        }
    }
    let mut mask = vec![0.0_f32; row.len()];
    is_out_of_gamut_mask_simd(&row, &mut mask);
    for (i, px) in row.iter().enumerate() {
        let scalar = is_out_of_gamut(*px);
        let simd_flag = mask[i] > 0.5;
        assert_eq!(
            scalar, simd_flag,
            "is_out_of_gamut disagreement at {px:?}: scalar={scalar} simd={simd_flag}"
        );
    }
}

// ============================================================================
// 10. Pseudo-random fuzz — deterministic LCG, no external deps.
// ============================================================================

/// Tiny deterministic LCG for reproducible fuzz seeds.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Uniform in [-3.0, 12.0] with occasional out-of-range spikes.
    fn next_f32_color(&mut self) -> f32 {
        let r = self.next_u32();
        // 1/256 chance of edge case
        if (r & 0xff) == 0 {
            let pick = (r >> 8) % 6;
            return [0.0, 1.0, -1.0, f32::EPSILON, 1.0 + f32::EPSILON, 1e20][pick as usize];
        }
        let f = (r as f32) / (u32::MAX as f32);
        -3.0 + f * 15.0
    }
}

#[test]
fn fuzz_scalar_simd_agreement() {
    let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
    // Generate 8000 random pixels (covers many SIMD strip alignments).
    let n = 8000;
    let mut row = Vec::with_capacity(n);
    for _ in 0..n {
        row.push([
            rng.next_f32_color(),
            rng.next_f32_color(),
            rng.next_f32_color(),
        ]);
    }
    assert_scalar_simd_agree_soft_clip(&row, "fuzz");

    // Also sweep apply_matrix: scalar vs SIMD agreement.
    let mut simd = row.clone();
    apply_matrix_row_simd(&BT2020_TO_BT709, &mut simd);
    for (i, px) in row.iter().enumerate() {
        let expected = apply_matrix(&BT2020_TO_BT709, *px);
        // Skip non-finite outputs (input had NaN/inf — both paths can diverge harmlessly).
        if !expected.iter().all(|x| x.is_finite()) {
            continue;
        }
        assert_pixel_close(simd[i], expected, &format!("fuzz apply pix={px:?}"));
    }

    // Also sweep is_out_of_gamut.
    let mut mask = vec![0.0_f32; n];
    is_out_of_gamut_mask_simd(&row, &mut mask);
    for (i, px) in row.iter().enumerate() {
        let scalar = is_out_of_gamut(*px);
        let simd_flag = mask[i] > 0.5;
        assert_eq!(scalar, simd_flag, "fuzz is_out_of_gamut disagree {px:?}");
    }
}

// ============================================================================
// 11. Strip-length sweep — exercise the scalar tail at every length 0..32.
// ============================================================================

#[test]
fn strip_length_sweep_tail_correctness() {
    // For every length 0..32, build a row of that length and confirm scalar
    // and SIMD agree. This catches off-by-ones in the chunks_exact tail loop.
    let template: Vec<[f32; 3]> = (0..32)
        .map(|i| {
            let f = i as f32;
            [f * 0.13 - 1.5, f * 0.07 + 0.5, f * 0.21 - 0.3]
        })
        .collect();
    for len in 0..=32 {
        let row: Vec<[f32; 3]> = template[..len].to_vec();
        assert_scalar_simd_agree_soft_clip(&row, &format!("len={len}"));

        // apply_matrix_row_simd on the same length.
        let mut simd = row.clone();
        apply_matrix_row_simd(&BT2020_TO_BT709, &mut simd);
        for (i, px) in row.iter().enumerate() {
            let expected = apply_matrix(&BT2020_TO_BT709, *px);
            assert_pixel_close(simd[i], expected, &format!("apply len={len} i={i}"));
        }
    }
}
