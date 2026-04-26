//! SIMD building blocks composed by the fused tone-mapping pipeline.
//!
//! Each kernel here is a strip-form sibling to a per-pixel reference function
//! in `gamut` or `hlg`. The per-pixel functions remain the parity surface; the
//! `*_row_simd` variants live in [`crate::gamut`] / [`crate::hlg`] and route
//! into these `#[archmage::magetypes]`-generated tier kernels via `incant!`.
//!
//! Layout: each kernel processes 8 RGB pixels per chunk in SOA form (gather
//! interleaved channels into three `f32x8`, do the math, scatter back). The
//! tail of the strip falls through to the per-pixel reference for correctness
//! parity at the boundary.
//!
//! Tiers: `v4(cfg(avx512))`, `v3`, `neon`, `wasm128`, `scalar`. Kernels that
//! call `pow_midp` (HLG OOTF) skip `v4` because magetypes' AVX-512 backend
//! does not yet implement `F32x8Convert` for `f32x8`; AVX-512-capable CPUs
//! still execute the V3 path safely (they all support AVX2).

use crate::hlg;

// ============================================================================
// Gamut: 3x3 matrix apply, RGB and RGBA.
// ============================================================================

/// 3x3 matrix multiply over 8 SOA pixels and the scalar tail of an RGB strip.
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
pub(crate) fn apply_matrix_rgb_tier(token: Token, m: &[[f32; 3]; 3], row: &mut [[f32; 3]]) {
    // Hoist matrix lanes once per strip — none depend on the chunk.
    let m00 = f32x8::splat(token, m[0][0]);
    let m01 = f32x8::splat(token, m[0][1]);
    let m02 = f32x8::splat(token, m[0][2]);
    let m10 = f32x8::splat(token, m[1][0]);
    let m11 = f32x8::splat(token, m[1][1]);
    let m12 = f32x8::splat(token, m[1][2]);
    let m20 = f32x8::splat(token, m[2][0]);
    let m21 = f32x8::splat(token, m[2][1]);
    let m22 = f32x8::splat(token, m[2][2]);

    let mut iter = row.chunks_exact_mut(8);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let nr = (m00 * r + m01 * g + m02 * b).to_array();
        let ng = (m10 * r + m11 * g + m12 * b).to_array();
        let nb = (m20 * r + m21 * g + m22 * b).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = nr[i];
            px[1] = ng[i];
            px[2] = nb[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        let r = m[0][0] * px[0] + m[0][1] * px[1] + m[0][2] * px[2];
        let g = m[1][0] * px[0] + m[1][1] * px[1] + m[1][2] * px[2];
        let b = m[2][0] * px[0] + m[2][1] * px[1] + m[2][2] * px[2];
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
}

/// 3x3 matrix multiply over 8 SOA pixels of an RGBA strip; alpha untouched.
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
pub(crate) fn apply_matrix_rgba_tier(token: Token, m: &[[f32; 3]; 3], row: &mut [[f32; 4]]) {
    let m00 = f32x8::splat(token, m[0][0]);
    let m01 = f32x8::splat(token, m[0][1]);
    let m02 = f32x8::splat(token, m[0][2]);
    let m10 = f32x8::splat(token, m[1][0]);
    let m11 = f32x8::splat(token, m[1][1]);
    let m12 = f32x8::splat(token, m[1][2]);
    let m20 = f32x8::splat(token, m[2][0]);
    let m21 = f32x8::splat(token, m[2][1]);
    let m22 = f32x8::splat(token, m[2][2]);

    let mut iter = row.chunks_exact_mut(8);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let nr = (m00 * r + m01 * g + m02 * b).to_array();
        let ng = (m10 * r + m11 * g + m12 * b).to_array();
        let nb = (m20 * r + m21 * g + m22 * b).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = nr[i];
            px[1] = ng[i];
            px[2] = nb[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        let r = m[0][0] * px[0] + m[0][1] * px[1] + m[0][2] * px[2];
        let g = m[1][0] * px[0] + m[1][1] * px[1] + m[1][2] * px[2];
        let b = m[2][0] * px[0] + m[2][1] * px[1] + m[2][2] * px[2];
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
}

// ============================================================================
// Gamut: hue-preserving soft clip and out-of-gamut mask.
// ============================================================================

/// Hue-preserving soft clip applied to 8 SOA pixels and the scalar tail.
///
/// Vectorized form of [`crate::gamut::soft_clip`]: max(0, c) on negatives,
/// then if any channel exceeds 1.0, scale all three with a uniform formula
/// derived from the sorted `(hi, mid, lo)`. With `hi = max(r,g,b)`,
/// `lo = min(r,g,b)`, `new_hi = min(hi,1)`, `new_lo = min(lo,1)`, every
/// channel obeys
/// `out = new_lo + (c - lo) * (new_hi - new_lo) / (hi - lo)` which collapses
/// to `out = new_lo` for `c == lo` and `out = new_hi` for `c == hi`. Clamping
/// `lo` is required: when every channel exceeds 1 (e.g. `[1.5, 2.0, 1.2]`)
/// the lane with `c == lo` would otherwise stay at its raw value > 1. The
/// pixels where `hi <= 1` pass through untouched via a lane-wise blend, and
/// the degenerate `hi == lo` case is also handled by the same blend (mapped
/// to `new_hi`).
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
pub(crate) fn soft_clip_tier(token: Token, row: &mut [[f32; 3]]) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    // EPS guards 1.0 / (hi - lo) when hi == lo. The blend below replaces
    // those lanes anyway, so the value chosen only needs to keep the
    // intermediate finite.
    let denom_eps = f32x8::splat(token, f32::EPSILON);

    let mut iter = row.chunks_exact_mut(8);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        // Clamp negatives to 0 first (under-range from gamut matrix).
        let r = f32x8::load(token, &ra).max(zero);
        let g = f32x8::load(token, &ga).max(zero);
        let b = f32x8::load(token, &ba).max(zero);

        let hi = r.max(g).max(b);
        let lo = r.min(g).min(b);
        let new_hi = hi.min(one);
        // Clamp lo too — when every channel exceeds 1 (e.g. `[1.5, 2.0, 1.2]`)
        // an unclamped `lo` leaves the `c == lo` lane at its original value
        // (1.2 in the example), producing out-of-gamut output. The scalar
        // `clip_sorted` clamps `new_lo = lo.min(1.0)` for exactly this reason.
        let new_lo = lo.min(one);
        let denom = hi - lo;
        // Safe denominator — lanes where denom == 0 get replaced by the
        // hi==lo branch below.
        let factor = (new_hi - new_lo) / denom.max(denom_eps);

        // Uniform per-channel formula: out = new_lo + (c - lo) * factor.
        // For c == lo: out = new_lo (clamped). For c == hi: out = new_hi.
        let cr = new_lo + (r - lo) * factor;
        let cg = new_lo + (g - lo) * factor;
        let cb = new_lo + (b - lo) * factor;

        // Where hi == lo (all channels equal-ish), every channel maps to new_hi.
        let eq = denom.simd_le(denom_eps);
        let cr = f32x8::blend(eq, new_hi, cr);
        let cg = f32x8::blend(eq, new_hi, cg);
        let cb = f32x8::blend(eq, new_hi, cb);

        // Where hi <= 1 (in gamut after the negative clamp), pass through.
        let needs = hi.simd_gt(one);
        let or = f32x8::blend(needs, cr, r).to_array();
        let og = f32x8::blend(needs, cg, g).to_array();
        let ob = f32x8::blend(needs, cb, b).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or[i];
            px[1] = og[i];
            px[2] = ob[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        let out = crate::gamut::soft_clip(*px);
        *px = out;
    }
}

/// Per-pixel out-of-gamut mask (1.0 if any channel is outside `[0, 1]`, else 0.0).
///
/// Output lane format: 1.0 / 0.0 floats (caller can reload into `f32x8` and
/// `simd_gt(0.5)` to recover a SIMD mask, or use directly as a scaling factor).
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
pub(crate) fn is_out_of_gamut_mask_tier(token: Token, row: &[[f32; 3]], out: &mut [f32]) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);

    let chunks = row.chunks_exact(8);
    let row_tail = chunks.remainder();
    let n_full = row.len() - row_tail.len();
    let (out_chunked, out_tail) = out.split_at_mut(n_full);

    for (chunk, dst) in chunks.zip(out_chunked.chunks_exact_mut(8)) {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        // any (c < 0 || c > 1)
        let any_lt = r.simd_lt(zero) | g.simd_lt(zero) | b.simd_lt(zero);
        let any_gt = r.simd_gt(one) | g.simd_gt(one) | b.simd_gt(one);
        let mask = any_lt | any_gt;
        let result = f32x8::blend(mask, one, zero);
        let arr = result.to_array();
        let dst8: &mut [f32; 8] = dst.try_into().unwrap();
        *dst8 = arr;
    }
    for (px, dst) in row_tail.iter().zip(out_tail.iter_mut()) {
        *dst = if crate::gamut::is_out_of_gamut(*px) {
            1.0
        } else {
            0.0
        };
    }
}

// ============================================================================
// HLG: chromaticity-preserving and per-channel approx OOTF, both directions.
// ============================================================================

const LR: f32 = 0.2627;
const LG: f32 = 0.6780;
const LB: f32 = 0.0593;

/// Chromaticity-preserving HLG OOTF over 8 SOA pixels and the scalar tail.
///
/// `gamma_minus_one` is `gamma - 1.0` for the forward direction and
/// `(1.0 - gamma) / gamma` for the inverse direction; both share the same
/// kernel shape (luminance, then `pow(Y, k)`, then per-channel multiply).
#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn hlg_ootf_exact_tier(token: Token, row: &mut [[f32; 3]], k: f32) {
    let lr = f32x8::splat(token, LR);
    let lg = f32x8::splat(token, LG);
    let lb = f32x8::splat(token, LB);
    let zero = f32x8::zero(token);
    let pos_eps = f32x8::splat(token, f32::MIN_POSITIVE);

    let mut iter = row.chunks_exact_mut(8);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let y = lr * r + lg * g + lb * b;
        // pow_midp expects positive inputs. Lanes where Y <= 0 are forced to
        // an all-zero pixel below via blend.
        let y_safe = y.max(pos_eps);
        let scale = y_safe.pow_midp(k);
        let nz = y.simd_gt(zero);
        let nr = (r * scale).to_array();
        let ng = (g * scale).to_array();
        let nb = (b * scale).to_array();
        let or_arr = f32x8::blend(nz, f32x8::load(token, &nr), zero).to_array();
        let og_arr = f32x8::blend(nz, f32x8::load(token, &ng), zero).to_array();
        let ob_arr = f32x8::blend(nz, f32x8::load(token, &nb), zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or_arr[i];
            px[1] = og_arr[i];
            px[2] = ob_arr[i];
        }
    }
    let tail = iter.into_remainder();
    // Tail uses the per-pixel reference. The tail kernel takes `gamma`, so we
    // recover it from `k` based on the caller's intent: not enough info
    // without context, so callers pass either `forward_gamma` or `None`.
    // Here we just call the per-pixel `pow_midp` formulation directly so the
    // tail uses the same math as the SIMD chunk.
    for px in tail.iter_mut() {
        let y = LR * px[0] + LG * px[1] + LB * px[2];
        if y <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let scale = crate::math::powf(y, k);
        px[0] *= scale;
        px[1] *= scale;
        px[2] *= scale;
    }
}

/// Per-channel HLG OOTF approx over 8 SOA pixels and the scalar tail.
///
/// `exponent` is `gamma` for the forward direction and `1.0 / gamma` for the
/// inverse. Three per-channel `pow_midp` calls (no luminance coupling).
#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn hlg_ootf_approx_tier(token: Token, row: &mut [[f32; 3]], exponent: f32) {
    let mut iter = row.chunks_exact_mut(8);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        let r = f32x8::load(token, &ra).pow_midp(exponent).to_array();
        let g = f32x8::load(token, &ga).pow_midp(exponent).to_array();
        let b = f32x8::load(token, &ba).pow_midp(exponent).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = r[i];
            px[1] = g[i];
            px[2] = b[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        px[0] = crate::math::powf(px[0], exponent);
        px[1] = crate::math::powf(px[1], exponent);
        px[2] = crate::math::powf(px[2], exponent);
    }
    // The `hlg` import keeps the doc link below resolvable in `cargo doc`.
    let _ = hlg::hlg_ootf_approx;
}
