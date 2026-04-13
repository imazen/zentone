//! SIMD-accelerated tone mapping kernels.
//!
//! AVX2+FMA (x86-64-v3) via `#[archmage::arcane]` with f32x8. Scalar
//! fallback for all other architectures. `archmage::incant!` handles
//! safe runtime dispatch. Respects `#![forbid(unsafe_code)]`.

// Use generic f32x8 (not polyfill) for transcendental support (log2_lowp, etc.)
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

// ============================================================================
// Public dispatch
// ============================================================================

#[inline]
pub(crate) fn reinhard_simple_row(row: &mut [f32], ch: usize) {
    match ch {
        3 => archmage::incant!(reinhard_3(row)),
        4 => archmage::incant!(reinhard_4(row)),
        _ => {}
    }
}

#[inline]
pub(crate) fn narkowicz_row(row: &mut [f32], ch: usize) {
    match ch {
        3 => archmage::incant!(narkowicz_3(row)),
        4 => archmage::incant!(narkowicz_4(row)),
        _ => {}
    }
}

#[inline]
pub(crate) fn hable_row(row: &mut [f32], ch: usize) {
    match ch {
        3 => archmage::incant!(hable_3(row)),
        4 => archmage::incant!(hable_4(row)),
        _ => {}
    }
}

#[inline]
pub(crate) fn aces_ap1_row(row: &mut [f32], ch: usize) {
    // ACES has cross-channel matrix — process per-pixel.
    // Still benefits from incant! because LLVM auto-vectorizes the
    // rational polynomial within each pixel when target_feature is set.
    match ch {
        3 => archmage::incant!(aces_3(row)),
        4 => archmage::incant!(aces_4(row)),
        _ => {}
    }
}

#[inline]
pub(crate) fn agx_row(row: &mut [f32], ch: usize, look: crate::AgxLook) {
    match look {
        crate::AgxLook::Default => match ch {
            3 => archmage::incant!(agx_default_3(row)),
            4 => archmage::incant!(agx_default_4(row)),
            _ => {}
        },
        // Non-default looks need pow — fall back to per-pixel scalar.
        _ => {
            for c in row.chunks_exact_mut(ch) {
                let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], look);
                c[0] = out[0];
                c[1] = out[1];
                c[2] = out[2];
            }
        }
    }
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

fn reinhard_3_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        c[0] = (c[0] / (1.0 + c[0])).min(1.0);
        c[1] = (c[1] / (1.0 + c[1])).min(1.0);
        c[2] = (c[2] / (1.0 + c[2])).min(1.0);
    }
}
fn reinhard_4_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        c[0] = (c[0] / (1.0 + c[0])).min(1.0);
        c[1] = (c[1] / (1.0 + c[1])).min(1.0);
        c[2] = (c[2] / (1.0 + c[2])).min(1.0);
    }
}
fn narkowicz_3_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        c[0] = crate::curves::filmic_narkowicz(c[0]);
        c[1] = crate::curves::filmic_narkowicz(c[1]);
        c[2] = crate::curves::filmic_narkowicz(c[2]);
    }
}
fn narkowicz_4_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        c[0] = crate::curves::filmic_narkowicz(c[0]);
        c[1] = crate::curves::filmic_narkowicz(c[1]);
        c[2] = crate::curves::filmic_narkowicz(c[2]);
    }
}
fn hable_3_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        c[0] = crate::curves::hable_filmic(c[0]);
        c[1] = crate::curves::hable_filmic(c[1]);
        c[2] = crate::curves::hable_filmic(c[2]);
    }
}
fn hable_4_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        c[0] = crate::curves::hable_filmic(c[0]);
        c[1] = crate::curves::hable_filmic(c[1]);
        c[2] = crate::curves::hable_filmic(c[2]);
    }
}
fn agx_default_3_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], crate::AgxLook::Default);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}
fn agx_default_4_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], crate::AgxLook::Default);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}
fn aces_3_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
fn aces_4_scalar(_t: archmage::ScalarToken, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

// ============================================================================
// AVX2+FMA via #[arcane]
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn reinhard_scalar(x: f32) -> f32 {
    (x / (1.0 + x)).min(1.0)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn reinhard_3_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let ones = f32x8::splat(t, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let v = f32x8::load(t, chunk);
        (v / (ones + v)).min(ones).store(chunk);
    }
    for v in tail.iter_mut() {
        *v = reinhard_scalar(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn reinhard_4_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let ones = f32x8::splat(t, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let a0 = chunk[3];
        let a1 = chunk[7];
        let v = f32x8::load(t, chunk);
        (v / (ones + v)).min(ones).store(chunk);
        chunk[3] = a0;
        chunk[7] = a1;
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = reinhard_scalar(c[0]);
        c[1] = reinhard_scalar(c[1]);
        c[2] = reinhard_scalar(c[2]);
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn narkowicz_3_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let a = f32x8::splat(t, 2.51);
    let b = f32x8::splat(t, 0.03);
    let c = f32x8::splat(t, 2.43);
    let d = f32x8::splat(t, 0.59);
    let e = f32x8::splat(t, 0.14);
    let z = f32x8::splat(t, 0.0);
    let o = f32x8::splat(t, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let x = f32x8::load(t, chunk);
        (x * (a * x + b) / (x * (c * x + d) + e))
            .max(z)
            .min(o)
            .store(chunk);
    }
    for v in tail.iter_mut() {
        *v = crate::curves::filmic_narkowicz(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn narkowicz_4_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let a = f32x8::splat(t, 2.51);
    let b = f32x8::splat(t, 0.03);
    let c = f32x8::splat(t, 2.43);
    let d = f32x8::splat(t, 0.59);
    let e = f32x8::splat(t, 0.14);
    let z = f32x8::splat(t, 0.0);
    let o = f32x8::splat(t, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let a0 = chunk[3];
        let a1 = chunk[7];
        let x = f32x8::load(t, chunk);
        (x * (a * x + b) / (x * (c * x + d) + e))
            .max(z)
            .min(o)
            .store(chunk);
        chunk[3] = a0;
        chunk[7] = a1;
    }
    for cv in tail.chunks_exact_mut(4) {
        cv[0] = crate::curves::filmic_narkowicz(cv[0]);
        cv[1] = crate::curves::filmic_narkowicz(cv[1]);
        cv[2] = crate::curves::filmic_narkowicz(cv[2]);
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn hable_3_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let a = f32x8::splat(t, 0.15);
    let bv = f32x8::splat(t, 0.50);
    let cb = f32x8::splat(t, 0.05);
    let de = f32x8::splat(t, 0.004);
    let df = f32x8::splat(t, 0.06);
    let ef = f32x8::splat(t, 0.02 / 0.30);
    let exp = f32x8::splat(t, 2.0);
    let ws = f32x8::splat(t, {
        const fn p(x: f32) -> f32 {
            ((x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.50) + 0.06)) - 0.02 / 0.30
        }
        1.0 / p(11.2)
    });
    let o = f32x8::splat(t, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let x = f32x8::load(t, chunk) * exp;
        let r = ((x * (a * x + cb) + de) / (x * (a * x + bv) + df) - ef) * ws;
        r.min(o).store(chunk);
    }
    for v in tail.iter_mut() {
        *v = crate::curves::hable_filmic(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn hable_4_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let a = f32x8::splat(t, 0.15);
    let bv = f32x8::splat(t, 0.50);
    let cb = f32x8::splat(t, 0.05);
    let de = f32x8::splat(t, 0.004);
    let df = f32x8::splat(t, 0.06);
    let ef = f32x8::splat(t, 0.02 / 0.30);
    let exp = f32x8::splat(t, 2.0);
    let ws = f32x8::splat(t, {
        const fn p(x: f32) -> f32 {
            ((x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.50) + 0.06)) - 0.02 / 0.30
        }
        1.0 / p(11.2)
    });
    let o = f32x8::splat(t, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let a0 = chunk[3];
        let a1 = chunk[7];
        let x = f32x8::load(t, chunk) * exp;
        let r = ((x * (a * x + cb) + de) / (x * (a * x + bv) + df) - ef) * ws;
        r.min(o).store(chunk);
        chunk[3] = a0;
        chunk[7] = a1;
    }
    for cv in tail.chunks_exact_mut(4) {
        cv[0] = crate::curves::hable_filmic(cv[0]);
        cv[1] = crate::curves::hable_filmic(cv[1]);
        cv[2] = crate::curves::hable_filmic(cv[2]);
    }
}

// ACES: cross-channel (matrix), dispatched per-pixel but still benefits
// from #[arcane] auto-vectorization of the rational polynomial.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn aces_3_v3(_t: archmage::X64V3Token, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn aces_4_v3(_t: archmage::X64V3Token, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

// ============================================================================
// AgX Default — 8-pixel SOA kernel with vectorized log2 + polynomial
// ============================================================================

/// Process 8 RGB pixels in SOA layout through the AgX Default pipeline.
/// Gather interleaved RGB → 3 × f32x8, compute, scatter back.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn agx_default_3_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    // Process 8 pixels (24 floats) at a time
    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        // Gather: deinterleave RGB → R[0..7], G[0..7], B[0..7]
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 3].abs();
            ga[i] = chunk[i * 3 + 1].abs();
            ba[i] = chunk[i * 3 + 2].abs();
        }
        let (r, g, b) = agx_default_8px(t, &ra, &ga, &ba);
        // Scatter: interleave back
        let ro = r.to_array();
        let go = g.to_array();
        let bo = b.to_array();
        for i in 0..8 {
            chunk[i * 3] = ro[i];
            chunk[i * 3 + 1] = go[i];
            chunk[i * 3 + 2] = bo[i];
        }
    }
    // Tail: scalar fallback
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], crate::AgxLook::Default);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn agx_default_4_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    // Process 8 RGBA pixels (32 floats) at a time
    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 4].abs();
            ga[i] = chunk[i * 4 + 1].abs();
            ba[i] = chunk[i * 4 + 2].abs();
        }
        let (r, g, b) = agx_default_8px(t, &ra, &ga, &ba);
        let ro = r.to_array();
        let go = g.to_array();
        let bo = b.to_array();
        for i in 0..8 {
            chunk[i * 4] = ro[i];
            chunk[i * 4 + 1] = go[i];
            chunk[i * 4 + 2] = bo[i];
            // Alpha at chunk[i*4+3] is preserved (untouched).
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(4) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], crate::AgxLook::Default);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}

/// Core AgX Default pipeline on 8 pixels (SOA: R, G, B as f32x8).
/// Returns (R_out, G_out, B_out) clamped to [0, 1].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn agx_default_8px(
    t: archmage::X64V3Token,
    ra: &[f32; 8],
    ga: &[f32; 8],
    ba: &[f32; 8],
) -> (f32x8, f32x8, f32x8) {
    const AGX_MIN_EV: f32 = -12.47393;
    const AGX_MAX_EV: f32 = 4.026069;
    const RECIP_EV: f32 = 1.0 / (AGX_MAX_EV - AGX_MIN_EV);

    let r = f32x8::load(t, ra);
    let g = f32x8::load(t, ga);
    let b = f32x8::load(t, ba);

    // Inset matrix (3x3 multiply, each output channel is a dot product)
    let m00 = f32x8::splat(t, 0.856627153315983);
    let m01 = f32x8::splat(t, 0.137318972929847);
    let m02 = f32x8::splat(t, 0.11189821299995);
    let m10 = f32x8::splat(t, 0.0951212405381588);
    let m11 = f32x8::splat(t, 0.761241990602591);
    let m12 = f32x8::splat(t, 0.0767994186031903);
    let m20 = f32x8::splat(t, 0.0482516061458583);
    let m21 = f32x8::splat(t, 0.101439036467562);
    let m22 = f32x8::splat(t, 0.811302368396859);

    let z0r = m00 * r + m01 * g + m02 * b;
    let z0g = m10 * r + m11 * g + m12 * b;
    let z0b = m20 * r + m21 * g + m22 * b;

    // log2(max(x, 1e-10)), clamped to [AGX_MIN_EV, AGX_MAX_EV]
    let floor = f32x8::splat(t, 1e-10);
    let min_ev = f32x8::splat(t, AGX_MIN_EV);
    let max_ev = f32x8::splat(t, AGX_MAX_EV);

    let z1r = z0r.max(floor).log2_midp().max(min_ev).min(max_ev);
    let z1g = z0g.max(floor).log2_midp().max(min_ev).min(max_ev);
    let z1b = z0b.max(floor).log2_midp().max(min_ev).min(max_ev);

    // Normalize to [0, 1]
    let recip = f32x8::splat(t, RECIP_EV);
    let z2r = (z1r - min_ev) * recip;
    let z2g = (z1g - min_ev) * recip;
    let z2b = (z1b - min_ev) * recip;

    // Polynomial contrast (degree 7, Horner-style with even powers)
    let z3r = agx_contrast_v8(t, z2r);
    let z3g = agx_contrast_v8(t, z2g);
    let z3b = agx_contrast_v8(t, z2b);

    // Outset matrix (3x3 multiply)
    let o00 = f32x8::splat(t, 1.19687900512017);
    let o01 = f32x8::splat(t, -0.0528968517574562);
    let o02 = f32x8::splat(t, -0.0529716355144438);
    let o10 = f32x8::splat(t, -0.0980208811401368);
    let o11 = f32x8::splat(t, 1.15190312990417);
    let o12 = f32x8::splat(t, -0.0505349770312032);
    let o20 = f32x8::splat(t, -0.0990297440797205);
    let o21 = f32x8::splat(t, -0.0989611768448433);
    let o22 = f32x8::splat(t, 1.15107367264116);

    let zero = f32x8::splat(t, 0.0);
    let one = f32x8::splat(t, 1.0);

    let or = (o00 * z3r + o01 * z3g + o02 * z3b).max(zero).min(one);
    let og = (o10 * z3r + o11 * z3g + o12 * z3b).max(zero).min(one);
    let ob = (o20 * z3r + o21 * z3g + o22 * z3b).max(zero).min(one);

    (or, og, ob)
}

/// Vectorized AgX contrast polynomial: degree 7, even-power decomposition.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn agx_contrast_v8(t: archmage::X64V3Token, x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;

    let c0 = f32x8::splat(t, 0.002857);
    let c1 = f32x8::splat(t, -0.1718);
    let c2 = f32x8::splat(t, 4.361);
    let c3 = f32x8::splat(t, -28.72);
    let c4 = f32x8::splat(t, 92.06);
    let c5 = f32x8::splat(t, -126.7);
    let c6 = f32x8::splat(t, 78.01);
    let c7 = f32x8::splat(t, -17.86);

    // w0 = 0.002857 - 0.1718*x
    // w1 = 4.361 - 28.72*x
    // w2 = 92.06 - 126.7*x
    // w3 = 78.01 - 17.86*x
    // result = w0 + w1*x² + w2*x⁴ + w3*x⁶
    let w0 = c0 + c1 * x;
    let w1 = c2 + c3 * x;
    let w2 = c4 + c5 * x;
    let w3 = c6 + c7 * x;

    w0 + w1 * x2 + w2 * x4 + w3 * x6
}
