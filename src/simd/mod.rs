//! SIMD-accelerated tone mapping kernels.
//!
//! AVX2+FMA (x86-64-v3) via `#[archmage::arcane]` with f32x8. Scalar
//! fallback for all other architectures. `archmage::incant!` handles
//! safe runtime dispatch. Respects `#![forbid(unsafe_code)]`.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use magetypes::simd::polyfill::v3::f32x8;

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
    for c in row.chunks_exact_mut(ch) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], look);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn reinhard_scalar(x: f32) -> f32 {
    (x / (1.0 + x)).min(1.0)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[archmage::arcane]
fn aces_3_v3(_t: archmage::X64V3Token, r: &mut [f32]) {
    for c in r.chunks_exact_mut(3) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[archmage::arcane]
fn aces_4_v3(_t: archmage::X64V3Token, r: &mut [f32]) {
    for c in r.chunks_exact_mut(4) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
