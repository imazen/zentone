//! SIMD-accelerated tone mapping kernels.
//!
//! Uses `archmage::incant!` for safe runtime dispatch to AVX2+FMA on x86
//! (8-wide f32x8), with scalar fallback on all other architectures.
//! Respects `#![forbid(unsafe_code)]`.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use magetypes::simd::polyfill::v3::f32x8;

/// SIMD Reinhard simple row: `x / (1 + x)` per channel.
#[inline]
pub(crate) fn reinhard_simple_row(row: &mut [f32], channels: usize) {
    match channels {
        3 => archmage::incant!(reinhard_3(row)),
        4 => archmage::incant!(reinhard_4(row)),
        _ => {}
    }
}

/// SIMD Narkowicz filmic row.
#[inline]
pub(crate) fn narkowicz_row(row: &mut [f32], channels: usize) {
    match channels {
        3 => archmage::incant!(narkowicz_3(row)),
        4 => archmage::incant!(narkowicz_4(row)),
        _ => {}
    }
}

/// SIMD Hable filmic row.
#[inline]
pub(crate) fn hable_row(row: &mut [f32], channels: usize) {
    match channels {
        3 => archmage::incant!(hable_3(row)),
        4 => archmage::incant!(hable_4(row)),
        _ => {}
    }
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

fn reinhard_3_scalar(_t: archmage::ScalarToken, row: &mut [f32]) {
    for c in row.chunks_exact_mut(3) {
        c[0] = (c[0] / (1.0 + c[0])).min(1.0);
        c[1] = (c[1] / (1.0 + c[1])).min(1.0);
        c[2] = (c[2] / (1.0 + c[2])).min(1.0);
    }
}
fn reinhard_4_scalar(_t: archmage::ScalarToken, row: &mut [f32]) {
    for c in row.chunks_exact_mut(4) {
        c[0] = (c[0] / (1.0 + c[0])).min(1.0);
        c[1] = (c[1] / (1.0 + c[1])).min(1.0);
        c[2] = (c[2] / (1.0 + c[2])).min(1.0);
    }
}

fn narkowicz_3_scalar(_t: archmage::ScalarToken, row: &mut [f32]) {
    for c in row.chunks_exact_mut(3) {
        c[0] = crate::curves::filmic_narkowicz(c[0]);
        c[1] = crate::curves::filmic_narkowicz(c[1]);
        c[2] = crate::curves::filmic_narkowicz(c[2]);
    }
}
fn narkowicz_4_scalar(_t: archmage::ScalarToken, row: &mut [f32]) {
    for c in row.chunks_exact_mut(4) {
        c[0] = crate::curves::filmic_narkowicz(c[0]);
        c[1] = crate::curves::filmic_narkowicz(c[1]);
        c[2] = crate::curves::filmic_narkowicz(c[2]);
    }
}

fn hable_3_scalar(_t: archmage::ScalarToken, row: &mut [f32]) {
    for c in row.chunks_exact_mut(3) {
        c[0] = crate::curves::hable_filmic(c[0]);
        c[1] = crate::curves::hable_filmic(c[1]);
        c[2] = crate::curves::hable_filmic(c[2]);
    }
}
fn hable_4_scalar(_t: archmage::ScalarToken, row: &mut [f32]) {
    for c in row.chunks_exact_mut(4) {
        c[0] = crate::curves::hable_filmic(c[0]);
        c[1] = crate::curves::hable_filmic(c[1]);
        c[2] = crate::curves::hable_filmic(c[2]);
    }
}

// ============================================================================
// AVX2+FMA (x86-64-v3) — process 8 contiguous f32s at a time
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[archmage::arcane]
fn reinhard_3_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let ones = f32x8::splat(t, 1.0);
    // For RGB (CN=3), process 8 f32s at a time (not pixel-aligned but
    // mathematically correct since reinhard is per-channel).
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        let v = f32x8::load(t, chunk);
        (v / (ones + v)).min(ones).store(chunk);
    }
    for v in tail.iter_mut() {
        *v = (*v / (1.0 + *v)).min(1.0);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[archmage::arcane]
fn reinhard_4_v3(t: archmage::X64V3Token, row: &mut [f32]) {
    let ones = f32x8::splat(t, 1.0);
    // For RGBA, save alpha every 4th element, process all, restore alpha.
    let (chunks, tail) = f32x8::partition_slice_mut(t, row);
    for chunk in chunks.iter_mut() {
        // In a chunk of 8 f32s with CN=4: positions 3 and 7 are alpha
        let a0 = chunk[3];
        let a1 = chunk[7];
        let v = f32x8::load(t, chunk);
        (v / (ones + v)).min(ones).store(chunk);
        chunk[3] = a0;
        chunk[7] = a1;
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = (c[0] / (1.0 + c[0])).min(1.0);
        c[1] = (c[1] / (1.0 + c[1])).min(1.0);
        c[2] = (c[2] / (1.0 + c[2])).min(1.0);
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
        let num = x * (a * x + b);
        let den = x * (c * x + d) + e;
        (num / den).max(z).min(o).store(chunk);
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
        let num = x * (a * x + b);
        let den = x * (c * x + d) + e;
        (num / den).max(z).min(o).store(chunk);
        chunk[3] = a0;
        chunk[7] = a1;
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = crate::curves::filmic_narkowicz(c[0]);
        c[1] = crate::curves::filmic_narkowicz(c[1]);
        c[2] = crate::curves::filmic_narkowicz(c[2]);
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
        let num = x * (a * x + cb) + de;
        let den = x * (a * x + bv) + df;
        let r = ((num / den) - ef) * ws;
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
        let num = x * (a * x + cb) + de;
        let den = x * (a * x + bv) + df;
        let r = ((num / den) - ef) * ws;
        r.min(o).store(chunk);
        chunk[3] = a0;
        chunk[7] = a1;
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = crate::curves::hable_filmic(c[0]);
        c[1] = crate::curves::hable_filmic(c[1]);
        c[2] = crate::curves::hable_filmic(c[2]);
    }
}
