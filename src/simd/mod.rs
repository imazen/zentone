//! SIMD-accelerated tone mapping kernels.
//!
//! Each curve has a single `#[archmage::magetypes(define(f32x8), ...)]`
//! kernel that the macro expands into per-tier variants (AVX-512, AVX2+FMA,
//! NEON, WASM128, scalar). `archmage::incant!` selects the best tier at
//! runtime. Respects `#![forbid(unsafe_code)]`.

use crate::ToneMap;

use archmage::incant;

// ============================================================================
// Public dispatch — match on channel count, route to a magetypes _tier fn.
// Symbol names and signatures are part of the crate-internal API and must
// not change here.
// ============================================================================

#[inline]
pub(crate) fn reinhard_simple_row(row: &mut [f32], ch: usize) {
    match ch {
        3 => incant!(reinhard_3_tier(row), [v4, v3, neon, wasm128, scalar]),
        4 => incant!(reinhard_4_tier(row), [v4, v3, neon, wasm128, scalar]),
        _ => {}
    }
}

#[inline]
pub(crate) fn narkowicz_row(row: &mut [f32], ch: usize) {
    match ch {
        3 => incant!(narkowicz_3_tier(row), [v4, v3, neon, wasm128, scalar]),
        4 => incant!(narkowicz_4_tier(row), [v4, v3, neon, wasm128, scalar]),
        _ => {}
    }
}

#[inline]
pub(crate) fn hable_row(row: &mut [f32], ch: usize) {
    match ch {
        3 => incant!(hable_3_tier(row), [v4, v3, neon, wasm128, scalar]),
        4 => incant!(hable_4_tier(row), [v4, v3, neon, wasm128, scalar]),
        _ => {}
    }
}

#[inline]
pub(crate) fn aces_ap1_row(row: &mut [f32], ch: usize) {
    // ACES is a cross-channel matrix → per-pixel; the tier's target_feature
    // context still lets LLVM auto-vectorize the rational polynomial.
    match ch {
        3 => incant!(aces_3_tier(row), [v4, v3, neon, wasm128, scalar]),
        4 => incant!(aces_4_tier(row), [v4, v3, neon, wasm128, scalar]),
        _ => {}
    }
}

#[inline]
pub(crate) fn agx_row(row: &mut [f32], ch: usize, look: crate::AgxLook) {
    // Encode look as (slope, power, saturation) tuples for SIMD dispatch.
    // None = Default (no look), Some(_) carries Punchy/Golden parameters.
    let params: Option<([f32; 3], [f32; 3], [f32; 3])> = match look {
        crate::AgxLook::Default => None,
        crate::AgxLook::Punchy => Some(([1.0, 1.0, 1.0], [1.35, 1.35, 1.35], [1.4, 1.4, 1.4])),
        crate::AgxLook::Golden => Some(([1.0, 0.9, 0.5], [0.8, 0.8, 0.8], [1.3, 1.3, 1.3])),
    };
    // AgX kernel uses f32x8::log2_midp / pow_midp which require F32x8Convert.
    // V4 (AVX-512) doesn't implement F32x8Convert for the f32x8 type, so the
    // chunk-of-8 SIMD path runs on V3 (AVX2). AVX-512 CPUs also support AVX2,
    // so this only changes which tier the dispatcher picks — never user-
    // visible. The `agx_*_tier` wrappers do the gather/scatter outside the
    // tier context and call the magetypes-generated kernel via `incant!`.
    match ch {
        3 => agx_3_tier(row, params),
        4 => agx_4_tier(row, params),
        _ => {}
    }
}

#[inline]
pub(crate) fn ext_reinhard_row(row: &mut [f32], ch: usize, l_max: f32, luma: [f32; 3]) {
    match ch {
        3 => incant!(
            ext_reinhard_3_tier(row, l_max, luma),
            [v4, v3, neon, wasm128, scalar]
        ),
        4 => incant!(
            ext_reinhard_4_tier(row, l_max, luma),
            [v4, v3, neon, wasm128, scalar]
        ),
        _ => {}
    }
}

#[inline]
pub(crate) fn reinhard_jodie_row(row: &mut [f32], ch: usize, luma: [f32; 3]) {
    match ch {
        3 => incant!(
            reinhard_jodie_3_tier(row, luma),
            [v4, v3, neon, wasm128, scalar]
        ),
        4 => incant!(
            reinhard_jodie_4_tier(row, luma),
            [v4, v3, neon, wasm128, scalar]
        ),
        _ => {}
    }
}

#[inline]
pub(crate) fn tuned_reinhard_row(
    row: &mut [f32],
    ch: usize,
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    match ch {
        3 => incant!(
            tuned_reinhard_3_tier(row, content_max, display_max, luma),
            [v4, v3, neon, wasm128, scalar]
        ),
        4 => incant!(
            tuned_reinhard_4_tier(row, content_max, display_max, luma),
            [v4, v3, neon, wasm128, scalar]
        ),
        _ => {}
    }
}

// ============================================================================
// Reinhard simple — splat-mul-fma kernel, RGB and RGBA.
// ============================================================================

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn reinhard_3_tier(token: Token, row: &mut [f32]) {
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);
    let (chunks, tail) = f32x8::partition_slice_mut(token, row);
    for chunk in chunks.iter_mut() {
        let v = f32x8::load(token, chunk).max(zero);
        (v / (one + v)).min(one).store(chunk);
    }
    for v in tail.iter_mut() {
        *v = crate::curves::reinhard_simple(*v);
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn reinhard_4_tier(token: Token, row: &mut [f32]) {
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);
    let (chunks, tail) = f32x8::partition_slice_mut(token, row);
    for chunk in chunks.iter_mut() {
        let a = [chunk[3], chunk[7]];
        let v = f32x8::load(token, chunk).max(zero);
        (v / (one + v)).min(one).store(chunk);
        chunk[3] = a[0];
        chunk[7] = a[1];
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = crate::curves::reinhard_simple(c[0]);
        c[1] = crate::curves::reinhard_simple(c[1]);
        c[2] = crate::curves::reinhard_simple(c[2]);
    }
}

// ============================================================================
// Narkowicz — `x * (a*x + b) / (x * (c*x + d) + e)`, RGB and RGBA.
// ============================================================================

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn narkowicz_3_tier(token: Token, row: &mut [f32]) {
    let a = f32x8::splat(token, 2.51);
    let b = f32x8::splat(token, 0.03);
    let c = f32x8::splat(token, 2.43);
    let d = f32x8::splat(token, 0.59);
    let e = f32x8::splat(token, 0.14);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(token, row);
    for chunk in chunks.iter_mut() {
        let x = f32x8::load(token, chunk);
        (x * (a * x + b) / (x * (c * x + d) + e))
            .max(zero)
            .min(one)
            .store(chunk);
    }
    for v in tail.iter_mut() {
        *v = crate::curves::filmic_narkowicz(*v);
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn narkowicz_4_tier(token: Token, row: &mut [f32]) {
    let a = f32x8::splat(token, 2.51);
    let b = f32x8::splat(token, 0.03);
    let c = f32x8::splat(token, 2.43);
    let d = f32x8::splat(token, 0.59);
    let e = f32x8::splat(token, 0.14);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(token, row);
    for chunk in chunks.iter_mut() {
        let alphas = [chunk[3], chunk[7]];
        let x = f32x8::load(token, chunk);
        (x * (a * x + b) / (x * (c * x + d) + e))
            .max(zero)
            .min(one)
            .store(chunk);
        chunk[3] = alphas[0];
        chunk[7] = alphas[1];
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = crate::curves::filmic_narkowicz(c[0]);
        c[1] = crate::curves::filmic_narkowicz(c[1]);
        c[2] = crate::curves::filmic_narkowicz(c[2]);
    }
}

// ============================================================================
// Hable filmic — rational polynomial with white-point scale, RGB and RGBA.
// ============================================================================

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn hable_3_tier(token: Token, row: &mut [f32]) {
    let a = f32x8::splat(token, 0.15);
    let bv = f32x8::splat(token, 0.50);
    let cb = f32x8::splat(token, 0.05);
    let de = f32x8::splat(token, 0.004);
    let df = f32x8::splat(token, 0.06);
    let ef = f32x8::splat(token, 0.02 / 0.30);
    let exp = f32x8::splat(token, 2.0);
    let ws = f32x8::splat(token, {
        const fn p(x: f32) -> f32 {
            ((x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.50) + 0.06)) - 0.02 / 0.30
        }
        1.0 / p(11.2)
    });
    let one = f32x8::splat(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(token, row);
    for chunk in chunks.iter_mut() {
        let x = f32x8::load(token, chunk) * exp;
        let r = ((x * (a * x + cb) + de) / (x * (a * x + bv) + df) - ef) * ws;
        r.min(one).store(chunk);
    }
    for v in tail.iter_mut() {
        *v = crate::curves::hable_filmic(*v);
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn hable_4_tier(token: Token, row: &mut [f32]) {
    let a = f32x8::splat(token, 0.15);
    let bv = f32x8::splat(token, 0.50);
    let cb = f32x8::splat(token, 0.05);
    let de = f32x8::splat(token, 0.004);
    let df = f32x8::splat(token, 0.06);
    let ef = f32x8::splat(token, 0.02 / 0.30);
    let exp = f32x8::splat(token, 2.0);
    let ws = f32x8::splat(token, {
        const fn p(x: f32) -> f32 {
            ((x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.50) + 0.06)) - 0.02 / 0.30
        }
        1.0 / p(11.2)
    });
    let one = f32x8::splat(token, 1.0);
    let (chunks, tail) = f32x8::partition_slice_mut(token, row);
    for chunk in chunks.iter_mut() {
        let alphas = [chunk[3], chunk[7]];
        let x = f32x8::load(token, chunk) * exp;
        let r = ((x * (a * x + cb) + de) / (x * (a * x + bv) + df) - ef) * ws;
        r.min(one).store(chunk);
        chunk[3] = alphas[0];
        chunk[7] = alphas[1];
    }
    for c in tail.chunks_exact_mut(4) {
        c[0] = crate::curves::hable_filmic(c[0]);
        c[1] = crate::curves::hable_filmic(c[1]);
        c[2] = crate::curves::hable_filmic(c[2]);
    }
}

// ============================================================================
// ACES AP1 — cross-channel matrix, processed per-pixel. Lifted into the
// tier-specific target_feature context so LLVM auto-vectorizes the polynomial
// inside aces_ap1.
// ============================================================================

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn aces_3_tier(_token: Token, row: &mut [f32]) {
    for c in row.chunks_exact_mut(3) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn aces_4_tier(_token: Token, row: &mut [f32]) {
    for c in row.chunks_exact_mut(4) {
        let o = crate::curves::aces_ap1([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

// ============================================================================
// Luma-based Reinhard variants — 8-pixel SOA gather/scatter.
// ============================================================================

/// Extended Reinhard: luma-preserving with white point.
/// scale = (1 + l/l_max²) / (1 + l), applied to RGB.
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn ext_reinhard_3_tier(token: Token, row: &mut [f32], l_max: f32, luma: [f32; 3]) {
    let lr = f32x8::splat(token, luma[0]);
    let lg = f32x8::splat(token, luma[1]);
    let lb = f32x8::splat(token, luma[2]);
    let lmax_sq = f32x8::splat(token, l_max * l_max);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 3];
            ga[i] = chunk[i * 3 + 1];
            ba[i] = chunk[i * 3 + 2];
        }
        let r = f32x8::load(token, &ra).max(zero);
        let g = f32x8::load(token, &ga).max(zero);
        let b = f32x8::load(token, &ba).max(zero);
        let l = r * lr + g * lg + b * lb;
        let scale = ((one + l / lmax_sq) / (one + l)).max(zero);
        let ro = (r * scale).min(one).to_array();
        let go = (g * scale).min(one).to_array();
        let bo = (b * scale).min(one).to_array();
        for i in 0..8 {
            chunk[i * 3] = ro[i];
            chunk[i * 3 + 1] = go[i];
            chunk[i * 3 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let o = crate::curves::ToneMapCurve::ExtendedReinhard { l_max, luma }
            .map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn ext_reinhard_4_tier(token: Token, row: &mut [f32], l_max: f32, luma: [f32; 3]) {
    let lr = f32x8::splat(token, luma[0]);
    let lg = f32x8::splat(token, luma[1]);
    let lb = f32x8::splat(token, luma[2]);
    let lmax_sq = f32x8::splat(token, l_max * l_max);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 4];
            ga[i] = chunk[i * 4 + 1];
            ba[i] = chunk[i * 4 + 2];
        }
        let r = f32x8::load(token, &ra).max(zero);
        let g = f32x8::load(token, &ga).max(zero);
        let b = f32x8::load(token, &ba).max(zero);
        let l = r * lr + g * lg + b * lb;
        let scale = ((one + l / lmax_sq) / (one + l)).max(zero);
        let ro = (r * scale).min(one).to_array();
        let go = (g * scale).min(one).to_array();
        let bo = (b * scale).min(one).to_array();
        for i in 0..8 {
            chunk[i * 4] = ro[i];
            chunk[i * 4 + 1] = go[i];
            chunk[i * 4 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(4) {
        let o = crate::curves::ToneMapCurve::ExtendedReinhard { l_max, luma }
            .map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

/// Reinhard Jodie: per-channel Reinhard blended with luma-based Reinhard.
/// out = (1-tv) * (rgb * luma_scale) + tv², tv = rgb/(1+rgb)
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn reinhard_jodie_3_tier(token: Token, row: &mut [f32], luma: [f32; 3]) {
    let lr = f32x8::splat(token, luma[0]);
    let lg = f32x8::splat(token, luma[1]);
    let lb = f32x8::splat(token, luma[2]);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 3];
            ga[i] = chunk[i * 3 + 1];
            ba[i] = chunk[i * 3 + 2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let l = r * lr + g * lg + b * lb;
        let luma_scale = one / (one + l);
        let tvr = r / (one + r);
        let tvg = g / (one + g);
        let tvb = b / (one + b);
        let ro = (((one - tvr) * (r * luma_scale) + tvr * tvr)
            .min(one)
            .max(zero))
        .to_array();
        let go = (((one - tvg) * (g * luma_scale) + tvg * tvg)
            .min(one)
            .max(zero))
        .to_array();
        let bo = (((one - tvb) * (b * luma_scale) + tvb * tvb)
            .min(one)
            .max(zero))
        .to_array();
        for i in 0..8 {
            chunk[i * 3] = ro[i];
            chunk[i * 3 + 1] = go[i];
            chunk[i * 3 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let o = crate::curves::reinhard_jodie([c[0], c[1], c[2]], luma);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn reinhard_jodie_4_tier(token: Token, row: &mut [f32], luma: [f32; 3]) {
    let lr = f32x8::splat(token, luma[0]);
    let lg = f32x8::splat(token, luma[1]);
    let lb = f32x8::splat(token, luma[2]);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 4];
            ga[i] = chunk[i * 4 + 1];
            ba[i] = chunk[i * 4 + 2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let l = r * lr + g * lg + b * lb;
        let luma_scale = one / (one + l);
        let tvr = r / (one + r);
        let tvg = g / (one + g);
        let tvb = b / (one + b);
        let ro = (((one - tvr) * (r * luma_scale) + tvr * tvr)
            .min(one)
            .max(zero))
        .to_array();
        let go = (((one - tvg) * (g * luma_scale) + tvg * tvg)
            .min(one)
            .max(zero))
        .to_array();
        let bo = (((one - tvb) * (b * luma_scale) + tvb * tvb)
            .min(one)
            .max(zero))
        .to_array();
        for i in 0..8 {
            chunk[i * 4] = ro[i];
            chunk[i * 4 + 1] = go[i];
            chunk[i * 4 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(4) {
        let o = crate::curves::reinhard_jodie([c[0], c[1], c[2]], luma);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

/// Tuned Reinhard: display-aware with content/display peak.
/// scale = (1 + w_a * l) / (1 + w_b * l), applied to RGB.
#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn tuned_reinhard_3_tier(
    token: Token,
    row: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    let white_point = 203.0_f32;
    let ld = content_max / white_point;
    let lr = f32x8::splat(token, luma[0]);
    let lg = f32x8::splat(token, luma[1]);
    let lb = f32x8::splat(token, luma[2]);
    let w_a = f32x8::splat(token, (display_max / white_point) / (ld * ld));
    let w_b = f32x8::splat(token, 1.0 / (display_max / white_point));
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 3];
            ga[i] = chunk[i * 3 + 1];
            ba[i] = chunk[i * 3 + 2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let l = r * lr + g * lg + b * lb;
        let scale = ((one + w_a * l) / (one + w_b * l)).max(zero);
        let ro = (r * scale).min(one).to_array();
        let go = (g * scale).min(one).to_array();
        let bo = (b * scale).min(one).to_array();
        for i in 0..8 {
            chunk[i * 3] = ro[i];
            chunk[i * 3 + 1] = go[i];
            chunk[i * 3 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let curve = crate::curves::ToneMapCurve::TunedReinhard {
            content_max_nits: content_max,
            display_max_nits: display_max,
            luma,
        };
        let o = curve.map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn tuned_reinhard_4_tier(
    token: Token,
    row: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    let white_point = 203.0_f32;
    let ld = content_max / white_point;
    let lr = f32x8::splat(token, luma[0]);
    let lg = f32x8::splat(token, luma[1]);
    let lb = f32x8::splat(token, luma[2]);
    let w_a = f32x8::splat(token, (display_max / white_point) / (ld * ld));
    let w_b = f32x8::splat(token, 1.0 / (display_max / white_point));
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 4];
            ga[i] = chunk[i * 4 + 1];
            ba[i] = chunk[i * 4 + 2];
        }
        let r = f32x8::load(token, &ra);
        let g = f32x8::load(token, &ga);
        let b = f32x8::load(token, &ba);
        let l = r * lr + g * lg + b * lb;
        let scale = ((one + w_a * l) / (one + w_b * l)).max(zero);
        let ro = (r * scale).min(one).to_array();
        let go = (g * scale).min(one).to_array();
        let bo = (b * scale).min(one).to_array();
        for i in 0..8 {
            chunk[i * 4] = ro[i];
            chunk[i * 4 + 1] = go[i];
            chunk[i * 4 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(4) {
        let curve = crate::curves::ToneMapCurve::TunedReinhard {
            content_max_nits: content_max,
            display_max_nits: display_max,
            luma,
        };
        let o = curve.map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

// ============================================================================
// AgX — 8-pixel SOA kernel with vectorized log2 + polynomial + optional look.
//
// Note: AgX uses `f32x8::log2_midp` and `f32x8::pow_midp` which require the
// `F32x8Convert` trait. AVX-512 (V4) doesn't implement that for the f32x8
// type, so AgX skips V4 and runs on V3 (AVX2) on x86. AVX-512-capable CPUs
// still support AVX2, so this is a no-op on the user side — the dispatcher
// just picks the V3 path.
// ============================================================================

/// Recover the AgxLook from optional params (used by the scalar tail loop).
fn params_to_look(params: Option<([f32; 3], [f32; 3], [f32; 3])>) -> crate::AgxLook {
    match params {
        None => crate::AgxLook::Default,
        Some(([1.0, 1.0, 1.0], [1.35, 1.35, 1.35], [1.4, 1.4, 1.4])) => crate::AgxLook::Punchy,
        Some(_) => crate::AgxLook::Golden,
    }
}

/// Process 8 SOA RGB pixels through the full AgX pipeline:
/// inset → log2 → contrast → look → outset → clamp.
///
/// `params == None` skips the look step (Default look).
#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn agx_kernel_8(
    token: Token,
    rin: [f32; 8],
    gin: [f32; 8],
    bin: [f32; 8],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    const AGX_MIN_EV: f32 = -12.47393;
    const AGX_MAX_EV: f32 = 4.026069;
    const RECIP_EV: f32 = 1.0 / (AGX_MAX_EV - AGX_MIN_EV);

    let r = f32x8::from_array(token, rin);
    let g = f32x8::from_array(token, gin);
    let b = f32x8::from_array(token, bin);

    // Inset matrix (Blender AgX, OCIO row-major — rows sum to 1.0).
    let z0r = f32x8::splat(token, 0.856627153315983) * r
        + f32x8::splat(token, 0.0951212405381588) * g
        + f32x8::splat(token, 0.0482516061458583) * b;
    let z0g = f32x8::splat(token, 0.137318972929847) * r
        + f32x8::splat(token, 0.761241990602591) * g
        + f32x8::splat(token, 0.101439036467562) * b;
    let z0b = f32x8::splat(token, 0.11189821299995) * r
        + f32x8::splat(token, 0.0767994186031903) * g
        + f32x8::splat(token, 0.811302368396859) * b;

    // log2(max(x, 1e-10)), clamped to EV range, normalized to [0, 1].
    let floor = f32x8::splat(token, 1e-10);
    let min_ev = f32x8::splat(token, AGX_MIN_EV);
    let max_ev = f32x8::splat(token, AGX_MAX_EV);
    let recip = f32x8::splat(token, RECIP_EV);

    let z2r = (z0r.max(floor).log2_midp().max(min_ev).min(max_ev) - min_ev) * recip;
    let z2g = (z0g.max(floor).log2_midp().max(min_ev).min(max_ev) - min_ev) * recip;
    let z2b = (z0b.max(floor).log2_midp().max(min_ev).min(max_ev) - min_ev) * recip;

    // Contrast polynomial with endpoint normalization (poly(0)=0.002857,
    // poly(1)=0.982059 → renormalize so 0→0, 1→1 matches Blender's LUT).
    let contrast = |x: f32x8| -> f32x8 {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        let w0 = f32x8::splat(token, 0.002857) + f32x8::splat(token, -0.1718) * x;
        let w1 = f32x8::splat(token, 4.361) + f32x8::splat(token, -28.72) * x;
        let w2 = f32x8::splat(token, 92.06) + f32x8::splat(token, -126.7) * x;
        let w3 = f32x8::splat(token, 78.01) + f32x8::splat(token, -17.86) * x;
        let raw = w0 + w1 * x2 + w2 * x4 + w3 * x6;
        let p0 = f32x8::splat(token, 0.002857);
        let scale = f32x8::splat(token, 1.0 / (0.982059 - 0.002857));
        (raw - p0) * scale
    };
    let cr = contrast(z2r);
    let cg = contrast(z2g);
    let cb = contrast(z2b);

    // Look transform: slope → pow → saturation blend. None = no-op (Default).
    let zero = f32x8::zero(token);
    let (lr, lg, lb) = match params {
        None => (cr, cg, cb),
        Some((slope, power, saturation)) => {
            let dr = (f32x8::splat(token, slope[0]) * cr).max(zero);
            let dg = (f32x8::splat(token, slope[1]) * cg).max(zero);
            let db = (f32x8::splat(token, slope[2]) * cb).max(zero);

            let (zr, zg, zb) = if power == [1.0, 1.0, 1.0] {
                (dr, dg, db)
            } else {
                (
                    dr.pow_midp(power[0]),
                    dg.pow_midp(power[1]),
                    db.pow_midp(power[2]),
                )
            };

            let luma = f32x8::splat(token, 0.2126) * zr
                + f32x8::splat(token, 0.7152) * zg
                + f32x8::splat(token, 0.0722) * zb;

            let or = f32x8::splat(token, saturation[0]) * (zr - luma) + luma;
            let og = f32x8::splat(token, saturation[1]) * (zg - luma) + luma;
            let ob = f32x8::splat(token, saturation[2]) * (zb - luma) + luma;
            (or, og, ob)
        }
    };

    // Outset matrix + clamp to [0, 1].
    let one = f32x8::splat(token, 1.0);
    let or = (f32x8::splat(token, 1.19744107688770) * lr
        + f32x8::splat(token, -0.144261512698001) * lg
        + f32x8::splat(token, -0.0531795641897042) * lb)
        .max(zero)
        .min(one);
    let og = (f32x8::splat(token, -0.196474626321346) * lr
        + f32x8::splat(token, 1.35409513146973) * lg
        + f32x8::splat(token, -0.157620505148385) * lb)
        .max(zero)
        .min(one);
    let ob = (f32x8::splat(token, -0.146557417106601) * lr
        + f32x8::splat(token, -0.108284058788469) * lg
        + f32x8::splat(token, 1.25484147589507) * lb)
        .max(zero)
        .min(one);

    (or.to_array(), og.to_array(), ob.to_array())
}

#[inline]
fn agx_3_tier(row: &mut [f32], params: Option<([f32; 3], [f32; 3], [f32; 3])>) {
    let look = params_to_look(params);
    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        for i in 0..8 {
            ra[i] = chunk[i * 3].abs();
            ga[i] = chunk[i * 3 + 1].abs();
            ba[i] = chunk[i * 3 + 2].abs();
        }
        let (ro, go, bo) = incant!(
            agx_kernel_8(ra, ga, ba, params),
            [v3, neon, wasm128, scalar]
        );
        for i in 0..8 {
            chunk[i * 3] = ro[i];
            chunk[i * 3 + 1] = go[i];
            chunk[i * 3 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], look);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}

#[inline]
fn agx_4_tier(row: &mut [f32], params: Option<([f32; 3], [f32; 3], [f32; 3])>) {
    let look = params_to_look(params);
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
        let (ro, go, bo) = incant!(
            agx_kernel_8(ra, ga, ba, params),
            [v3, neon, wasm128, scalar]
        );
        for i in 0..8 {
            chunk[i * 4] = ro[i];
            chunk[i * 4 + 1] = go[i];
            chunk[i * 4 + 2] = bo[i];
        }
    }
    for c in iter.into_remainder().chunks_exact_mut(4) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], look);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}
