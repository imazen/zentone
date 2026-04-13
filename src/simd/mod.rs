//! SIMD-accelerated tone mapping kernels.
//!
//! AVX2+FMA (x86-64-v3) via `#[archmage::arcane]` with f32x8. Scalar
//! fallback for all other architectures. `archmage::incant!` handles
//! safe runtime dispatch. Respects `#![forbid(unsafe_code)]`.

use crate::ToneMap;

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
    // Encode look as (slope, power, saturation) arrays for SIMD dispatch.
    // Default returns early (no look applied), Punchy/Golden have params.
    let params: Option<([f32; 3], [f32; 3], [f32; 3])> = match look {
        crate::AgxLook::Default => None,
        crate::AgxLook::Punchy => Some(([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.4, 1.4, 1.4])),
        crate::AgxLook::Golden => Some(([1.0, 0.9, 0.5], [0.8, 0.8, 0.8], [1.2, 1.2, 1.2])),
    };
    match ch {
        3 => archmage::incant!(agx_3(row, params)),
        4 => archmage::incant!(agx_4(row, params)),
        _ => {}
    }
}

#[inline]
pub(crate) fn ext_reinhard_row(row: &mut [f32], ch: usize, l_max: f32, luma: [f32; 3]) {
    match ch {
        3 => archmage::incant!(ext_reinhard_3(row, l_max, luma)),
        4 => archmage::incant!(ext_reinhard_4(row, l_max, luma)),
        _ => {}
    }
}

#[inline]
pub(crate) fn reinhard_jodie_row(row: &mut [f32], ch: usize, luma: [f32; 3]) {
    match ch {
        3 => archmage::incant!(reinhard_jodie_3(row, luma)),
        4 => archmage::incant!(reinhard_jodie_4(row, luma)),
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
        3 => archmage::incant!(tuned_reinhard_3(row, content_max, display_max, luma)),
        4 => archmage::incant!(tuned_reinhard_4(row, content_max, display_max, luma)),
        _ => {}
    }
}

// ============================================================================
// Scalar fallbacks + NEON stubs (delegate to scalar on aarch64)
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
fn ext_reinhard_3_scalar(_t: archmage::ScalarToken, r: &mut [f32], l_max: f32, luma: [f32; 3]) {
    for c in r.chunks_exact_mut(3) {
        let o = crate::curves::ToneMapCurve::ExtendedReinhard { l_max, luma }
            .map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
fn ext_reinhard_4_scalar(_t: archmage::ScalarToken, r: &mut [f32], l_max: f32, luma: [f32; 3]) {
    for c in r.chunks_exact_mut(4) {
        let o = crate::curves::ToneMapCurve::ExtendedReinhard { l_max, luma }
            .map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
fn reinhard_jodie_3_scalar(_t: archmage::ScalarToken, r: &mut [f32], luma: [f32; 3]) {
    for c in r.chunks_exact_mut(3) {
        let o = crate::curves::reinhard_jodie([c[0], c[1], c[2]], luma);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
fn reinhard_jodie_4_scalar(_t: archmage::ScalarToken, r: &mut [f32], luma: [f32; 3]) {
    for c in r.chunks_exact_mut(4) {
        let o = crate::curves::reinhard_jodie([c[0], c[1], c[2]], luma);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
fn tuned_reinhard_3_scalar(
    _t: archmage::ScalarToken,
    r: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    let curve = crate::curves::ToneMapCurve::TunedReinhard {
        content_max_nits: content_max,
        display_max_nits: display_max,
        luma,
    };
    for c in r.chunks_exact_mut(3) {
        let o = curve.map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}
fn tuned_reinhard_4_scalar(
    _t: archmage::ScalarToken,
    r: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    let curve = crate::curves::ToneMapCurve::TunedReinhard {
        content_max_nits: content_max,
        display_max_nits: display_max,
        luma,
    };
    for c in r.chunks_exact_mut(4) {
        let o = curve.map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
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
fn agx_3_scalar(
    _t: archmage::ScalarToken,
    r: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
    let look = params_to_look(params);
    for c in r.chunks_exact_mut(3) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], look);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}
fn agx_4_scalar(
    _t: archmage::ScalarToken,
    r: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
    let look = params_to_look(params);
    for c in r.chunks_exact_mut(4) {
        let out = crate::curves::agx_tonemap([c[0], c[1], c[2]], look);
        c[0] = out[0];
        c[1] = out[1];
        c[2] = out[2];
    }
}

/// Recover the AgxLook from optional params (for scalar fallback only).
fn params_to_look(params: Option<([f32; 3], [f32; 3], [f32; 3])>) -> crate::AgxLook {
    match params {
        None => crate::AgxLook::Default,
        Some((_, [1.0, 1.0, 1.0], [1.4, 1.4, 1.4])) => crate::AgxLook::Punchy,
        Some(_) => crate::AgxLook::Golden,
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
// NEON stubs — delegate to scalar on aarch64.
// `incant!` requires `_neon` variants on aarch64; these ensure compilation
// while we only have x86_64 SIMD kernels. They'll be replaced with real
// NEON implementations when magetypes aarch64 support is ready.
// ============================================================================

#[cfg(target_arch = "aarch64")]
fn reinhard_3_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    reinhard_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn reinhard_4_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    reinhard_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn ext_reinhard_3_neon(_t: archmage::NeonToken, r: &mut [f32], l_max: f32, luma: [f32; 3]) {
    ext_reinhard_3_scalar(archmage::ScalarToken, r, l_max, luma);
}
#[cfg(target_arch = "aarch64")]
fn ext_reinhard_4_neon(_t: archmage::NeonToken, r: &mut [f32], l_max: f32, luma: [f32; 3]) {
    ext_reinhard_4_scalar(archmage::ScalarToken, r, l_max, luma);
}
#[cfg(target_arch = "aarch64")]
fn reinhard_jodie_3_neon(_t: archmage::NeonToken, r: &mut [f32], luma: [f32; 3]) {
    reinhard_jodie_3_scalar(archmage::ScalarToken, r, luma);
}
#[cfg(target_arch = "aarch64")]
fn reinhard_jodie_4_neon(_t: archmage::NeonToken, r: &mut [f32], luma: [f32; 3]) {
    reinhard_jodie_4_scalar(archmage::ScalarToken, r, luma);
}
#[cfg(target_arch = "aarch64")]
fn tuned_reinhard_3_neon(
    _t: archmage::NeonToken,
    r: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    tuned_reinhard_3_scalar(archmage::ScalarToken, r, content_max, display_max, luma);
}
#[cfg(target_arch = "aarch64")]
fn tuned_reinhard_4_neon(
    _t: archmage::NeonToken,
    r: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    tuned_reinhard_4_scalar(archmage::ScalarToken, r, content_max, display_max, luma);
}
#[cfg(target_arch = "aarch64")]
fn narkowicz_3_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    narkowicz_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn narkowicz_4_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    narkowicz_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn hable_3_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    hable_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn hable_4_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    hable_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn aces_3_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    aces_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn aces_4_neon(_t: archmage::NeonToken, r: &mut [f32]) {
    aces_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "aarch64")]
fn agx_3_neon(
    _t: archmage::NeonToken,
    r: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
    agx_3_scalar(archmage::ScalarToken, r, params);
}
#[cfg(target_arch = "aarch64")]
fn agx_4_neon(
    _t: archmage::NeonToken,
    r: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
    agx_4_scalar(archmage::ScalarToken, r, params);
}

// ============================================================================
// WASM128 stubs — delegate to scalar on wasm32.
// ============================================================================

#[cfg(target_arch = "wasm32")]
fn reinhard_3_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    reinhard_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn reinhard_4_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    reinhard_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn ext_reinhard_3_wasm128(_t: archmage::Wasm128Token, r: &mut [f32], l_max: f32, luma: [f32; 3]) {
    ext_reinhard_3_scalar(archmage::ScalarToken, r, l_max, luma);
}
#[cfg(target_arch = "wasm32")]
fn ext_reinhard_4_wasm128(_t: archmage::Wasm128Token, r: &mut [f32], l_max: f32, luma: [f32; 3]) {
    ext_reinhard_4_scalar(archmage::ScalarToken, r, l_max, luma);
}
#[cfg(target_arch = "wasm32")]
fn reinhard_jodie_3_wasm128(_t: archmage::Wasm128Token, r: &mut [f32], luma: [f32; 3]) {
    reinhard_jodie_3_scalar(archmage::ScalarToken, r, luma);
}
#[cfg(target_arch = "wasm32")]
fn reinhard_jodie_4_wasm128(_t: archmage::Wasm128Token, r: &mut [f32], luma: [f32; 3]) {
    reinhard_jodie_4_scalar(archmage::ScalarToken, r, luma);
}
#[cfg(target_arch = "wasm32")]
fn tuned_reinhard_3_wasm128(
    _t: archmage::Wasm128Token,
    r: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    tuned_reinhard_3_scalar(archmage::ScalarToken, r, content_max, display_max, luma);
}
#[cfg(target_arch = "wasm32")]
fn tuned_reinhard_4_wasm128(
    _t: archmage::Wasm128Token,
    r: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    tuned_reinhard_4_scalar(archmage::ScalarToken, r, content_max, display_max, luma);
}
#[cfg(target_arch = "wasm32")]
fn narkowicz_3_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    narkowicz_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn narkowicz_4_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    narkowicz_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn hable_3_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    hable_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn hable_4_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    hable_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn aces_3_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    aces_3_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn aces_4_wasm128(_t: archmage::Wasm128Token, r: &mut [f32]) {
    aces_4_scalar(archmage::ScalarToken, r);
}
#[cfg(target_arch = "wasm32")]
fn agx_3_wasm128(
    _t: archmage::Wasm128Token,
    r: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
    agx_3_scalar(archmage::ScalarToken, r, params);
}
#[cfg(target_arch = "wasm32")]
fn agx_4_wasm128(
    _t: archmage::Wasm128Token,
    r: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
    agx_4_scalar(archmage::ScalarToken, r, params);
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
// Luma-based Reinhard variants — 8-pixel SOA kernels
// ============================================================================

/// Helper: gather 8 RGB pixels from stride-3 interleaved data into SOA.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn gather_rgb3(chunk: &[f32], ra: &mut [f32; 8], ga: &mut [f32; 8], ba: &mut [f32; 8]) {
    for i in 0..8 {
        ra[i] = chunk[i * 3];
        ga[i] = chunk[i * 3 + 1];
        ba[i] = chunk[i * 3 + 2];
    }
}

/// Helper: gather 8 RGBA pixels from stride-4 interleaved data into SOA.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn gather_rgb4(chunk: &[f32], ra: &mut [f32; 8], ga: &mut [f32; 8], ba: &mut [f32; 8]) {
    for i in 0..8 {
        ra[i] = chunk[i * 4];
        ga[i] = chunk[i * 4 + 1];
        ba[i] = chunk[i * 4 + 2];
    }
}

/// Helper: scatter 3 f32x8 back into stride-3 interleaved data.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scatter_rgb3(chunk: &mut [f32], r: f32x8, g: f32x8, b: f32x8) {
    let ro = r.to_array();
    let go = g.to_array();
    let bo = b.to_array();
    for i in 0..8 {
        chunk[i * 3] = ro[i];
        chunk[i * 3 + 1] = go[i];
        chunk[i * 3 + 2] = bo[i];
    }
}

/// Helper: scatter 3 f32x8 back into stride-4 interleaved data (alpha untouched).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scatter_rgb4(chunk: &mut [f32], r: f32x8, g: f32x8, b: f32x8) {
    let ro = r.to_array();
    let go = g.to_array();
    let bo = b.to_array();
    for i in 0..8 {
        chunk[i * 4] = ro[i];
        chunk[i * 4 + 1] = go[i];
        chunk[i * 4 + 2] = bo[i];
    }
}

/// Extended Reinhard: luma-preserving with white point.
/// scale = l * (1 + l/l_max²) / (1 + l) / l = (1 + l/l_max²) / (1 + l)
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn ext_reinhard_3_v3(t: archmage::X64V3Token, row: &mut [f32], l_max: f32, luma: [f32; 3]) {
    let lr = f32x8::splat(t, luma[0]);
    let lg = f32x8::splat(t, luma[1]);
    let lb = f32x8::splat(t, luma[2]);
    let lmax_sq = f32x8::splat(t, l_max * l_max);
    let one = f32x8::splat(t, 1.0);
    let zero = f32x8::splat(t, 0.0);

    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        gather_rgb3(chunk, &mut ra, &mut ga, &mut ba);
        let r = f32x8::load(t, &ra);
        let g = f32x8::load(t, &ga);
        let b = f32x8::load(t, &ba);
        let l = r * lr + g * lg + b * lb;
        // scale = reinhard_extended(l) / l = (1 + l/l_max²) / (1 + l)
        // When l <= 0, output is 0 (mask with zero).
        let scale = ((one + l / lmax_sq) / (one + l)).max(zero);
        scatter_rgb3(
            chunk,
            (r * scale).min(one),
            (g * scale).min(one),
            (b * scale).min(one),
        );
    }
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let o = crate::curves::ToneMapCurve::ExtendedReinhard { l_max, luma }
            .map_rgb([c[0], c[1], c[2]]);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn ext_reinhard_4_v3(t: archmage::X64V3Token, row: &mut [f32], l_max: f32, luma: [f32; 3]) {
    let lr = f32x8::splat(t, luma[0]);
    let lg = f32x8::splat(t, luma[1]);
    let lb = f32x8::splat(t, luma[2]);
    let lmax_sq = f32x8::splat(t, l_max * l_max);
    let one = f32x8::splat(t, 1.0);
    let zero = f32x8::splat(t, 0.0);

    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        gather_rgb4(chunk, &mut ra, &mut ga, &mut ba);
        let r = f32x8::load(t, &ra);
        let g = f32x8::load(t, &ga);
        let b = f32x8::load(t, &ba);
        let l = r * lr + g * lg + b * lb;
        let scale = ((one + l / lmax_sq) / (one + l)).max(zero);
        scatter_rgb4(
            chunk,
            (r * scale).min(one),
            (g * scale).min(one),
            (b * scale).min(one),
        );
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
/// out[i] = (1-tv) * (rgb[i] * luma_scale) + tv * tv, tv = rgb[i]/(1+rgb[i])
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn reinhard_jodie_3_v3(t: archmage::X64V3Token, row: &mut [f32], luma: [f32; 3]) {
    let lr = f32x8::splat(t, luma[0]);
    let lg = f32x8::splat(t, luma[1]);
    let lb = f32x8::splat(t, luma[2]);
    let one = f32x8::splat(t, 1.0);
    let zero = f32x8::splat(t, 0.0);

    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        gather_rgb3(chunk, &mut ra, &mut ga, &mut ba);
        let r = f32x8::load(t, &ra);
        let g = f32x8::load(t, &ga);
        let b = f32x8::load(t, &ba);
        let l = r * lr + g * lg + b * lb;
        let luma_scale = one / (one + l);
        // Per-channel: tv = x/(1+x), out = (1-tv)*(x*luma_scale) + tv*tv
        let tvr = r / (one + r);
        let tvg = g / (one + g);
        let tvb = b / (one + b);
        let or = ((one - tvr) * (r * luma_scale) + tvr * tvr)
            .min(one)
            .max(zero);
        let og = ((one - tvg) * (g * luma_scale) + tvg * tvg)
            .min(one)
            .max(zero);
        let ob = ((one - tvb) * (b * luma_scale) + tvb * tvb)
            .min(one)
            .max(zero);
        scatter_rgb3(chunk, or, og, ob);
    }
    for c in iter.into_remainder().chunks_exact_mut(3) {
        let o = crate::curves::reinhard_jodie([c[0], c[1], c[2]], luma);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn reinhard_jodie_4_v3(t: archmage::X64V3Token, row: &mut [f32], luma: [f32; 3]) {
    let lr = f32x8::splat(t, luma[0]);
    let lg = f32x8::splat(t, luma[1]);
    let lb = f32x8::splat(t, luma[2]);
    let one = f32x8::splat(t, 1.0);
    let zero = f32x8::splat(t, 0.0);

    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        gather_rgb4(chunk, &mut ra, &mut ga, &mut ba);
        let r = f32x8::load(t, &ra);
        let g = f32x8::load(t, &ga);
        let b = f32x8::load(t, &ba);
        let l = r * lr + g * lg + b * lb;
        let luma_scale = one / (one + l);
        let tvr = r / (one + r);
        let tvg = g / (one + g);
        let tvb = b / (one + b);
        let or = ((one - tvr) * (r * luma_scale) + tvr * tvr)
            .min(one)
            .max(zero);
        let og = ((one - tvg) * (g * luma_scale) + tvg * tvg)
            .min(one)
            .max(zero);
        let ob = ((one - tvb) * (b * luma_scale) + tvb * tvb)
            .min(one)
            .max(zero);
        scatter_rgb4(chunk, or, og, ob);
    }
    for c in iter.into_remainder().chunks_exact_mut(4) {
        let o = crate::curves::reinhard_jodie([c[0], c[1], c[2]], luma);
        c[0] = o[0];
        c[1] = o[1];
        c[2] = o[2];
    }
}

/// Tuned Reinhard: display-aware with content/display peak.
/// scale = (1 + w_a * l) / (1 + w_b * l), applied to luma then RGB scaled.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn tuned_reinhard_3_v3(
    t: archmage::X64V3Token,
    row: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    let white_point = 203.0_f32;
    let ld = content_max / white_point;
    let lr = f32x8::splat(t, luma[0]);
    let lg = f32x8::splat(t, luma[1]);
    let lb = f32x8::splat(t, luma[2]);
    let w_a = f32x8::splat(t, (display_max / white_point) / (ld * ld));
    let w_b = f32x8::splat(t, 1.0 / (display_max / white_point));
    let one = f32x8::splat(t, 1.0);
    let zero = f32x8::splat(t, 0.0);

    let mut iter = row.chunks_exact_mut(24);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        gather_rgb3(chunk, &mut ra, &mut ga, &mut ba);
        let r = f32x8::load(t, &ra);
        let g = f32x8::load(t, &ga);
        let b = f32x8::load(t, &ba);
        let l = r * lr + g * lg + b * lb;
        // scale = tuned_reinhard(l) = (1 + w_a*l) / (1 + w_b*l)
        let scale = ((one + w_a * l) / (one + w_b * l)).max(zero);
        scatter_rgb3(
            chunk,
            (r * scale).min(one),
            (g * scale).min(one),
            (b * scale).min(one),
        );
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

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn tuned_reinhard_4_v3(
    t: archmage::X64V3Token,
    row: &mut [f32],
    content_max: f32,
    display_max: f32,
    luma: [f32; 3],
) {
    let white_point = 203.0_f32;
    let ld = content_max / white_point;
    let lr = f32x8::splat(t, luma[0]);
    let lg = f32x8::splat(t, luma[1]);
    let lb = f32x8::splat(t, luma[2]);
    let w_a = f32x8::splat(t, (display_max / white_point) / (ld * ld));
    let w_b = f32x8::splat(t, 1.0 / (display_max / white_point));
    let one = f32x8::splat(t, 1.0);
    let zero = f32x8::splat(t, 0.0);

    let mut iter = row.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut ra = [0.0_f32; 8];
        let mut ga = [0.0_f32; 8];
        let mut ba = [0.0_f32; 8];
        gather_rgb4(chunk, &mut ra, &mut ga, &mut ba);
        let r = f32x8::load(t, &ra);
        let g = f32x8::load(t, &ga);
        let b = f32x8::load(t, &ba);
        let l = r * lr + g * lg + b * lb;
        let scale = ((one + w_a * l) / (one + w_b * l)).max(zero);
        scatter_rgb4(
            chunk,
            (r * scale).min(one),
            (g * scale).min(one),
            (b * scale).min(one),
        );
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
// AgX — 8-pixel SOA kernel with vectorized log2 + polynomial + optional look
// ============================================================================

/// Process 8 RGB pixels in SOA layout through the full AgX pipeline.
/// Gather interleaved RGB → 3 × f32x8, compute core + look, scatter back.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn agx_3_v3(
    t: archmage::X64V3Token,
    row: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
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
        // Order: inset → log2 → contrast → look → outset → clamp
        let (r, g, b) = agx_pre_outset_8px(t, &ra, &ga, &ba);
        let (r, g, b) = agx_look_8px(t, r, g, b, params);
        let (r, g, b) = agx_outset_8px(t, r, g, b);
        let ro = r.to_array();
        let go = g.to_array();
        let bo = b.to_array();
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

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn agx_4_v3(
    t: archmage::X64V3Token,
    row: &mut [f32],
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) {
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
        let (r, g, b) = agx_pre_outset_8px(t, &ra, &ga, &ba);
        let (r, g, b) = agx_look_8px(t, r, g, b, params);
        let (r, g, b) = agx_outset_8px(t, r, g, b);
        let ro = r.to_array();
        let go = g.to_array();
        let bo = b.to_array();
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

/// AgX pipeline up to (but not including) outset matrix:
/// inset matrix → log2 → contrast. Returns contrast-mapped R, G, B.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn agx_pre_outset_8px(
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

    // Inset matrix
    let z0r = f32x8::splat(t, 0.856627153315983) * r
        + f32x8::splat(t, 0.137318972929847) * g
        + f32x8::splat(t, 0.11189821299995) * b;
    let z0g = f32x8::splat(t, 0.0951212405381588) * r
        + f32x8::splat(t, 0.761241990602591) * g
        + f32x8::splat(t, 0.0767994186031903) * b;
    let z0b = f32x8::splat(t, 0.0482516061458583) * r
        + f32x8::splat(t, 0.101439036467562) * g
        + f32x8::splat(t, 0.811302368396859) * b;

    // log2(max(x, 1e-10)), clamped to EV range
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

    // Polynomial contrast
    (
        agx_contrast_v8(t, z2r),
        agx_contrast_v8(t, z2g),
        agx_contrast_v8(t, z2b),
    )
}

/// AgX outset matrix + clamp to [0, 1]. Applied AFTER the look transform.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn agx_outset_8px(t: archmage::X64V3Token, r: f32x8, g: f32x8, b: f32x8) -> (f32x8, f32x8, f32x8) {
    let zero = f32x8::splat(t, 0.0);
    let one = f32x8::splat(t, 1.0);

    let or = (f32x8::splat(t, 1.19687900512017) * r
        + f32x8::splat(t, -0.0528968517574562) * g
        + f32x8::splat(t, -0.0529716355144438) * b)
        .max(zero)
        .min(one);
    let og = (f32x8::splat(t, -0.0980208811401368) * r
        + f32x8::splat(t, 1.15190312990417) * g
        + f32x8::splat(t, -0.0505349770312032) * b)
        .max(zero)
        .min(one);
    let ob = (f32x8::splat(t, -0.0990297440797205) * r
        + f32x8::splat(t, -0.0989611768448433) * g
        + f32x8::splat(t, 1.15107367264116) * b)
        .max(zero)
        .min(one);

    (or, og, ob)
}

/// Apply AgX look transform in SIMD. For Default (None), returns input unchanged.
/// For Punchy/Golden, applies slope → pow → saturation blend.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn agx_look_8px(
    t: archmage::X64V3Token,
    r: f32x8,
    g: f32x8,
    b: f32x8,
    params: Option<([f32; 3], [f32; 3], [f32; 3])>,
) -> (f32x8, f32x8, f32x8) {
    let (slope, power, saturation) = match params {
        None => return (r, g, b),
        Some(p) => p,
    };

    let zero = f32x8::splat(t, 0.0);

    // slope * x, clamp to 0
    let dr = (f32x8::splat(t, slope[0]) * r).max(zero);
    let dg = (f32x8::splat(t, slope[1]) * g).max(zero);
    let db = (f32x8::splat(t, slope[2]) * b).max(zero);

    // pow(x, power) — skip if power == 1.0 for all channels
    let (zr, zg, zb) = if power == [1.0, 1.0, 1.0] {
        (dr, dg, db)
    } else {
        (
            dr.pow_midp(power[0]),
            dg.pow_midp(power[1]),
            db.pow_midp(power[2]),
        )
    };

    // Saturation blend: luma + sat * (channel - luma)
    let luma =
        f32x8::splat(t, 0.2126) * zr + f32x8::splat(t, 0.7152) * zg + f32x8::splat(t, 0.0722) * zb;

    let or = f32x8::splat(t, saturation[0]) * (zr - luma) + luma;
    let og = f32x8::splat(t, saturation[1]) * (zg - luma) + luma;
    let ob = f32x8::splat(t, saturation[2]) * (zb - luma) + luma;

    (or, og, ob)
}

/// Vectorized AgX contrast polynomial with endpoint normalization.
/// Raw polynomial has poly(0)=0.002857, poly(1)=0.982059; we normalize
/// so 0→0, 1→1 to match Blender's actual sigmoid LUT.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn agx_contrast_v8(t: archmage::X64V3Token, x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;

    let w0 = f32x8::splat(t, 0.002857) + f32x8::splat(t, -0.1718) * x;
    let w1 = f32x8::splat(t, 4.361) + f32x8::splat(t, -28.72) * x;
    let w2 = f32x8::splat(t, 92.06) + f32x8::splat(t, -126.7) * x;
    let w3 = f32x8::splat(t, 78.01) + f32x8::splat(t, -17.86) * x;

    let raw = w0 + w1 * x2 + w2 * x4 + w3 * x6;
    let p0 = f32x8::splat(t, 0.002857);
    let scale = f32x8::splat(t, 1.0 / (0.982059 - 0.002857));
    (raw - p0) * scale
}
