//! `#[archmage::magetypes]` strip kernels for the transcendental-using
//! tonemapper curves: BT.2408, BT.2446 A/B/C, and `CompiledFilmicSpline`.
//!
//! These are wired up by `ToneMap::map_strip_simd` overrides in each curve's
//! own module — see the trait method default in `tone_map.rs` for the
//! contract. Each kernel processes 8 RGB pixels per chunk in SOA form
//! (gather, math, scatter); the strip's scalar tail falls back to the
//! per-pixel `ToneMap::map_rgb` reference for boundary parity.
//!
//! ## Tier annotations
//!
//! Kernels that call `f32x8::pow_midp` / `log2_midp` / `exp2_midp` omit
//! `v4(cfg(avx512))` — magetypes 0.9.22 doesn't implement
//! `F32x8Convert for X64V4Token`, so AVX-512 cannot resolve those entry
//! points on a 256-bit `f32x8`. AVX-512 hardware still picks the V3 (AVX2)
//! path safely. Once magetypes adds the impl, every kernel here gains the
//! v4 tier on the next bump with zero source changes.

use crate::math::{expf, log2f, powf, sqrtf};

// ============================================================================
// BT.2408 — PQ-domain Hermite EETF, YRGB and MaxRGB application spaces.
// ============================================================================

/// SMPTE ST.2084 PQ OETF coefficients (linear → PQ).
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

/// All BT.2408 spline coefficients pre-extracted from `Bt2408Tonemapper`.
#[derive(Clone, Copy)]
pub(crate) struct Bt2408Params {
    pub luma: [f32; 3],
    pub content_max_nits: f32,
    pub display_max_nits: f32,
    pub inv_display_max: f32,
    pub content_min_pq: f32,
    pub content_range_pq: f32,
    pub inv_content_range_pq: f32,
    pub min_lum: f32,
    pub max_lum: f32,
    pub ks: f32,
    pub one_minus_ks: f32,
    pub inv_one_minus_ks: f32,
    pub normalizer: f32,
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn bt2408_yrgb_tier(token: Token, row: &mut [[f32; 3]], p: &Bt2408Params) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let pos_eps = f32x8::splat(token, f32::MIN_POSITIVE);
    let inv_10000 = f32x8::splat(token, 1.0 / 10000.0);
    let lr = f32x8::splat(token, p.luma[0]);
    let lg = f32x8::splat(token, p.luma[1]);
    let lb = f32x8::splat(token, p.luma[2]);
    let cmax = f32x8::splat(token, p.content_max_nits);
    let dmax = f32x8::splat(token, p.display_max_nits);
    let inv_dmax = f32x8::splat(token, p.inv_display_max);
    let cmin_pq = f32x8::splat(token, p.content_min_pq);
    let inv_range = f32x8::splat(token, p.inv_content_range_pq);
    let range_pq = f32x8::splat(token, p.content_range_pq);
    let min_lum = f32x8::splat(token, p.min_lum);
    let max_lum = f32x8::splat(token, p.max_lum);
    let ks = f32x8::splat(token, p.ks);
    let one_minus_ks = f32x8::splat(token, p.one_minus_ks);
    let inv_one_minus_ks = f32x8::splat(token, p.inv_one_minus_ks);
    let normalizer = f32x8::splat(token, p.normalizer);
    let pq_c1 = f32x8::splat(token, PQ_C1);
    let pq_c2 = f32x8::splat(token, PQ_C2);
    let pq_c3 = f32x8::splat(token, PQ_C3);
    let two = f32x8::splat(token, 2.0);
    let three = f32x8::splat(token, 3.0);
    let neg_two = f32x8::splat(token, -2.0);
    let cutoff = f32x8::splat(token, 1e-6);
    let ten_thousand = f32x8::splat(token, 10000.0);

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
        let signal = lr * r + lg * g + lb * b;
        let signal_nits = signal * cmax;

        // PQ-encode (signal_nits / 10000), normalize to [0, 1].
        let safe_in = signal_nits.max(zero) * inv_10000;
        let xm1 = safe_in.pow_midp(PQ_M1);
        let s = ((pq_c1 + pq_c2 * xm1) / (one + pq_c3 * xm1)).pow_midp(PQ_M2);
        let normalized_pq = ((s - cmin_pq) * inv_range).min(one).max(zero);

        // Hermite spline above KS, passthrough below.
        let t_b = (normalized_pq - ks) * inv_one_minus_ks;
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        let spline = (two * t_b_3 - three * t_b_2 + one) * ks
            + (t_b_3 - two * t_b_2 + t_b) * one_minus_ks
            + (neg_two * t_b_3 + three * t_b_2) * max_lum;
        let in_knee = normalized_pq.simd_lt(ks);
        let e2 = f32x8::blend(in_knee, normalized_pq, spline);

        // Black-lift: e3 = min_lum * (1 - e2)^4 + e2.
        let one_minus_e2 = one - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;

        // PQ-decode to nits.
        let e4 = (e3 * range_pq + cmin_pq).max(zero).min(one);
        let ep = e4.pow_midp(1.0 / PQ_M2);
        let num = (ep - pq_c1).max(zero);
        let den = pq_c2 - pq_c3 * ep;
        let d4 = (num / den).pow_midp(1.0 / PQ_M1) * ten_thousand;

        // Convert to scale factor.
        let new_lum = d4.min(dmax).max(zero);
        let above = signal_nits.simd_gt(cutoff);
        let safe_signal = signal_nits.max(pos_eps);
        let scale_above = (new_lum / safe_signal) * normalizer;
        let scale_below = new_lum * inv_dmax;
        let scale = f32x8::blend(above, scale_above, scale_below);

        let pos = signal_nits.simd_gt(zero);
        let nr = f32x8::blend(pos, r * scale, zero).to_array();
        let ng = f32x8::blend(pos, g * scale, zero).to_array();
        let nb = f32x8::blend(pos, b * scale, zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = nr[i];
            px[1] = ng[i];
            px[2] = nb[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        let signal = p.luma[0] * px[0] + p.luma[1] * px[1] + p.luma[2] * px[2];
        let signal_nits = signal * p.content_max_nits;
        if signal_nits <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let scale = bt2408_make_luma_scale_scalar(signal_nits, p);
        px[0] *= scale;
        px[1] *= scale;
        px[2] *= scale;
    }
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn bt2408_maxrgb_tier(token: Token, row: &mut [[f32; 3]], p: &Bt2408Params) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let pos_eps = f32x8::splat(token, f32::MIN_POSITIVE);
    let inv_10000 = f32x8::splat(token, 1.0 / 10000.0);
    let cmax = f32x8::splat(token, p.content_max_nits);
    let dmax = f32x8::splat(token, p.display_max_nits);
    let inv_dmax = f32x8::splat(token, p.inv_display_max);
    let cmin_pq = f32x8::splat(token, p.content_min_pq);
    let inv_range = f32x8::splat(token, p.inv_content_range_pq);
    let range_pq = f32x8::splat(token, p.content_range_pq);
    let min_lum = f32x8::splat(token, p.min_lum);
    let max_lum = f32x8::splat(token, p.max_lum);
    let ks = f32x8::splat(token, p.ks);
    let one_minus_ks = f32x8::splat(token, p.one_minus_ks);
    let inv_one_minus_ks = f32x8::splat(token, p.inv_one_minus_ks);
    let normalizer = f32x8::splat(token, p.normalizer);
    let pq_c1 = f32x8::splat(token, PQ_C1);
    let pq_c2 = f32x8::splat(token, PQ_C2);
    let pq_c3 = f32x8::splat(token, PQ_C3);
    let two = f32x8::splat(token, 2.0);
    let three = f32x8::splat(token, 3.0);
    let neg_two = f32x8::splat(token, -2.0);
    let cutoff = f32x8::splat(token, 1e-6);
    let ten_thousand = f32x8::splat(token, 10000.0);

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
        let signal = r.max(g).max(b);
        let signal_nits = signal * cmax;

        let safe_in = signal_nits.max(zero) * inv_10000;
        let xm1 = safe_in.pow_midp(PQ_M1);
        let s = ((pq_c1 + pq_c2 * xm1) / (one + pq_c3 * xm1)).pow_midp(PQ_M2);
        let normalized_pq = ((s - cmin_pq) * inv_range).min(one).max(zero);

        let t_b = (normalized_pq - ks) * inv_one_minus_ks;
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        let spline = (two * t_b_3 - three * t_b_2 + one) * ks
            + (t_b_3 - two * t_b_2 + t_b) * one_minus_ks
            + (neg_two * t_b_3 + three * t_b_2) * max_lum;
        let in_knee = normalized_pq.simd_lt(ks);
        let e2 = f32x8::blend(in_knee, normalized_pq, spline);

        let one_minus_e2 = one - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;

        let e4 = (e3 * range_pq + cmin_pq).max(zero).min(one);
        let ep = e4.pow_midp(1.0 / PQ_M2);
        let num = (ep - pq_c1).max(zero);
        let den = pq_c2 - pq_c3 * ep;
        let d4 = (num / den).pow_midp(1.0 / PQ_M1) * ten_thousand;

        let new_lum = d4.min(dmax).max(zero);
        let above = signal_nits.simd_gt(cutoff);
        let safe_signal = signal_nits.max(pos_eps);
        let scale_above = (new_lum / safe_signal) * normalizer;
        let scale_below = new_lum * inv_dmax;
        let scale = f32x8::blend(above, scale_above, scale_below);

        let pos = signal_nits.simd_gt(zero);
        let nr = f32x8::blend(pos, r * scale, zero).to_array();
        let ng = f32x8::blend(pos, g * scale, zero).to_array();
        let nb = f32x8::blend(pos, b * scale, zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = nr[i];
            px[1] = ng[i];
            px[2] = nb[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        let signal = px[0].max(px[1]).max(px[2]);
        let signal_nits = signal * p.content_max_nits;
        if signal_nits <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let scale = bt2408_make_luma_scale_scalar(signal_nits, p);
        px[0] *= scale;
        px[1] *= scale;
        px[2] *= scale;
    }
}

#[inline]
fn bt2408_make_luma_scale_scalar(signal_nits: f32, p: &Bt2408Params) -> f32 {
    let s = linear_srgb::tf::linear_to_pq(signal_nits / 10000.0);
    let normalized_pq = ((s - p.content_min_pq) * p.inv_content_range_pq).clamp(0.0, 1.0);
    let e2 = if normalized_pq < p.ks {
        normalized_pq
    } else {
        let t_b = (normalized_pq - p.ks) * p.inv_one_minus_ks;
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        (2.0 * t_b_3 - 3.0 * t_b_2 + 1.0) * p.ks
            + (t_b_3 - 2.0 * t_b_2 + t_b) * p.one_minus_ks
            + (-2.0 * t_b_3 + 3.0 * t_b_2) * p.max_lum
    };
    let one_minus_e2 = 1.0 - e2;
    let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
    let e3 = p.min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;
    let e4 = e3 * p.content_range_pq + p.content_min_pq;
    let d4 = linear_srgb::tf::pq_to_linear(e4) * 10000.0;
    let new_lum = d4.min(p.display_max_nits).max(0.0);
    let min_luminance = 1e-6;
    if signal_nits <= min_luminance {
        new_lum * p.inv_display_max
    } else {
        (new_lum / signal_nits.max(min_luminance)) * p.normalizer
    }
}

// ============================================================================
// BT.2446 Method A — gamma-domain piecewise polynomial + Hunt color correction.
// ============================================================================

/// BT.2020 luma weights (used by all BT.2446 methods).
const LR_2020: f32 = 0.2627;
const LG_2020: f32 = 0.6780;
const LB_2020: f32 = 0.0593;

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn bt2446a_tier(
    token: Token,
    row: &mut [[f32; 3]],
    rho_hdr: f32,
    inv_log_rho_hdr: f32,
    rho_sdr: f32,
    inv_rho_sdr_minus_1: f32,
) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let lr = f32x8::splat(token, LR_2020);
    let lg = f32x8::splat(token, LG_2020);
    let lb = f32x8::splat(token, LB_2020);
    let inv_24 = 1.0_f32 / 2.4;
    let rho_hdr_minus_1 = f32x8::splat(token, rho_hdr - 1.0);
    let inv_log_rho_hdr_v = f32x8::splat(token, inv_log_rho_hdr);
    let inv_rho_sdr_m1_v = f32x8::splat(token, inv_rho_sdr_minus_1);
    let log2_rho_sdr = f32x8::splat(token, libm::log2f(rho_sdr));
    let pos_eps = f32x8::splat(token, f32::MIN_POSITIVE);
    let ln2 = f32x8::splat(token, core::f32::consts::LN_2);

    let t1 = f32x8::splat(token, 0.7399);
    let t2 = f32x8::splat(token, 0.9909);
    let a1 = f32x8::splat(token, 1.0770);
    let a2_a = f32x8::splat(token, -1.1510);
    let a2_b = f32x8::splat(token, 2.7811);
    let a2_c = f32x8::splat(token, -0.6302);
    let a3_a = f32x8::splat(token, 0.5);
    let a3_b = f32x8::splat(token, 0.5);
    let one_p_one = f32x8::splat(token, 1.1);
    let inv_1_8814 = f32x8::splat(token, 1.0 / 1.8814);
    let inv_1_4746 = f32x8::splat(token, 1.0 / 1.4746);
    let mat_g_b = f32x8::splat(token, 0.16455 / 0.6780);
    let mat_g_r = f32x8::splat(token, 0.57135 / 0.6780);
    let mat_r_cr = f32x8::splat(token, 1.4746);
    let mat_b_cb = f32x8::splat(token, 1.8814);
    let zero_one = f32x8::splat(token, 0.1);

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
        let r = f32x8::load(token, &ra).max(zero);
        let g = f32x8::load(token, &ga).max(zero);
        let b = f32x8::load(token, &ba).max(zero);

        let r_p = r.pow_midp(inv_24);
        let g_p = g.pow_midp(inv_24);
        let b_p = b.pow_midp(inv_24);

        let y_p = lr * r_p + lg * g_p + lb * b_p;

        let arg = one + rho_hdr_minus_1 * y_p;
        let ln_arg = arg.max(pos_eps).log2_midp() * ln2;
        let y_p_lin = ln_arg * inv_log_rho_hdr_v;

        let lo_branch = a1 * y_p_lin;
        let mid_branch = a2_a * y_p_lin * y_p_lin + a2_b * y_p_lin + a2_c;
        let hi_branch = a3_a * y_p_lin + a3_b;
        let in_lo = y_p_lin.simd_le(t1);
        let in_hi = y_p_lin.simd_ge(t2);
        let mid_or_hi = f32x8::blend(in_hi, hi_branch, mid_branch);
        let y_c = f32x8::blend(in_lo, lo_branch, mid_or_hi);

        // pow(rho_sdr, y_c) = exp2(y_c * log2(rho_sdr)).
        let y_sdr = ((y_c * log2_rho_sdr).exp2_midp() - one) * inv_rho_sdr_m1_v;

        let y_p_safe = y_p.max(pos_eps);
        let f = y_sdr / (one_p_one * y_p_safe);
        let cb = f * (b_p - y_p) * inv_1_8814;
        let cr = f * (r_p - y_p) * inv_1_4746;
        let cr_pos = cr.max(zero);
        let y_tmo = y_sdr - zero_one * cr_pos;

        let r_out = (y_tmo + mat_r_cr * cr).max(zero).min(one);
        let g_out = (y_tmo - mat_g_b * cb - mat_g_r * cr).max(zero).min(one);
        let b_out = (y_tmo + mat_b_cb * cb).max(zero).min(one);

        let valid = y_p.simd_gt(zero);
        let or_arr = f32x8::blend(valid, r_out, zero).to_array();
        let og_arr = f32x8::blend(valid, g_out, zero).to_array();
        let ob_arr = f32x8::blend(valid, b_out, zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or_arr[i];
            px[1] = og_arr[i];
            px[2] = ob_arr[i];
        }
    }

    for px in iter.into_remainder().iter_mut() {
        let r_p = powf(px[0].max(0.0), 1.0 / 2.4);
        let g_p = powf(px[1].max(0.0), 1.0 / 2.4);
        let b_p = powf(px[2].max(0.0), 1.0 / 2.4);
        let y_p = LR_2020 * r_p + LG_2020 * g_p + LB_2020 * b_p;
        if y_p <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let y_p_lin = libm::logf(1.0 + (rho_hdr - 1.0) * y_p) * inv_log_rho_hdr;
        let y_c = if y_p_lin <= 0.7399 {
            1.0770 * y_p_lin
        } else if y_p_lin < 0.9909 {
            -1.1510 * y_p_lin * y_p_lin + 2.7811 * y_p_lin - 0.6302
        } else {
            0.5000 * y_p_lin + 0.5000
        };
        let y_sdr = (powf(rho_sdr, y_c) - 1.0) * inv_rho_sdr_minus_1;
        let f = y_sdr / (1.1 * y_p);
        let cb = f * (b_p - y_p) / 1.8814;
        let cr = f * (r_p - y_p) / 1.4746;
        let y_tmo = y_sdr - 0.1_f32.max(0.0) * cr.max(0.0);
        let r_out = (y_tmo + 1.4746 * cr).clamp(0.0, 1.0);
        let g_out = (y_tmo - (0.16455 / 0.6780) * cb - (0.57135 / 0.6780) * cr).clamp(0.0, 1.0);
        let b_out = (y_tmo + 1.8814 * cb).clamp(0.0, 1.0);
        *px = [r_out, g_out, b_out];
    }
}

// ============================================================================
// BT.2446 Method B — chromaticity-preserving log roll-off.
// ============================================================================

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn bt2446b_tier(
    token: Token,
    row: &mut [[f32; 3]],
    breakpoint: f32,
    gain: f32,
    log_scale: f32,
    log_offset: f32,
) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let lr = f32x8::splat(token, LR_2020);
    let lg = f32x8::splat(token, LG_2020);
    let lb = f32x8::splat(token, LB_2020);
    let bp = f32x8::splat(token, breakpoint);
    let inv_bp = f32x8::splat(token, 1.0 / breakpoint);
    let gain_v = f32x8::splat(token, gain);
    let log_scale_v = f32x8::splat(token, log_scale);
    let log_offset_v = f32x8::splat(token, log_offset);
    let pos_eps = f32x8::splat(token, f32::MIN_POSITIVE);
    let ln2 = f32x8::splat(token, core::f32::consts::LN_2);

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
        let y_safe = y.max(pos_eps);

        let lo_branch = gain_v * y;
        let log_arg = (y_safe * inv_bp).max(pos_eps);
        let ln_v = log_arg.log2_midp() * ln2;
        let hi_branch = log_scale_v * ln_v + log_offset_v;
        let below = y.simd_lt(bp);
        let y_sdr = f32x8::blend(below, lo_branch, hi_branch);

        let ratio = y_sdr / y_safe;
        let r_out = (r * ratio).max(zero).min(one);
        let g_out = (g * ratio).max(zero).min(one);
        let b_out = (b * ratio).max(zero).min(one);

        let pos = y.simd_gt(zero);
        let or_arr = f32x8::blend(pos, r_out, zero).to_array();
        let og_arr = f32x8::blend(pos, g_out, zero).to_array();
        let ob_arr = f32x8::blend(pos, b_out, zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or_arr[i];
            px[1] = og_arr[i];
            px[2] = ob_arr[i];
        }
    }
    for px in iter.into_remainder().iter_mut() {
        let y = LR_2020 * px[0] + LG_2020 * px[1] + LB_2020 * px[2];
        if y <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let y_sdr = if y < breakpoint {
            gain * y
        } else {
            log_scale * libm::logf(y / breakpoint) + log_offset
        };
        let ratio = y_sdr / y;
        px[0] = (px[0] * ratio).clamp(0.0, 1.0);
        px[1] = (px[1] * ratio).clamp(0.0, 1.0);
        px[2] = (px[2] * ratio).clamp(0.0, 1.0);
    }
}

// ============================================================================
// BT.2446 Method C — piecewise linear+log with crosstalk + inverse crosstalk.
// ============================================================================

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn bt2446c_tier(
    token: Token,
    row: &mut [[f32; 3]],
    k1: f32,
    k2: f32,
    k4: f32,
    y_ip: f32,
    alpha: f32,
) {
    let zero = f32x8::zero(token);
    let lr = f32x8::splat(token, LR_2020);
    let lg = f32x8::splat(token, LG_2020);
    let lb = f32x8::splat(token, LB_2020);

    let a = alpha;
    let d = 1.0 - 2.0 * a;
    let a_v = f32x8::splat(token, a);
    let d_v = f32x8::splat(token, d);
    let do_crosstalk = a > 0.0;

    let inv_a = if do_crosstalk {
        -a / (1.0 - 3.0 * a)
    } else {
        0.0
    };
    let inv_d = 1.0 - 2.0 * inv_a;
    let inv_a_v = f32x8::splat(token, inv_a);
    let inv_d_v = f32x8::splat(token, inv_d);

    let k1_v = f32x8::splat(token, k1);
    let k2_v = f32x8::splat(token, k2);
    let k4_v = f32x8::splat(token, k4);
    let y_ip_v = f32x8::splat(token, y_ip);
    let inv_y_ip = f32x8::splat(token, 1.0 / y_ip);
    let one_hundred = f32x8::splat(token, 100.0);
    let inv_one_hundred = f32x8::splat(token, 1.0 / 100.0);
    let pct_clamp = f32x8::splat(token, 100.0);
    let lo = f32x8::splat(token, 0.0);
    let hi = f32x8::splat(token, 1.09);
    let pos_eps = f32x8::splat(token, f32::MIN_POSITIVE);
    let ln2 = f32x8::splat(token, core::f32::consts::LN_2);

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
        let r_pct = f32x8::load(token, &ra).max(zero).min(pct_clamp) * one_hundred;
        let g_pct = f32x8::load(token, &ga).max(zero).min(pct_clamp) * one_hundred;
        let b_pct = f32x8::load(token, &ba).max(zero).min(pct_clamp) * one_hundred;

        let (cr_, cg_, cb_) = if do_crosstalk {
            (
                d_v * r_pct + a_v * g_pct + a_v * b_pct,
                a_v * r_pct + d_v * g_pct + a_v * b_pct,
                a_v * r_pct + a_v * g_pct + d_v * b_pct,
            )
        } else {
            (r_pct, g_pct, b_pct)
        };

        let y = lr * cr_ + lg * cg_ + lb * cb_;
        let y_safe = y.max(pos_eps);

        let lo_branch = k1_v * y;
        let log_arg = (y_safe * inv_y_ip).max(pos_eps);
        let ln_v = log_arg.log2_midp() * ln2;
        let hi_branch = k2_v * ln_v + k4_v;
        let below = y.simd_lt(y_ip_v);
        let y_sdr = f32x8::blend(below, lo_branch, hi_branch);

        let ratio = y_sdr / y_safe;
        let sdr_r = cr_ * ratio;
        let sdr_g = cg_ * ratio;
        let sdr_b = cb_ * ratio;

        let (or_pct, og_pct, ob_pct) = if do_crosstalk {
            (
                inv_d_v * sdr_r + inv_a_v * sdr_g + inv_a_v * sdr_b,
                inv_a_v * sdr_r + inv_d_v * sdr_g + inv_a_v * sdr_b,
                inv_a_v * sdr_r + inv_a_v * sdr_g + inv_d_v * sdr_b,
            )
        } else {
            (sdr_r, sdr_g, sdr_b)
        };

        let r_out = (or_pct * inv_one_hundred).max(lo).min(hi);
        let g_out = (og_pct * inv_one_hundred).max(lo).min(hi);
        let b_out = (ob_pct * inv_one_hundred).max(lo).min(hi);

        let pos = y.simd_gt(zero);
        let or_arr = f32x8::blend(pos, r_out, zero).to_array();
        let og_arr = f32x8::blend(pos, g_out, zero).to_array();
        let ob_arr = f32x8::blend(pos, b_out, zero).to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or_arr[i];
            px[1] = og_arr[i];
            px[2] = ob_arr[i];
        }
    }

    for px in iter.into_remainder().iter_mut() {
        let rgb_pct = [
            px[0].clamp(0.0, 100.0) * 100.0,
            px[1].clamp(0.0, 100.0) * 100.0,
            px[2].clamp(0.0, 100.0) * 100.0,
        ];
        let ct = if do_crosstalk {
            [
                d * rgb_pct[0] + a * rgb_pct[1] + a * rgb_pct[2],
                a * rgb_pct[0] + d * rgb_pct[1] + a * rgb_pct[2],
                a * rgb_pct[0] + a * rgb_pct[1] + d * rgb_pct[2],
            ]
        } else {
            rgb_pct
        };
        let y = LR_2020 * ct[0] + LG_2020 * ct[1] + LB_2020 * ct[2];
        if y <= 0.0 {
            *px = [0.0, 0.0, 0.0];
            continue;
        }
        let y_sdr = if y < y_ip {
            k1 * y
        } else {
            k2 * libm::logf(y / y_ip) + k4
        };
        let ratio = y_sdr / y;
        let sdr_pct = [ct[0] * ratio, ct[1] * ratio, ct[2] * ratio];
        let sdr = if do_crosstalk {
            [
                inv_d * sdr_pct[0] + inv_a * sdr_pct[1] + inv_a * sdr_pct[2],
                inv_a * sdr_pct[0] + inv_d * sdr_pct[1] + inv_a * sdr_pct[2],
                inv_a * sdr_pct[0] + inv_a * sdr_pct[1] + inv_d * sdr_pct[2],
            ]
        } else {
            sdr_pct
        };
        *px = [
            (sdr[0] / 100.0).clamp(0.0, 1.09),
            (sdr[1] / 100.0).clamp(0.0, 1.09),
            (sdr[2] / 100.0).clamp(0.0, 1.09),
        ];
    }
}

// ============================================================================
// CompiledFilmicSpline — log shaper + piecewise rational + desaturate.
// ============================================================================

/// Pre-extracted spline coefficients passed in by the host.
#[derive(Clone, Copy)]
pub(crate) struct FilmicSimdParams {
    pub m1_toe: f32,
    pub m1_shoulder: f32,
    pub m2_toe: f32,
    pub m2_shoulder: f32,
    pub m2_lin: f32,
    pub m3_toe: f32,
    pub m3_shoulder: f32,
    pub m4_toe: f32,
    pub m4_shoulder: f32,
    pub m1_lin: f32,
    pub latitude_min: f32,
    pub latitude_max: f32,
    pub grey_source: f32,
    pub black_source: f32,
    pub dynamic_range: f32,
    pub sigma_toe: f32,
    pub sigma_shoulder: f32,
    pub saturation: f32,
    pub luma: [f32; 3],
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(crate) fn filmic_spline_tier(token: Token, row: &mut [[f32; 3]], p: &FilmicSimdParams) {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let lr = f32x8::splat(token, p.luma[0]);
    let lg = f32x8::splat(token, p.luma[1]);
    let lb = f32x8::splat(token, p.luma[2]);
    let norm_floor = f32x8::splat(token, 1.525879e-05);
    let inv_grey = f32x8::splat(token, 1.0 / p.grey_source);
    let black_src = f32x8::splat(token, p.black_source);
    let inv_dyn = f32x8::splat(token, 1.0 / p.dynamic_range);
    let lat_min = f32x8::splat(token, p.latitude_min);
    let lat_max = f32x8::splat(token, p.latitude_max);
    let m1_toe = f32x8::splat(token, p.m1_toe);
    let m1_shoulder = f32x8::splat(token, p.m1_shoulder);
    let m2_toe = f32x8::splat(token, p.m2_toe);
    let m2_shoulder = f32x8::splat(token, p.m2_shoulder);
    let m2_lin = f32x8::splat(token, p.m2_lin);
    let m3_toe = f32x8::splat(token, p.m3_toe);
    let m3_shoulder = f32x8::splat(token, p.m3_shoulder);
    let m4_toe = f32x8::splat(token, p.m4_toe);
    let m4_shoulder = f32x8::splat(token, p.m4_shoulder);
    let m1_lin = f32x8::splat(token, p.m1_lin);
    let inv_sigma_toe = f32x8::splat(token, 1.0 / p.sigma_toe);
    let inv_sigma_shoulder = f32x8::splat(token, 1.0 / p.sigma_shoulder);
    let sat = f32x8::splat(token, p.saturation);
    let sat2 = f32x8::splat(token, 0.5 / sqrtf(p.saturation));
    let inv_ln2 = f32x8::splat(token, 1.0 / core::f32::consts::LN_2);

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

        let norm0 = (lr * r + lg * g + lb * b).max(norm_floor);
        let inv_norm = one / norm0;
        let mut rr = r * inv_norm;
        let mut rg = g * inv_norm;
        let mut rb = b * inv_norm;
        let min_ratio = rr.min(rg).min(rb);
        let neg_mask = min_ratio.simd_lt(zero);
        rr = f32x8::blend(neg_mask, rr - min_ratio, rr);
        rg = f32x8::blend(neg_mask, rg - min_ratio, rg);
        rb = f32x8::blend(neg_mask, rb - min_ratio, rb);

        let log2_norm = (norm0 * inv_grey).log2_midp();
        let shaped = ((log2_norm - black_src) * inv_dyn).max(zero).min(one);

        // exp(z) = exp2(z / ln 2)
        let z_toe = -shaped * shaped * inv_sigma_toe * sat2 * inv_ln2;
        let z_sh = -(one - shaped) * (one - shaped) * inv_sigma_shoulder * sat2 * inv_ln2;
        let key_toe = z_toe.exp2_midp();
        let key_sh = z_sh.exp2_midp();
        let desat = sat - (key_toe + key_sh) * sat;

        let xi_toe = lat_min - shaped;
        let rat_toe = xi_toe * (xi_toe * m2_toe + one);
        let toe_branch = m4_toe - m1_toe * rat_toe / (rat_toe + m3_toe);

        let xi_sh = shaped - lat_max;
        let rat_sh = xi_sh * (xi_sh * m2_shoulder + one);
        let sh_branch = m4_shoulder + m1_shoulder * rat_sh / (rat_sh + m3_shoulder);

        let lin_branch = m1_lin + shaped * m2_lin;

        let in_toe = shaped.simd_lt(lat_min);
        let in_sh = shaped.simd_gt(lat_max);
        let mid = f32x8::blend(in_sh, sh_branch, lin_branch);
        let mapped = f32x8::blend(in_toe, toe_branch, mid).max(zero).min(one);

        let one_minus_desat = one - desat;
        let r_out = ((rr + (one - rr) * one_minus_desat) * mapped)
            .max(zero)
            .min(one);
        let g_out = ((rg + (one - rg) * one_minus_desat) * mapped)
            .max(zero)
            .min(one);
        let b_out = ((rb + (one - rb) * one_minus_desat) * mapped)
            .max(zero)
            .min(one);

        let or_arr = r_out.to_array();
        let og_arr = g_out.to_array();
        let ob_arr = b_out.to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = or_arr[i];
            px[1] = og_arr[i];
            px[2] = ob_arr[i];
        }
    }

    for px in iter.into_remainder().iter_mut() {
        let mut norm =
            (px[0] * p.luma[0] + px[1] * p.luma[1] + px[2] * p.luma[2]).max(1.525879e-05);
        let mut ratios = [px[0] / norm, px[1] / norm, px[2] / norm];
        let min_ratio = ratios[0].min(ratios[1]).min(ratios[2]);
        if min_ratio < 0.0 {
            ratios[0] -= min_ratio;
            ratios[1] -= min_ratio;
            ratios[2] -= min_ratio;
        }
        norm = ((log2f(norm / p.grey_source) - p.black_source) / p.dynamic_range).clamp(0.0, 1.0);
        let radius_toe = norm;
        let radius_shoulder = 1.0 - norm;
        let sat2_s = 0.5 / sqrtf(p.saturation);
        let key_toe = expf(-radius_toe * radius_toe / p.sigma_toe * sat2_s);
        let key_sh = expf(-radius_shoulder * radius_shoulder / p.sigma_shoulder * sat2_s);
        let desat = p.saturation - (key_toe + key_sh) * p.saturation;
        let mapped = filmic_apply_spline_scalar(p, norm).clamp(0.0, 1.0);
        px[0] = ((ratios[0] + (1.0 - ratios[0]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0);
        px[1] = ((ratios[1] + (1.0 - ratios[1]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0);
        px[2] = ((ratios[2] + (1.0 - ratios[2]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0);
    }
}

#[inline]
fn filmic_apply_spline_scalar(p: &FilmicSimdParams, x: f32) -> f32 {
    if x < p.latitude_min {
        let xi = p.latitude_min - x;
        let rat = xi * (xi * p.m2_toe + 1.0);
        p.m4_toe - p.m1_toe * rat / (rat + p.m3_toe)
    } else if x > p.latitude_max {
        let xi = x - p.latitude_max;
        let rat = xi * (xi * p.m2_shoulder + 1.0);
        p.m4_shoulder + p.m1_shoulder * rat / (rat + p.m3_shoulder)
    } else {
        p.m1_lin + x * p.m2_lin
    }
}
