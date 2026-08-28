//! Eight-pixel SIMD kernel for the ACES 2.0 forward output transform.
//!
//! A lane-parallel port of [`Aces2OutputTransform::forward`]: the same
//! stages (AP1 clamp → Hellwig JMh → tonescale + chroma compression → gamut
//! compression → JMh → display RGB) on eight pixels at once in SOA form,
//! with the CTL's data-dependent branches turned into masks. The
//! hue-indexed table lookups (reach M, display cusp, upper-hull gamma) are
//! scalar gathers per lane through the transform's own lookup methods, so
//! the tables and their search stay single-sourced.
//!
//! Transcendentals come from `magetypes` (`pow_midp`, `log2_midp`,
//! `exp2_midp`) plus in-file minimax polynomials for `atan2` (hue) and
//! `sin` / `cos` (hue → chroma axes), so the kernel is **not** bit-identical
//! to the scalar `libm` path: the parity test in this file pins it to
//! `2e-4` absolute + `2e-4` relative on every golden input and a random
//! sweep (measured worst 1.1e-4 normalised, 2026-08-28, aarch64 NEON), well
//! inside the `1e-3` the OpenColorIO parity gates allow — the kernel is also run through the forward output-transform
//! golden directly.
//!
//! Tier list follows `simd/curves.rs`: `v3` (AVX2), `neon`, `wasm128`,
//! `scalar`; the AVX-512 `v4` tier is omitted for the same magetypes
//! `pow_midp`-on-`f32x8` reason documented there.

use super::aces2::{
    Aces2OutputTransform, CAM_NL_OFFSET, COMPRESSION_THRESHOLD, J_SCALE, SMOOTH_CUSPS,
};
use crate::ToneMap;

const PI: f32 = core::f32::consts::PI;
const REF_LUMINANCE: f32 = 100.0;

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub(super) fn aces2_forward_tier(token: Token, strip: &mut [[f32; 3]], ot: &Aces2OutputTransform) {
    let splat = |v: f32| f32x8::splat(token, v);
    let zero = splat(0.0);
    let one = splat(1.0);
    let half = splat(0.5);
    let two = splat(2.0);

    // -- vector helpers ------------------------------------------------------

    // v · M with the CTL's row-vector convention.
    let mul_vm = |v: [f32x8; 3], m: &[[f32; 3]; 3]| -> [f32x8; 3] {
        [
            v[2].mul_add(
                splat(m[2][0]),
                v[1].mul_add(splat(m[1][0]), v[0] * splat(m[0][0])),
            ),
            v[2].mul_add(
                splat(m[2][1]),
                v[1].mul_add(splat(m[1][1]), v[0] * splat(m[0][1])),
            ),
            v[2].mul_add(
                splat(m[2][2]),
                v[1].mul_add(splat(m[1][2]), v[0] * splat(m[0][2])),
            ),
        ]
    };
    // Mask OR: a true lane is all-ones, so selecting `a` where `a` is true
    // and `b` elsewhere is `a | b`.
    let or_mask = |a: f32x8, b: f32x8| f32x8::blend(a, a, b);
    // `copysign(mag, sign_of)`.
    let copysign =
        |mag: f32x8, sign_of: f32x8| f32x8::blend(sign_of.simd_lt(zero), zero - mag, mag);
    // x^n for x ≥ 0 with a per-lane exponent (`pow_midp` takes a scalar).
    let pow_vec = |x: f32x8, n: f32x8| (x.log2_midp() * n).exp2_midp();

    // sin / cos on any argument: reduce to [-π, π], fold to [-π/2, π/2],
    // then odd/even minimax polynomials (|err| < 1e-6 over the fold).
    let sincos = |x: f32x8| -> (f32x8, f32x8) {
        let inv_2pi = splat(1.0 / (2.0 * PI));
        let x = x - (x * inv_2pi).round() * splat(2.0 * PI);
        let half_pi = splat(0.5 * PI);
        let fold = x.abs().simd_gt(half_pi);
        let folded = f32x8::blend(fold, copysign(splat(PI), x) - x, x);
        let x2 = folded * folded;
        let s = x2
            .mul_add(splat(2.752_556_2e-6), splat(-1.984_087_4e-4))
            .mul_add(x2, splat(8.333_331_0e-3))
            .mul_add(x2, splat(-1.666_666_7e-1))
            .mul_add(x2, one)
            * folded;
        let c = x2
            .mul_add(splat(2.476_049_5e-5), splat(-1.388_837_8e-3))
            .mul_add(x2, splat(4.166_663_8e-2))
            .mul_add(x2, splat(-0.5))
            .mul_add(x2, one);
        (s, f32x8::blend(fold, zero - c, c))
    };
    // atan2(y, x) in radians (|err| < 2e-6): atan on [0, 1] via an odd
    // degree-11 minimax polynomial, then octant / quadrant unfolding.
    let atan2 = |y: f32x8, x: f32x8| -> f32x8 {
        let ax = x.abs();
        let ay = y.abs();
        let swap = ay.simd_gt(ax);
        let num = f32x8::blend(swap, ax, ay);
        let den = f32x8::blend(swap, ay, ax);
        // Both zero → t = 0 (atan2(0, 0) = 0, as libm).
        let t = num / den.max(splat(f32::MIN_POSITIVE));
        let t2 = t * t;
        let a = t2
            .mul_add(splat(-1.172_120e-2), splat(5.265_332e-2))
            .mul_add(t2, splat(-1.164_328_7e-1))
            .mul_add(t2, splat(1.935_434_6e-1))
            .mul_add(t2, splat(-3.326_234_7e-1))
            .mul_add(t2, splat(9.999_772_6e-1))
            * t;
        let a = f32x8::blend(swap, splat(0.5 * PI) - a, a);
        let a = f32x8::blend(x.simd_lt(zero), splat(PI) - a, a);
        copysign(a, y)
    };

    // -- constants pulled out of the transform ----------------------------------
    let ip = &ot.input_params;
    let lp = &ot.limit_params;
    let ts = &ot.ts;
    let fwd_limit = splat(ts.forward_limit);
    let cam_nl_offset = splat(CAM_NL_OFFSET);
    let j_scale = splat(J_SCALE);
    let limit_j_max = splat(ot.limit_j_max);
    let inv_limit_j_max = splat(1.0 / ot.limit_j_max);
    let model_gamma_inv = ot.model_gamma_inv;
    let sat = splat(ot.sat);
    let sat_thr = splat(ot.sat_thr);
    let compr = splat(ot.compr);
    let ccs = splat(ot.chroma_compress_scale);
    let focus_dist = splat(ot.focus_dist);
    let deg_to_rad = splat(PI / 180.0);
    let rad_to_deg = splat(180.0 / PI);
    let three_sixty = splat(360.0);
    let eps_1e_3 = splat(0.001);
    let compression_threshold = splat(COMPRESSION_THRESHOLD);
    let log10_2 = splat(core::f32::consts::LOG10_2);

    let cone_fwd_abs = |rc: f32x8| {
        let f_l_y = rc.pow_midp(0.42);
        f_l_y / (cam_nl_offset + f_l_y)
    };
    let cone_fwd = |v: f32x8| copysign(cone_fwd_abs(v.abs()), v);
    let cone_inv_abs = |ra: f32x8| {
        let ra_lim = ra.min(splat(0.99));
        let f_l_y = (cam_nl_offset * ra_lim) / (one - ra_lim);
        f_l_y.pow_midp(1.0 / 0.42)
    };
    let cone_inv = |v: f32x8| copysign(cone_inv_abs(v.abs()), v);
    // `y_to_j` for the input CAM (used by the tonescale stage).
    let y_to_j_in = |y: f32x8| {
        let ra = cone_fwd_abs(y.abs() * splat(ip.f_l_n));
        copysign(j_scale * (ra * splat(ip.inv_a_w_j)).pow_midp(ip.cz), y)
    };
    let j_to_y_in = |j: f32x8| {
        let a = (j.abs() * splat(1.0 / J_SCALE)).pow_midp(ip.inv_cz);
        cone_inv_abs(splat(ip.a_w_j) * a) / splat(ip.f_l_n)
    };
    // The toe from the chroma-compression stage (forward only).
    let toe = |x: f32x8, limit: f32x8, k1_in: f32x8, k2_in: f32x8| {
        let k2 = k2_in.max(eps_1e_3);
        let k1 = (k1_in * k1_in + k2 * k2).sqrt();
        let k3 = (limit + k1) / (limit + k2);
        let minus_b = k3 * x - k1;
        let minus_c = k2 * k3 * x;
        let y = half * (minus_b + (minus_b * minus_b + splat(4.0) * minus_c).sqrt());
        f32x8::blend(x.simd_gt(limit), x, y)
    };

    let mut chunks = strip.chunks_exact_mut(8);
    for chunk in &mut chunks {
        let mut ra = [0.0f32; 8];
        let mut ga = [0.0f32; 8];
        let mut ba = [0.0f32; 8];
        for (i, px) in chunk.iter().enumerate() {
            ra[i] = px[0];
            ga[i] = px[1];
            ba[i] = px[2];
        }
        let aces = [
            f32x8::load(token, &ra),
            f32x8::load(token, &ga),
            f32x8::load(token, &ba),
        ];

        // -- clamp_AP0_to_AP1 (NaN → 0 via max-then-min) ---------------------
        let ap1 = mul_vm(aces, &ot.ap0_to_ap1);
        let ap1 = [
            ap1[0].max(zero).min(fwd_limit),
            ap1[1].max(zero).min(fwd_limit),
            ap1[2].max(zero).min(fwd_limit),
        ];
        let ap0 = mul_vm(ap1, &ot.ap1_to_ap0);

        // -- RGB → JMh (input CAM) --------------------------------------------
        let rgb_m = mul_vm(ap0, &ip.rgb_to_cam16_c);
        let rgb_a = [cone_fwd(rgb_m[0]), cone_fwd(rgb_m[1]), cone_fwd(rgb_m[2])];
        let aab = mul_vm(rgb_a, &ip.cone_response_to_aab);
        let achromatic = aab[0].simd_gt(zero);
        let j = f32x8::blend(achromatic, j_scale * aab[0].max(zero).pow_midp(ip.cz), zero);
        let m = f32x8::blend(achromatic, (aab[1] * aab[1] + aab[2] * aab[2]).sqrt(), zero);
        let h_raw = atan2(aab[2], aab[1]) * rad_to_deg;
        // wrap_to_360: fmod then fix negatives.
        let h_wrapped = h_raw - (h_raw * splat(1.0 / 360.0)).floor() * three_sixty;
        let h = f32x8::blend(achromatic, h_wrapped, zero);

        // -- tonescale of J -----------------------------------------------------
        let linear = j_to_y_in(j) * splat(1.0 / REF_LUMINANCE);
        let f = splat(ts.m_2) * (linear.max(zero) / (linear + splat(ts.s_2))).pow_midp(ts.g);
        let hts = (f * f / (f + splat(ts.t_1))).max(zero) * splat(ts.n_r);
        let j_ts = y_to_j_in(hts);

        // -- chroma compression --------------------------------------------------
        // Per-lane reach-M gather at hue h (1° table).
        let h_arr = h.to_array();
        let mut reach_arr = [0.0f32; 8];
        for (r, &hh) in reach_arr.iter_mut().zip(h_arr.iter()) {
            *r = ot.reach_m_from_table(hh);
        }
        let reach_m = f32x8::load(token, &reach_arr);
        let n_j = j_ts * inv_limit_j_max;
        let sn_j = (one - n_j).max(zero);
        let (sin_h, cos_h) = sincos(h * deg_to_rad);
        let cos_h2 = cos_h * cos_h - sin_h * sin_h;
        let sin_h2 = two * cos_h * sin_h;
        let cos_h3 = splat(4.0) * cos_h * cos_h * cos_h - splat(3.0) * cos_h;
        let sin_h3 = splat(3.0) * sin_h - splat(4.0) * sin_h * sin_h * sin_h;
        let m_norm = (splat(11.34072) * cos_h
            + splat(16.46899) * cos_h2
            + splat(7.88380) * cos_h3
            + splat(14.66441) * sin_h
            - splat(6.37224) * sin_h2
            + splat(9.19364) * sin_h3
            + splat(77.12896))
            * ccs;
        let limit = n_j.pow_midp(model_gamma_inv) * reach_m / m_norm;
        let toe_limit = limit - eps_1e_3;
        let toe_sn_j_sat = sn_j * sat;
        let toe_sqrt = (n_j * n_j + sat_thr).sqrt();
        let toe_n_j_compr = n_j * compr;
        // j == 0 ⇒ m == 0 (both come from aab[0] ≤ 0), so the ratio's NaN
        // there is masked by the `m != 0` blend below.
        let m_c = m * (j_ts / j).pow_midp(model_gamma_inv) / m_norm;
        let m_c = limit - toe(limit - m_c, toe_limit, toe_sn_j_sat, toe_sqrt);
        let m_c = toe(m_c, limit, toe_n_j_compr, sn_j) * m_norm;
        let m_tc = f32x8::blend(m.simd_ne(zero), m_c, m);

        // -- gamut compression -----------------------------------------------------
        // Per-lane hue-dependent parameter gather.
        let mut cusp_j = [0.0f32; 8];
        let mut cusp_m = [0.0f32; 8];
        let mut gamma_top_inv = [0.0f32; 8];
        let mut gamma_bottom_inv = [0.0f32; 8];
        let mut focus_j_a = [0.0f32; 8];
        let mut thr_a = [0.0f32; 8];
        for i in 0..8 {
            let p = ot.hue_dependent_gamut_params(h_arr[i]);
            cusp_j[i] = p.jm_cusp[0];
            cusp_m[i] = p.jm_cusp[1];
            gamma_top_inv[i] = p.gamma_top_inv;
            gamma_bottom_inv[i] = p.gamma_bottom_inv;
            focus_j_a[i] = p.focus_j;
            thr_a[i] = p.analytical_threshold;
        }
        let cusp_j = f32x8::load(token, &cusp_j);
        let cusp_m = f32x8::load(token, &cusp_m);
        let gamma_top_inv = f32x8::load(token, &gamma_top_inv);
        let gamma_bottom_inv = f32x8::load(token, &gamma_bottom_inv);
        let focus_j = f32x8::load(token, &focus_j_a);
        let analytical_threshold = f32x8::load(token, &thr_a);

        // get_focus_gain(J = j_ts).
        let gain_base = limit_j_max * focus_dist;
        let adj = ((limit_j_max - analytical_threshold) / (limit_j_max - j_ts).max(splat(0.0001)))
            .log2_midp()
            * log10_2;
        let adj = adj * adj + one;
        let slope_gain = f32x8::blend(
            j_ts.simd_gt(analytical_threshold),
            gain_base * adj,
            gain_base,
        );
        // solve_J_intersect for the source and the cusp.
        let solve_j_intersect = |jj: f32x8, mm: f32x8| {
            let m_scaled = mm / slope_gain;
            let a = m_scaled / focus_j;
            let b_lo = one - m_scaled;
            let c_lo = zero - jj;
            let root_lo = (b_lo * b_lo - splat(4.0) * a * c_lo).sqrt();
            let lo = splat(-2.0) * c_lo / (b_lo + root_lo);
            let b_hi = zero - (one + m_scaled + limit_j_max * a);
            let c_hi = limit_j_max * m_scaled + jj;
            let root_hi = (b_hi * b_hi - splat(4.0) * a * c_hi).sqrt();
            let hi = splat(-2.0) * c_hi / (b_hi - root_hi);
            f32x8::blend(jj.simd_lt(focus_j), lo, hi)
        };
        let j_int_src = solve_j_intersect(j_ts, m_tc);
        let j_int_cusp = solve_j_intersect(cusp_j, cusp_m);
        // compute_compression_vector_slope.
        let dir = f32x8::blend(
            j_int_src.simd_lt(focus_j),
            j_int_src,
            limit_j_max - j_int_src,
        );
        let gamut_slope = dir * (j_int_src - focus_j) / (focus_j * slope_gain);
        // find_gamut_boundary_intersection: lower hull, flipped upper hull,
        // smooth minimum.
        let est = |j_axis: f32x8,
                   slope: f32x8,
                   inv_gamma: f32x8,
                   j_max: f32x8,
                   m_max: f32x8,
                   j_ref: f32x8| {
            let shifted = j_ref * pow_vec(j_axis / j_ref, inv_gamma);
            shifted * m_max / (j_max - slope * m_max)
        };
        let m_lower = est(
            j_int_src,
            gamut_slope,
            gamma_bottom_inv,
            cusp_j,
            cusp_m,
            j_int_cusp,
        );
        let m_upper = est(
            limit_j_max - j_int_src,
            zero - gamut_slope,
            gamma_top_inv,
            limit_j_max - cusp_j,
            cusp_m,
            limit_j_max - j_int_cusp,
        );
        let s_scaled = splat(SMOOTH_CUSPS) * cusp_m;
        let hh = (s_scaled - (m_lower - m_upper).abs()).max(zero) / s_scaled;
        let gamut_boundary_m = m_lower.min(m_upper) - hh * hh * hh * s_scaled * splat(1.0 / 6.0);
        // Reach boundary along the same line.
        let reach_boundary_m = est(
            j_int_src,
            gamut_slope,
            splat(model_gamma_inv),
            limit_j_max,
            reach_m,
            limit_j_max,
        );
        // remap_M (forward).
        let boundary_ratio = gamut_boundary_m / reach_boundary_m;
        let proportion = boundary_ratio.max(compression_threshold);
        let threshold = proportion * gamut_boundary_m;
        let m_offset = m_tc - threshold;
        let gamut_offset = gamut_boundary_m - threshold;
        let reach_offset = reach_boundary_m - threshold;
        let scale = reach_offset / ((reach_offset / gamut_offset) - one);
        let nd = m_offset / scale;
        let remapped = threshold + scale * nd / (one + nd);
        let keep = or_mask(m_tc.simd_le(threshold), proportion.simd_ge(one));
        let remapped_m = f32x8::blend(keep, m_tc, remapped);
        let j_gc = j_int_src + remapped_m * gamut_slope;
        // Early exits of gamut_compress_fwd / compress_gamut.
        let boundary_ok = gamut_boundary_m.simd_gt(zero);
        let j_c = f32x8::blend(boundary_ok, j_gc, j_ts);
        let m_c = f32x8::blend(boundary_ok, remapped_m, zero);
        let skip = or_mask(m_tc.simd_le(zero), j_ts.simd_gt(limit_j_max));
        let j_c = f32x8::blend(skip, j_ts, j_c);
        let m_c = f32x8::blend(skip, zero, m_c);
        let black = j_ts.simd_le(zero);
        let j_c = f32x8::blend(black, zero, j_c);
        let m_c = f32x8::blend(black, zero, m_c);

        // -- JMh → display RGB (limiting CAM) ---------------------------------------
        let a_lim = (j_c.abs() * splat(1.0 / J_SCALE)).pow_midp(lp.inv_cz);
        let aab_out = [a_lim, m_c * cos_h, m_c * sin_h];
        let rgb_a = mul_vm(aab_out, &lp.aab_to_cone_response);
        let rgb_m = [cone_inv(rgb_a[0]), cone_inv(rgb_a[1]), cone_inv(rgb_a[2])];
        let out = mul_vm(rgb_m, &lp.cam16_c_to_rgb);

        let ro = out[0].to_array();
        let go = out[1].to_array();
        let bo = out[2].to_array();
        for (i, px) in chunk.iter_mut().enumerate() {
            px[0] = ro[i];
            px[1] = go[i];
            px[2] = bo[i];
        }
    }
    for px in chunks.into_remainder() {
        *px = ot.map_rgb(*px);
    }
}

#[cfg(test)]
mod tests {
    use super::super::aces2::{Aces2Config, Aces2OutputTransform, Chromaticities};
    use crate::ToneMap;
    use alloc::vec::Vec;

    const OT_CSV: &str =
        include_str!("../../reference-checks/golden/aces2_ocio_output_transform.csv");

    fn rows(csv: &str) -> Vec<Vec<&str>> {
        csv.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| l.split(',').collect())
            .collect()
    }

    fn f(s: &str) -> f32 {
        s.trim().parse().unwrap()
    }

    fn prims(name: &str) -> Chromaticities {
        match name {
            "rec709" => Chromaticities::REC709,
            "p3d65" => Chromaticities::P3_D65,
            "rec2020" => Chromaticities::REC2020,
            other => panic!("{other}"),
        }
    }

    /// Deterministic LCG so the sweep needs no dev-dependency.
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 40) as f32) / (1u64 << 24) as f32
    }

    /// SIMD strip vs scalar `forward` on the golden inputs, hostile values,
    /// and a random log-uniform sweep, for three display configurations.
    #[test]
    fn strip_matches_scalar() {
        let configs = [
            (100.0, Chromaticities::REC709),
            (100.0, Chromaticities::P3_D65),
            (1000.0, Chromaticities::REC2020),
            (4000.0, Chromaticities::REC2020),
        ];
        let mut seed = 0x5eed_2026u64;
        let mut worst = 0.0f32;
        for (peak, lim) in configs {
            let ot = Aces2OutputTransform::new(Aces2Config {
                peak_luminance: peak,
                limiting_primaries: lim,
            })
            .unwrap();
            let mut inputs: Vec<[f32; 3]> = rows(OT_CSV)
                .iter()
                .map(|r| [f(r[2]), f(r[3]), f(r[4])])
                .collect();
            for _ in 0..1000 {
                let lum = libm::powf(10.0, lcg(&mut seed) * 6.0 - 4.0);
                let (r, g, b) = (lcg(&mut seed), lcg(&mut seed), lcg(&mut seed));
                let mx = r.max(g).max(b).max(1e-6);
                inputs.push([lum * r / mx, lum * g / mx, lum * b / mx]);
            }
            for _ in 0..100 {
                inputs.push([
                    lcg(&mut seed) * 0.4 - 0.1,
                    lcg(&mut seed) * 0.4 - 0.1,
                    lcg(&mut seed) * 0.4 - 0.1,
                ]);
            }
            inputs.extend([
                [0.0; 3],
                [f32::NAN, 0.5, 0.5],
                [f32::INFINITY, 0.0, 0.0],
                [-1.0e30, 1.0e30, 0.0],
                [f32::MAX, f32::MIN, 0.0],
                [1e-9, 1e-9, 1e-9],
                [10_000.0; 3],
            ]);
            // Odd length so the scalar tail is exercised too.
            let mut via_strip = inputs.clone();
            ot.map_strip_simd(&mut via_strip);
            for (i, (got, src)) in via_strip.iter().zip(inputs.iter()).enumerate() {
                let want = ot.forward(*src);
                for c in 0..3 {
                    assert!(got[c].is_finite(), "{peak} {src:?} lane {c} = {}", got[c]);
                    let err = (got[c] - want[c]).abs();
                    let tol = 2e-4 + 2e-4 * want[c].abs();
                    assert!(
                        err <= tol,
                        "peak {peak} px {i} {src:?}: simd {got:?} vs scalar {want:?} (ch {c}, err {err:e})"
                    );
                    worst = worst.max(err / (1.0 + want[c].abs()));
                }
            }
        }
        // The documented bound; tighten here if the kernel improves.
        assert!(worst <= 1.5e-4, "worst normalised error {worst:e}");
    }

    /// The SIMD path must also hold the OpenColorIO forward golden on its own.
    #[test]
    fn strip_matches_ocio_forward() {
        let mut ots: Vec<(f32, &str, Aces2OutputTransform)> = Vec::new();
        let all = rows(OT_CSV);
        for r in &all {
            let peak = f(r[0]);
            if ots.iter().all(|(p, n, _)| !(*p == peak && *n == r[1])) {
                let cfg = Aces2Config {
                    peak_luminance: peak,
                    limiting_primaries: prims(r[1]),
                };
                ots.push((peak, r[1], Aces2OutputTransform::new(cfg).unwrap()));
            }
        }
        let mut checked = 0;
        for (peak, name, ot) in &ots {
            // OCIO's fixed function skips the CTL's AP1 clamp, so only rows
            // inside AP1 are comparable (as in the scalar golden test).
            let sel: Vec<&Vec<&str>> = all
                .iter()
                .filter(|r| f(r[0]) == *peak && r[1] == *name)
                .filter(|r| {
                    let rgb = [f(r[2]), f(r[3]), f(r[4])];
                    let c = ot.clamp_ap0_to_ap1(rgb);
                    (0..3).all(|k| (c[k] - rgb[k]).abs() <= 1e-6 + rgb[k].abs() * 1e-6)
                })
                .collect();
            let mut strip: Vec<[f32; 3]> =
                sel.iter().map(|r| [f(r[2]), f(r[3]), f(r[4])]).collect();
            ot.map_strip_simd(&mut strip);
            for (r, got) in sel.iter().zip(strip.iter()) {
                let want = [f(r[5]), f(r[6]), f(r[7])];
                for c in 0..3 {
                    assert!(
                        (got[c] - want[c]).abs() <= 1e-3 + 1e-3 * want[c].abs(),
                        "peak {peak} {name} {:?}: {got:?} vs {want:?}",
                        &r[2..5]
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 150, "only {checked}");
    }
}
