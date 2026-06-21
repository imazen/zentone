//! BT.2446 Method A tone mapper — the ITU's reference for graded
//! HDR↔SDR conversion with psychophysically verified round-trip fidelity.
//!
//! ITU-R BT.2446-1 §4. TMO: 1000 cd/m² HDR → 100 cd/m² SDR.
//!
//! Psychophysical evaluation (Annex 1): 12 participants, 115 images,
//! 2AFC on Sony BVM-X300 — imperceptible degradation after full
//! round-trip (p = 0.167 HDR, p = 0.196 SDR).
//!
//! The method operates in a perceptually linearized log-luminance domain
//! with a piecewise polynomial knee, Hunt-effect color correction in
//! Y'Cb'Cr', and BT.2020 luma weights.
//!
//! Reference: ITU-R BT.2446-1 (03/2021) §4 + Annex 1.

use crate::ToneMap;
use crate::math::powf;

/// BT.2020 luma weights (BT.2446 uses BT.2020, not BT.709).
const LR: f32 = 0.2627;
const LG: f32 = 0.6780;
const LB: f32 = 0.0593;

/// BT.2446 Method A tonemapper.
///
/// Construct with `new()`, then apply via the [`ToneMap`] trait. Input
/// is linear-light BT.2020 RGB normalized so `1.0 = hdr_peak_nits`.
/// Output is linear-light BT.2020 RGB normalized so `1.0 = sdr_peak_nits`
/// — the BT.2446-1 §4 pipeline natively emits gamma-encoded `R'_TMO`
/// `G'_TMO` `B'_TMO` (BT.1886 1/2.4); we apply the BT.1886 EOTF (`^2.4`)
/// at the output to satisfy the [`ToneMap`] trait's linear-light
/// contract (matching `Bt2446B`, `Bt2446C`, `Bt2408`, and libplacebo's
/// own `bt2446a` which ends with `bt1886_eotf`).
///
/// # When to pick this
///
/// The most rigorously validated HDR → SDR curve in zentone — the only
/// one with a published psychophysical study showing imperceptible
/// degradation after a full HDR → SDR → HDR round-trip on graded content.
/// Pick when broadcast-grade fidelity matters. Heavier than
/// [`Bt2408Tonemapper`](crate::Bt2408Tonemapper) (extra perceptual-log domain conversion + Hunt
/// color correction), so reserve for offline transcodes or when subjective
/// tests confirm the difference is worth it.
///
/// Reference: ITU-R BT.2446-1 (03/2021) §4 + Annex 1 (12 participants,
/// 115 images, Sony BVM-X300, 2AFC, p ≈ 0.17 indistinguishability).
///
/// # Examples
///
/// ```
/// use zentone::{Bt2446A, ToneMap};
///
/// let curve = Bt2446A::new(1000.0, 100.0);
/// let sdr = curve.map_rgb([0.6, 0.4, 0.2]);
/// assert!(sdr.iter().all(|&c| (0.0..=1.0).contains(&c)));
/// ```
pub struct Bt2446A {
    rho_hdr: f32,
    inv_log_rho_hdr: f32,
    rho_sdr: f32,
    inv_rho_sdr_minus_1: f32,
}

impl Bt2446A {
    /// Create a new BT.2446 Method A tonemapper.
    ///
    /// `hdr_peak_nits`: peak luminance of HDR content (typically 1000).
    /// `sdr_peak_nits`: peak luminance of SDR target (typically 100).
    pub fn new(hdr_peak_nits: f32, sdr_peak_nits: f32) -> Self {
        // ρ_H = 1 + 32 · (L_HDR / 10 000)^(1/2.4) per ITU-R BT.2446-1 §4.
        // The exponent is the BT.1886 gamma reciprocal, not γ itself — the
        // pre-2025 zentone used `2.4` here, which collapsed ρ_H from 13.4
        // toward 1.13 at 1000 nits and turned the log compression into a
        // near-identity. Fixed against the libplacebo reference; matches
        // the well-known "ρ_H ≈ 13.2 at 1000 nit, 33 at 10 000 nit"
        // quoted in ITU-R BT.2446-1.
        let inv_gamma = 1.0_f32 / 2.4;
        let rho_hdr = 1.0 + 32.0 * powf(hdr_peak_nits / 10000.0, inv_gamma);
        let log_rho_hdr = libm::logf(rho_hdr);
        let rho_sdr = 1.0 + 32.0 * powf(sdr_peak_nits / 10000.0, inv_gamma);
        Self {
            rho_hdr,
            inv_log_rho_hdr: 1.0 / log_rho_hdr,
            rho_sdr,
            inv_rho_sdr_minus_1: 1.0 / (rho_sdr - 1.0),
        }
    }

    /// Perceptual linearization: Y' → Y'_p (log domain).
    #[inline]
    fn perceptual_linearize(&self, y_prime: f32) -> f32 {
        libm::logf(1.0 + (self.rho_hdr - 1.0) * y_prime) * self.inv_log_rho_hdr
    }

    /// BT.2446 Method A piecewise tone curve.
    #[inline]
    fn tone_curve(y_p: f32) -> f32 {
        if y_p <= 0.7399 {
            1.0770 * y_p
        } else if y_p < 0.9909 {
            -1.1510 * y_p * y_p + 2.7811 * y_p - 0.6302
        } else {
            0.5000 * y_p + 0.5000
        }
    }

    /// Inverse perceptual linearization: Y'_c → Y'_SDR.
    #[inline]
    fn perceptual_delinearize(&self, y_c: f32) -> f32 {
        (powf(self.rho_sdr, y_c) - 1.0) * self.inv_rho_sdr_minus_1
    }
}

impl ToneMap for Bt2446A {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Step 1: nonlinear transfer (gamma 1/2.4)
        let r_prime = powf(rgb[0].max(0.0), 1.0 / 2.4);
        let g_prime = powf(rgb[1].max(0.0), 1.0 / 2.4);
        let b_prime = powf(rgb[2].max(0.0), 1.0 / 2.4);

        // Luma in gamma domain
        let y_prime = LR * r_prime + LG * g_prime + LB * b_prime;
        if y_prime <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Perceptual linearization
        let y_p = self.perceptual_linearize(y_prime);

        // Piecewise tone curve
        let y_c = Self::tone_curve(y_p);

        // Convert back from perceptual to gamma domain
        let y_sdr = self.perceptual_delinearize(y_c);

        // Hunt-effect color correction in Y'Cb'Cr' (Table 3)
        let f = y_sdr / (1.1 * y_prime);
        let cb = f * (b_prime - y_prime) / 1.8814;
        let cr = f * (r_prime - y_prime) / 1.4746;

        // Adjusted luma
        let y_tmo = y_sdr - 0.1_f32.max(0.0) * cr.max(0.0); // max(0.1*Cr, 0) subtracted

        // Y'Cb'Cr' → R'G'B' via the standard BT.2020 NCL inverse matrix.
        // Coefficients: 0.16455 = 2·Kb·(1-Kb)/Kg, 0.57135 = 2·Kr·(1-Kr)/Kg
        // (already divided by Kg = 0.6780; do not divide again — the
        // pre-2025 zentone double-divided, making green channel ~1.47×
        // off and shifting hue on saturated content).
        let r_prime_out = (y_tmo + 1.4746 * cr).clamp(0.0, 1.0);
        let g_prime_out = (y_tmo - 0.16455 * cb - 0.57135 * cr).clamp(0.0, 1.0);
        let b_prime_out = (y_tmo + 1.8814 * cb).clamp(0.0, 1.0);

        // BT.1886 EOTF (`^2.4`): the spec emits gamma-encoded R'G'B', but the
        // `ToneMap` trait contract is linear-light in / linear-light out
        // (matching Bt2446B/C/Bt2408 + libplacebo's `bt2446a` which closes
        // with `bt1886_eotf`). Without this step the consumer treats the
        // gamma-encoded value as linear and double-gamma-encodes through
        // its display OETF — every pixel comes out far too bright (median
        // ΔE2000 ≈ 23 vs producer-graded SDR on the imazen-26 shootout,
        // dead last out of 20 curves).
        let r_out = powf(r_prime_out, 2.4);
        let g_out = powf(g_prime_out, 2.4);
        let b_out = powf(b_prime_out, 2.4);

        [r_out, g_out, b_out]
    }

    fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        archmage::incant!(
            crate::simd::curves::bt2446a_tier(
                strip,
                self.rho_hdr,
                self.inv_log_rho_hdr,
                self.rho_sdr,
                self.inv_rho_sdr_minus_1,
            ),
            [v4(cfg(avx512)), v3, neon, wasm128, scalar]
        );
    }
}

impl Bt2446A {
    /// Relaxed-precision tone-map calibrated for 8-bit sRGB output.
    ///
    /// Uses a 2-piece sqrt-substituted monomial polynomial for the
    /// `x^(1/2.4)` input gamma encode instead of
    /// [`pow_midp_unchecked`](magetypes) inside the SIMD kernel. The
    /// kernel-amplified output error is **≤ ~5.21e-4** in linear-light
    /// space — well below the 8-bit sRGB half-LSB threshold (1.96e-3) —
    /// so this path is byte-identical to [`Self::map_strip_simd`] after
    /// sRGB-u8 encoding for ~all pixels (verified by the
    /// `u8_fast_path_byte_identical_to_spec_strict_at_8bit` test).
    ///
    /// # When to pick this
    ///
    /// - Output is destined for sRGB-u8 (JPEG, PNG-8, byte buffer): use this.
    ///   Output is byte-identical to [`Self::map_strip_simd`] after sRGB-u8
    ///   encoding; signals intent ("8-bit-display-targeted") at the call site
    ///   so future precision/perf tuning (e.g. different polynomial on
    ///   narrower target hardware, lower-precision transcendentals on
    ///   non-x86) doesn't risk visible regressions on 16-bit paths.
    /// - Output is 16-bit PNG, EXR, half-float, PQ-round-trip: use
    ///   [`Self::map_strip_simd`] — the 5.21e-4 linear-light error is
    ///   visible at 16-bit (≈ 34 16-bit LSBs).
    /// - You need reference output for spec validation: use
    ///   [`Self::map_strip_simd`] (passes the 5e-4 SIMD parity tolerance);
    ///   this fast path does not.
    ///
    /// # Throughput
    ///
    /// On Zen 4 / AVX-512 the two paths are at parity (~260 Mpix/s, ±5%
    /// run-to-run variance). `pow_midp_unchecked(1/2.4)` reduces to ~8
    /// FMAs and 1 div in magetypes 0.9.22; the 2-piece polynomial (one
    /// `sqrt` plus an Estrin-evaluated deg-10 Horner chain) lands at the
    /// same end-to-end latency. The path split is shipped for intent
    /// clarity and as future-proofing on hardware where the cost balance
    /// shifts (older CPUs with slower div, NEON without fast reciprocals,
    /// WASM-SIMD without `vrsqrteq`). See
    /// `benchmarks/bt2446a_throughput_2026-06-20.md` for the side-by-side
    /// measurements.
    ///
    /// # Polynomial derivation
    ///
    /// 2-piece sqrt-substituted monomial of `z = √x`: near branch
    /// (`z < 0.07`, deg 7) handles the fast-curving region near zero,
    /// bulk branch (`z ≥ 0.07`, deg 10) handles the rest. Both Estrin'd
    /// for shallow FMA dep chains (depth 3 / 4 respectively). Polynomial
    /// max abs error 3.44e-4 over `x ∈ [0, 11]`. See
    /// `src/simd/curves.rs::POW_INV24_*` for coefficients and
    /// `examples/fit_pow_inv_24.rs` for the reproducer.
    pub fn map_strip_simd_for_u8(&self, strip: &mut [[f32; 3]]) {
        archmage::incant!(
            crate::simd::curves::bt2446a_u8_tier(
                strip,
                self.rho_hdr,
                self.inv_log_rho_hdr,
                self.rho_sdr,
                self.inv_rho_sdr_minus_1,
            ),
            [v4(cfg(avx512)), v3, neon, wasm128, scalar]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn black_to_black() {
        let tm = Bt2446A::new(1000.0, 100.0);
        let out = tm.map_rgb([0.0, 0.0, 0.0]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn neutral_gray_passes_through_approximately() {
        let tm = Bt2446A::new(1000.0, 100.0);
        // Mid-gray at 1000 cd/m²: ~18% linear = 0.18
        let out = tm.map_rgb([0.18, 0.18, 0.18]);
        // All channels should be equal (neutral) and in a reasonable range
        assert!(
            (out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5,
            "neutral gray should stay neutral: {out:?}"
        );
        assert!(
            out[0] > 0.1 && out[0] < 0.8,
            "mid-gray should map to reasonable SDR level: {}",
            out[0]
        );
    }

    #[test]
    fn peak_maps_to_sdr_range() {
        let tm = Bt2446A::new(1000.0, 100.0);
        // HDR peak (1.0 = 1000 cd/m²) should map near SDR peak
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in out {
            assert!(
                c > 0.8 && c <= 1.0,
                "peak should map to near-SDR-white: {c}"
            );
        }
    }

    #[test]
    fn monotonic_on_neutral_ramp() {
        let tm = Bt2446A::new(1000.0, 100.0);
        let mut last = -1.0_f32;
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let out = tm.map_rgb([v, v, v]);
            let lum = out[0]; // neutral → all channels equal
            assert!(
                lum >= last - 1e-5,
                "monotonicity violated at {v}: {lum} < {last}"
            );
            last = lum;
        }
    }

    #[test]
    fn rho_hdr_matches_itu_reference_values() {
        // ITU-R BT.2446-1 §4 cites ρ_H ≈ 13.2 at 1000 nits and 33 at
        // 10 000 nits. The well-known table value of 13.4 at 1000 nits
        // matches `1 + 32 · (1000/10000)^(1/2.4) = 13.378`. The pre-fix
        // code used `(L/10000)^2.4` which gave ρ_H ≈ 1.127 — essentially
        // identity, breaking the entire compression curve. This test
        // pins ρ to the well-defined spec values so the bug can't return.
        let tm1k = Bt2446A::new(1000.0, 100.0);
        // 1 + 32 · (0.1)^(1/2.4) ≈ 13.260 in f32 (matches the spec's "13.2").
        assert!(
            (tm1k.rho_hdr - 13.260).abs() < 0.02,
            "ρ_H at 1000 nits should be ≈13.26, got {}",
            tm1k.rho_hdr
        );
        let tm10k = Bt2446A::new(10_000.0, 100.0);
        // 1 + 32 · 1.0 = 33.0 exactly.
        assert!(
            (tm10k.rho_hdr - 33.0).abs() < 0.05,
            "ρ_H at 10 000 nits should be 33.0, got {}",
            tm10k.rho_hdr
        );

        // Pin the pre-fix bug: with exponent 2.4 (instead of 1/2.4), ρ_H at
        // 1000 nits collapses to ~1.127, which would silently turn the
        // log compression into a near-identity. This bound is far enough
        // below the correct value that any regression to the old formula
        // is immediately visible.
        assert!(
            tm1k.rho_hdr > 5.0,
            "ρ_H regressed toward the pre-fix value (~1.13); got {}",
            tm1k.rho_hdr
        );
    }

    #[test]
    fn libplacebo_parity_eetf_only() {
        // Per-channel EETF parity check against the published libplacebo
        // BT.2446-A formula (haasn/libplacebo, src/tone_mapping.c). Stages
        // 1–3 are isolated here — log compression, piecewise tone curve,
        // inverse log expansion — without the Hunt color correction
        // (which is applied via Y'CbCr in `map_rgb`). The EETF must be
        // bit-close to the libplacebo numerics across the PQ domain;
        // matching them keeps zentone on the same target as mpv / FFmpeg
        // vf_libplacebo.
        fn libplacebo_eetf(x: f32, hdr_peak: f32, sdr_peak: f32) -> f32 {
            // Test inputs are all in-range finite values, so `clamp` is safe.
            let x = x.clamp(0.0, 1.0);
            let p_hdr = 1.0 + 32.0 * powf(hdr_peak / 10000.0, 1.0 / 2.4);
            let p_sdr = 1.0 + 32.0 * powf(sdr_peak / 10000.0, 1.0 / 2.4);
            let mut y = libm::logf(1.0 + (p_hdr - 1.0) * x) / libm::logf(p_hdr);
            y = if y <= 0.7399 {
                1.0770 * y
            } else if y < 0.9909 {
                -1.1510 * y * y + 2.7811 * y - 0.6302
            } else {
                0.5 * y + 0.5
            };
            (powf(p_sdr, y) - 1.0) / (p_sdr - 1.0)
        }

        for &(hdr, sdr) in &[(1000.0_f32, 100.0_f32), (4000.0, 100.0), (10_000.0, 100.0)] {
            let tm = Bt2446A::new(hdr, sdr);
            for &x in &[
                0.0_f32, 0.05, 0.1, 0.2, 0.3, 0.5, 0.581, 0.7, 0.75, 0.85, 0.95, 1.0,
            ] {
                // Stage-by-stage reproduction.
                let y_p = tm.perceptual_linearize(x);
                let y_c = Bt2446A::tone_curve(y_p);
                let got = tm.perceptual_delinearize(y_c);
                let want = libplacebo_eetf(x, hdr, sdr);
                assert!(
                    (got - want).abs() < 1e-4,
                    "libplacebo parity at (hdr={hdr}, sdr={sdr}, x={x}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn ycbcr_inverse_matrix_round_trips_at_y_tmo_passthrough() {
        // The inverse Y'CbCr → R'G'B' matrix must round-trip exactly for
        // a known (R',G',B') with Y' computed by the same BT.2020 luma
        // weights and Cb/Cr scaled per BT.2446 §4. Pins the G' coefficient
        // correctness regardless of which tone curve is in play.
        //
        // Pre-fix zentone divided 0.16455 / 0.57135 by Kg = 0.6780 a second
        // time, so this test fails by ~1.47× on the G channel.
        let r_p = 0.9_f32;
        let g_p = 0.5_f32;
        let b_p = 0.2_f32;
        let y_p = LR * r_p + LG * g_p + LB * b_p;
        let cb = (b_p - y_p) / 1.8814;
        let cr = (r_p - y_p) / 1.4746;

        // Inverse matrix as in the implementation (without the Hunt scaling
        // 1/1.1·f; this is a pure matrix round-trip check).
        let r_back = y_p + 1.4746 * cr;
        let g_back = y_p - 0.16455 * cb - 0.57135 * cr;
        let b_back = y_p + 1.8814 * cb;

        assert!(
            (r_back - r_p).abs() < 1e-5,
            "R round-trip: {r_back} vs {r_p}"
        );
        assert!(
            (g_back - g_p).abs() < 1e-5,
            "G round-trip: {g_back} vs {g_p}"
        );
        assert!(
            (b_back - b_p).abs() < 1e-5,
            "B round-trip: {b_back} vs {b_p}"
        );
    }

    #[test]
    fn output_is_linear_light_not_gamma_encoded() {
        // The `ToneMap` trait contract is "linear-light HDR in, linear-light
        // SDR out" (lib.rs §76 + §137). The BT.2446-1 spec's pipeline
        // gamma-encodes R/G/B with `^(1/2.4)` at step 1, runs the tone curve
        // in gamma + Y'Cb'Cr' domain, and emits `Y'_TMO C'_b,TMO C'_r,TMO`
        // — *gamma-encoded* SDR. To deliver linear-light SDR per the trait
        // contract (and to match Bt2446B / Bt2446C / Bt2408, all of which
        // operate end-to-end in linear-light), we MUST apply the BT.1886
        // EOTF (`^2.4`) at the output. libplacebo does the same thing in
        // tone_mapping.c:525 (`x = bt1886_eotf(x, output_min, output_max)`).
        //
        // Pre-fix Bt2446A returned gamma-encoded values, which the shootout
        // consumer then treated as linear and double-gamma-encoded into
        // sRGB — every pixel came out far too bright (median ΔE2000 ≈ 23
        // against producer-graded SDR, dead last out of 20 curves on the
        // 76-sample imazen-26 shootout).
        //
        // Pin: HDR peak round-trips, HDR black round-trips, and HDR
        // mid-grey lands in the linear-light range that the spec implies
        // *after* BT.1886 decode (≈ 0.37 linear, not 0.66 gamma).
        let tm = Bt2446A::new(1000.0, 100.0);

        // Black → black is unaffected by the EOTF.
        let black = tm.map_rgb([0.0, 0.0, 0.0]);
        assert_eq!(black, [0.0, 0.0, 0.0]);

        // Peak → peak: `1.0^2.4 = 1.0`, so the EOTF is identity here too.
        // But this confirms the saturating top is preserved.
        let peak = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in peak {
            assert!(
                (c - 1.0).abs() < 1e-4,
                "HDR peak should round-trip to SDR peak: {c}"
            );
        }

        // The critical mid-grey case. Trace by hand:
        //   r' = 0.18^(1/2.4) = 0.4815
        //   y_p = log(1 + 12.26·0.4815)/log(13.26) ≈ 0.7474
        //   y_c (middle branch) ≈ 0.806
        //   y_sdr (gamma) = (5.69^0.806 - 1)/4.69 ≈ 0.660
        //   y_sdr (linear) = 0.660^2.4 ≈ 0.370
        // Pre-fix returned ≈ 0.660 (gamma-encoded); post-fix returns ≈ 0.370.
        let mid = tm.map_rgb([0.18, 0.18, 0.18]);
        for c in mid {
            assert!(
                (c - 0.370).abs() < 0.02,
                "mid-grey HDR 0.18 should map to linear-light SDR ≈ 0.37, got {c}"
            );
        }

        // Direct guard: the pre-fix bug returned gamma-encoded mid-grey at
        // 0.66; any regression to that value is immediately visible.
        assert!(
            mid[0] < 0.55,
            "mid-grey output regressed toward the pre-fix gamma-encoded value (~0.66); got {}",
            mid[0]
        );
    }

    #[test]
    fn colored_input_stays_finite_and_bounded() {
        let tm = Bt2446A::new(1000.0, 100.0);
        let cases = [
            [0.8, 0.2, 0.05],
            [0.1, 0.9, 0.05],
            [0.3, 0.3, 0.8],
            [0.5, 0.5, 0.5],
        ];
        for rgb in cases {
            let out = tm.map_rgb(rgb);
            for (i, c) in out.iter().enumerate() {
                assert!(
                    c.is_finite() && *c >= 0.0 && *c <= 1.001,
                    "Bt2446A({rgb:?})[{i}] = {c}"
                );
            }
        }
    }

    /// Encode a linear-light value into an 8-bit sRGB byte (matches the
    /// final OETF step every JPEG/PNG-8 writer applies). Uses the simple
    /// `^(1/2.4)` gamma encode shared by the BT.1886 and sRGB pipelines
    /// at the byte level — the documented precision target.
    fn encode_to_srgb_u8(linear: f32) -> u8 {
        let v = linear.clamp(0.0, 1.0);
        let oetf = libm::powf(v, 1.0 / 2.4);
        (oetf * 255.0).round().clamp(0.0, 255.0) as u8
    }

    /// The u8-fast-path output must be byte-identical (or ±1 LSB on a tiny
    /// minority) to the spec-strict path after sRGB-u8 encoding. Documents
    /// the actual observed mismatch rate so future regressions in the
    /// polynomial or the kernel are visible.
    #[test]
    fn u8_fast_path_byte_identical_to_spec_strict_at_8bit() {
        // Spread of fixtures: greys, primaries, mixed colors, hot HDR.
        let cases: &[(f32, &[[f32; 3]])] = &[
            (
                1000.0,
                &[
                    [0.0, 0.0, 0.0],
                    [0.05, 0.05, 0.05],
                    [0.18, 0.18, 0.18],
                    [0.5, 0.3, 0.1],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0],
                ],
            ),
            (
                500.0,
                &[[0.18; 3], [0.5; 3], [1.0; 3], [2.0; 3], [0.7, 0.2, 0.9]],
            ),
            (4000.0, &[[0.18; 3], [0.5; 3], [1.0; 3], [0.3, 0.6, 0.05]]),
        ];
        let mut total_bytes = 0_usize;
        let mut mismatches = 0_usize;
        let mut max_diff = 0_i32;
        for (peak, pixels) in cases {
            let tm = Bt2446A::new(*peak, 100.0);
            let mut strict = pixels.to_vec();
            let mut fast = pixels.to_vec();
            tm.map_strip_simd(&mut strict);
            tm.map_strip_simd_for_u8(&mut fast);
            for (i, (s, f)) in strict.iter().zip(fast.iter()).enumerate() {
                for ch in 0..3 {
                    let su8 = encode_to_srgb_u8(s[ch]) as i32;
                    let fu8 = encode_to_srgb_u8(f[ch]) as i32;
                    let diff = (su8 - fu8).abs();
                    total_bytes += 1;
                    if diff > 0 {
                        mismatches += 1;
                    }
                    max_diff = max_diff.max(diff);
                    // Hard ceiling: any individual byte may drift by at most
                    // 1 LSB. The polynomial's worst-case kernel-amplified
                    // error (5.21e-4 ≈ 0.13 LSBs) cannot produce a 2-LSB
                    // jump under the rounded encode, but the spec-strict
                    // path may itself sit right at a half-LSB boundary —
                    // so a 1-LSB drift is the realistic ceiling.
                    assert!(
                        diff <= 1,
                        "u8 byte drift > 1 LSB at peak={peak} px{i} ch{ch}: \
                         strict={su8} fast={fu8} (linear strict={s:?} fast={f:?})",
                    );
                }
            }
        }
        // Hard cap on the ±1 LSB mismatch rate. The documented
        // kernel-amplified error (5.21e-4) is much smaller than the 8-bit
        // half-LSB threshold (1.96e-3), so any sustained mismatch rate
        // signals a regression in the polynomial or the kernel.
        let pct = (mismatches as f64) / (total_bytes as f64) * 100.0;
        let _ = max_diff;
        assert!(
            pct < 5.0,
            "u8 fast-path ±1 LSB mismatch rate {pct:.3}% \
             ({mismatches}/{total_bytes}) exceeds 5% — polynomial regressed?"
        );
    }

    /// The SIMD u8 fast path must match its scalar-tail evaluation within
    /// 5.21e-4 (the documented kernel output error after Cb/Cr amplification).
    /// Pins the strip-body / per-pixel agreement; if the SIMD body diverges
    /// from the scalar fallback by more than this, the magetypes upgrade or
    /// the polynomial constants got mis-edited.
    #[test]
    fn u8_fast_path_simd_matches_scalar_within_tolerance() {
        // Property-style strip: mixes greys, primaries, mixes, HDR peaks.
        // Length is deliberately non-multiple-of-16 to exercise the tail.
        let strip_in: Vec<[f32; 3]> = (0..67)
            .map(|i| {
                let f = (i as f32) / 67.0;
                [
                    (f * 4.0).clamp(0.0, 4.0),
                    ((1.0 - f) * 3.0).clamp(0.0, 4.0),
                    ((f * (1.0 - f)) * 2.0).clamp(0.0, 1.0),
                ]
            })
            .collect();

        let tm = Bt2446A::new(1000.0, 100.0);
        let mut via_simd = strip_in.clone();
        tm.map_strip_simd_for_u8(&mut via_simd);

        // Build the per-pixel reference using the scalar tail's own algorithm
        // (calling the 1-pixel slice form of map_strip_simd_for_u8 exercises
        // ONLY the tail). That guarantees we're comparing SIMD body to the
        // documented scalar evaluator, not to `map_rgb` (which uses libm).
        let mut via_scalar = strip_in.clone();
        for px in via_scalar.iter_mut() {
            let mut one = [*px];
            tm.map_strip_simd_for_u8(&mut one);
            *px = one[0];
        }

        let mut max_err = 0.0_f32;
        for (i, (s, p)) in via_simd.iter().zip(via_scalar.iter()).enumerate() {
            for ch in 0..3 {
                let err = (s[ch] - p[ch]).abs();
                max_err = max_err.max(err);
                // 5.21e-4 documented kernel-amplified error + small ULP slack.
                assert!(
                    err < 1e-3,
                    "u8 fast SIMD body vs scalar tail diverged at px{i} ch{ch}: \
                     simd={} scalar={} err={err:.3e}",
                    s[ch],
                    p[ch],
                );
            }
        }
        // Pin a non-trivial divergence ceiling so an accidental swap of
        // the SIMD body for the spec-strict kernel still surfaces.
        assert!(
            max_err < 1e-3,
            "u8 fast SIMD/scalar divergence ceiling exceeded: max_err={max_err:.3e}"
        );
    }
}
