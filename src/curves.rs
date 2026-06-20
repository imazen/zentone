//! Classical tone mapping curves and unified [`ToneMapCurve`] dispatch.
//!
//! All functions here operate in scene-linear light. Output is clamped to
//! `[0, 1]` or `[0, ~1]` depending on the curve; apply an OETF (sRGB, BT.709,
//! …) afterwards for display-encoded output.

use crate::ToneMap;
use crate::math::{log2f, powf};

// ============================================================================
// Simple per-channel curves
// ============================================================================

/// Simple per-channel Reinhard: `x / (1 + x)`.
///
/// Negative input is clamped to 0 (linear light is non-negative).
#[inline]
pub fn reinhard_simple(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    x / (1.0 + x)
}

/// Clamp tone map: clamps to `[0, 1]`.
#[inline]
pub(crate) fn clamp_tonemap(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Extended Reinhard with max luminance.
///
/// `L_out = L_in * (1 + L_in / L_max²) / (1 + L_in)`.
#[inline]
pub fn reinhard_extended(l_in: f32, l_max: f32) -> f32 {
    if l_in <= 0.0 {
        return 0.0;
    }
    let l_max_sq = l_max * l_max;
    // Reassociated from `l_in * (1 + l_in/l_max_sq) / (1 + l_in)` to keep every
    // intermediate finite for large finite `l_in`. The original numerator
    // `l_in * (1 + l_in/l_max_sq)` grows like l_in² and overflows f32 to +Inf
    // around l_in ≳ 1e19, which then propagates as ±Inf through the caller's
    // `rgb * scale` (the NaN/Inf-output fuzz failure, zentone#21). This form
    // factors out `l_in/(1 + l_in)` (bounded ≤ 1) first, so the product only
    // reaches ~l_in/l_max_sq — finite for all finite l_in. Algebraically
    // identical; differs only by ≤1 ULP in the normal range.
    (l_in / (1.0 + l_in)) * (1.0 + l_in / l_max_sq)
}

/// Reinhard-Jodie tone mapping (per-channel and luminance-based blend).
pub fn reinhard_jodie(rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
    // Clamp to non-negative linear light and clamp the result to [0, 1],
    // matching the SIMD row path (`simd::reinhard_jodie_3_tier`, which does
    // `.max(zero)` on inputs and `.min(one).max(zero)` on output). Without this
    // the scalar and SIMD paths diverged on negative input (zentone#21 sweep).
    let rgb = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let luma = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
    if luma <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let luma_scale = 1.0 / (1.0 + luma);
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let tv = rgb[i] / (1.0 + rgb[i]);
        // Inputs are clamped non-negative above, so the expression is finite;
        // `clamp` is equivalent to `.min(1.0).max(0.0)` here (no NaN to differ
        // on) and matches the SIMD path's [0, 1] output range.
        out[i] = ((1.0 - tv) * (rgb[i] * luma_scale) + tv * tv).clamp(0.0, 1.0);
    }
    out
}

/// Tuned Reinhard with display-aware parameters.
///
/// Derived from content and display peak luminance (nits), with a reference
/// white of 203 nits. Internal; used by `ToneMapCurve::TunedReinhard`.
pub(crate) fn tuned_reinhard(luma: f32, content_max: f32, display_max: f32) -> f32 {
    let white_point = 203.0_f32;
    let ld = content_max / white_point;
    let w_a = (display_max / white_point) / (ld * ld);
    let w_b = 1.0 / (display_max / white_point);
    (1.0 + w_a * luma) / (1.0 + w_b * luma)
}

// ============================================================================
// Filmic curves
// ============================================================================

/// Narkowicz ACES-inspired filmic S-curve.
///
/// Cheap scene-linear approximation of ACES RRT+ODT. Verified bit-exact
/// against Krzysztof Narkowicz's published formula (see
/// [`tests/cross_reference.rs::narkowicz_matches_reference`]).
///
/// # Input domain
///
/// **The curve expects pre-exposed scene-linear input** where:
/// - Scene mid-grey (18 % grey) sits at `x ≈ 0.18` → SDR output `≈ 0.267`
/// - Scene white sits at `x ≈ 1.0` → output `≈ 0.804`
/// - Specular highlights at `x ≈ 2.0` → output `≈ 0.915`
/// - `x ≥ ~10` saturates to 1.0
///
/// The curve has saturation asymptote `a/c = 2.51/2.43 ≈ 1.033` (clamped
/// to 1.0). Inputs above ~10 are mapped near 1.0 with vanishing
/// distinction.
///
/// **For HDR→SDR rendering**, the caller MUST apply an exposure scale
/// that places scene mid-grey at `x ≈ 0.18`. Two common mistakes:
///
/// - Passing `hdr / source_peak_nits` (e.g. nits / 1000): under-exposes
///   the curve. A 50-nit mid-grey lands at `x = 0.05`, output `≈ 0.044`
///   — too dark.
/// - Passing `hdr * (diffuse_white_nits / target_peak_nits)` (e.g.
///   `* 2.03`): over-exposes the curve. A 50-nit mid-grey lands at
///   `x = 0.50`, output `≈ 0.62` — washed out.
///
/// For an HDR buffer anchored `1.0 = diffuse_white_nits` (203 nits per
/// ultrahdr / heic convention), the buffer can be passed **directly,
/// unscaled** — scene mid-grey is already at 0.18 (=36.5 nits / 203
/// nits) and diffuse white is at 1.0. The shootout in
/// `examples/hdr_tone_map_shootout_full.rs` multiplies by
/// `diffuse_white_nits / target_peak_nits = 2.03` before invoking
/// Narkowicz; this places scene mid-grey at `x ≈ 0.366` (output ≈
/// 0.51), which is too bright and is the over-exposed frame above.
///
/// # Look character
///
/// Narkowicz has a strong "filmic" shape: deep shadow toe + saturated
/// highlight shoulder + S-curve through the midtones. It does NOT match
/// producer-graded SDR (camera ISP / colorist-graded output), which is
/// typically more linear through the midtones. A 76-sample HDR→SDR
/// shootout against producer-graded SDR scored Narkowicz at ΔE2000 ≈
/// 22.4, well behind BT.2390 and BT.2446B. The curve is implemented
/// correctly; it just isn't a faithful target for ISP-graded content.
/// Prefer [`bt2390_tonemap`] or [`Bt2446B`](crate::Bt2446B) when the
/// SDR target is producer-graded camera output.
///
/// # Stability
///
/// At extreme values (`x > ~1e18`), `x²` overflows `f32` → `Inf/Inf =
/// NaN`. The early return for `x > 65536` (well past the 1.0 clamp
/// threshold) keeps the curve panic- and NaN-free across all inputs.
///
/// Reference: Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve"
/// <https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/>
#[inline]
pub fn filmic_narkowicz(x: f32) -> f32 {
    // Asymptote: a/c = 2.51/2.43 ≈ 1.033 → clamped to 1.0.
    // At extreme values (>1e18), x² overflows f32 → Inf/Inf = NaN.
    // Early return for x > 65536 (well past the 1.0 clamp threshold).
    if x > 65536.0 {
        return 1.0;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let a = 2.51_f32;
    let b = 0.03_f32;
    let c = 2.43_f32;
    let d = 0.59_f32;
    let e = 0.14_f32;
    let num = x * (a * x + b);
    let den = x * (c * x + d) + e;
    (num / den).clamp(0.0, 1.0)
}

/// Hable filmic tone mapping (John Hable, GDC 2010).
pub fn hable_filmic(v: f32) -> f32 {
    #[inline(always)]
    const fn partial(x: f32) -> f32 {
        const A: f32 = 0.15;
        const B: f32 = 0.50;
        const C: f32 = 0.10;
        const D: f32 = 0.20;
        const E: f32 = 0.02;
        const F: f32 = 0.30;
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }
    const EXPOSURE_BIAS: f32 = 2.0;
    const W: f32 = 11.2;
    const W_SCALE: f32 = 1.0 / partial(W);
    (partial(v * EXPOSURE_BIAS) * W_SCALE).min(1.0)
}

/// ACES AP1 filmic tone mapping (Krzysztof Narkowicz fit, RRT+ODT).
///
/// **Near-black negativity:** The RRT's `−0.000090537` offset causes
/// output to go slightly negative (≈ −2.7e-4) at very low input (< 0.002).
/// This is inherent to the Narkowicz approximation. Clamp to 0 if your
/// pipeline requires non-negative output.
#[allow(clippy::excessive_precision)]
pub fn aces_ap1(rgb: [f32; 3]) -> [f32; 3] {
    let a = 0.59719 * rgb[0] + 0.35458 * rgb[1] + 0.04823 * rgb[2];
    let b = 0.07600 * rgb[0] + 0.90834 * rgb[1] + 0.01566 * rgb[2];
    let c = 0.02840 * rgb[0] + 0.13383 * rgb[1] + 0.83777 * rgb[2];
    let ra = a * (a + 0.0245786) - 0.000090537;
    let rb = a * (a * 0.983729 + 0.4329510) + 0.238081;
    let ga = b * (b + 0.0245786) - 0.000090537;
    let gb = b * (b * 0.983729 + 0.4329510) + 0.238081;
    let ba = c * (c + 0.0245786) - 0.000090537;
    let bb = c * (c * 0.983729 + 0.4329510) + 0.238081;
    let mr = ra / rb;
    let mg = ga / gb;
    let mb = ba / bb;
    [
        (1.60475 * mr - 0.53108 * mg - 0.07367 * mb).min(1.0),
        (-0.10208 * mr + 1.10813 * mg - 0.00605 * mb).min(1.0),
        (-0.00327 * mr - 0.07276 * mg + 1.07602 * mb).min(1.0),
    ]
}

// ============================================================================
// AgX
// ============================================================================

/// AgX look preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AgxLook {
    /// Default AgX (no look applied).
    Default,
    /// Punchy: increased saturation (1.4x).
    Punchy,
    /// Golden: warm tone with reduced blue.
    Golden,
}

/// AgX tone mapping (Blender).
///
/// Operates in a log2 domain with a polynomial contrast curve and an
/// inset/outset matrix pair.
#[allow(clippy::excessive_precision)]
pub fn agx_tonemap(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    const AGX_MIN_EV: f32 = -12.47393;
    const AGX_MAX_EV: f32 = 4.026069;
    const RECIP_EV: f32 = 1.0 / (AGX_MAX_EV - AGX_MIN_EV);

    let z = [rgb[0].abs(), rgb[1].abs(), rgb[2].abs()];
    // Inset matrix (Blender AgX, OCIO row-major — rows sum to 1.0).
    // Previous version was transposed, causing non-neutral gray output.
    let z0 = [
        0.856627153315983 * z[0] + 0.0951212405381588 * z[1] + 0.0482516061458583 * z[2],
        0.137318972929847 * z[0] + 0.761241990602591 * z[1] + 0.101439036467562 * z[2],
        0.11189821299995 * z[0] + 0.0767994186031903 * z[1] + 0.811302368396859 * z[2],
    ];
    let z1 = [
        log2f(z0[0].max(1e-10)).clamp(AGX_MIN_EV, AGX_MAX_EV),
        log2f(z0[1].max(1e-10)).clamp(AGX_MIN_EV, AGX_MAX_EV),
        log2f(z0[2].max(1e-10)).clamp(AGX_MIN_EV, AGX_MAX_EV),
    ];
    let z2 = [
        (z1[0] - AGX_MIN_EV) * RECIP_EV,
        (z1[1] - AGX_MIN_EV) * RECIP_EV,
        (z1[2] - AGX_MIN_EV) * RECIP_EV,
    ];
    let z3 = [
        agx_contrast(z2[0]),
        agx_contrast(z2[1]),
        agx_contrast(z2[2]),
    ];
    let z4 = agx_apply_look(z3, look);
    // Outset matrix (analytical inverse of inset, rows sum to 1.0)
    [
        (1.19744107688770 * z4[0] - 0.144261512698001 * z4[1] - 0.0531795641897042 * z4[2])
            .clamp(0.0, 1.0),
        (-0.196474626321346 * z4[0] + 1.35409513146973 * z4[1] - 0.157620505148385 * z4[2])
            .clamp(0.0, 1.0),
        (-0.146557417106601 * z4[0] - 0.108284058788469 * z4[1] + 1.25484147589507 * z4[2])
            .clamp(0.0, 1.0),
    ]
}

#[inline]
fn agx_contrast(x: f32) -> f32 {
    // Degree-7 polynomial fitted to the Blender AgX sigmoid.
    // Coefficients from the Blender source, verified against gainforge.
    // Raw polynomial: 0.002857 − 0.1718x + 4.361x² − 28.72x³
    //                 + 92.06x⁴ − 126.7x⁵ + 78.01x⁶ − 17.86x⁷
    //
    // The raw polynomial has endpoint residuals: poly(0) = 0.002857,
    // poly(1) = 0.982059. Blender's actual sigmoid maps 0→0, 1→1
    // (it's a LUT). Normalize to fix the endpoints.
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let w0 = 0.002857 - 0.1718 * x;
    let w1 = 4.361 - 28.72 * x;
    let w2 = 92.06 - 126.7 * x;
    let w3 = 78.01 - 17.86 * x;
    let raw = w0 + w1 * x2 + w2 * x4 + w3 * x6;
    // Normalize: (raw - poly(0)) / (poly(1) - poly(0))
    const P0: f32 = 0.002857;
    const SCALE: f32 = 1.0 / (0.982059 - 0.002857); // 1 / 0.979202
    (raw - P0) * SCALE
}

fn agx_apply_look(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    let (slope, power, saturation) = match look {
        AgxLook::Default => return rgb,
        AgxLook::Punchy => ([1.0, 1.0, 1.0], [1.35, 1.35, 1.35], [1.4, 1.4, 1.4]),
        AgxLook::Golden => ([1.0, 0.9, 0.5], [0.8, 0.8, 0.8], [1.3, 1.3, 1.3]),
    };
    let dot = [
        (slope[0] * rgb[0]).max(0.0),
        (slope[1] * rgb[1]).max(0.0),
        (slope[2] * rgb[2]).max(0.0),
    ];
    let z = [
        powf(dot[0], power[0]),
        powf(dot[1], power[1]),
        powf(dot[2], power[2]),
    ];
    let luma = 0.2126 * z[0] + 0.7152 * z[1] + 0.0722 * z[2];
    [
        saturation[0] * (z[0] - luma) + luma,
        saturation[1] * (z[1] - luma) + luma,
        saturation[2] * (z[2] - luma) + luma,
    ]
}

// ============================================================================
// BT.2390 EETF
// ============================================================================

/// BT.2390 EETF (EOTF-based tone mapping) in scene-linear domain.
///
/// Maps scene-linear input with a given source peak to a lower target peak
/// using the reference Hermite-spline EETF from ITU-R BT.2390.
#[inline]
pub fn bt2390_tonemap(scene_linear: f32, source_peak: f32, target_peak: f32) -> f32 {
    bt2390_tonemap_ext(scene_linear, source_peak, target_peak, None)
}

/// BT.2390 EETF with optional min-luminance black-crush correction.
///
/// The `min_lum` term adds `e3 = min_lum * (1 - e2)^4 + e2`, lifting near-black
/// values to preserve shadow detail.
///
/// **Input domain:** `scene_linear` is in `[0, 1]`, already normalized so
/// that 1.0 = source peak. The `source_peak` and `target_peak` parameters
/// configure the knee and output scaling but do NOT rescale the input.
/// For scene-linear nits, normalize before calling:
/// `bt2390_tonemap(nits / source_nits, source_nits, target_nits)`.
#[inline]
pub fn bt2390_tonemap_ext(
    scene_linear: f32,
    source_peak: f32,
    target_peak: f32,
    min_lum: Option<f32>,
) -> f32 {
    if source_peak <= target_peak {
        return scene_linear;
    }

    let ks = (1.5 * target_peak / source_peak - 0.5).clamp(0.0, 1.0);
    let e1 = scene_linear;
    let e2 = if e1 < ks {
        e1
    } else {
        let t = (e1 - ks) / (1.0 - ks);
        let t2 = t * t;
        let t3 = t2 * t;
        let p0 = ks;
        let p1 = 1.0_f32;
        let m0 = 1.0 - ks;
        let m1 = 0.0_f32;
        let a = 2.0 * t3 - 3.0 * t2 + 1.0;
        let b = t3 - 2.0 * t2 + t;
        let c = -2.0 * t3 + 3.0 * t2;
        let d = t3 - t2;
        a * p0 + b * m0 + c * p1 + d * m1
    };

    let e3 = if let Some(ml) = min_lum {
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        ml * (one_minus_e2_2 * one_minus_e2_2) + e2
    } else {
        e2
    };

    e3 * target_peak / source_peak
}

// ============================================================================
// Möbius (libplacebo)
// ============================================================================

/// Coefficients for the Möbius tone-map. Depend only on
/// `(source_peak, knee)`; per-pixel work is one branch + one add + one
/// divide + one multiply once these are precomputed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MobiusCoeffs {
    /// Knee point (clamped to `(eps, 1 - eps)`).
    pub j: f32,
    /// Numerator offset `a`.
    pub a: f32,
    /// Denominator offset `b`.
    pub b: f32,
    /// Output scale.
    pub scale: f32,
    /// `true` if the input already fits the target (`source_peak <= 1.0`)
    /// — in that case [`mobius_apply`] is identity.
    pub identity: bool,
}

/// Compute Möbius coefficients per libplacebo
/// (`src/tone_mapping.c:638-665`).
///
/// `source_peak` is the normalized source peak — `source_peak_nits /
/// target_peak_nits`. `knee` is the linear knee in the normalized output
/// range `[0, 1]` (libplacebo default `0.30`).
///
/// Edge cases:
/// - `source_peak <= 1.0` returns `identity = true`; no tone mapping
///   needed.
/// - `knee` outside `(eps, 1 - eps)` is silently clamped to that range.
/// - The denominator `peak - 1` is floored at `1e-6` (matches the
///   `fmaxf` in the libplacebo source) to avoid divide-by-zero when the
///   source peak is very close to (but above) the target peak.
#[inline]
pub(crate) fn mobius_coefficients(source_peak: f32, knee: f32) -> MobiusCoeffs {
    const EPS: f32 = 1.0e-6;
    // Identity guard: source already fits the target, or input is NaN.
    // Written as `is_nan() || <= 1.0` to keep clippy happy and make the
    // NaN branch explicit (the original `!(source_peak > 1.0)` is correct
    // for both but trips `neg_cmp_op_on_partial_ord`).
    if source_peak.is_nan() || source_peak <= 1.0 {
        return MobiusCoeffs {
            j: knee.clamp(EPS, 1.0 - EPS),
            a: 0.0,
            b: 1.0,
            scale: 1.0,
            identity: true,
        };
    }
    let j = knee.clamp(EPS, 1.0 - EPS);
    let p = source_peak;
    let j2 = j * j;
    let denom_a = j2 - 2.0 * j + p;
    let a = -j2 * (p - 1.0) / denom_a;
    let b = (j2 - 2.0 * j * p + p) / (p - 1.0).max(EPS);
    let scale = (b * b + 2.0 * b * j + j2) / (b - a);
    MobiusCoeffs {
        j,
        a,
        b,
        scale,
        identity: false,
    }
}

/// Apply a precomputed Möbius curve to a single channel.
///
/// `M(x) = x` for `x <= j`, `scale * (x + a) / (x + b)` for `x > j`.
/// Negative input is clamped to 0 (linear light is non-negative).
///
/// **Algebraic reform for f32 safety:** the published form
/// `scale * (x + a) / (x + b)` overflows for `x` near `f32::MAX`
/// (`x + a ≈ x ≈ 3.4e38`, then `scale * x` exceeds the f32 range and
/// returns `+Inf`). We use the algebraically identical
/// `scale * (1 + (a - b) / (x + b))`, which evaluates the divide first
/// (bounded for `x > 0` since `b` is finite and small), so the result
/// stays finite for every finite input. As `x → ∞`, the divide → 0 and
/// the output asymptotes to `scale` — the true mathematical limit.
#[inline]
pub(crate) fn mobius_apply(x: f32, c: MobiusCoeffs) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if c.identity || x <= c.j {
        return x;
    }
    // Equivalent to `scale * (x + a) / (x + b)` but stays finite at
    // f32::MAX. Algebra:
    //   (x + a) / (x + b) = 1 + (a - b) / (x + b)
    // which approaches 1 as x → ∞.
    c.scale * (1.0 + (c.a - c.b) / (x + c.b))
}

// ============================================================================
// Unified dispatch enum
// ============================================================================

/// Enumeration of all supported tone mapping curves.
///
/// Variants that need per-pixel luminance carry luma coefficients inline so
/// the curve is self-contained — a caller constructs it once with the desired
/// RGB→Y weights ([`LUMA_BT709`](crate::LUMA_BT709) or
/// [`LUMA_BT2020`](crate::LUMA_BT2020)) and applies it without threading luma
/// through every call.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ToneMapCurve {
    /// Simple per-channel Reinhard: `x / (1 + x)`.
    Reinhard,
    /// Extended Reinhard with max luminance.
    ExtendedReinhard {
        /// Maximum expected luminance.
        l_max: f32,
        /// RGB→luminance weights.
        luma: [f32; 3],
    },
    /// Reinhard-Jodie (luminance-aware per-channel).
    ReinhardJodie {
        /// RGB→luminance weights.
        luma: [f32; 3],
    },
    /// Tuned Reinhard with display-aware weights (content/display peak nits).
    TunedReinhard {
        /// Content peak luminance in nits.
        content_max_nits: f32,
        /// Display peak luminance in nits.
        display_max_nits: f32,
        /// RGB→luminance weights.
        luma: [f32; 3],
    },
    /// Narkowicz filmic (ACES-inspired S-curve).
    ///
    /// **Input must be pre-exposed scene-linear** with scene mid-grey
    /// (~18 % grey) at `x ≈ 0.18` and scene white at `x ≈ 1.0`. See
    /// [`filmic_narkowicz`] for the full input-domain contract and the
    /// reason this curve is a poor match for producer-graded SDR
    /// targets. For HDR→SDR pipelines targeting camera-ISP-graded SDR,
    /// prefer [`Bt2390`](Self::Bt2390) or
    /// [`Bt2446B`](crate::Bt2446B).
    Narkowicz,
    /// Hable filmic (GDC 2010).
    HableFilmic,
    /// ACES AP1 RRT+ODT fit.
    AcesAp1,
    /// BT.2390 EETF in scene-linear domain.
    ///
    /// **Input must be normalized** to `[0, 1]` where `1.0 = source_peak`.
    /// Passing raw HDR linear-light values (e.g. nits/10000) without
    /// dividing by `source_peak` produces out-of-domain results.
    Bt2390 {
        /// Source peak luminance (normalized, in `[0, 1]`).
        source_peak: f32,
        /// Target peak luminance (normalized, in `[0, 1]`).
        target_peak: f32,
    },
    /// AgX (Blender) with a named look.
    Agx(AgxLook),
    /// Möbius tone-mapping curve (libplacebo's `mobius`).
    ///
    /// `M(x) = scale · (x + a) / (x + b)` for `x > knee`, identity below.
    /// Coefficients solve `M(knee) = knee`, `M(source_peak) = 1.0`,
    /// `M'(knee) = 1.0` — continuous and `C¹`-smooth at the knee.
    ///
    /// **Input must be normalized** to `[0, source_peak]` where `1.0` is
    /// the SDR / target peak. Pass `source_peak = source_peak_nits /
    /// target_peak_nits` (e.g. `1000 / 100 = 10.0`); `knee` is in the
    /// normalized output range (default `0.30` per libplacebo).
    ///
    /// Operates per-channel; caller composes with
    /// [`gamut::soft_clip_knee`](crate::gamut::soft_clip_knee) for hue
    /// preservation at the gamut boundary.
    ///
    /// Default in production HDR playback (mpv, VLC, FFmpeg, Plex via
    /// libplacebo).
    Mobius {
        /// Normalized source peak (`source_peak_nits / target_peak_nits`).
        source_peak: f32,
        /// Linear knee in `[eps, 1.0 - eps]` (libplacebo default `0.30`).
        knee: f32,
    },
    /// Clamp to `[0, 1]`.
    Clamp,
}

impl ToneMap for ToneMapCurve {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        match *self {
            ToneMapCurve::Reinhard => [
                reinhard_simple(rgb[0]).min(1.0),
                reinhard_simple(rgb[1]).min(1.0),
                reinhard_simple(rgb[2]).min(1.0),
            ],
            ToneMapCurve::ExtendedReinhard { l_max, luma } => {
                // Clamp to non-negative linear light first, matching the SIMD row
                // path (`simd::ext_reinhard_3_tier`, which does `.max(zero)` per
                // channel). Negative scene-linear input is unphysical; without
                // this the scalar `map_rgb` and the SIMD `map_row` diverged on
                // negative channels — and within one SIMD row the main loop
                // (clamped) and the scalar remainder (unclamped) disagreed.
                // A large negative channel times the scale also produced -Inf
                // (fuzz #21). reinhard_extended is now overflow-safe, so with
                // non-negative inputs the output is finite and in [0, 1].
                let r = rgb[0].max(0.0);
                let g = rgb[1].max(0.0);
                let b = rgb[2].max(0.0);
                let l = r * luma[0] + g * luma[1] + b * luma[2];
                if l <= 0.0 {
                    return [0.0, 0.0, 0.0];
                }
                let scale = reinhard_extended(l, l_max) / l;
                [
                    (r * scale).min(1.0),
                    (g * scale).min(1.0),
                    (b * scale).min(1.0),
                ]
            }
            ToneMapCurve::ReinhardJodie { luma } => reinhard_jodie(rgb, luma),
            ToneMapCurve::TunedReinhard {
                content_max_nits,
                display_max_nits,
                luma,
            } => {
                // Clamp to non-negative linear light first, consistent with the
                // other Reinhard variants and required for scalar/SIMD parity.
                // The pre-fix SIMD kernel fed raw channels into the rational
                // `(1 + w_a·l)/(1 + w_b·l)`; a negative luminance `l` made both
                // terms negative and the ratio produced garbage (≈ -1.3e36),
                // while scalar guarded `l <= 0` → black. Clamping per channel
                // makes `l ≥ 0`, so both paths agree.
                let r = rgb[0].max(0.0);
                let g = rgb[1].max(0.0);
                let b = rgb[2].max(0.0);
                let l = r * luma[0] + g * luma[1] + b * luma[2];
                if l <= 0.0 {
                    return [0.0, 0.0, 0.0];
                }
                let scale = tuned_reinhard(l, content_max_nits, display_max_nits);
                [
                    (r * scale).min(1.0),
                    (g * scale).min(1.0),
                    (b * scale).min(1.0),
                ]
            }
            ToneMapCurve::Narkowicz => [
                filmic_narkowicz(rgb[0]),
                filmic_narkowicz(rgb[1]),
                filmic_narkowicz(rgb[2]),
            ],
            ToneMapCurve::HableFilmic => [
                hable_filmic(rgb[0]),
                hable_filmic(rgb[1]),
                hable_filmic(rgb[2]),
            ],
            ToneMapCurve::AcesAp1 => aces_ap1(rgb),
            ToneMapCurve::Bt2390 {
                source_peak,
                target_peak,
            } => {
                // bt2390_tonemap expects input normalized to [0,1] where
                // 1.0 = source_peak. Normalize scene-linear input here.
                let inv = 1.0 / source_peak;
                [
                    bt2390_tonemap((rgb[0] * inv).min(1.0), source_peak, target_peak),
                    bt2390_tonemap((rgb[1] * inv).min(1.0), source_peak, target_peak),
                    bt2390_tonemap((rgb[2] * inv).min(1.0), source_peak, target_peak),
                ]
            }
            ToneMapCurve::Agx(look) => agx_tonemap(rgb, look),
            ToneMapCurve::Mobius { source_peak, knee } => {
                let c = mobius_coefficients(source_peak, knee);
                [
                    mobius_apply(rgb[0], c),
                    mobius_apply(rgb[1], c),
                    mobius_apply(rgb[2], c),
                ]
            }
            ToneMapCurve::Clamp => [
                clamp_tonemap(rgb[0]),
                clamp_tonemap(rgb[1]),
                clamp_tonemap(rgb[2]),
            ],
        }
    }

    fn map_row(&self, row: &mut [f32], channels: u8) {
        assert!(
            channels == 3 || channels == 4,
            "channels must be 3 or 4, got {channels}"
        );
        let ch = channels as usize;
        match self {
            // SIMD-accelerated per-channel curves
            ToneMapCurve::Reinhard => {
                crate::simd::reinhard_simple_row(row, ch);
                return;
            }
            ToneMapCurve::ExtendedReinhard { l_max, luma } => {
                crate::simd::ext_reinhard_row(row, ch, *l_max, *luma);
                return;
            }
            ToneMapCurve::ReinhardJodie { luma } => {
                crate::simd::reinhard_jodie_row(row, ch, *luma);
                return;
            }
            ToneMapCurve::TunedReinhard {
                content_max_nits,
                display_max_nits,
                luma,
            } => {
                crate::simd::tuned_reinhard_row(
                    row,
                    ch,
                    *content_max_nits,
                    *display_max_nits,
                    *luma,
                );
                return;
            }
            ToneMapCurve::Narkowicz => {
                crate::simd::narkowicz_row(row, ch);
                return;
            }
            ToneMapCurve::HableFilmic => {
                crate::simd::hable_row(row, ch);
                return;
            }
            ToneMapCurve::AcesAp1 => {
                crate::simd::aces_ap1_row(row, ch);
                return;
            }
            ToneMapCurve::Agx(look) => {
                crate::simd::agx_row(row, ch, *look);
                return;
            }
            // All other variants use the default trait impl (per-pixel map_rgb)
            _ => {}
        }
        // Fallback to default trait implementation
        match channels {
            3 => crate::tone_map::map_row_cn::<3, Self>(self, row),
            4 => crate::tone_map::map_row_cn::<4, Self>(self, row),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LUMA_BT709;

    #[test]
    fn reinhard_monotonic() {
        let mut last = 0.0f32;
        for i in 1..100 {
            let x = i as f32 / 10.0;
            let y = reinhard_simple(x);
            assert!(y >= last, "reinhard not monotonic at x={x}");
            assert!(y < 1.0, "reinhard exceeds 1 at x={x}: {y}");
            last = y;
        }
    }

    #[test]
    fn clamp_fixed_points() {
        assert_eq!(clamp_tonemap(-1.0), 0.0);
        assert_eq!(clamp_tonemap(0.5), 0.5);
        assert_eq!(clamp_tonemap(2.0), 1.0);
    }

    #[test]
    fn narkowicz_black_and_saturation() {
        assert!(filmic_narkowicz(0.0).abs() < 1e-6);
        assert!((filmic_narkowicz(100.0) - 1.0).abs() < 0.01);
    }

    /// Pins the input-domain contract documented on [`filmic_narkowicz`].
    ///
    /// Locks the "correct" frame (pre-exposed scene-linear, mid-grey at
    /// 0.18 in, ~0.27 out) and the two common "wrong" frames so future
    /// investigators don't re-derive these by hand. If any of these
    /// numbers shift, the curve's published formula changed (highly
    /// unlikely) OR the input-domain doc on `filmic_narkowicz` needs an
    /// update.
    #[test]
    fn narkowicz_input_domain_pins() {
        // Pre-exposed scene-linear ("correct frame"):
        // Scene mid-grey (18% of scene white) at x = 0.18 → ~0.267 SDR.
        let y_grey = filmic_narkowicz(0.18);
        assert!(
            (y_grey - 0.2669).abs() < 0.001,
            "scene-grey at x=0.18 should map to ~0.267, got {y_grey}"
        );
        // Scene white at x = 1.0 → ~0.804 SDR.
        let y_white = filmic_narkowicz(1.0);
        assert!(
            (y_white - 0.8038).abs() < 0.001,
            "scene-white at x=1.0 should map to ~0.804, got {y_white}"
        );
        // Specular highlight at x = 2.0 → near saturation (~0.915).
        let y_highlight = filmic_narkowicz(2.0);
        assert!(
            (y_highlight - 0.9148).abs() < 0.001,
            "highlight at x=2.0 should map to ~0.915, got {y_highlight}"
        );

        // "Under-exposed" frame (raw HDR / source_peak, e.g. 1000-nit
        // peak buffer normalized to [0, 1]): a 50-nit mid-grey lands at
        // x = 0.05 → output ~0.044 (looks like deep shadow). This is the
        // input-domain MISUSE we warn about in the doc.
        let y_underexposed = filmic_narkowicz(0.05);
        assert!(
            y_underexposed < 0.10,
            "x=0.05 under-exposes to deep shadow (<0.10), got {y_underexposed}"
        );

        // "Over-exposed" frame (HDR anchored 1.0 = diffuse_white_nits=
        // 203 then scaled by 203/100=2.03 to "target-norm"): a 50-nit
        // mid-grey lands at x = 0.50 → output ~0.616 (washed out). This
        // is the frame the shootout_full example uses; documented as
        // misuse because mid-grey should be ~0.27, not ~0.62.
        let y_overexposed = filmic_narkowicz(0.50);
        assert!(
            y_overexposed > 0.55,
            "x=0.50 over-exposes mid-grey (>0.55), got {y_overexposed}"
        );
    }

    #[test]
    fn aces_ap1_zero_to_zero() {
        let out = aces_ap1([0.0, 0.0, 0.0]);
        for c in out {
            assert!(c.abs() < 1e-3, "aces_ap1 of black should be ~0, got {c}");
        }
    }

    #[test]
    fn agx_default_clamps_to_unit() {
        for &look in &[AgxLook::Default, AgxLook::Punchy, AgxLook::Golden] {
            let out = agx_tonemap([4.0, 2.0, 0.5], look);
            for c in out {
                assert!((0.0..=1.0).contains(&c), "agx out of range: {c}");
            }
        }
    }

    #[test]
    fn bt2390_passthrough_when_source_below_target() {
        let y = bt2390_tonemap(0.5, 100.0, 1000.0);
        assert!((y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bt2390_reduces_peak() {
        // Input 1.0 (= source peak in normalized domain) → should map
        // to target_peak / source_peak = 0.1.
        let y = bt2390_tonemap(1.0, 1000.0, 100.0);
        assert!(
            y > 0.09 && y <= 0.11,
            "BT.2390 should map 1.0 to ~target/source: {y}"
        );
    }

    #[test]
    fn bt2390_min_lum_lifts_shadows() {
        // min_lum adds a soft shadow lift: very dark normalized input
        // should be nudged upward by the min_lum factor.
        let dark_input = 0.001_f32;
        let plain = bt2390_tonemap(dark_input, 1000.0, 100.0);
        let lifted = bt2390_tonemap_ext(dark_input, 1000.0, 100.0, Some(0.05));

        assert!(
            lifted > plain,
            "min_lum should lift near-black: plain={plain}, lifted={lifted}"
        );
    }

    #[test]
    fn bt2390_min_lum_does_not_affect_highlights() {
        // At the top end (normalized ~1.0), (1 - e2)^4 ≈ 0, so the
        // min_lum correction should barely move the output.
        let plain = bt2390_tonemap(0.99, 1000.0, 100.0);
        let lifted = bt2390_tonemap_ext(0.99, 1000.0, 100.0, Some(0.05));
        assert!(
            (plain - lifted).abs() < 0.005,
            "min_lum should leave highlights ~unchanged: plain={plain}, lifted={lifted}"
        );
    }

    #[test]
    fn map_row_rgb_in_place() {
        let mut row = [0.1_f32, 0.5, 2.0, 0.3, 0.8, 4.0];
        ToneMapCurve::Reinhard.map_row(&mut row, 3);
        for v in row {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn map_row_rgba_preserves_alpha() {
        let mut row = [0.5_f32, 0.5, 0.5, 0.42, 1.0, 2.0, 3.0, 0.77];
        ToneMapCurve::Reinhard.map_row(&mut row, 4);
        assert!((row[3] - 0.42).abs() < 1e-6);
        assert!((row[7] - 0.77).abs() < 1e-6);
    }

    #[test]
    fn map_into_copies_and_preserves_alpha() {
        let src = [0.5_f32, 0.5, 0.5, 0.42];
        let mut dst = [0.0_f32; 4];
        ToneMapCurve::Reinhard.map_into(&src, &mut dst, 4);
        assert!((dst[3] - 0.42).abs() < 1e-6);
        assert!(dst[0] > 0.0 && dst[0] < 1.0);
    }

    #[test]
    fn reinhard_jodie_zero_input() {
        let out = reinhard_jodie([0.0, 0.0, 0.0], LUMA_BT709);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn all_variants_dispatch() {
        // Scene-linear in [0, ~3]. BT.2390 wants input normalized to
        // source_peak, so excluded here and tested separately.
        let rgb = [1.5_f32, 2.5, 0.8];
        let curves = [
            ToneMapCurve::Reinhard,
            ToneMapCurve::ExtendedReinhard {
                l_max: 4.0,
                luma: LUMA_BT709,
            },
            ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 },
            ToneMapCurve::TunedReinhard {
                content_max_nits: 1000.0,
                display_max_nits: 250.0,
                luma: LUMA_BT709,
            },
            ToneMapCurve::Narkowicz,
            ToneMapCurve::HableFilmic,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Agx(AgxLook::Punchy),
            ToneMapCurve::Mobius {
                source_peak: 4.0,
                knee: 0.30,
            },
            ToneMapCurve::Clamp,
        ];
        for c in curves {
            let out = c.map_rgb(rgb);
            for v in out {
                assert!(v.is_finite(), "curve {c:?} produced non-finite {v}");
                assert!((0.0..=1.0).contains(&v), "curve {c:?} out of [0,1]: {v}");
            }
        }
    }

    // ------------------------------------------------------------------
    // Möbius (libplacebo port)
    // ------------------------------------------------------------------

    #[test]
    fn mobius_identity_when_source_below_target() {
        // source_peak <= 1.0 → input already fits target. The curve must
        // be a pure identity (no scaling, no knee).
        let c = mobius_coefficients(0.5, 0.30);
        assert!(c.identity, "source_peak <= 1.0 should set identity flag");
        for x in [0.0_f32, 0.1, 0.5, 0.8, 1.0] {
            assert!(
                (mobius_apply(x, c) - x).abs() < 1e-7,
                "mobius identity failed at x={x}"
            );
        }
    }

    #[test]
    fn mobius_continuity_at_knee() {
        // The libplacebo formula solves M(j) = j; the output must equal
        // the input at the knee point.
        for peak in [2.0_f32, 4.0, 10.0, 20.0] {
            for knee in [0.05_f32, 0.30, 0.7] {
                let c = mobius_coefficients(peak, knee);
                let y = mobius_apply(knee, c);
                assert!(
                    (y - knee).abs() < 1e-5,
                    "discontinuity at knee={knee} (peak={peak}): M({knee})={y}"
                );
            }
        }
    }

    #[test]
    fn mobius_maps_peak_to_unit() {
        // M(peak) = 1.0 by construction.
        for peak in [2.0_f32, 4.0, 10.0, 20.0, 100.0] {
            let c = mobius_coefficients(peak, 0.30);
            let y = mobius_apply(peak, c);
            assert!(
                (y - 1.0).abs() < 1e-4,
                "M(peak={peak}) should be 1.0, got {y}"
            );
        }
    }

    #[test]
    fn mobius_below_knee_is_identity() {
        // Below the knee, the curve is literally `x`.
        let c = mobius_coefficients(10.0, 0.30);
        for x in [0.0_f32, 0.05, 0.1, 0.2, 0.29] {
            assert!(
                (mobius_apply(x, c) - x).abs() < 1e-7,
                "below-knee identity failed at x={x}"
            );
        }
    }

    #[test]
    fn mobius_monotonic() {
        // Strictly non-decreasing on [0, peak].
        let c = mobius_coefficients(10.0, 0.30);
        let mut last = 0.0_f32;
        for i in 0..=200 {
            let x = (i as f32) / 20.0; // 0 .. 10
            let y = mobius_apply(x, c);
            assert!(
                y >= last - 1e-6,
                "mobius not monotonic at x={x}: y={y} < last={last}"
            );
            assert!(y.is_finite(), "mobius non-finite at x={x}: y={y}");
            last = y;
        }
    }

    #[test]
    fn mobius_negative_input_clamped_to_zero() {
        let c = mobius_coefficients(10.0, 0.30);
        assert_eq!(mobius_apply(-1.0, c), 0.0);
        assert_eq!(mobius_apply(-1e20, c), 0.0);
    }

    #[test]
    fn mobius_knee_outside_range_clamped() {
        // knee should be silently clamped to (eps, 1 - eps).
        let c_lo = mobius_coefficients(10.0, -0.5);
        let c_hi = mobius_coefficients(10.0, 1.5);
        assert!(c_lo.j > 0.0 && c_lo.j < 1.0);
        assert!(c_hi.j > 0.0 && c_hi.j < 1.0);
        // Both should still produce finite output everywhere.
        for x in [0.0_f32, 0.5, 1.0, 5.0, 10.0] {
            assert!(mobius_apply(x, c_lo).is_finite());
            assert!(mobius_apply(x, c_hi).is_finite());
        }
    }

    #[test]
    fn mobius_dispatch_via_curve() {
        // Variant in the ToneMapCurve enum dispatch.
        let curve = ToneMapCurve::Mobius {
            source_peak: 10.0,
            knee: 0.30,
        };
        let out = curve.map_rgb([5.0, 8.0, 0.1]);
        for v in out {
            assert!(v.is_finite() && v >= 0.0);
        }
        // 0.1 is below the knee → identity passthrough.
        assert!((out[2] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn mobius_dispatch_identity_when_no_compression_needed() {
        // source_peak == target_peak → identity.
        let curve = ToneMapCurve::Mobius {
            source_peak: 1.0,
            knee: 0.30,
        };
        let rgb = [0.4_f32, 0.7, 0.9];
        let out = curve.map_rgb(rgb);
        for i in 0..3 {
            assert!((out[i] - rgb[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn mobius_dispatch_extreme_peak_finite() {
        // Stress: very high peak, normal input — should remain finite.
        let curve = ToneMapCurve::Mobius {
            source_peak: 100.0,
            knee: 0.30,
        };
        for x in [0.0_f32, 0.5, 5.0, 50.0, 100.0] {
            let out = curve.map_rgb([x; 3]);
            for v in out {
                assert!(v.is_finite(), "non-finite at x={x}");
            }
        }
    }

    #[test]
    fn mobius_dispatch_peak_just_above_target_finite() {
        // peak - 1 close to zero — the `max(1e-6, ...)` floor keeps
        // coefficients finite.
        let curve = ToneMapCurve::Mobius {
            source_peak: 1.0 + 1e-7,
            knee: 0.30,
        };
        for x in [0.0_f32, 0.5, 1.0] {
            let out = curve.map_rgb([x; 3]);
            for v in out {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn bt2390_dispatch_in_range_input() {
        let rgb = [0.5_f32, 0.7, 0.2];
        let curve = ToneMapCurve::Bt2390 {
            source_peak: 1.0,
            target_peak: 0.5,
        };
        let out = curve.map_rgb(rgb);
        for v in out {
            assert!(v.is_finite(), "bt2390 non-finite {v}");
            assert!((0.0..=0.5).contains(&v), "bt2390 out of [0, target]: {v}");
        }
    }
}
