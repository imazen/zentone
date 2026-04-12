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
#[inline]
pub fn reinhard_simple(x: f32) -> f32 {
    x / (1.0 + x)
}

/// Clamp tone map: clamps to `[0, 1]`.
#[inline]
pub fn clamp_tonemap(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Extended Reinhard with max luminance.
///
/// `L_out = L_in * (1 + L_in / L_max²) / (1 + L_in)`.
#[inline]
pub fn reinhard_extended(l_in: f32, l_max: f32) -> f32 {
    let l_max_sq = l_max * l_max;
    l_in * (1.0 + l_in / l_max_sq) / (1.0 + l_in)
}

/// Reinhard-Jodie tone mapping (per-channel and luminance-based blend).
pub fn reinhard_jodie(rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
    let luma = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
    if luma <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let luma_scale = 1.0 / (1.0 + luma);
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let tv = rgb[i] / (1.0 + rgb[i]);
        out[i] = ((1.0 - tv) * (rgb[i] * luma_scale) + tv * tv).min(1.0);
    }
    out
}

/// Tuned Reinhard with display-aware parameters.
///
/// Derived from content and display peak luminance (nits), with a reference
/// white of 203 nits.
pub fn tuned_reinhard(luma: f32, content_max: f32, display_max: f32) -> f32 {
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
/// Cheap scene-linear approximation of ACES RRT+ODT.
#[inline]
pub fn filmic_narkowicz(x: f32) -> f32 {
    let a = 2.51_f32;
    let b = 0.03_f32;
    let c = 2.43_f32;
    let d = 0.59_f32;
    let e = 0.14_f32;
    let num = x * (a * x + b);
    let den = x * (c * x + d) + e;
    (num / den).clamp(0.0, 1.0)
}

/// Uncharted 2 filmic tone mapping (Hable).
pub fn uncharted2_filmic(v: f32) -> f32 {
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
    // Inset matrix
    let z0 = [
        0.856627153315983 * z[0] + 0.137318972929847 * z[1] + 0.11189821299995 * z[2],
        0.0951212405381588 * z[0] + 0.761241990602591 * z[1] + 0.0767994186031903 * z[2],
        0.0482516061458583 * z[0] + 0.101439036467562 * z[1] + 0.811302368396859 * z[2],
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
    // Outset matrix
    [
        (1.19687900512017 * z4[0] - 0.0528968517574562 * z4[1] - 0.0529716355144438 * z4[2])
            .clamp(0.0, 1.0),
        (-0.0980208811401368 * z4[0] + 1.15190312990417 * z4[1] - 0.0505349770312032 * z4[2])
            .clamp(0.0, 1.0),
        (-0.0990297440797205 * z4[0] - 0.0989611768448433 * z4[1] + 1.15107367264116 * z4[2])
            .clamp(0.0, 1.0),
    ]
}

#[inline]
fn agx_contrast(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let w0 = 0.002857 * x - 0.1718;
    let w1 = 4.361 * x - 28.72;
    let w2 = 92.06 * x - 126.7;
    let w3 = 78.01 * x - 17.86;
    let z0 = w0 * x2 + w1;
    let z1 = x4 * w2 * x6 + w3;
    z1 + z0
}

fn agx_apply_look(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    let (slope, power, saturation) = match look {
        AgxLook::Default => return rgb,
        AgxLook::Punchy => ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.4, 1.4, 1.4]),
        AgxLook::Golden => ([1.0, 0.9, 0.5], [0.8, 0.8, 0.8], [1.2, 1.2, 1.2]),
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
    Narkowicz,
    /// Uncharted 2 filmic (Hable).
    Uncharted2,
    /// ACES AP1 RRT+ODT fit.
    AcesAp1,
    /// BT.2390 EETF in scene-linear domain.
    Bt2390 {
        /// Source peak luminance (normalized, in `[0, 1]`).
        source_peak: f32,
        /// Target peak luminance (normalized, in `[0, 1]`).
        target_peak: f32,
    },
    /// AgX (Blender) with a named look.
    Agx(AgxLook),
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
                let l = rgb[0] * luma[0] + rgb[1] * luma[1] + rgb[2] * luma[2];
                if l <= 0.0 {
                    return [0.0, 0.0, 0.0];
                }
                let scale = reinhard_extended(l, l_max) / l;
                [
                    (rgb[0] * scale).min(1.0),
                    (rgb[1] * scale).min(1.0),
                    (rgb[2] * scale).min(1.0),
                ]
            }
            ToneMapCurve::ReinhardJodie { luma } => reinhard_jodie(rgb, luma),
            ToneMapCurve::TunedReinhard {
                content_max_nits,
                display_max_nits,
                luma,
            } => {
                let l = rgb[0] * luma[0] + rgb[1] * luma[1] + rgb[2] * luma[2];
                if l <= 0.0 {
                    return [0.0, 0.0, 0.0];
                }
                let scale = tuned_reinhard(l, content_max_nits, display_max_nits);
                [
                    (rgb[0] * scale).min(1.0),
                    (rgb[1] * scale).min(1.0),
                    (rgb[2] * scale).min(1.0),
                ]
            }
            ToneMapCurve::Narkowicz => [
                filmic_narkowicz(rgb[0]),
                filmic_narkowicz(rgb[1]),
                filmic_narkowicz(rgb[2]),
            ],
            ToneMapCurve::Uncharted2 => [
                uncharted2_filmic(rgb[0]),
                uncharted2_filmic(rgb[1]),
                uncharted2_filmic(rgb[2]),
            ],
            ToneMapCurve::AcesAp1 => aces_ap1(rgb),
            ToneMapCurve::Bt2390 {
                source_peak,
                target_peak,
            } => [
                bt2390_tonemap(rgb[0], source_peak, target_peak),
                bt2390_tonemap(rgb[1], source_peak, target_peak),
                bt2390_tonemap(rgb[2], source_peak, target_peak),
            ],
            ToneMapCurve::Agx(look) => agx_tonemap(rgb, look),
            ToneMapCurve::Clamp => [
                clamp_tonemap(rgb[0]),
                clamp_tonemap(rgb[1]),
                clamp_tonemap(rgb[2]),
            ],
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
        let y = bt2390_tonemap(1.0, 1000.0, 100.0);
        assert!(
            y <= 0.11,
            "BT.2390 should map 1.0 to roughly target/source: {y}"
        );
    }

    #[test]
    fn bt2390_min_lum_lifts_shadows() {
        // min_lum adds a soft shadow lift: very dark input that would
        // tonemap to near-zero should be nudged upward by the min_lum
        // factor.
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
        // At the top end, (1 - e2)^4 ≈ 0, so the min_lum correction
        // should barely move the output.
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
            ToneMapCurve::Uncharted2,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Agx(AgxLook::Punchy),
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
