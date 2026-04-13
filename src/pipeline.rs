//! HDR→SDR pipeline helpers.
//!
//! Convenience functions that compose transfer-function decoding,
//! tone mapping, gamut conversion, and sRGB encoding into single calls.
//!
//! # Configuration
//!
//! [`PipelineConfig`] controls the tone mapping application space via
//! [`ToneMapSpace`](crate::gamut::ToneMapSpace): RGB (per-channel) vs
//! luma-preserving (better color, fewer out-of-gamut). Out-of-gamut colors
//! after the BT.2020→BT.709 matrix are always handled with hue-preserving
//! [`soft_clip`](crate::gamut::soft_clip).
//!
//! The simple functions ([`tonemap_pq_to_linear_srgb`], etc.) use defaults
//! (RGB per-channel). For control, use [`tonemap_pq_to_linear_srgb_config`].

use crate::ToneMap;
use crate::gamut::{
    BT2020_TO_BT709, ToneMapSpace, apply_matrix, is_out_of_gamut, soft_clip,
    tonemap_luma_preserving,
};

/// Pipeline configuration for HDR→SDR conversion.
///
/// Controls which color space the tone curve is applied in.
/// Out-of-gamut colors after gamut matrix conversion are always
/// handled with hue-preserving [`soft_clip`](crate::gamut::soft_clip).
///
/// For principled perceptual gamut compression (rather than post-matrix
/// clipping), use ACES 2.0 ([issue #14](https://github.com/imazen/zentone/issues/14))
/// which compresses in Hellwig 2022 JMh space.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PipelineConfig {
    /// Color space for tone curve application.
    pub tone_map_space: ToneMapSpace,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tone_map_space: ToneMapSpace::Rgb,
        }
    }
}

/// Apply a tone curve to a linear RGB triple respecting the configured space.
#[inline]
fn apply_tonemap(rgb: [f32; 3], tm: &dyn ToneMap, space: ToneMapSpace) -> [f32; 3] {
    match space {
        ToneMapSpace::Rgb => tm.map_rgb(rgb),
        ToneMapSpace::LumaPreserving { luma } => tonemap_luma_preserving(rgb, luma, tm),
    }
}

/// Tonemap a PQ-encoded BT.2020 RGB row to linear sRGB.
///
/// Uses default config (RGB per-channel + SoftClip). For control, use
/// [`tonemap_pq_to_linear_srgb_config`].
pub fn tonemap_pq_to_linear_srgb(pq_row: &[f32], out: &mut [f32], tm: &dyn ToneMap, channels: u8) {
    tonemap_pq_to_linear_srgb_config(pq_row, out, tm, channels, &PipelineConfig::default());
}

/// Tonemap a PQ-encoded BT.2020 RGB row to linear sRGB with explicit config.
///
/// Pipeline: PQ EOTF → linear BT.2020 → tonemap (in configured space) →
/// BT.2020→BT.709 → gamut clip.
pub fn tonemap_pq_to_linear_srgb_config(
    pq_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    channels: u8,
    cfg: &PipelineConfig,
) {
    debug_assert_eq!(pq_row.len(), out.len());
    let ch = channels as usize;

    for (src, dst) in pq_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];

        let tonemapped = apply_tonemap(linear_2020, tm, cfg.tone_map_space);
        let bt709 = {
            let rgb = apply_matrix(&BT2020_TO_BT709, tonemapped);
            if is_out_of_gamut(rgb) {
                soft_clip(rgb)
            } else {
                rgb
            }
        };

        dst[0] = bt709[0];
        dst[1] = bt709[1];
        dst[2] = bt709[2];
        if ch == 4 {
            dst[3] = src[3];
        }
    }
}

/// Tonemap a PQ-encoded BT.2020 RGB row to sRGB-encoded u8.
///
/// Uses default config. For control, use [`tonemap_pq_to_srgb8_config`].
pub fn tonemap_pq_to_srgb8(pq_row: &[f32], out: &mut [u8], tm: &dyn ToneMap, channels: u8) {
    tonemap_pq_to_srgb8_config(pq_row, out, tm, channels, &PipelineConfig::default());
}

/// Tonemap a PQ-encoded BT.2020 RGB row to sRGB u8 with explicit config.
pub fn tonemap_pq_to_srgb8_config(
    pq_row: &[f32],
    out: &mut [u8],
    tm: &dyn ToneMap,
    channels: u8,
    cfg: &PipelineConfig,
) {
    debug_assert_eq!(pq_row.len(), out.len());
    let ch = channels as usize;

    for (src, dst) in pq_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let linear_2020 = [
            linear_srgb::tf::pq_to_linear(src[0]),
            linear_srgb::tf::pq_to_linear(src[1]),
            linear_srgb::tf::pq_to_linear(src[2]),
        ];

        let tonemapped = apply_tonemap(linear_2020, tm, cfg.tone_map_space);
        let bt709 = {
            let rgb = apply_matrix(&BT2020_TO_BT709, tonemapped);
            if is_out_of_gamut(rgb) {
                soft_clip(rgb)
            } else {
                rgb
            }
        };

        dst[0] = linear_to_srgb_u8(bt709[0]);
        dst[1] = linear_to_srgb_u8(bt709[1]);
        dst[2] = linear_to_srgb_u8(bt709[2]);
        if ch == 4 {
            dst[3] = (src[3] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Tonemap an HLG-encoded BT.2020 RGB row to linear sRGB.
///
/// Uses default config. For control, use [`tonemap_hlg_to_linear_srgb_config`].
pub fn tonemap_hlg_to_linear_srgb(
    hlg_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
    channels: u8,
) {
    tonemap_hlg_to_linear_srgb_config(
        hlg_row,
        out,
        tm,
        display_peak_nits,
        channels,
        &PipelineConfig::default(),
    );
}

/// Tonemap an HLG-encoded BT.2020 RGB row to linear sRGB with explicit config.
pub fn tonemap_hlg_to_linear_srgb_config(
    hlg_row: &[f32],
    out: &mut [f32],
    tm: &dyn ToneMap,
    display_peak_nits: f32,
    channels: u8,
    cfg: &PipelineConfig,
) {
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);
    let ch = channels as usize;

    for (src, dst) in hlg_row.chunks_exact(ch).zip(out.chunks_exact_mut(ch)) {
        let scene = [
            linear_srgb::tf::hlg_to_linear(src[0]),
            linear_srgb::tf::hlg_to_linear(src[1]),
            linear_srgb::tf::hlg_to_linear(src[2]),
        ];

        let display = crate::hlg::hlg_ootf(scene, gamma);
        let tonemapped = apply_tonemap(display, tm, cfg.tone_map_space);
        let bt709 = {
            let rgb = apply_matrix(&BT2020_TO_BT709, tonemapped);
            if is_out_of_gamut(rgb) {
                soft_clip(rgb)
            } else {
                rgb
            }
        };

        dst[0] = bt709[0];
        dst[1] = bt709[1];
        dst[2] = bt709[2];
        if ch == 4 {
            dst[3] = src[3];
        }
    }
}

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = linear_srgb::tf::linear_to_srgb(v);
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bt2408Tonemapper;

    #[test]
    fn pq_to_linear_srgb_black() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.0_f32; 3];
        let mut out = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb(&pq, &mut out, &tm, 3);
        for c in out {
            assert!(c.abs() < 1e-3, "black should stay black: {c}");
        }
    }

    #[test]
    fn pq_to_srgb8_produces_valid_bytes() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        // SDR reference white in PQ: ~0.58
        let pq = [0.58_f32, 0.58, 0.58];
        let mut out = [0u8; 3];
        tonemap_pq_to_srgb8(&pq, &mut out, &tm, 3);
        // 203 nits in a 4000-nit content range → modest gray after EETF
        assert!(out[0] > 30 && out[0] < 255, "SDR white byte: {}", out[0]);
    }

    #[test]
    fn hlg_to_linear_srgb_at_reference_white() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        // HLG 75% = reference white
        let hlg = [0.75_f32, 0.75, 0.75];
        let mut out = [0.0_f32; 3];
        tonemap_hlg_to_linear_srgb(&hlg, &mut out, &tm, 1000.0, 3);
        for c in out {
            assert!(c.is_finite() && c > 0.0, "HLG ref white: {c}");
        }
    }

    #[test]
    fn rgba_alpha_preserved() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.5_f32, 0.5, 0.5, 0.42];
        let mut out = [0.0_f32; 4];
        tonemap_pq_to_linear_srgb(&pq, &mut out, &tm, 4);
        assert!((out[3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn luma_preserving_config_produces_valid_output() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let cfg = PipelineConfig {
            tone_map_space: ToneMapSpace::LumaPreserving {
                luma: crate::LUMA_BT709,
            },
        };
        // Bright saturated PQ red
        let pq = [0.7_f32, 0.3, 0.2];
        let mut out = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq, &mut out, &tm, 3, &cfg);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 0.0 && v <= 1.01,
                "luma-preserving cfg: ch {i} = {v}"
            );
        }
        // Channel ordering should be preserved (R > G > B since input R > G > B)
        assert!(
            out[0] >= out[1] && out[1] >= out[2],
            "luma-preserving should preserve channel ordering: {out:?}"
        );
    }

    #[test]
    fn luma_preserving_less_oog_than_rgb() {
        let tm = crate::ToneMapCurve::Reinhard;
        let pq_green = [0.2_f32, 0.7, 0.1];

        let cfg_rgb = PipelineConfig {
            tone_map_space: ToneMapSpace::Rgb,
        };
        let mut out_rgb = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq_green, &mut out_rgb, &tm, 3, &cfg_rgb);

        let cfg_luma = PipelineConfig {
            tone_map_space: ToneMapSpace::LumaPreserving {
                luma: crate::LUMA_BT709,
            },
        };
        let mut out_luma = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq_green, &mut out_luma, &tm, 3, &cfg_luma);

        for &v in out_rgb.iter().chain(out_luma.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn pipeline_config_default_matches_simple_function() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let pq = [0.5_f32, 0.4, 0.3];

        let mut out_simple = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb(&pq, &mut out_simple, &tm, 3);

        let mut out_config = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq, &mut out_config, &tm, 3, &PipelineConfig::default());

        for i in 0..3 {
            assert!(
                (out_simple[i] - out_config[i]).abs() < 1e-7,
                "default config should match simple fn: ch {i}: {} vs {}",
                out_simple[i],
                out_config[i]
            );
        }
    }

    #[test]
    fn soft_clip_pipeline_output_in_range() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let cfg = PipelineConfig {
            tone_map_space: ToneMapSpace::Rgb,
        };
        let pq = [0.6_f32, 0.6, 0.6];
        let mut out = [0.0_f32; 3];
        tonemap_pq_to_linear_srgb_config(&pq, &mut out, &tm, 3, &cfg);
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "soft clip: ch {i} = {v} out of [0,1]");
        }
    }
}
