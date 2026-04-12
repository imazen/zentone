//! Adaptive tonemapper that fits a LUT from an HDR/SDR pair.
//!
//! Use case: re-encoding edited HDR content while preserving the artistic
//! intent of the original SDR rendition. The fitter samples the HDR→SDR
//! relationship in the training pair and fits either a luminance-domain LUT
//! (hue-preserving) or per-channel LUTs (highest fidelity for round-trips).
//!
//! # Status
//!
//! Lightly tested. API may change. Inputs are tightly-packed interleaved
//! linear f32 buffers of `pixels * channels` elements.

use alloc::boxed::Box;

use crate::error::{Error, Result};
use crate::math::roundf;
use crate::{LUMA_BT709, ToneMap};

/// LUT resolution for adaptive tone curves.
const LUT_SIZE: usize = 4096;

/// Adaptive tonemapper fitted from an HDR/SDR training pair.
///
/// Implements [`ToneMap`] — once fitted, use `map_rgb` / `map_row` /
/// `map_into` to apply the learned curve.
#[derive(Debug, Clone)]
pub struct AdaptiveTonemapper {
    mode: TonemapMode,
    max_hdr_observed: f32,
    stats: FitStats,
}

/// Representation of a fitted tonemapping curve.
#[derive(Debug, Clone)]
pub enum TonemapMode {
    /// Hue-preserving luminance-based curve.
    Luminance(LuminanceCurve),
    /// Per-channel LUTs; highest fidelity, may shift hues.
    PerChannel(PerChannelLut),
}

/// Luminance-based tone curve.
#[derive(Debug, Clone)]
pub struct LuminanceCurve {
    /// LUT mapping HDR luminance `[0, max_hdr]` → SDR luminance `[0, 1]`.
    lut: Box<[f32; LUT_SIZE]>,
    /// Maximum HDR luminance covered by the LUT.
    max_hdr: f32,
    /// Saturation adjustment (1.0 = preserve, >1 = boost, <1 = reduce).
    saturation: f32,
    /// RGB→luminance weights captured from the fit config.
    luma: [f32; 3],
}

/// Per-channel tone curves.
#[derive(Debug, Clone)]
pub struct PerChannelLut {
    lut_r: Box<[f32; LUT_SIZE]>,
    lut_g: Box<[f32; LUT_SIZE]>,
    lut_b: Box<[f32; LUT_SIZE]>,
    max_hdr: f32,
}

/// Statistics from the fitting process.
#[derive(Debug, Clone, Default)]
pub struct FitStats {
    /// Number of pixel samples used (after sub-sampling).
    pub samples: usize,
    /// Maximum observed HDR luminance (luminance fit) or RGB value
    /// (per-channel fit).
    pub max_hdr: f32,
    /// Detected saturation change (SDR_sat / HDR_sat). `1.0` if saturation
    /// detection was disabled or no valid samples were found.
    pub saturation_ratio: f32,
    /// Mean absolute error of the fit. `None` unless
    /// [`FitConfig::compute_mae`] was enabled.
    pub mae: Option<f32>,
}

/// Configuration for fitting an adaptive tonemapper.
///
/// Defaults: 100k sub-sampled pixels, saturation detection on, MAE off,
/// BT.709 luminance weights.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Maximum number of samples to process. `0` = use every pixel.
    pub max_samples: usize,
    /// Whether to detect and apply saturation changes
    /// ([`LuminanceCurve`] only).
    pub detect_saturation: bool,
    /// If true, compute mean absolute error as a diagnostic
    /// ([`LuminanceCurve`] only). Adds one extra pass over the source.
    pub compute_mae: bool,
    /// RGB→luminance weights ([`LuminanceCurve`] only).
    pub luma_coeffs: [f32; 3],
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_samples: 100_000,
            detect_saturation: true,
            compute_mae: false,
            luma_coeffs: LUMA_BT709,
        }
    }
}

impl AdaptiveTonemapper {
    /// Fit a luminance-based tonemapper from an HDR/SDR pair.
    ///
    /// Both buffers must have the same length and contain `pixels * channels`
    /// interleaved linear f32 elements (`channels` = 3 for RGB, 4 for RGBA;
    /// alpha is ignored during fit). Pixel count is inferred from the slice
    /// length.
    ///
    /// The result implements [`ToneMap`] — apply with
    /// [`map_rgb`](ToneMap::map_rgb), [`map_row`](ToneMap::map_row), or
    /// [`map_into`](ToneMap::map_into).
    pub fn fit_luminance(hdr: &[f32], sdr: &[f32], channels: u8, cfg: &FitConfig) -> Result<Self> {
        let (ch, pixels) = validate_pair(hdr, sdr, channels)?;

        // Pass 1: find max_hdr (+ optional saturation detection)
        let mut max_hdr = 0.0_f32;
        let mut saturation_sum = 0.0_f32;
        let mut saturation_count = 0_usize;
        let mut sample_count = 0_usize;
        let step = sample_step(pixels, cfg.max_samples);
        let luma = cfg.luma_coeffs;

        for i in (0..pixels).step_by(step) {
            let off = i * ch;
            let hdr_rgb = [hdr[off], hdr[off + 1], hdr[off + 2]];
            let sdr_rgb = [sdr[off], sdr[off + 1], sdr[off + 2]];
            let l_hdr = dot3(hdr_rgb, luma);
            let l_sdr = dot3(sdr_rgb, luma);
            if l_hdr > 0.001 && l_sdr > 0.001 {
                sample_count += 1;
                if l_hdr > max_hdr {
                    max_hdr = l_hdr;
                }
                if cfg.detect_saturation && l_hdr > 0.01 && l_sdr > 0.01 {
                    let sat_hdr = compute_saturation(hdr_rgb, l_hdr);
                    let sat_sdr = compute_saturation(sdr_rgb, l_sdr);
                    if sat_hdr > 0.01 {
                        saturation_sum += sat_sdr / sat_hdr;
                        saturation_count += 1;
                    }
                }
            }
        }

        if sample_count == 0 || max_hdr <= 0.0 {
            return Err(Error::NoValidSamples);
        }

        // Pass 2: bucket directly into the LUT (no pair vector, no sort)
        let mut lut = Box::new([0.0_f32; LUT_SIZE]);
        let mut counts = [0_u32; LUT_SIZE];
        let max_idx = (LUT_SIZE - 1) as f32;

        for i in (0..pixels).step_by(step) {
            let off = i * ch;
            let l_hdr = dot3([hdr[off], hdr[off + 1], hdr[off + 2]], luma);
            let l_sdr = dot3([sdr[off], sdr[off + 1], sdr[off + 2]], luma);
            if l_hdr > 0.001 && l_sdr > 0.001 {
                let idx = roundf((l_hdr / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut[idx] += l_sdr;
                counts[idx] += 1;
            }
        }
        finalize_bucketed_lut(&mut lut, &counts);

        let saturation = if saturation_count > 0 {
            (saturation_sum / saturation_count as f32).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // Optional pass 3: MAE (off by default)
        let mae = if cfg.compute_mae {
            let mut mae_sum = 0.0_f32;
            let mut mae_count = 0_usize;
            for i in (0..pixels).step_by(step) {
                let off = i * ch;
                let l_hdr = dot3([hdr[off], hdr[off + 1], hdr[off + 2]], luma);
                let l_sdr = dot3([sdr[off], sdr[off + 1], sdr[off + 2]], luma);
                if l_hdr > 0.001 && l_sdr > 0.001 {
                    let idx = roundf((l_hdr / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                    mae_sum += (lut[idx] - l_sdr).abs();
                    mae_count += 1;
                }
            }
            if mae_count > 0 {
                Some(mae_sum / mae_count as f32)
            } else {
                Some(0.0)
            }
        } else {
            None
        };

        Ok(Self {
            mode: TonemapMode::Luminance(LuminanceCurve {
                lut,
                max_hdr,
                saturation,
                luma,
            }),
            max_hdr_observed: max_hdr,
            stats: FitStats {
                samples: sample_count,
                max_hdr,
                saturation_ratio: saturation,
                mae,
            },
        })
    }

    /// Fit per-channel tonemappers from an HDR/SDR pair.
    ///
    /// Returns [`Error::EmptyChannel`] if any of R/G/B has no valid samples
    /// — per-channel mode can't fabricate a mapping from nothing. Fall back
    /// to [`fit_luminance`](Self::fit_luminance) in that case.
    ///
    /// `cfg.detect_saturation`, `cfg.compute_mae`, and `cfg.luma_coeffs` are
    /// ignored in per-channel mode.
    pub fn fit_per_channel(
        hdr: &[f32],
        sdr: &[f32],
        channels: u8,
        cfg: &FitConfig,
    ) -> Result<Self> {
        let (ch, pixels) = validate_pair(hdr, sdr, channels)?;
        let step = sample_step(pixels, cfg.max_samples);
        let max_idx = (LUT_SIZE - 1) as f32;

        // Pass 1: find max_hdr (shared across channels) + per-channel counts
        let mut max_hdr = 0.0_f32;
        let mut n_r = 0_usize;
        let mut n_g = 0_usize;
        let mut n_b = 0_usize;
        for i in (0..pixels).step_by(step) {
            let off = i * ch;
            let hr = hdr[off];
            let hg = hdr[off + 1];
            let hb = hdr[off + 2];
            if hr > 0.001 {
                n_r += 1;
            }
            if hg > 0.001 {
                n_g += 1;
            }
            if hb > 0.001 {
                n_b += 1;
            }
            if hr > max_hdr {
                max_hdr = hr;
            }
            if hg > max_hdr {
                max_hdr = hg;
            }
            if hb > max_hdr {
                max_hdr = hb;
            }
        }
        if max_hdr <= 0.0 {
            return Err(Error::NoValidSamples);
        }
        if n_r == 0 {
            return Err(Error::EmptyChannel { channel: 0 });
        }
        if n_g == 0 {
            return Err(Error::EmptyChannel { channel: 1 });
        }
        if n_b == 0 {
            return Err(Error::EmptyChannel { channel: 2 });
        }

        // Pass 2: bucket each channel directly into its LUT
        let mut lut_r = Box::new([0.0_f32; LUT_SIZE]);
        let mut lut_g = Box::new([0.0_f32; LUT_SIZE]);
        let mut lut_b = Box::new([0.0_f32; LUT_SIZE]);
        let mut cnt_r = [0_u32; LUT_SIZE];
        let mut cnt_g = [0_u32; LUT_SIZE];
        let mut cnt_b = [0_u32; LUT_SIZE];

        for i in (0..pixels).step_by(step) {
            let off = i * ch;
            let hr = hdr[off];
            let hg = hdr[off + 1];
            let hb = hdr[off + 2];
            let sr = sdr[off];
            let sg = sdr[off + 1];
            let sb = sdr[off + 2];
            if hr > 0.001 {
                let idx = roundf((hr / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut_r[idx] += sr;
                cnt_r[idx] += 1;
            }
            if hg > 0.001 {
                let idx = roundf((hg / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut_g[idx] += sg;
                cnt_g[idx] += 1;
            }
            if hb > 0.001 {
                let idx = roundf((hb / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut_b[idx] += sb;
                cnt_b[idx] += 1;
            }
        }

        finalize_bucketed_lut(&mut lut_r, &cnt_r);
        finalize_bucketed_lut(&mut lut_g, &cnt_g);
        finalize_bucketed_lut(&mut lut_b, &cnt_b);

        Ok(Self {
            mode: TonemapMode::PerChannel(PerChannelLut {
                lut_r,
                lut_g,
                lut_b,
                max_hdr,
            }),
            max_hdr_observed: max_hdr,
            stats: FitStats {
                samples: n_r + n_g + n_b,
                max_hdr,
                saturation_ratio: 1.0,
                mae: None,
            },
        })
    }

    /// Fitting statistics.
    pub fn stats(&self) -> &FitStats {
        &self.stats
    }

    /// Maximum HDR value observed during fitting.
    pub fn max_hdr_observed(&self) -> f32 {
        self.max_hdr_observed
    }

    /// Representation of the fitted curve.
    pub fn mode(&self) -> &TonemapMode {
        &self.mode
    }
}

impl ToneMap for AdaptiveTonemapper {
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        match &self.mode {
            TonemapMode::Luminance(curve) => curve.apply(rgb),
            TonemapMode::PerChannel(luts) => luts.apply(rgb),
        }
    }
}

impl LuminanceCurve {
    fn apply(&self, hdr: [f32; 3]) -> [f32; 3] {
        let l_hdr = dot3(hdr, self.luma);
        if l_hdr <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        let idx_f = (l_hdr / self.max_hdr) * (LUT_SIZE - 1) as f32;
        let l_sdr = if idx_f >= (LUT_SIZE - 1) as f32 {
            let slope = self.lut[LUT_SIZE - 1] - self.lut[LUT_SIZE - 2];
            self.lut[LUT_SIZE - 1] + slope * (idx_f - (LUT_SIZE - 1) as f32)
        } else if idx_f <= 0.0 {
            self.lut[0]
        } else {
            let idx = idx_f as usize;
            let frac = idx_f - idx as f32;
            self.lut[idx] * (1.0 - frac) + self.lut[idx + 1] * frac
        };

        let ratio = (l_sdr / l_hdr).clamp(0.0, 10.0);
        let sdr = [hdr[0] * ratio, hdr[1] * ratio, hdr[2] * ratio];

        let l_sdr_actual = dot3(sdr, self.luma);
        let adjusted = if self.saturation != 1.0 && l_sdr_actual > 0.001 {
            [
                l_sdr_actual + (sdr[0] - l_sdr_actual) * self.saturation,
                l_sdr_actual + (sdr[1] - l_sdr_actual) * self.saturation,
                l_sdr_actual + (sdr[2] - l_sdr_actual) * self.saturation,
            ]
        } else {
            sdr
        };

        [
            adjusted[0].clamp(0.0, 1.0),
            adjusted[1].clamp(0.0, 1.0),
            adjusted[2].clamp(0.0, 1.0),
        ]
    }
}

impl PerChannelLut {
    fn apply(&self, hdr: [f32; 3]) -> [f32; 3] {
        [
            lookup_lut(&self.lut_r, hdr[0], self.max_hdr),
            lookup_lut(&self.lut_g, hdr[1], self.max_hdr),
            lookup_lut(&self.lut_b, hdr[2], self.max_hdr),
        ]
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sample_step(pixels: usize, max_samples: usize) -> usize {
    if max_samples > 0 && pixels > max_samples {
        (pixels / max_samples).max(1)
    } else {
        1
    }
}

fn validate_pair(hdr: &[f32], sdr: &[f32], channels: u8) -> Result<(usize, usize)> {
    if channels != 3 && channels != 4 {
        return Err(Error::InvalidConfig("channels must be 3 or 4"));
    }
    if hdr.len() != sdr.len() {
        return Err(Error::LengthMismatch {
            hdr: hdr.len(),
            sdr: sdr.len(),
        });
    }
    let ch = channels as usize;
    if !hdr.len().is_multiple_of(ch) {
        return Err(Error::NotAligned {
            len: hdr.len(),
            channels,
        });
    }
    Ok((ch, hdr.len() / ch))
}

fn compute_saturation(rgb: [f32; 3], luminance: f32) -> f32 {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    if luminance > 0.001 {
        (max - min) / luminance
    } else {
        0.0
    }
}

fn finalize_bucketed_lut(lut: &mut [f32; LUT_SIZE], counts: &[u32; LUT_SIZE]) {
    for i in 0..LUT_SIZE {
        if counts[i] > 0 {
            lut[i] /= counts[i] as f32;
        }
    }
    fill_lut_gaps(lut, counts);
    enforce_monotonicity(lut);
}

fn fill_lut_gaps(lut: &mut [f32; LUT_SIZE], counts: &[u32; LUT_SIZE]) {
    let mut last_valid = 0_usize;
    let mut last_value = lut[0];

    for i in 0..LUT_SIZE {
        if counts[i] > 0 {
            if i > last_valid + 1 {
                let start_value = last_value;
                let end_value = lut[i];
                let gap_size = (i - last_valid) as f32;
                #[allow(clippy::needless_range_loop)]
                for j in (last_valid + 1)..i {
                    let t = (j - last_valid) as f32 / gap_size;
                    lut[j] = start_value * (1.0 - t) + end_value * t;
                }
            }
            last_valid = i;
            last_value = lut[i];
        }
    }
    for slot in lut.iter_mut().skip(last_valid + 1) {
        *slot = last_value;
    }
}

fn enforce_monotonicity(lut: &mut [f32; LUT_SIZE]) {
    let mut max_so_far = lut[0];
    for slot in lut.iter_mut().skip(1) {
        if *slot < max_so_far {
            *slot = max_so_far;
        } else {
            max_so_far = *slot;
        }
    }
}

fn lookup_lut(lut: &[f32; LUT_SIZE], value: f32, max_hdr: f32) -> f32 {
    let idx_f = (value / max_hdr).clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
    if idx_f >= (LUT_SIZE - 1) as f32 {
        lut[LUT_SIZE - 1]
    } else {
        let idx = idx_f as usize;
        let frac = idx_f - idx as f32;
        (lut[idx] * (1.0 - frac) + lut[idx + 1] * frac).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn make_pair_rgb(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
        let n = (w * h) as usize;
        let mut hdr = Vec::with_capacity(n * 3);
        let mut sdr = Vec::with_capacity(n * 3);
        for i in 0..n {
            let v = (i as f32 / n as f32) * 4.0;
            hdr.extend_from_slice(&[v, v, v]);
            let s = v / (1.0 + v);
            sdr.extend_from_slice(&[s, s, s]);
        }
        (hdr, sdr)
    }

    #[test]
    fn fit_luminance_reproduces_training() {
        let (hdr, sdr) = make_pair_rgb(16, 16);
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &FitConfig::default()).unwrap();
        assert!(tm.stats().samples > 0);
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in out {
            assert!(
                (c - 0.5).abs() < 0.1,
                "lum fit should reproduce training data, got {c}"
            );
        }
    }

    #[test]
    fn fit_per_channel_reproduces_training() {
        let (hdr, sdr) = make_pair_rgb(16, 16);
        let tm = AdaptiveTonemapper::fit_per_channel(&hdr, &sdr, 3, &FitConfig::default()).unwrap();
        let out = tm.map_rgb([1.0, 1.0, 1.0]);
        for c in out {
            assert!((c - 0.5).abs() < 0.1, "per-channel fit wrong: {c}");
        }
    }

    #[test]
    fn fit_rejects_bad_channels() {
        let buf = vec![0.0_f32; 12];
        let err =
            AdaptiveTonemapper::fit_luminance(&buf, &buf, 2, &FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    #[test]
    fn fit_rejects_length_mismatch() {
        let a = vec![0.0_f32; 12];
        let b = vec![0.0_f32; 9];
        let err = AdaptiveTonemapper::fit_luminance(&a, &b, 3, &FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch { .. }));
    }

    #[test]
    fn fit_rejects_misaligned_length() {
        let buf = vec![0.0_f32; 13];
        let err =
            AdaptiveTonemapper::fit_luminance(&buf, &buf, 3, &FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::NotAligned { .. }));
    }

    #[test]
    fn per_channel_empty_red_errors() {
        // Green/blue have content, red is ~zero
        let mut hdr = Vec::new();
        let mut sdr = Vec::new();
        for i in 0..64 {
            let v = (i as f32 / 64.0) * 2.0;
            hdr.extend_from_slice(&[0.0, v, v]);
            let s = v / (1.0 + v);
            sdr.extend_from_slice(&[0.0, s, s]);
        }
        let err =
            AdaptiveTonemapper::fit_per_channel(&hdr, &sdr, 3, &FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::EmptyChannel { channel: 0 }));
    }

    #[test]
    fn map_row_rgba_preserves_alpha() {
        let n = 16 * 16;
        let mut hdr_rgba = Vec::with_capacity(n * 4);
        let mut sdr_rgba = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = (i as f32 / n as f32) * 4.0;
            hdr_rgba.extend_from_slice(&[v, v, v, 1.0]);
            let s = v / (1.0 + v);
            sdr_rgba.extend_from_slice(&[s, s, s, 1.0]);
        }
        let tm = AdaptiveTonemapper::fit_luminance(&hdr_rgba, &sdr_rgba, 4, &FitConfig::default())
            .unwrap();

        let mut row: Vec<f32> = (0..16).flat_map(|_| [0.5_f32, 0.5, 0.5, 0.42]).collect();
        tm.map_row(&mut row, 4);
        for pixel in row.chunks_exact(4) {
            assert!((pixel[3] - 0.42).abs() < 1e-6, "alpha not preserved");
        }
    }

    #[test]
    fn map_into_copies_and_preserves_alpha() {
        let (hdr_rgb, sdr_rgb) = make_pair_rgb(8, 8);
        let tm = AdaptiveTonemapper::fit_luminance(&hdr_rgb, &sdr_rgb, 3, &FitConfig::default())
            .unwrap();

        let src = [0.5_f32, 0.5, 0.5, 0.42];
        let mut dst = [0.0_f32; 4];
        tm.map_into(&src, &mut dst, 4);
        assert!((dst[3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn compute_mae_populates_option() {
        let (hdr, sdr) = make_pair_rgb(16, 16);
        let cfg = FitConfig {
            compute_mae: true,
            ..Default::default()
        };
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &cfg).unwrap();
        assert!(tm.stats().mae.is_some());
    }

    #[test]
    fn mae_none_by_default() {
        let (hdr, sdr) = make_pair_rgb(16, 16);
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &FitConfig::default()).unwrap();
        assert!(tm.stats().mae.is_none());
    }

    #[test]
    fn detect_saturation_false_leaves_saturation_at_one() {
        let (hdr, sdr) = make_pair_rgb(16, 16);
        let cfg = FitConfig {
            detect_saturation: false,
            ..Default::default()
        };
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &cfg).unwrap();
        assert_eq!(tm.stats().saturation_ratio, 1.0);
    }

    #[test]
    fn full_sampling_max_samples_zero() {
        // max_samples = 0 means "use every pixel" — sample_step = 1.
        let (hdr, sdr) = make_pair_rgb(8, 8);
        let cfg = FitConfig {
            max_samples: 0,
            ..Default::default()
        };
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &cfg).unwrap();
        // 64 pixels, minus the first one which is v=0 (below the 0.001
        // threshold), = 63 valid samples.
        assert_eq!(tm.stats().samples, 63);
    }

    #[test]
    fn apply_extrapolates_beyond_fitted_max_hdr() {
        // Fit on data up to max_hdr = 4.0. Apply to an input at 5.0
        // which is beyond the fit range. The luminance curve should
        // extrapolate linearly from the last two LUT entries instead of
        // clamping to the last value.
        let (hdr, sdr) = make_pair_rgb(32, 32);
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &FitConfig::default()).unwrap();
        assert!(tm.max_hdr_observed() <= 4.0);

        // An input above max_hdr_observed must still produce finite
        // output (not NaN, not clamped to zero).
        let out = tm.map_rgb([5.0, 5.0, 5.0]);
        for c in out {
            assert!(c.is_finite(), "extrapolation produced non-finite {c}");
            assert!(
                (0.0..=1.0).contains(&c),
                "extrapolation should stay in SDR range, got {c}"
            );
        }
    }

    #[test]
    fn per_channel_empty_blue_errors_with_correct_index() {
        let mut hdr = Vec::new();
        let mut sdr = Vec::new();
        for i in 0..64 {
            let v = (i as f32 / 64.0) * 2.0;
            hdr.extend_from_slice(&[v, v, 0.0]);
            let s = v / (1.0 + v);
            sdr.extend_from_slice(&[s, s, 0.0]);
        }
        let err =
            AdaptiveTonemapper::fit_per_channel(&hdr, &sdr, 3, &FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::EmptyChannel { channel: 2 }));
    }

    #[test]
    fn per_channel_empty_green_errors_with_correct_index() {
        let mut hdr = Vec::new();
        let mut sdr = Vec::new();
        for i in 0..64 {
            let v = (i as f32 / 64.0) * 2.0;
            hdr.extend_from_slice(&[v, 0.0, v]);
            let s = v / (1.0 + v);
            sdr.extend_from_slice(&[s, 0.0, s]);
        }
        let err =
            AdaptiveTonemapper::fit_per_channel(&hdr, &sdr, 3, &FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::EmptyChannel { channel: 1 }));
    }

    #[test]
    fn fit_on_known_curve_recovers_it_approximately() {
        // Generate an HDR image and a matching SDR image computed
        // through a simple Reinhard curve. Fit luminance adaptive and
        // verify the LUT approximates the curve when we apply it to
        // unseen intermediate values.
        let n = 64 * 64;
        let mut hdr = Vec::with_capacity(n * 3);
        let mut sdr = Vec::with_capacity(n * 3);
        for i in 0..n {
            let v = (i as f32 / n as f32) * 4.0;
            hdr.extend_from_slice(&[v, v, v]);
            let s = v / (1.0 + v); // Reinhard
            sdr.extend_from_slice(&[s, s, s]);
        }
        let tm = AdaptiveTonemapper::fit_luminance(&hdr, &sdr, 3, &FitConfig::default()).unwrap();

        // Probe at values the fit did see.
        for probe in [0.5_f32, 1.0, 2.0, 3.0] {
            let expected = probe / (1.0 + probe);
            let out = tm.map_rgb([probe, probe, probe]);
            let err = (out[0] - expected).abs();
            assert!(
                err < 0.02,
                "adaptive fit at {probe}: got {}, expected {expected} (err {err})",
                out[0]
            );
        }
    }
}
