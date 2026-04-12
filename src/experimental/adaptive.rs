//! Adaptive tonemapper that fits a LUT from an HDR/SDR pair.
//!
//! Use case: re-encoding edited HDR content while preserving the artistic
//! intent of the original SDR rendition. The fitter samples the HDR→SDR
//! relationship in the source pair and fits either a luminance-domain LUT
//! (hue-preserving) or per-channel LUTs (highest fidelity for round-trips).
//!
//! # Status
//!
//! Lightly tested. API may change. Inputs are tightly-packed interleaved
//! linear f32 buffers; stride equals `width * channels`.

use alloc::boxed::Box;

use crate::error::{Error, Result};
use crate::math::roundf;

/// LUT resolution for adaptive tone curves.
const LUT_SIZE: usize = 4096;

/// Adaptive tonemapper that learns from HDR/SDR pairs.
#[derive(Debug, Clone)]
pub struct AdaptiveTonemapper {
    mode: TonemapMode,
    max_hdr_observed: f32,
    stats: FitStats,
}

/// Tonemapping mode (how the curve is represented).
#[derive(Debug, Clone)]
pub enum TonemapMode {
    /// Luminance-based curve with saturation preservation.
    /// Hue-preserving, most natural for edits.
    Luminance(LuminanceCurve),
    /// Per-channel LUTs. Most accurate for round-trips; may shift hues.
    PerChannel(PerChannelLut),
}

/// Luminance-based tone curve.
#[derive(Debug, Clone)]
pub struct LuminanceCurve {
    /// LUT mapping HDR luminance [0, max_hdr] → SDR luminance [0, 1].
    lut: Box<[f32; LUT_SIZE]>,
    /// Maximum HDR luminance covered by the LUT.
    max_hdr: f32,
    /// Saturation adjustment (1.0 = preserve, >1 = boost, <1 = reduce).
    saturation: f32,
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
    /// Number of pixel samples used.
    pub samples: usize,
    /// Mean absolute error of the fit.
    pub mae: f32,
    /// Maximum observed HDR luminance.
    pub max_hdr_luminance: f32,
    /// Detected saturation change (SDR_sat / HDR_sat).
    pub saturation_ratio: f32,
}

/// Configuration for fitting an adaptive tonemapper.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Fitting mode.
    pub mode: FitMode,
    /// Maximum number of samples (0 = all pixels).
    pub max_samples: usize,
    /// Whether to detect and apply saturation changes (one extra pass).
    pub detect_saturation: bool,
    /// If true, compute mean absolute error as a diagnostic. Default: false.
    /// Enabling this adds a third pass over the source buffers.
    pub compute_mae: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            mode: FitMode::Luminance,
            max_samples: 100_000,
            detect_saturation: true,
            compute_mae: false,
        }
    }
}

/// Which type of curve to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FitMode {
    /// Luminance-based (recommended for most use cases).
    #[default]
    Luminance,
    /// Per-channel LUTs.
    PerChannel,
}

impl AdaptiveTonemapper {
    /// Fit a tonemapper from an HDR/SDR pair using default configuration.
    ///
    /// Both buffers are interleaved linear f32, tightly packed (stride =
    /// `width * channels`). `channels` must be 3 or 4; alpha is ignored.
    pub fn fit(hdr: &[f32], sdr: &[f32], width: u32, height: u32, channels: u8) -> Result<Self> {
        Self::fit_with_config(hdr, sdr, width, height, channels, &FitConfig::default())
    }

    /// Fit with a custom configuration.
    pub fn fit_with_config(
        hdr: &[f32],
        sdr: &[f32],
        width: u32,
        height: u32,
        channels: u8,
        config: &FitConfig,
    ) -> Result<Self> {
        if channels != 3 && channels != 4 {
            return Err(Error::InvalidConfig("channels must be 3 or 4"));
        }
        let required = (width as usize) * (height as usize) * channels as usize;
        if hdr.len() < required {
            return Err(Error::BufferTooSmall {
                required,
                actual: hdr.len(),
            });
        }
        if sdr.len() < required {
            return Err(Error::BufferTooSmall {
                required,
                actual: sdr.len(),
            });
        }
        match config.mode {
            FitMode::Luminance => Self::fit_luminance(hdr, sdr, width, height, channels, config),
            FitMode::PerChannel => Self::fit_per_channel(hdr, sdr, width, height, channels, config),
        }
    }

    /// Apply the tonemapper to a pre-allocated SDR buffer.
    ///
    /// `hdr_in` and `sdr_out` must have matching layout: interleaved f32,
    /// tightly packed, `width * height * channels` elements.
    pub fn apply(
        &self,
        hdr_in: &[f32],
        sdr_out: &mut [f32],
        width: u32,
        height: u32,
        channels: u8,
    ) -> Result<()> {
        if channels != 3 && channels != 4 {
            return Err(Error::InvalidConfig("channels must be 3 or 4"));
        }
        let required = (width as usize) * (height as usize) * channels as usize;
        if hdr_in.len() < required {
            return Err(Error::BufferTooSmall {
                required,
                actual: hdr_in.len(),
            });
        }
        if sdr_out.len() < required {
            return Err(Error::BufferTooSmall {
                required,
                actual: sdr_out.len(),
            });
        }
        let ch = channels as usize;
        for (src, dst) in hdr_in.chunks_exact(ch).zip(sdr_out.chunks_exact_mut(ch)) {
            let hdr = [src[0], src[1], src[2]];
            let mapped = self.tonemap_pixel(hdr);
            dst[0] = mapped[0];
            dst[1] = mapped[1];
            dst[2] = mapped[2];
            if ch == 4 {
                dst[3] = src[3];
            }
        }
        Ok(())
    }

    /// Tonemap a single linear HDR pixel to linear SDR.
    pub fn tonemap_pixel(&self, hdr_linear: [f32; 3]) -> [f32; 3] {
        match &self.mode {
            TonemapMode::Luminance(curve) => curve.apply(hdr_linear),
            TonemapMode::PerChannel(luts) => luts.apply(hdr_linear),
        }
    }

    /// Fitting statistics.
    pub fn stats(&self) -> &FitStats {
        &self.stats
    }

    /// Maximum HDR value observed during fitting.
    pub fn max_hdr_observed(&self) -> f32 {
        self.max_hdr_observed
    }

    /// Representation of the fit curve.
    pub fn mode(&self) -> &TonemapMode {
        &self.mode
    }

    fn fit_luminance(
        hdr: &[f32],
        sdr: &[f32],
        width: u32,
        height: u32,
        channels: u8,
        config: &FitConfig,
    ) -> Result<Self> {
        let ch = channels as usize;
        let total_pixels = (width as usize) * (height as usize);
        let step = sample_step(total_pixels, config.max_samples);

        // Pass 1: find max_hdr (+ optional saturation detection)
        let mut max_hdr = 0.0_f32;
        let mut saturation_sum = 0.0_f32;
        let mut saturation_count = 0_usize;
        let mut sample_count = 0_usize;

        for i in (0..total_pixels).step_by(step) {
            let off = i * ch;
            let hdr_rgb = [hdr[off], hdr[off + 1], hdr[off + 2]];
            let sdr_rgb = [sdr[off], sdr[off + 1], sdr[off + 2]];
            let l_hdr = luminance(hdr_rgb);
            let l_sdr = luminance(sdr_rgb);
            if l_hdr > 0.001 && l_sdr > 0.001 {
                sample_count += 1;
                if l_hdr > max_hdr {
                    max_hdr = l_hdr;
                }
                if config.detect_saturation && l_hdr > 0.01 && l_sdr > 0.01 {
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

        for i in (0..total_pixels).step_by(step) {
            let off = i * ch;
            let l_hdr = luminance([hdr[off], hdr[off + 1], hdr[off + 2]]);
            let l_sdr = luminance([sdr[off], sdr[off + 1], sdr[off + 2]]);
            if l_hdr > 0.001 && l_sdr > 0.001 {
                let idx = roundf((l_hdr / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut[idx] += l_sdr;
                counts[idx] += 1;
            }
        }
        for i in 0..LUT_SIZE {
            if counts[i] > 0 {
                lut[i] /= counts[i] as f32;
            }
        }
        fill_lut_gaps(&mut lut, &counts);
        enforce_monotonicity(&mut lut);

        let saturation = if saturation_count > 0 {
            (saturation_sum / saturation_count as f32).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // Optional pass 3: MAE (off by default)
        let mae = if config.compute_mae {
            let mut mae_sum = 0.0_f32;
            let mut mae_count = 0_usize;
            for i in (0..total_pixels).step_by(step) {
                let off = i * ch;
                let l_hdr = luminance([hdr[off], hdr[off + 1], hdr[off + 2]]);
                let l_sdr = luminance([sdr[off], sdr[off + 1], sdr[off + 2]]);
                if l_hdr > 0.001 && l_sdr > 0.001 {
                    let idx = roundf((l_hdr / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                    mae_sum += (lut[idx] - l_sdr).abs();
                    mae_count += 1;
                }
            }
            if mae_count > 0 {
                mae_sum / mae_count as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(Self {
            mode: TonemapMode::Luminance(LuminanceCurve {
                lut,
                max_hdr,
                saturation,
            }),
            max_hdr_observed: max_hdr,
            stats: FitStats {
                samples: sample_count,
                mae,
                max_hdr_luminance: max_hdr,
                saturation_ratio: saturation,
            },
        })
    }

    fn fit_per_channel(
        hdr: &[f32],
        sdr: &[f32],
        width: u32,
        height: u32,
        channels: u8,
        config: &FitConfig,
    ) -> Result<Self> {
        let ch = channels as usize;
        let total_pixels = (width as usize) * (height as usize);
        let step = sample_step(total_pixels, config.max_samples);
        let max_idx = (LUT_SIZE - 1) as f32;

        // Pass 1: find max_hdr (shared across channels)
        let mut max_hdr = 0.0_f32;
        let mut any_sample = false;
        for i in (0..total_pixels).step_by(step) {
            let off = i * ch;
            let hr = hdr[off];
            let hg = hdr[off + 1];
            let hb = hdr[off + 2];
            if hr > 0.001 || hg > 0.001 || hb > 0.001 {
                any_sample = true;
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
        if !any_sample || max_hdr <= 0.0 {
            return Err(Error::NoValidSamples);
        }

        // Pass 2: bucket each channel directly into its LUT
        let mut lut_r = Box::new([0.0_f32; LUT_SIZE]);
        let mut lut_g = Box::new([0.0_f32; LUT_SIZE]);
        let mut lut_b = Box::new([0.0_f32; LUT_SIZE]);
        let mut cnt_r = [0_u32; LUT_SIZE];
        let mut cnt_g = [0_u32; LUT_SIZE];
        let mut cnt_b = [0_u32; LUT_SIZE];
        let mut sample_count = 0_usize;

        for i in (0..total_pixels).step_by(step) {
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
                sample_count += 1;
            }
            if hg > 0.001 {
                let idx = roundf((hg / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut_g[idx] += sg;
                cnt_g[idx] += 1;
                sample_count += 1;
            }
            if hb > 0.001 {
                let idx = roundf((hb / max_hdr) * max_idx).clamp(0.0, max_idx) as usize;
                lut_b[idx] += sb;
                cnt_b[idx] += 1;
                sample_count += 1;
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
                samples: sample_count,
                mae: 0.0,
                max_hdr_luminance: max_hdr,
                saturation_ratio: 1.0,
            },
        })
    }
}

#[inline]
fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

#[inline]
fn sample_step(total_pixels: usize, max_samples: usize) -> usize {
    if max_samples > 0 && total_pixels > max_samples {
        (total_pixels / max_samples).max(1)
    } else {
        1
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

impl LuminanceCurve {
    fn apply(&self, hdr_linear: [f32; 3]) -> [f32; 3] {
        let l_hdr = 0.2126 * hdr_linear[0] + 0.7152 * hdr_linear[1] + 0.0722 * hdr_linear[2];
        if l_hdr <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        let idx_f = (l_hdr / self.max_hdr) * (LUT_SIZE - 1) as f32;
        let l_sdr = if idx_f >= (LUT_SIZE - 1) as f32 {
            // Extrapolate beyond LUT end
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
        let sdr = [
            hdr_linear[0] * ratio,
            hdr_linear[1] * ratio,
            hdr_linear[2] * ratio,
        ];

        let l_sdr_actual = 0.2126 * sdr[0] + 0.7152 * sdr[1] + 0.0722 * sdr[2];
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
    fn apply(&self, hdr_linear: [f32; 3]) -> [f32; 3] {
        [
            lookup_lut(&self.lut_r, hdr_linear[0], self.max_hdr),
            lookup_lut(&self.lut_g, hdr_linear[1], self.max_hdr),
            lookup_lut(&self.lut_b, hdr_linear[2], self.max_hdr),
        ]
    }
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

    fn make_pair(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
        let n = (w * h) as usize;
        let mut hdr = Vec::with_capacity(n * 3);
        let mut sdr = Vec::with_capacity(n * 3);
        for i in 0..n {
            let v = (i as f32 / n as f32) * 4.0; // HDR up to 4.0
            hdr.extend_from_slice(&[v, v, v]);
            // Synthetic SDR: reinhard-like
            let s = v / (1.0 + v);
            sdr.extend_from_slice(&[s, s, s]);
        }
        (hdr, sdr)
    }

    #[test]
    fn fit_luminance_basic() {
        let (hdr, sdr) = make_pair(16, 16);
        let tm = AdaptiveTonemapper::fit(&hdr, &sdr, 16, 16, 3).unwrap();
        assert!(tm.stats().samples > 0);
        let out = tm.tonemap_pixel([1.0, 1.0, 1.0]);
        // At x=1.0, SDR synthetic = 0.5, so LUT should reproduce ~0.5
        for c in out {
            assert!(
                (c - 0.5).abs() < 0.1,
                "lum fit should reproduce training data, got {c}"
            );
        }
    }

    #[test]
    fn fit_per_channel_basic() {
        let (hdr, sdr) = make_pair(16, 16);
        let cfg = FitConfig {
            mode: FitMode::PerChannel,
            ..Default::default()
        };
        let tm = AdaptiveTonemapper::fit_with_config(&hdr, &sdr, 16, 16, 3, &cfg).unwrap();
        let out = tm.tonemap_pixel([1.0, 1.0, 1.0]);
        for c in out {
            assert!((c - 0.5).abs() < 0.1, "per-channel fit wrong: {c}");
        }
    }

    #[test]
    fn fit_rejects_bad_channels() {
        let buf = vec![0.0_f32; 12];
        let err = AdaptiveTonemapper::fit(&buf, &buf, 2, 2, 2).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    #[test]
    fn fit_rejects_too_small() {
        let buf = vec![0.0_f32; 3];
        let err = AdaptiveTonemapper::fit(&buf, &buf, 4, 4, 3).unwrap_err();
        assert!(matches!(err, Error::BufferTooSmall { .. }));
    }

    #[test]
    fn apply_rgba_preserves_alpha() {
        let n = 16 * 16;
        let mut hdr_rgba = Vec::with_capacity(n * 4);
        let mut sdr_rgba = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = (i as f32 / n as f32) * 4.0;
            hdr_rgba.extend_from_slice(&[v, v, v, 1.0]);
            let s = v / (1.0 + v);
            sdr_rgba.extend_from_slice(&[s, s, s, 1.0]);
        }
        let tm = AdaptiveTonemapper::fit(&hdr_rgba, &sdr_rgba, 16, 16, 4).unwrap();
        let hdr_in: Vec<f32> = (0..n).flat_map(|_| [0.5_f32, 0.5, 0.5, 0.42]).collect();
        let mut sdr_out = vec![0.0_f32; n * 4];
        tm.apply(&hdr_in, &mut sdr_out, 16, 16, 4).unwrap();
        for pixel in sdr_out.chunks_exact(4) {
            assert!((pixel[3] - 0.42).abs() < 1e-6, "alpha not preserved");
        }
    }
}
