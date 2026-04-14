//! Luma gain map splitter — round-trippable HDR ↔ (SDR, log2 gain).
//!
//! Splits a linear-light HDR frame into an SDR base frame plus a per-pixel
//! `log2` gain map, using the ISO 21496-1 / Adobe Gain Map / Apple Ultra
//! HDR decode form:
//!
//! ```text
//! HDR_i = (SDR_i + base_offset_i) · 2^g − alternate_offset_i      (per channel i)
//! ```
//!
//! That is the exact formula implemented by `ultrahdr-core`
//! (`gainmap/apply.rs`, `apply_gain`) and consumed by
//! `zencodec::GainMapParams`. Round-trip is exact within float precision
//! when (a) the curve is strictly monotonic, (b) the observed gain fits
//! in `[min_log2, max_log2]`, and (c) the SDR rescale stays in `[0, 1]`.
//! Out-of-gamut highlights clamp on the SDR side and become lossy — the
//! splitter reports the count via [`SplitStats`].
//!
//! ## Required input descriptor
//!
//! Inputs are interleaved `&[f32]` rows with channels = 3 or 4.
//!
//! - **Transfer**: linear light. Linearize PQ/HLG/sRGB beforehand using
//!   `crate::pipeline` or `linear-srgb`.
//! - **Signal range**: full range. Expand narrow-range first.
//! - **Primaries**: arbitrary, but `SplitConfig::luma_weights` MUST match.
//!   Use [`LUMA_BT709`](crate::LUMA_BT709), [`LUMA_BT2020`](crate::LUMA_BT2020),
//!   or [`LUMA_P3`](crate::LUMA_P3).
//! - **Alpha**: passed through unchanged on RGBA rows.
//!
//! ## Wire-format alignment with `zencodec::GainMapParams`
//!
//! After splitting, build the gain map metadata as:
//!
//! | `GainMapChannel` field | source |
//! |---|---|
//! | `base_offset` | [`SplitConfig::base_offset`] (typ. `1.0/64.0`) |
//! | `alternate_offset` | [`SplitConfig::alternate_offset`] (typ. `1.0/64.0`) |
//! | `min` | `SplitConfig::min_log2` (or `stats.observed_min_log2` if tighter) |
//! | `max` | `SplitConfig::max_log2` (or `stats.observed_max_log2` if tighter) |
//! | `gamma` | encoder choice (typ. `1.0`); the splitter emits raw `log2` |
//!
//! The splitter intentionally emits **raw f32 log2 gain**. u8 quantization
//! and gamma encoding are zencodec's responsibility.
//!
//! ## Multi-channel gain maps
//!
//! Out of scope here — this module produces a single scalar gain per
//! pixel (luminance gain map). For per-channel RGB gain maps, see
//! `ultrahdr-core::compute_multichannel_gainmap`.

use crate::math::{exp2f, log2f};
use crate::{
    Bt2408Tonemapper, Bt2446A, Bt2446B, Bt2446C, CompiledFilmicSpline, ToneMap, ToneMapCurve,
};
use alloc::vec;
use linear_srgb::tf::{linear_to_pq, pq_to_linear};

/// A scalar luma tone curve: `Y_HDR` (linear, ≥0) → `Y_SDR` (linear).
///
/// Implementors must be **strictly monotonic** on the operating range
/// and produce output in `[0, 1]`. The trait is intentionally distinct
/// from [`ToneMap`] — only curves whose natural mode of use is luma-only
/// should implement it. Per-channel and matrix-based curves do not.
///
/// To wrap an ad-hoc closure, use [`LumaFn`].
pub trait LumaToneMap {
    /// Map a single linear-light luminance sample.
    fn map_luma(&self, y_hdr: f32) -> f32;
}

/// Adapt a closure as a [`LumaToneMap`]. Caller is responsible for
/// monotonicity and `[0, 1]` output range.
pub struct LumaFn<F: Fn(f32) -> f32>(pub F);

impl<F: Fn(f32) -> f32> LumaToneMap for LumaFn<F> {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        (self.0)(y)
    }
}

// Reference & boxed dispatch so the splitter can hold trait objects.
impl<T: LumaToneMap + ?Sized> LumaToneMap for &T {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        (**self).map_luma(y)
    }
}

impl<T: LumaToneMap + ?Sized> LumaToneMap for alloc::boxed::Box<T> {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        (**self).map_luma(y)
    }
}

/// Newtype guaranteeing a [`Bt2408Tonemapper`] is in `Yrgb` (luma) mode.
///
/// `MaxRgb` mode is also chromaticity-preserving, but the splitter
/// applies the curve as a Y function — wrapping the `MaxRgb` configuration
/// would silently change its meaning. Use this newtype to make the choice
/// explicit and unmistakable.
pub struct Bt2408Yrgb(Bt2408Tonemapper);

impl Bt2408Yrgb {
    /// Construct from `content_max_nits` and `display_max_nits`. Always uses `Yrgb`.
    pub fn new(content_max_nits: f32, display_max_nits: f32) -> Self {
        // Bt2408Tonemapper::new defaults to Yrgb.
        Self(Bt2408Tonemapper::new(content_max_nits, display_max_nits))
    }

    /// Construct with explicit luma weights. Always uses `Yrgb`.
    pub fn with_luma(content_max_nits: f32, display_max_nits: f32, luma: [f32; 3]) -> Self {
        // Bt2408Tonemapper::with_luma defaults to Yrgb.
        Self(Bt2408Tonemapper::with_luma(
            content_max_nits,
            display_max_nits,
            luma,
        ))
    }
}

impl LumaToneMap for Bt2408Yrgb {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        // Yrgb mode: scalar = luma·rgb; for grayscale [y,y,y] this is y.
        self.0.map_rgb([y, y, y])[0]
    }
}

impl LumaToneMap for Bt2446C {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        self.map_rgb([y, y, y])[0]
    }
}

impl LumaToneMap for Bt2446A {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        self.map_rgb([y, y, y])[0]
    }
}

impl LumaToneMap for Bt2446B {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        self.map_rgb([y, y, y])[0]
    }
}

/// Wraps a [`ToneMapCurve::ExtendedReinhard`] for use as a luma curve.
///
/// Other `ToneMapCurve` variants are intentionally not wrapped: they are
/// per-channel or matrix-based and would silently misbehave as luma curves.
pub struct ExtendedReinhardLuma {
    curve: ToneMapCurve,
}

impl ExtendedReinhardLuma {
    /// `l_max` is the white point in linear light; `luma` weights must
    /// match the input primaries.
    pub fn new(l_max: f32, luma: [f32; 3]) -> Self {
        Self {
            curve: ToneMapCurve::ExtendedReinhard { l_max, luma },
        }
    }
}

impl LumaToneMap for ExtendedReinhardLuma {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        self.curve.map_rgb([y, y, y])[0]
    }
}

/// [`CompiledFilmicSpline`] as a luma curve.
///
/// Grayscale-invariant: for `[y,y,y]` input, `ratios = [1,1,1]` so the
/// desaturation term has no effect and all three output channels are equal.
/// On chromatic input the desaturation DOES remix chrominance — but that
/// only affects `map_rgb`, not `map_luma` which always feeds grayscale.
impl LumaToneMap for CompiledFilmicSpline {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        self.map_rgb([y, y, y])[0]
    }
}

// ----- Splitter -------------------------------------------------------------

/// Splitter configuration.
///
/// Defaults are chosen so the splitter "just works" with any qualifying
/// curve, including shadow-lifters like BT.2446 Method A. The `min_log2`
/// and `max_log2` fields are **safety clamps** that prevent infinite gain
/// at black or unreasonable values from numerical edge cases — they are
/// not quality knobs. The actual gain range to store in
/// `zencodec::GainMapChannel.{min,max}` should come from the
/// [`SplitStats::observed_min_log2`] / [`SplitStats::observed_max_log2`]
/// after a pass over the image.
#[derive(Debug, Clone, Copy)]
pub struct SplitConfig {
    /// RGB → Y weights. Must match the input primaries.
    /// Default: [`LUMA_BT2020`](crate::LUMA_BT2020).
    pub luma_weights: [f32; 3],
    /// Offset on the base (SDR) image. Maps to `GainMapChannel.base_offset`.
    /// Default `1/64` matches `ultrahdr-core::GainMapConfig`.
    pub base_offset: f32,
    /// Offset on the alternate (HDR) image. Maps to `GainMapChannel.alternate_offset`.
    /// Default `1/64` matches `ultrahdr-core::GainMapConfig`.
    pub alternate_offset: f32,
    /// Sanity floor on `log2` gain. Default `-4.0` (1/16×) tolerates
    /// shadow-lifting curves like BT.2446 Method A.
    pub min_log2: f32,
    /// Sanity ceiling on `log2` gain. Default `6.0` (64×) covers Apple
    /// Ultra HDR's ~9-stop headroom plus margin.
    pub max_log2: f32,
    /// Pre-desaturation (crosstalk) parameter in `[0.0, 0.33)`.
    ///
    /// Before the chromaticity-preserving RGB rescale, each HDR channel
    /// is blended toward the pixel's mean:
    /// ```text
    /// R' = (1 − 2α)·R + α·G + α·B
    /// ```
    /// This pulls saturated primaries toward gray, reducing the chance
    /// that the SDR rescale pushes a channel above 1.0 (out-of-gamut).
    /// The inverse matrix is applied after the gain is computed, so the
    /// desaturation is transparent to the round-trip.
    ///
    /// `0.0` = disabled (default). `0.10` is a conservative starting
    /// point; BT.2446-C uses up to `0.33` (but `0.33` is degenerate —
    /// collapses to grayscale).
    pub pre_desaturate: f32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            luma_weights: crate::LUMA_BT2020,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            min_log2: -4.0,
            max_log2: 6.0,
            pre_desaturate: 0.0,
        }
    }
}

/// Per-row statistics. Accumulate across rows; pass into `GainMapParams`
/// metadata or use to tighten `min_log2`/`max_log2` on a second pass.
#[derive(Debug, Clone, Copy)]
pub struct SplitStats {
    /// Smallest pre-clamp `log2` gain seen. Initialize to `f32::INFINITY`.
    pub observed_min_log2: f32,
    /// Largest pre-clamp `log2` gain seen. Initialize to `f32::NEG_INFINITY`.
    pub observed_max_log2: f32,
    /// Pixels where any SDR channel had to be clamped to `[0, 1]`. These
    /// pixels are NOT exactly invertible (HDR roundtrip will differ).
    pub clipped_sdr_pixels: u32,
}

impl Default for SplitStats {
    fn default() -> Self {
        Self {
            observed_min_log2: f32::INFINITY,
            observed_max_log2: f32::NEG_INFINITY,
            clipped_sdr_pixels: 0,
        }
    }
}

/// Splits HDR rows into (SDR, log2-gain) pairs around a [`LumaToneMap`].
///
/// Stateless after construction. Safe to share across threads (`Sync`)
/// when the inner curve is `Sync`.
pub struct LumaGainMapSplitter<T: LumaToneMap> {
    curve: T,
    cfg: SplitConfig,
}

impl<T: LumaToneMap> LumaGainMapSplitter<T> {
    /// Construct from a curve and config.
    pub fn new(curve: T, cfg: SplitConfig) -> Self {
        Self { curve, cfg }
    }

    /// Borrow the configuration (e.g. to feed into `GainMapParams`).
    pub fn config(&self) -> &SplitConfig {
        &self.cfg
    }

    /// Borrow the curve.
    pub fn curve(&self) -> &T {
        &self.curve
    }

    /// Encode one row.
    ///
    /// `hdr`: linear-light HDR (length = pixels · `channels`).
    /// `sdr_out`: receives linear SDR; must match `hdr.len()`.
    /// `gain_out`: receives one `log2` gain per pixel; length = pixels.
    /// `stats`: accumulator; updated in place.
    ///
    /// Panics if `channels` is not 3 or 4, or if lengths mismatch.
    pub fn split_row(
        &self,
        hdr: &[f32],
        sdr_out: &mut [f32],
        gain_out: &mut [f32],
        channels: u8,
        stats: &mut SplitStats,
    ) {
        match channels {
            3 => self.split_cn::<3>(hdr, sdr_out, gain_out, stats),
            4 => self.split_cn::<4>(hdr, sdr_out, gain_out, stats),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }

    /// Decode one row: SDR + log2 gain → HDR.
    ///
    /// Inverse of [`Self::split_row`] (modulo SDR clipping recorded in
    /// [`SplitStats::clipped_sdr_pixels`] and any external u8 quantization
    /// of the gain map).
    pub fn apply_row(&self, sdr: &[f32], gain: &[f32], hdr_out: &mut [f32], channels: u8) {
        match channels {
            3 => self.apply_cn::<3>(sdr, gain, hdr_out),
            4 => self.apply_cn::<4>(sdr, gain, hdr_out),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }

    #[inline]
    fn split_cn<const CN: usize>(
        &self,
        hdr: &[f32],
        sdr: &mut [f32],
        gain: &mut [f32],
        st: &mut SplitStats,
    ) {
        debug_assert!(CN == 3 || CN == 4);
        assert_eq!(hdr.len(), sdr.len());
        assert_eq!(hdr.len() / CN, gain.len());
        let [wr, wg, wb] = self.cfg.luma_weights;
        let (b, a) = (self.cfg.base_offset, self.cfg.alternate_offset);
        let (lo, hi) = (self.cfg.min_log2, self.cfg.max_log2);
        let alpha = self.cfg.pre_desaturate;
        let has_ct = alpha > 0.0;

        for ((h, s), gp) in hdr
            .chunks_exact(CN)
            .zip(sdr.chunks_exact_mut(CN))
            .zip(gain.iter_mut())
        {
            let r = h[0].max(0.0);
            let gc = h[1].max(0.0);
            let bl = h[2].max(0.0);

            // Optional pre-desaturation (crosstalk matrix).
            // Blends each channel toward the pixel mean to keep the
            // chromaticity-preserving rescale in-gamut.
            let (cr, cg, cb) = if has_ct {
                let d = 1.0 - 2.0 * alpha;
                (
                    d * r + alpha * gc + alpha * bl,
                    alpha * r + d * gc + alpha * bl,
                    alpha * r + alpha * gc + d * bl,
                )
            } else {
                (r, gc, bl)
            };

            let y_hdr = wr * cr + wg * cg + wb * cb;
            let y_sdr = self.curve.map_luma(y_hdr).clamp(0.0, 1.0);

            // Choose gain from the luma ratio. Both offsets prevent 0/0
            // at black and align with ultrahdr `compute_luminance_gainmap`.
            let raw_log2 = log2f((y_hdr + a) / (y_sdr + b));
            if raw_log2 < st.observed_min_log2 {
                st.observed_min_log2 = raw_log2;
            }
            if raw_log2 > st.observed_max_log2 {
                st.observed_max_log2 = raw_log2;
            }
            let g_log2 = raw_log2.clamp(lo, hi);
            let m = exp2f(g_log2);

            // Per-channel SDR from the (possibly desaturated) HDR channels.
            //   HDR_i = (SDR_i + b) · 2^g − a   ⇒   SDR_i = (HDR_i + a) / 2^g − b
            let d0 = (cr + a) / m - b;
            let d1 = (cg + a) / m - b;
            let d2 = (cb + a) / m - b;

            // Inverse crosstalk to recover original chromaticity.
            let (s0, s1, s2) = if has_ct {
                let inv_a = -alpha / (1.0 - 3.0 * alpha);
                let id = 1.0 - 2.0 * inv_a;
                (
                    id * d0 + inv_a * d1 + inv_a * d2,
                    inv_a * d0 + id * d1 + inv_a * d2,
                    inv_a * d0 + inv_a * d1 + id * d2,
                )
            } else {
                (d0, d1, d2)
            };

            // Clip detection uses a small tolerance so float roundoff in the
            // log2/exp2 round of `m` doesn't get flagged as a real out-of-gamut
            // event. Real out-of-gamut highlights overshoot 1.0 by far more.
            const CLIP_EPS: f32 = 1.0e-4;
            let clipped = s0 < -CLIP_EPS
                || s1 < -CLIP_EPS
                || s2 < -CLIP_EPS
                || s0 > 1.0 + CLIP_EPS
                || s1 > 1.0 + CLIP_EPS
                || s2 > 1.0 + CLIP_EPS;
            if clipped {
                st.clipped_sdr_pixels = st.clipped_sdr_pixels.saturating_add(1);
            }
            s[0] = s0.clamp(0.0, 1.0);
            s[1] = s1.clamp(0.0, 1.0);
            s[2] = s2.clamp(0.0, 1.0);
            if CN == 4 {
                s[3] = h[3];
            }
            *gp = g_log2;
        }
    }

    #[inline]
    fn apply_cn<const CN: usize>(&self, sdr: &[f32], gain: &[f32], hdr: &mut [f32]) {
        debug_assert!(CN == 3 || CN == 4);
        assert_eq!(sdr.len(), hdr.len());
        assert_eq!(sdr.len() / CN, gain.len());
        let (b, a) = (self.cfg.base_offset, self.cfg.alternate_offset);

        for ((s, &g), h) in sdr
            .chunks_exact(CN)
            .zip(gain.iter())
            .zip(hdr.chunks_exact_mut(CN))
        {
            let m = exp2f(g);
            h[0] = ((s[0] + b) * m - a).max(0.0);
            h[1] = ((s[1] + b) * m - a).max(0.0);
            h[2] = ((s[2] + b) * m - a).max(0.0);
            if CN == 4 {
                h[3] = s[3];
            }
        }
    }

    /// Encode one PQ-encoded HDR row (BT.2020 or BT.709 wire), normalizing
    /// internally so `1.0 = content_peak_nits` before applying the curve.
    ///
    /// PQ samples range `[0, 1]` mapping to `[0, 10000 nits]` (ST.2084).
    /// This method linearizes, rescales to `1.0 = content_peak_nits`, runs
    /// [`Self::split_row`], and emits **linear-light SDR** in `[0, 1]`.
    /// Caller may then sRGB-encode for storage. Allocates a transient
    /// linear buffer; for hot paths, call [`pq_to_normalized_linear_row`]
    /// + [`Self::split_row`] directly with reused buffers.
    pub fn split_pq_row(
        &self,
        pq: &[f32],
        sdr_linear: &mut [f32],
        gain: &mut [f32],
        channels: u8,
        content_peak_nits: f32,
        stats: &mut SplitStats,
    ) {
        let mut linear = vec![0.0_f32; pq.len()];
        linear.copy_from_slice(pq);
        pq_to_normalized_linear_row(&mut linear, channels, content_peak_nits);
        self.split_row(&linear, sdr_linear, gain, channels, stats);
    }

    /// Encode one HLG-encoded HDR row. Linearizes via the HLG inverse OETF +
    /// OOTF, rescales so `1.0 = content_peak_nits`, then splits.
    ///
    /// Allocates a transient linear buffer; for hot paths, call
    /// [`hlg_to_normalized_linear_row`] + [`Self::split_row`] with reused
    /// buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn split_hlg_row(
        &self,
        hlg: &[f32],
        sdr_linear: &mut [f32],
        gain: &mut [f32],
        channels: u8,
        display_peak_nits: f32,
        content_peak_nits: f32,
        stats: &mut SplitStats,
    ) {
        let mut linear = vec![0.0_f32; hlg.len()];
        linear.copy_from_slice(hlg);
        hlg_to_normalized_linear_row(&mut linear, channels, display_peak_nits, content_peak_nits);
        self.split_row(&linear, sdr_linear, gain, channels, stats);
    }

    /// Decode a linear-light SDR row + log2 gain row back to HLG-encoded HDR.
    /// Inverse of [`Self::split_hlg_row`].
    pub fn apply_hlg_row(
        &self,
        sdr_linear: &[f32],
        gain: &[f32],
        hlg_out: &mut [f32],
        channels: u8,
        display_peak_nits: f32,
        content_peak_nits: f32,
    ) {
        self.apply_row(sdr_linear, gain, hlg_out, channels);
        normalized_linear_to_hlg_row(hlg_out, channels, display_peak_nits, content_peak_nits);
    }

    /// Decode a linear-light SDR row + log2 gain row back to PQ-encoded HDR.
    /// Inverse of [`Self::split_pq_row`].
    pub fn apply_pq_row(
        &self,
        sdr_linear: &[f32],
        gain: &[f32],
        pq_out: &mut [f32],
        channels: u8,
        content_peak_nits: f32,
    ) {
        self.apply_row(sdr_linear, gain, pq_out, channels);
        normalized_linear_to_pq_row(pq_out, channels, content_peak_nits);
    }
}

// ----- PQ helpers -----------------------------------------------------------

/// Linearize a PQ row in place and rescale so `1.0 = content_peak_nits`.
///
/// Use as a preprocessing step before [`LumaGainMapSplitter::split_row`]
/// when your HDR source is PQ (ST.2084) wire format. RGBA alpha in
/// channel 3 is passed through unchanged.
///
/// `content_peak_nits` is the source content peak (e.g. 1000 for an
/// HDR10 1000-nit master). Curves like [`Bt2446A`](crate::Bt2446A)
/// expect `1.0` to mean their constructor's `hdr_peak_nits`.
pub fn pq_to_normalized_linear_row(in_out: &mut [f32], channels: u8, content_peak_nits: f32) {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    assert!(
        content_peak_nits > 0.0,
        "content_peak_nits must be positive"
    );
    let scale = 10000.0 / content_peak_nits;
    let ch = channels as usize;
    for px in in_out.chunks_exact_mut(ch) {
        px[0] = pq_to_linear(px[0]) * scale;
        px[1] = pq_to_linear(px[1]) * scale;
        px[2] = pq_to_linear(px[2]) * scale;
        // alpha (px[3]) untouched
    }
}

/// Inverse of [`pq_to_normalized_linear_row`]. Encodes a linear-light
/// row (where `1.0 = content_peak_nits`) into PQ wire format in place.
pub fn normalized_linear_to_pq_row(in_out: &mut [f32], channels: u8, content_peak_nits: f32) {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    assert!(
        content_peak_nits > 0.0,
        "content_peak_nits must be positive"
    );
    let inv_scale = content_peak_nits / 10000.0;
    let ch = channels as usize;
    for px in in_out.chunks_exact_mut(ch) {
        px[0] = linear_to_pq(px[0] * inv_scale);
        px[1] = linear_to_pq(px[1] * inv_scale);
        px[2] = linear_to_pq(px[2] * inv_scale);
        // alpha untouched
    }
}

// ----- HLG helpers ----------------------------------------------------------

/// Linearize an HLG-encoded row in place, apply the OOTF, and rescale
/// so `1.0 = content_peak_nits`.
///
/// HLG wire values in `[0, 1]` are decoded via the HLG inverse OETF
/// (from `linear-srgb`) to scene-linear, then the BT.2100 OOTF is
/// applied with [`hlg_system_gamma`](crate::hlg::hlg_system_gamma) for
/// `display_peak_nits`. The result is display-linear, normalized so
/// `1.0 = content_peak_nits`.
///
/// RGBA alpha in channel 3 is passed through unchanged.
pub fn hlg_to_normalized_linear_row(
    in_out: &mut [f32],
    channels: u8,
    display_peak_nits: f32,
    content_peak_nits: f32,
) {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    assert!(
        content_peak_nits > 0.0,
        "content_peak_nits must be positive"
    );
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);
    // After OOTF, output is display-linear in [0, 1] where 1.0 = display_peak_nits.
    // Rescale so 1.0 = content_peak_nits.
    let scale = display_peak_nits / content_peak_nits;
    let ch = channels as usize;
    for px in in_out.chunks_exact_mut(ch) {
        let scene = [
            linear_srgb::tf::hlg_to_linear(px[0]),
            linear_srgb::tf::hlg_to_linear(px[1]),
            linear_srgb::tf::hlg_to_linear(px[2]),
        ];
        let display = crate::hlg::hlg_ootf(scene, gamma);
        px[0] = display[0] * scale;
        px[1] = display[1] * scale;
        px[2] = display[2] * scale;
        // alpha untouched
    }
}

/// Inverse of [`hlg_to_normalized_linear_row`]. Encodes a display-linear
/// row (where `1.0 = content_peak_nits`) back to HLG wire format in place.
pub fn normalized_linear_to_hlg_row(
    in_out: &mut [f32],
    channels: u8,
    display_peak_nits: f32,
    content_peak_nits: f32,
) {
    assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    assert!(
        content_peak_nits > 0.0,
        "content_peak_nits must be positive"
    );
    let gamma = crate::hlg::hlg_system_gamma(display_peak_nits);
    let inv_scale = content_peak_nits / display_peak_nits;
    let ch = channels as usize;
    for px in in_out.chunks_exact_mut(ch) {
        // Undo display normalization.
        let display = [px[0] * inv_scale, px[1] * inv_scale, px[2] * inv_scale];
        // Inverse OOTF → scene-linear.
        let scene = crate::hlg::hlg_inverse_ootf(display, gamma);
        // Scene-linear → HLG wire via OETF.
        px[0] = linear_srgb::tf::linear_to_hlg(scene[0]);
        px[1] = linear_srgb::tf::linear_to_hlg(scene[1]);
        px[2] = linear_srgb::tf::linear_to_hlg(scene[2]);
        // alpha untouched
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FilmicSplineConfig, LUMA_BT2020, LUMA_BT709};
    use alloc::vec;
    use alloc::vec::Vec;

    /// Contract every `LumaToneMap` impl must satisfy:
    ///   - `T(0)` is near zero (some curves like filmic lift blacks slightly)
    ///   - strictly monotonic on the operating range
    ///   - finite output
    ///
    /// Output is **not** required to stay within `[0, 1]` — some curves
    /// (e.g. `Bt2446C` past its inflection point) overshoot slightly. The
    /// splitter clamps before computing gain.
    fn assert_luma_curve_well_behaved<T: LumaToneMap>(curve: &T, max_input: f32, name: &str) {
        let zero = curve.map_luma(0.0);
        // Filmic curves lift black to a small target (e.g. 0.015% of SDR peak).
        assert!(zero.abs() < 2e-3, "{name}: T(0) = {zero}, expected near 0");

        let mut prev = curve.map_luma(0.0);
        let steps = 256;
        for i in 1..=steps {
            let y = (i as f32 / steps as f32) * max_input;
            let s = curve.map_luma(y);
            assert!(s.is_finite(), "{name}: T({y}) = {s} not finite");
            assert!(
                s >= prev - 1.0e-5,
                "{name}: not monotonic at y={y}: {prev} -> {s}"
            );
            prev = s;
        }
    }

    #[test]
    fn bt2446c_well_behaved() {
        let c = Bt2446C::new(1000.0, 100.0);
        assert_luma_curve_well_behaved(&c, 10.0, "Bt2446C");
    }

    #[test]
    fn bt2446a_well_behaved() {
        let c = Bt2446A::new(1000.0, 100.0);
        assert_luma_curve_well_behaved(&c, 10.0, "Bt2446A");
    }

    #[test]
    fn bt2446b_well_behaved() {
        let c = Bt2446B::new(1000.0, 100.0);
        assert_luma_curve_well_behaved(&c, 10.0, "Bt2446B");
    }

    #[test]
    fn bt2408_yrgb_well_behaved() {
        let c = Bt2408Yrgb::new(4000.0, 1000.0);
        assert_luma_curve_well_behaved(&c, 10.0, "Bt2408Yrgb");
    }

    #[test]
    fn extended_reinhard_well_behaved() {
        let c = ExtendedReinhardLuma::new(4.0, LUMA_BT709);
        assert_luma_curve_well_behaved(&c, 10.0, "ExtendedReinhardLuma");
    }

    #[test]
    fn filmic_spline_well_behaved() {
        let c = CompiledFilmicSpline::new(&FilmicSplineConfig::default());
        assert_luma_curve_well_behaved(&c, 10.0, "CompiledFilmicSpline");
    }

    /// Build a grayscale HDR row in `[0, max]`. Grayscale is the exact
    /// round-trip case: no SDR channel is pushed out of `[0, 1]` by the
    /// chromaticity-preserving rescale.
    fn synth_grayscale_hdr_row(pixels: usize, channels: usize, max: f32) -> Vec<f32> {
        let mut row = Vec::with_capacity(pixels * channels);
        for i in 0..pixels {
            let y = (i as f32 / pixels.max(1) as f32) * max;
            row.push(y);
            row.push(y);
            row.push(y);
            if channels == 4 {
                row.push(0.25 + (i as f32 / pixels.max(1) as f32) * 0.5);
            }
        }
        row
    }

    /// End-to-end exact round-trip on grayscale HDR with Bt2446C.
    /// Grayscale + wide gain range = no clamping anywhere.
    #[test]
    fn round_trip_bt2446c_grayscale_exact() {
        let split = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                luma_weights: LUMA_BT709,
                max_log2: 10.0,
                ..Default::default()
            },
        );
        let hdr = synth_grayscale_hdr_row(16, 3, 4.0);
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 3];
        let mut rec = vec![0.0; hdr.len()];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        split.apply_row(&sdr, &gain, &mut rec, 3);
        assert_eq!(stats.clipped_sdr_pixels, 0, "grayscale should never clip");
        for (a, b) in hdr.iter().zip(&rec) {
            assert!((a - b).abs() < 1e-4, "round-trip drift: {a} vs {b}");
        }
    }

    /// Same exact-round-trip property for all other qualifying curves.
    #[test]
    fn round_trip_qualifying_curves_grayscale_exact() {
        let cfg = SplitConfig {
            luma_weights: LUMA_BT2020,
            max_log2: 10.0,
            ..Default::default()
        };
        let hdr = synth_grayscale_hdr_row(16, 3, 1.5);

        let curves: Vec<(alloc::boxed::Box<dyn LumaToneMap>, &'static str)> = vec![
            (alloc::boxed::Box::new(Bt2446A::new(1000.0, 100.0)), "Bt2446A"),
            (alloc::boxed::Box::new(Bt2446B::new(1000.0, 100.0)), "Bt2446B"),
            (
                alloc::boxed::Box::new(Bt2408Yrgb::with_luma(4000.0, 1000.0, LUMA_BT2020)),
                "Bt2408Yrgb",
            ),
            (
                alloc::boxed::Box::new(ExtendedReinhardLuma::new(4.0, LUMA_BT2020)),
                "ExtendedReinhardLuma",
            ),
            (
                alloc::boxed::Box::new(CompiledFilmicSpline::with_luma(
                    &FilmicSplineConfig::default(),
                    LUMA_BT2020,
                )),
                "CompiledFilmicSpline",
            ),
        ];

        for (curve, name) in curves {
            let split = LumaGainMapSplitter::new(curve, cfg);
            let mut sdr = vec![0.0; hdr.len()];
            let mut gain = vec![0.0; hdr.len() / 3];
            let mut rec = vec![0.0; hdr.len()];
            let mut stats = SplitStats::default();
            split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
            split.apply_row(&sdr, &gain, &mut rec, 3);
            assert_eq!(
                stats.clipped_sdr_pixels, 0,
                "{name}: grayscale should never clip"
            );
            for (i, (a, b)) in hdr.iter().zip(&rec).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "{name}: drift at [{i}]: {a} vs {b}"
                );
            }
        }
    }

    /// Exercise the clipping path: chromatic HDR with a narrow gain range
    /// is allowed to drift on clipped pixels, but un-clipped pixels still
    /// round-trip exactly.
    #[test]
    fn chromatic_hdr_clipped_pixels_isolated() {
        let split = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                luma_weights: LUMA_BT709,
                max_log2: 10.0,
                ..Default::default()
            },
        );
        // Mix of grayscale (round-trippable) and saturated (will clip).
        let hdr: Vec<f32> = vec![
            0.5, 0.5, 0.5, // grayscale, exact
            0.1, 0.1, 0.1, // grayscale, exact
            4.0, 0.05, 0.05, // saturated red highlight: SDR rescale will clip
        ];
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 3];
        let mut rec = vec![0.0; hdr.len()];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        split.apply_row(&sdr, &gain, &mut rec, 3);
        assert!(
            stats.clipped_sdr_pixels >= 1,
            "expected the saturated pixel to clip, stats: {stats:?}"
        );
        // First two grayscale pixels MUST be exact even though pixel 2 clipped.
        for i in 0..6 {
            assert!(
                (hdr[i] - rec[i]).abs() < 1e-4,
                "grayscale drift at [{i}]: {} vs {}",
                hdr[i],
                rec[i]
            );
        }
        // Third pixel: SDR was clamped, so HDR doesn't round-trip — but the
        // recovered values must still be finite and non-negative.
        for v in &rec[6..9] {
            assert!(v.is_finite() && *v >= 0.0, "pixel 2 invalid: {v}");
        }
    }

    /// Alpha is preserved through encode and decode on RGBA rows.
    #[test]
    fn rgba_alpha_passthrough() {
        let split = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig::default(),
        );
        let hdr = synth_grayscale_hdr_row(4, 4, 0.6);
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 4];
        let mut rec = vec![0.0; hdr.len()];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 4, &mut stats);
        split.apply_row(&sdr, &gain, &mut rec, 4);
        for (i, src) in hdr.chunks_exact(4).enumerate() {
            assert_eq!(sdr[i * 4 + 3], src[3], "sdr alpha drift at pixel {i}");
            assert_eq!(rec[i * 4 + 3], src[3], "rec alpha drift at pixel {i}");
        }
    }

    /// Pixels above the `max_log2` clamp must still produce valid output
    /// and bump `clipped_sdr_pixels` rather than panic.
    #[test]
    fn extreme_highlights_clamp_gracefully() {
        let split = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                max_log2: 3.0, // 8× cap; HDR with luma 100 will exceed
                ..Default::default()
            },
        );
        let hdr = vec![100.0_f32, 100.0, 100.0, 0.5, 0.5, 0.5];
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 3];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        for v in &sdr {
            assert!((0.0..=1.0).contains(v), "SDR out of range: {v}");
        }
        for v in &gain {
            assert!((0.0..=3.0).contains(v), "gain out of clamp: {v}");
        }
        assert!(
            stats.observed_max_log2 > 3.0,
            "observed_max_log2 should report pre-clamp value, got {}",
            stats.observed_max_log2
        );
    }

    /// PQ HDR → split → apply → PQ HDR round-trips on grayscale within
    /// PQ wire-format precision (~3e-6 per linear-srgb's `linear_to_pq`).
    #[test]
    fn pq_round_trip_grayscale() {
        let split = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                luma_weights: LUMA_BT2020,
                ..Default::default()
            },
        );
        // Eight grayscale PQ samples spanning shadows to highlights.
        // PQ 0.58 ≈ 203 nits (HDR Reference White); PQ 0.75 ≈ 1000 nits.
        let pq: Vec<f32> = (0..8)
            .flat_map(|i| {
                let v = i as f32 / 7.0 * 0.75; // up to 1000 nits @ peak=1000
                [v, v, v]
            })
            .collect();
        let mut sdr = vec![0.0; pq.len()];
        let mut gain = vec![0.0; pq.len() / 3];
        let mut rec = vec![0.0; pq.len()];
        let mut stats = SplitStats::default();
        split.split_pq_row(&pq, &mut sdr, &mut gain, 3, 1000.0, &mut stats);
        split.apply_pq_row(&sdr, &gain, &mut rec, 3, 1000.0);
        for (i, (a, b)) in pq.iter().zip(&rec).enumerate() {
            // PQ encode/decode itself is ~3e-6 accurate — round-trip is
            // dominated by that, not the splitter.
            assert!(
                (a - b).abs() < 1e-3,
                "PQ round-trip drift at [{i}]: {a} vs {b}"
            );
        }
    }

    /// Standalone PQ helper round-trip.
    #[test]
    fn pq_helpers_round_trip() {
        let data: Vec<f32> = (0..15).map(|i| i as f32 / 14.0).collect();
        let mut roundtrip = data.clone();
        pq_to_normalized_linear_row(&mut roundtrip, 3, 1000.0);
        normalized_linear_to_pq_row(&mut roundtrip, 3, 1000.0);
        for (i, (a, b)) in data.iter().zip(&roundtrip).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "PQ helper drift at [{i}]: {a} vs {b}"
            );
        }
    }

    /// PQ helper preserves alpha untouched on RGBA rows.
    #[test]
    fn pq_helper_alpha_untouched() {
        let mut row: Vec<f32> = vec![0.5, 0.5, 0.5, 0.42, 0.7, 0.7, 0.7, 0.99];
        pq_to_normalized_linear_row(&mut row, 4, 1000.0);
        assert!((row[3] - 0.42).abs() < 1e-9);
        assert!((row[7] - 0.99).abs() < 1e-9);
    }

    /// HLG HDR → split → apply → HLG HDR round-trips on grayscale.
    #[test]
    fn hlg_round_trip_grayscale() {
        let split = LumaGainMapSplitter::new(
            Bt2446B::new(1000.0, 100.0),
            SplitConfig {
                luma_weights: LUMA_BT2020,
                ..Default::default()
            },
        );
        // HLG samples: 0.75 = reference white; up to 0.9.
        let hlg: Vec<f32> = (0..8)
            .flat_map(|i| {
                let v = 0.1 + i as f32 / 7.0 * 0.8;
                [v, v, v]
            })
            .collect();
        let mut sdr = vec![0.0; hlg.len()];
        let mut gain = vec![0.0; hlg.len() / 3];
        let mut rec = vec![0.0; hlg.len()];
        let mut stats = SplitStats::default();
        split.split_hlg_row(&hlg, &mut sdr, &mut gain, 3, 1000.0, 1000.0, &mut stats);
        split.apply_hlg_row(&sdr, &gain, &mut rec, 3, 1000.0, 1000.0);
        for (i, (a, b)) in hlg.iter().zip(&rec).enumerate() {
            assert!(
                (a - b).abs() < 5e-3,
                "HLG round-trip drift at [{i}]: {a} vs {b}"
            );
        }
    }

    /// HLG helpers round-trip.
    #[test]
    fn hlg_helpers_round_trip() {
        let data: Vec<f32> = (0..15).map(|i| 0.05 + i as f32 / 14.0 * 0.9).collect();
        let mut roundtrip = data.clone();
        hlg_to_normalized_linear_row(&mut roundtrip, 3, 1000.0, 1000.0);
        normalized_linear_to_hlg_row(&mut roundtrip, 3, 1000.0, 1000.0);
        for (i, (a, b)) in data.iter().zip(&roundtrip).enumerate() {
            assert!(
                (a - b).abs() < 5e-3,
                "HLG helper drift at [{i}]: {a} vs {b}"
            );
        }
    }

    /// Pre-desaturation reduces clipping on saturated highlights.
    #[test]
    fn pre_desaturation_reduces_clipping() {
        let hdr: Vec<f32> = vec![
            4.0, 0.05, 0.05, // saturated red — will clip without desaturation
            0.05, 4.0, 0.05, // saturated green
            0.05, 0.05, 4.0, // saturated blue
        ];
        let no_desat = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                pre_desaturate: 0.0,
                ..Default::default()
            },
        );
        let with_desat = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                pre_desaturate: 0.10,
                ..Default::default()
            },
        );
        let mut sdr_a = vec![0.0; hdr.len()];
        let mut sdr_b = vec![0.0; hdr.len()];
        let mut gain_a = vec![0.0; hdr.len() / 3];
        let mut gain_b = vec![0.0; hdr.len() / 3];
        let mut stats_a = SplitStats::default();
        let mut stats_b = SplitStats::default();
        no_desat.split_row(&hdr, &mut sdr_a, &mut gain_a, 3, &mut stats_a);
        with_desat.split_row(&hdr, &mut sdr_b, &mut gain_b, 3, &mut stats_b);
        assert!(
            stats_a.clipped_sdr_pixels > 0,
            "saturated pixels should clip without desaturation"
        );
        assert!(
            stats_b.clipped_sdr_pixels <= stats_a.clipped_sdr_pixels,
            "desaturation should reduce or equal clipping: {} vs {}",
            stats_b.clipped_sdr_pixels,
            stats_a.clipped_sdr_pixels
        );
    }

    /// Pre-desaturation preserves exact grayscale round-trip.
    #[test]
    fn pre_desaturation_grayscale_exact() {
        let split = LumaGainMapSplitter::new(
            Bt2446C::new(1000.0, 100.0),
            SplitConfig {
                pre_desaturate: 0.15,
                ..Default::default()
            },
        );
        let hdr = synth_grayscale_hdr_row(16, 3, 2.0);
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 3];
        let mut rec = vec![0.0; hdr.len()];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        split.apply_row(&sdr, &gain, &mut rec, 3);
        assert_eq!(stats.clipped_sdr_pixels, 0);
        for (a, b) in hdr.iter().zip(&rec) {
            assert!(
                (a - b).abs() < 1e-4,
                "grayscale drift with desaturation: {a} vs {b}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "channels must be 3 or 4")]
    fn bad_channel_count_panics() {
        let split =
            LumaGainMapSplitter::new(Bt2446C::new(1000.0, 100.0), SplitConfig::default());
        let mut sdr = [0.0_f32; 12];
        let mut gain = [0.0_f32; 6];
        let mut stats = SplitStats::default();
        split.split_row(&[0.0_f32; 12], &mut sdr, &mut gain, 2, &mut stats);
    }
}
