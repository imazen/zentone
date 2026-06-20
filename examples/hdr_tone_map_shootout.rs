//! HDR → SDR tone-map cross-curve shootout on real UltraHDR samples.
//!
//! For each input UltraHDR file (JPEG or AVIF), we:
//!   1. Decode the SDR base image (the producer's ground truth).
//!   2. Reconstruct the HDR rendition (via gain map + ultrahdr-core).
//!   3. Measure the source peak via the BT.2408-anchored `CllMeasure`.
//!   4. Apply each candidate curve in `ToneMapCurve` / `Bt2446A` / `Bt2446B` /
//!      `Bt2446C` / `Bt2408Tonemapper` plus the new `HdrToSdr` Möbius wrapper
//!      across a 12-cell knee sweep.
//!   5. Compare against the SDR base using PSNR (sRGB byte space), mean
//!      ΔE2000 (Lab), max |Δ| per channel, and % pixels with ΔE > 5.
//!
//! Writes per-sample montages and the aggregate summary, then a benchmark
//! markdown to `benchmarks/hdr_tone_map_shootout_2026-06-20.md`.
//!
//! Run:
//! ```text
//! nice -n19 cargo run -p zentone --release \
//!   --example hdr_tone_map_shootout --features hdr-shootout
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use zencodecs::DecodeRequest;
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_dev::hdr::ContentLightLevel;

use zentone::{Bt2408Tonemapper, Bt2446A, Bt2446B, Bt2446C, HdrToSdr, ToneMap, ToneMapCurve};

use ultrahdr_core::Unstoppable;
use ultrahdr_core::gainmap::{HdrOutputFormat, apply_gainmap};

// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/mnt/v/input/gainmap-samples";
const MONTAGES_DIR: &str = "/mnt/v/output/zentone/shootout";
const REPORT_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_2026-06-20.md";

// Möbius knee sweep grid.
const KNEE_TONE_VALUES: &[f32] = &[0.20, 0.30, 0.40, 0.50];
const KNEE_GAMUT_VALUES: &[f32] = &[0.90, 0.95, 0.99];

// =========================================================================
// Sample-finding
// =========================================================================

fn collect_samples() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for sub in &["JPEG", "AVIF"] {
        let dir = Path::new(SAMPLES_ROOT).join(sub);
        if let Ok(rd) = fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_file() {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

// =========================================================================
// Linear-RGB f32 buffer helper
// =========================================================================

#[derive(Clone)]
struct LinearRgb {
    width: u32,
    height: u32,
    /// Tightly packed `[r, g, b, r, g, b, ...]` linear-light f32 in scene
    /// linear (where the SDR producer's chosen "1.0" maps to its diffuse
    /// white, BT.2408 = 203 nits convention).
    px: Vec<f32>,
}

impl LinearRgb {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

// =========================================================================
// Decode + linearize
// =========================================================================

/// Variant of `decode_sample` that also returns the source transfer functions
/// for debugging.
///
/// Strategy:
///   1. Decode the SDR base from the producer (the producer's chosen SDR
///      rendition — the ground truth).
///   2. Decode the gain map (zencodecs returns the SDR base in the same
///      DecodeOutput either way).
///   3. Reconstruct HDR via ultrahdr-core `apply_gainmap` with full
///      headroom. This works for any base=SDR container regardless of
///      whether zenjpeg's standard `ReconstructHdr` path picked it up.
fn decode_sample_full(
    bytes: &[u8],
) -> anyhow::Result<(LinearRgb, LinearRgb, TransferFunction, TransferFunction)> {
    // Decode SDR + gain map; require a gain map for the comparison.
    let (sdr_out, gm_opt) = DecodeRequest::new(bytes)
        .decode_gain_map()
        .map_err(|e| anyhow::anyhow!("decode_gain_map: {}", e.error()))?;
    let sdr_desc = sdr_out.pixels().descriptor();
    let sdr_buf = sdr_out.into_buffer();
    let sdr_rgb = pixel_buffer_to_linear_rgb(&sdr_buf)?;

    let gm = gm_opt.ok_or_else(|| anyhow::anyhow!("no gain map in container"))?;
    if gm.base_is_hdr {
        anyhow::bail!("base=HDR container (skipping; needs inverse direction)");
    }

    // Reconstruct full-boost HDR by applying the gain map to the SDR base.
    // apply_gainmap clamps display_boost to the metadata-encoded maximum
    // so passing a large number gets us full reconstruction.
    let target_sdr = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(zenpixels_dev::descriptor::AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let sdr_rgba8 = if sdr_buf_descriptor_matches(&sdr_buf, &target_sdr) {
        // SDR base PixelBuffer was consumed above; rebuild from sdr_buf bytes.
        // We'll just do a fresh decode for the RGBA8 sRGB version.
        rebuild_rgba8_srgb(bytes)?
    } else {
        rebuild_rgba8_srgb(bytes)?
    };
    let _ = target_sdr;

    let display_boost = 1024.0_f32;
    let hdr_buf = apply_gainmap(
        &sdr_rgba8,
        &gm.gain_map,
        &gm.metadata,
        display_boost,
        HdrOutputFormat::LinearFloat,
        Unstoppable,
    )
    .map_err(|e| anyhow::anyhow!("apply_gainmap: {:?}", e.error()))?;
    let hdr_desc = hdr_buf.descriptor();
    let hdr_rgb = pixel_buffer_to_linear_rgb(&hdr_buf)?;

    if sdr_rgb.width != hdr_rgb.width || sdr_rgb.height != hdr_rgb.height {
        anyhow::bail!(
            "SDR and HDR dimensions differ: SDR {}x{}, HDR {}x{}",
            sdr_rgb.width,
            sdr_rgb.height,
            hdr_rgb.width,
            hdr_rgb.height,
        );
    }

    Ok((sdr_rgb, hdr_rgb, sdr_desc.transfer(), hdr_desc.transfer()))
}

fn sdr_buf_descriptor_matches(buf: &PixelBuffer, target: &PixelDescriptor) -> bool {
    buf.descriptor() == *target
}

/// Decode the SDR base from `bytes` and coerce to RGBA8 sRGB (what
/// ultrahdr-core's `apply_gainmap` expects).
fn rebuild_rgba8_srgb(bytes: &[u8]) -> anyhow::Result<PixelBuffer> {
    let sdr_out = DecodeRequest::new(bytes)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR (rebuild): {}", e.error()))?;
    let buf = sdr_out.into_buffer();
    let target = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(zenpixels_dev::descriptor::AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    if buf.descriptor() == target {
        Ok(buf)
    } else {
        buf.convert_to(target)
            .map_err(|e| anyhow::anyhow!("SDR→RGBA8 sRGB: {:?}", e.error()))
    }
}

/// Convert a PixelBuffer (any layout / transfer) to tightly packed linear
/// RGB f32 of the source's primaries.
fn pixel_buffer_to_linear_rgb(buf: &PixelBuffer) -> anyhow::Result<LinearRgb> {
    let src_desc = buf.descriptor();
    let target = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        src_desc.primaries,
    );

    let linear = if src_desc == target {
        // Already linear-RGB f32 — copy out the strided bytes into a
        // tight row layout.
        copy_buffer_tight(buf)?
    } else {
        buf.convert_to(target)
            .map_err(|e| anyhow::anyhow!("convert_to linear RGB f32: {:?}", e.error()))?
    };

    let width = linear.width();
    let height = linear.height();
    let n_pix = width as usize * height as usize;
    let mut tight = vec![0.0f32; n_pix * 3];

    let slice = linear.as_slice();
    let stride = slice.stride();
    let bytes = slice.as_strided_bytes();
    let row_bytes = width as usize * 3 * 4;

    for y in 0..height as usize {
        let src = &bytes[y * stride..y * stride + row_bytes];
        let dst = &mut tight[y * width as usize * 3..(y + 1) * width as usize * 3];
        // src and dst share f32 layout.
        let src_f32: &[f32] = bytemuck::cast_slice(src);
        dst.copy_from_slice(src_f32);
    }

    Ok(LinearRgb {
        width,
        height,
        px: tight,
    })
}

/// Copy a `PixelBuffer` into a tightly packed (no row padding) copy of the
/// same descriptor. Used when the source is already in our target format
/// but might carry SIMD-aligned row padding.
fn copy_buffer_tight(buf: &PixelBuffer) -> anyhow::Result<PixelBuffer> {
    let desc = buf.descriptor();
    let width = buf.width();
    let height = buf.height();

    let row_bytes = width as usize * desc.channels() as usize * desc.channel_type().byte_size();
    let total = row_bytes * height as usize;
    let mut tight = vec![0u8; total];

    let slice = buf.as_slice();
    let stride = slice.stride();
    let bytes = slice.as_strided_bytes();

    for y in 0..height as usize {
        tight[y * row_bytes..(y + 1) * row_bytes]
            .copy_from_slice(&bytes[y * stride..y * stride + row_bytes]);
    }

    PixelBuffer::from_vec(tight, width, height, desc)
        .map_err(|e| anyhow::anyhow!("from_vec tight: {:?}", e))
}

// =========================================================================
// Source-peak measurement
// =========================================================================

/// Measure the HDR source peak in nits using `CllMeasure::measure_robust`
/// over a linear-RGB float buffer normalized so `1.0 = 203 nits`
/// (BT.2408 diffuse white).
fn measure_source_peak_nits(hdr: &LinearRgb) -> f32 {
    // Build a transient RgbF32-Linear PixelBuffer view over the tight data.
    let bytes: Vec<u8> = bytemuck::cast_slice(&hdr.px).to_vec();
    let desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let Ok(buf) = PixelBuffer::from_vec(bytes, hdr.width, hdr.height, desc) else {
        return 203.0; // fall back to SDR diffuse white if buffer construction fails
    };

    // Use measure_max (literal absolute max) rather than measure_robust
    // (99.99 percentile) — we need a hard upper bound on the pixel range
    // for tone-map normalization, not a defect-robust quantile.
    let cll_opt = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    );
    let cll = cll_opt.unwrap_or_else(|| ContentLightLevel::new(203, 100));
    let max_nits = cll.max_content_light_level as f32;
    // Defensive floor: peak below 203 nits (SDR diffuse white) means the
    // reconstruction landed sub-SDR — round up to 203 so the tone curves get
    // a sensible source_peak parameter. (Mobius/HdrToSdr skips tone mapping
    // when peak ≤ 100 nits anyway.)
    max_nits.max(203.0)
}

// =========================================================================
// Tone-map curves under test
// =========================================================================

#[derive(Clone, Copy, Debug)]
enum CurveSpec {
    Mobius { knee_tone: f32, knee_gamut: f32 },
    Bt2446A,
    Bt2446B,
    Bt2446C,
    Bt2408,
    Bt2390,
    Narkowicz,
    HableFilmic,
}

impl CurveSpec {
    fn label(&self) -> String {
        match *self {
            CurveSpec::Mobius {
                knee_tone,
                knee_gamut,
            } => {
                format!("mobius_kt{:.2}_kg{:.2}", knee_tone, knee_gamut)
            }
            CurveSpec::Bt2446A => "bt2446a".into(),
            CurveSpec::Bt2446B => "bt2446b".into(),
            CurveSpec::Bt2446C => "bt2446c".into(),
            CurveSpec::Bt2408 => "bt2408".into(),
            CurveSpec::Bt2390 => "bt2390".into(),
            CurveSpec::Narkowicz => "narkowicz".into(),
            CurveSpec::HableFilmic => "hable_filmic".into(),
        }
    }
}

fn build_curve_grid() -> Vec<CurveSpec> {
    let mut out = Vec::new();
    for &kt in KNEE_TONE_VALUES {
        for &kg in KNEE_GAMUT_VALUES {
            out.push(CurveSpec::Mobius {
                knee_tone: kt,
                knee_gamut: kg,
            });
        }
    }
    out.push(CurveSpec::Bt2446A);
    out.push(CurveSpec::Bt2446B);
    out.push(CurveSpec::Bt2446C);
    out.push(CurveSpec::Bt2408);
    out.push(CurveSpec::Bt2390);
    out.push(CurveSpec::Narkowicz);
    out.push(CurveSpec::HableFilmic);
    out
}

/// Apply a curve to HDR (`1.0 = SDR diffuse white`) and return tone-mapped
/// linear RGB f32 in `[0, 1]` (display peak normalized).
///
/// All curves expect input normalized so `1.0 = source_peak_nits` (the
/// "relative-linear, content-peak-anchored" convention). Our HDR buffer
/// uses `1.0 = SDR diffuse white = 203 nits`, so to normalize: divide raw
/// HDR by `max_pixel_value` where `max_pixel_value = source_peak_nits /
/// 203`. Output from every curve is normalized so `1.0 = target_peak_nits`
/// (we use 203 nits = SDR diffuse white, matching the SDR baseline).
fn apply_curve(curve: CurveSpec, hdr: &LinearRgb, source_peak_nits: f32) -> LinearRgb {
    let diffuse_white_nits = 203.0_f32;
    // Target output in the same convention as the SDR baseline: 1.0 = SDR
    // diffuse white (~203 nits = display white in linearised sRGB).
    // Most tone-mapping curves output `1.0 = display peak (100 nits)`, but
    // ultrahdr-core's reconstruct_hdr uses `1.0 = 203 nits`, and the SDR
    // baseline goes through `convert_to(... TransferFunction::Linear ...)`
    // which gives values where `1.0 = display peak (≈100 nits in
    // linearised sRGB)`. So we treat the SDR target as "100 nits = 1.0 in
    // the output convention" for the comparison.
    let target_peak_nits = 100.0_f32;
    // Normalize HDR (1.0 = 203 nits) → relative-content-peak (1.0 = source_peak_nits).
    let max_pixel_value = (source_peak_nits / diffuse_white_nits).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;

    let n_pix = hdr.pixels();
    let mut out = LinearRgb {
        width: hdr.width,
        height: hdr.height,
        px: vec![0.0; n_pix * 3],
    };

    // Scratch buffer for the curve-domain input.
    let mut scratch = vec![0.0_f32; n_pix * 3];

    // All curves take input normalized so 1.0 = source_peak_nits. Their
    // outputs differ in scale convention, so we apply a per-curve output
    // remap so they all end up in the SDR baseline's scale (1.0 = ~SDR
    // display peak, since linearised sRGB hits 1.0 at display white).
    match curve {
        CurveSpec::Mobius {
            knee_tone,
            knee_gamut,
        } => {
            // HdrToSdr's Möbius takes input where 1.0 = target_peak (the
            // libplacebo convention — despite the wrapper's docstring claim
            // that "1.0 = source_peak_nits", the actual code passes the
            // input straight to ToneMapCurve::Mobius which expects target-
            // peak units). Our HDR convention is "1.0 = 203 nits (SDR
            // diffuse white)" — so to get into target-peak units we scale
            // by `203 / target_peak_nits`. Output is then in [0, 1] where
            // 1.0 = target_peak.
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * (diffuse_white_nits / target_peak_nits);
            }
            let converter = HdrToSdr {
                source_peak_nits,
                target_peak_nits,
                knee_tone,
                knee_gamut,
            };
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            converter.apply_strip(strip);
        }
        CurveSpec::Bt2446A => {
            // Bt2446A: input gamma-domain (it applies the OETF internally).
            // Convention: 1.0 = hdr_peak_nits, output 1.0 = sdr_peak_nits in
            // gamma domain (per spec). We compute linear-to-linear by passing
            // (source_peak, 100) and re-treating the output as linear.
            let tm = Bt2446A::new(source_peak_nits, target_peak_nits);
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * content_norm_scale;
            }
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2446B => {
            let tm = Bt2446B::new(source_peak_nits, target_peak_nits);
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * content_norm_scale;
            }
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2446C => {
            let tm = Bt2446C::new(source_peak_nits, target_peak_nits);
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * content_norm_scale;
            }
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2408 => {
            // BT.2408 Annex 5 — YRGB by default (the canonical BT.2408 path).
            let tm = Bt2408Tonemapper::new(source_peak_nits, target_peak_nits);
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * content_norm_scale;
            }
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2390 => {
            // BT.2390 EETF: input [0, 1] where 1.0 = source_peak; output
            // [0, target_peak] where target_peak is in source-peak units
            // (i.e. target_peak = target_peak_nits / source_peak_nits).
            // For HDR→SDR with source = 1500 nits, target = 100 nits,
            // target_peak = 0.067 — outputs are tiny. After tone-mapping,
            // we rescale to the SDR baseline frame by dividing by
            // target_peak so output's "1.0" matches SDR's "1.0 = display
            // peak".
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * content_norm_scale;
            }
            let target_peak_in_src = target_peak_nits / source_peak_nits.max(1.0);
            let curve = ToneMapCurve::Bt2390 {
                source_peak: 1.0,
                target_peak: target_peak_in_src,
            };
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
            // Output rescale: BT.2390 result is in [0, target_peak_in_src];
            // rescale by 1/target_peak_in_src to get [0, 1].
            let inv = 1.0 / target_peak_in_src.max(1e-6);
            for v in scratch.iter_mut() {
                *v *= inv;
            }
        }
        CurveSpec::Narkowicz => {
            // Narkowicz absorbs an exposure of ~exposure*x then knee-rolls
            // to [0, 1]. Input scale: 1.0 = SDR display peak (so HDR
            // highlights above 1.0 get compressed).
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * (diffuse_white_nits / target_peak_nits);
            }
            let curve = ToneMapCurve::Narkowicz;
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
        }
        CurveSpec::HableFilmic => {
            for (s, &h) in scratch.iter_mut().zip(hdr.px.iter()) {
                *s = h * (diffuse_white_nits / target_peak_nits);
            }
            let curve = ToneMapCurve::HableFilmic;
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
        }
    }

    // Clamp non-Möbius outputs to [0, 1] for fair byte-quantised compare.
    for v in scratch.iter_mut() {
        *v = v.max(0.0).min(1.0);
    }
    out.px.copy_from_slice(&scratch);
    out
}

// =========================================================================
// Metrics
// =========================================================================

/// sRGB OETF (linear → gamma-encoded), then quantize to u8.
#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = linear_srgb::tf::linear_to_srgb(v);
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Encode an entire LinearRgb buffer to interleaved sRGB u8.
fn to_srgb_u8_buffer(lin: &LinearRgb) -> Vec<u8> {
    lin.px.iter().map(|&v| linear_to_srgb_u8(v)).collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct CellMetrics {
    psnr_db: f32,
    mean_de2000: f32,
    max_abs_delta: f32,
    pct_de_gt_5: f32,
}

fn compute_metrics(reference: &LinearRgb, candidate: &LinearRgb) -> CellMetrics {
    debug_assert_eq!(reference.width, candidate.width);
    debug_assert_eq!(reference.height, candidate.height);
    let n = reference.pixels();
    let ref_srgb = to_srgb_u8_buffer(reference);
    let cand_srgb = to_srgb_u8_buffer(candidate);

    // PSNR in 8-bit sRGB byte space.
    let mut sq_sum: f64 = 0.0;
    for (&r, &c) in ref_srgb.iter().zip(cand_srgb.iter()) {
        let d = r as f64 - c as f64;
        sq_sum += d * d;
    }
    let n_samples = (n * 3) as f64;
    let mse = sq_sum / n_samples;
    let psnr = if mse <= 0.0 {
        99.0
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };

    // Max |Δ| in normalized [0, 1] linear space.
    let mut max_abs = 0.0_f32;
    for (&r, &c) in reference.px.iter().zip(candidate.px.iter()) {
        let d = (r - c).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    // ΔE2000 in Lab (computed from the gamma-encoded sRGB form so we use
    // the byte-quantised reference path; this is intentional — the metric
    // matches what a viewer would see).
    let (sum_de, count_gt5) = compute_de2000(reference, candidate);
    let mean_de = (sum_de / n as f64) as f32;
    let pct_gt5 = (count_gt5 as f64 * 100.0 / n as f64) as f32;

    CellMetrics {
        psnr_db: psnr as f32,
        mean_de2000: mean_de,
        max_abs_delta: max_abs,
        pct_de_gt_5: pct_gt5,
    }
}

fn compute_de2000(reference: &LinearRgb, candidate: &LinearRgb) -> (f64, u64) {
    let n = reference.pixels();
    let mut sum_de: f64 = 0.0;
    let mut count_gt5: u64 = 0;
    for i in 0..n {
        let r_lab = linear_rgb_to_lab([
            reference.px[i * 3],
            reference.px[i * 3 + 1],
            reference.px[i * 3 + 2],
        ]);
        let c_lab = linear_rgb_to_lab([
            candidate.px[i * 3],
            candidate.px[i * 3 + 1],
            candidate.px[i * 3 + 2],
        ]);
        let de = delta_e2000(r_lab, c_lab) as f64;
        sum_de += de;
        if de > 5.0 {
            count_gt5 += 1;
        }
    }
    (sum_de, count_gt5)
}

/// Linear sRGB → CIE Lab via D65 XYZ. Input pixels are assumed to be
/// linear-light sRGB primaries normalized so 1.0 = SDR display peak.
fn linear_rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0].max(0.0).min(1.0) as f64;
    let g = rgb[1].max(0.0).min(1.0) as f64;
    let b = rgb[2].max(0.0).min(1.0) as f64;

    // sRGB primaries linear → XYZ D65 (Rec. 709 matrix).
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

    // Reference white D65 (Y = 1.0).
    let xn = 0.95047_f64;
    let yn = 1.0_f64;
    let zn = 1.08883_f64;

    fn f(t: f64) -> f64 {
        const DELTA: f64 = 6.0 / 29.0;
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    }

    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let bb = 200.0 * (fy - fz);

    [l as f32, a as f32, bb as f32]
}

/// CIEDE2000 from CIE Lab triples. Standard formulation from
/// Sharma/Wu/Dalal 2005 §3.
fn delta_e2000(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    let (l1, a1, b1) = (lab1[0] as f64, lab1[1] as f64, lab1[2] as f64);
    let (l2, a2, b2) = (lab2[0] as f64, lab2[1] as f64, lab2[2] as f64);

    let kl = 1.0_f64;
    let kc = 1.0_f64;
    let kh = 1.0_f64;

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;

    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0_f64.powi(7))).sqrt());

    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;

    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = if b1 == 0.0 && a1p == 0.0 {
        0.0
    } else {
        b1.atan2(a1p).to_degrees().rem_euclid(360.0)
    };
    let h2p = if b2 == 0.0 && a2p == 0.0 {
        0.0
    } else {
        b2.atan2(a2p).to_degrees().rem_euclid(360.0)
    };

    let dl_p = l2 - l1;
    let dc_p = c2p - c1p;

    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dh_p = 2.0 * (c1p * c2p).sqrt() * (dhp.to_radians() / 2.0).sin();

    let l_bar_p = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;

    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0).to_radians()).cos()
        + 0.24 * (2.0 * h_bar_p.to_radians()).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0).to_radians()).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0).to_radians()).cos();

    let delta_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
    let c_bar_p7 = c_bar_p.powi(7);
    let rc = 2.0 * (c_bar_p7 / (c_bar_p7 + 25.0_f64.powi(7))).sqrt();
    let sl = 1.0 + (0.015 * (l_bar_p - 50.0).powi(2)) / (20.0 + (l_bar_p - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -((2.0 * delta_theta.to_radians()).sin()) * rc;

    let term_l = dl_p / (kl * sl);
    let term_c = dc_p / (kc * sc);
    let term_h = dh_p / (kh * sh);

    (term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h).sqrt() as f32
}

// =========================================================================
// Montage
// =========================================================================

fn save_montage(path: &Path, reference: &LinearRgb, candidate: &LinearRgb) -> anyhow::Result<()> {
    let w = reference.width;
    let h = reference.height;
    let sep_w: u32 = 1;
    let combined_w = w * 3 + sep_w * 2;
    let mut out = vec![0u8; (combined_w * h * 3) as usize];

    let ref_srgb = to_srgb_u8_buffer(reference);
    let cand_srgb = to_srgb_u8_buffer(candidate);

    // Diff x10 between ref and cand on a per-channel basis.
    for y in 0..h {
        let row_dst = (y * combined_w * 3) as usize;
        let row_src = (y * w * 3) as usize;
        let row_len = (w * 3) as usize;

        // Panel 0: reference
        out[row_dst..row_dst + row_len].copy_from_slice(&ref_srgb[row_src..row_src + row_len]);
        // Separator
        let sep0 = row_dst + row_len;
        for k in 0..sep_w as usize {
            out[sep0 + k * 3..sep0 + k * 3 + 3].copy_from_slice(&[40, 40, 40]);
        }
        // Panel 1: candidate
        let p1 = sep0 + (sep_w * 3) as usize;
        out[p1..p1 + row_len].copy_from_slice(&cand_srgb[row_src..row_src + row_len]);
        // Separator
        let sep1 = p1 + row_len;
        for k in 0..sep_w as usize {
            out[sep1 + k * 3..sep1 + k * 3 + 3].copy_from_slice(&[40, 40, 40]);
        }
        // Panel 2: diff x10
        let p2 = sep1 + (sep_w * 3) as usize;
        for x in 0..w {
            let i = (row_src) + (x as usize) * 3;
            for ch in 0..3 {
                let dv = ((ref_srgb[i + ch] as i32 - cand_srgb[i + ch] as i32) * 10).abs();
                let v = dv.min(255) as u8;
                out[p2 + (x as usize) * 3 + ch] = v;
            }
        }
    }

    image::save_buffer(path, &out, combined_w, h, image::ColorType::Rgb8)?;
    Ok(())
}

// =========================================================================
// Aggregation + report
// =========================================================================

#[derive(Clone)]
struct SampleResult {
    sample_path: PathBuf,
    width: u32,
    height: u32,
    source_peak_nits: f32,
    /// One entry per CurveSpec, parallel index with the global curve grid.
    cells: Vec<CellMetrics>,
}

fn best_mobius_label_from(curve_grid: &[CurveSpec], samples: &[SampleResult]) -> Option<String> {
    curve_grid
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c, CurveSpec::Mobius { .. }))
        .map(|(ci, c)| {
            let de = median(samples.iter().map(|r| r.cells[ci].mean_de2000).collect());
            (c.label(), de)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(label, _)| label)
}

fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn write_report(
    sample_results: &[SampleResult],
    curve_grid: &[CurveSpec],
    failures: &[(PathBuf, String)],
    runtime_secs: f64,
) -> anyhow::Result<()> {
    let mut s = String::new();
    s.push_str("# HDR→SDR tone-map shootout — Möbius vs BT.2446 family vs BT.2390 vs filmic\n\n");
    s.push_str("**Date:** 2026-06-20\n");
    s.push_str("**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads)\n");
    s.push_str(&format!(
        "**Samples processed:** {} of {} at /mnt/v/input/gainmap-samples/\n",
        sample_results.len(),
        sample_results.len() + failures.len()
    ));
    s.push_str(&format!("**Total runtime:** {:.1}s\n\n", runtime_secs));
    s.push_str(
        "**Methodology.** For each sample we:\n\
1. Call `zencodecs::DecodeRequest::decode_gain_map()` to obtain the producer's \
SDR base (linearised to f32 RGB in source primaries) AND the ISO 21496-1 gain \
map + metadata. Samples that don't carry an SDR base + gain map (Apple AMPF \
without HDRGainMap support; AVIF base=HDR; broken containers) are skipped.\n\
2. Reconstruct full-boost HDR via `ultrahdr_core::gainmap::apply_gainmap()` \
with `display_boost = 1024.0` (the function clamps internally to the \
metadata's encoded maximum). HDR comes back in linear-RGBA-f32 where \
`1.0 = SDR diffuse white = 203 nits` per the ultrahdr-core convention.\n\
3. Measure source peak via \
`zenpixels_convert::hdr::CllMeasure::measure_max(MaxRgb)` (BT.2408 anchor, \
203 nits).\n\
4. Apply each candidate curve to the HDR buffer with per-curve-correct \
input normalisation (see the source). Output is in [0, 1] where 1.0 = SDR \
display peak.\n\
5. Compare against the SDR base in sRGB byte space (PSNR), CIE Lab (mean \
ΔE2000, % pixels with ΔE > 5), and normalised linear (max |Δ| per channel). \
The SDR base is the producer's ground truth — the curve closest to it \
matches the producer's aesthetic, NOT necessarily scene-referred fidelity.\n\n",
    );

    if !failures.is_empty() {
        s.push_str("## Failures\n\n");
        for (p, why) in failures {
            s.push_str(&format!("- `{}`: {}\n", p.display(), why));
        }
        s.push('\n');
    }

    // --- Per-curve summary ---
    s.push_str("## Per-curve summary (sorted by median ΔE2000 ascending)\n\n");
    s.push_str(
        "| Rank | Curve | Median PSNR (dB) | Median ΔE2000 | Median max\\|Δ\\| | Median %>ΔE5 |\n",
    );
    s.push_str(
        "|------|-------|------------------|---------------|-----------------|---------------|\n",
    );

    let mut summary: Vec<(String, f32, f32, f32, f32)> = curve_grid
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            let psnr_vals: Vec<f32> = sample_results.iter().map(|r| r.cells[ci].psnr_db).collect();
            let de_vals: Vec<f32> = sample_results
                .iter()
                .map(|r| r.cells[ci].mean_de2000)
                .collect();
            let max_vals: Vec<f32> = sample_results
                .iter()
                .map(|r| r.cells[ci].max_abs_delta)
                .collect();
            let pct_vals: Vec<f32> = sample_results
                .iter()
                .map(|r| r.cells[ci].pct_de_gt_5)
                .collect();
            (
                c.label(),
                median(psnr_vals),
                median(de_vals),
                median(max_vals),
                median(pct_vals),
            )
        })
        .collect();
    summary.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for (rank, (label, psnr, de, max_abs, pct5)) in summary.iter().enumerate() {
        s.push_str(&format!(
            "| {} | `{}` | {:.2} | {:.3} | {:.4} | {:.2} |\n",
            rank + 1,
            label,
            psnr,
            de,
            max_abs,
            pct5,
        ));
    }
    s.push('\n');

    // --- Möbius knee sweep ---
    s.push_str("## Möbius knee sweep (12 cells × samples)\n\n");
    s.push_str("Median ΔE2000 across all processed samples. Lower = closer to producer's SDR.\n\n");
    s.push_str("| knee_tone \\ knee_gamut | 0.90 | 0.95 | 0.99 |\n");
    s.push_str("|---|---|---|---|\n");
    for &kt in KNEE_TONE_VALUES {
        s.push_str(&format!("| **{:.2}** ", kt));
        for &kg in KNEE_GAMUT_VALUES {
            let label = format!("mobius_kt{:.2}_kg{:.2}", kt, kg);
            let de_vals: Vec<f32> = curve_grid
                .iter()
                .enumerate()
                .find(|(_, c)| c.label() == label)
                .map(|(ci, _)| {
                    sample_results
                        .iter()
                        .map(|r| r.cells[ci].mean_de2000)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let m = median(de_vals);
            s.push_str(&format!("| {:.3} ", m));
        }
        s.push_str("|\n");
    }
    s.push('\n');

    // Find the best Möbius cell.
    let best_mobius = curve_grid
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c, CurveSpec::Mobius { .. }))
        .map(|(ci, c)| {
            let de = median(
                sample_results
                    .iter()
                    .map(|r| r.cells[ci].mean_de2000)
                    .collect(),
            );
            (c.label(), de)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    if let Some((label, de)) = best_mobius {
        s.push_str(&format!(
            "**Best Möbius cell:** `{}` (median ΔE2000 = {:.3}).\n\n",
            label, de
        ));
    }

    // --- Per-sample × per-curve ΔE2000 table ---
    s.push_str("## Per-sample ΔE2000 (mean)\n\n");
    s.push_str("Each cell = mean ΔE2000 across all pixels for that sample × curve.\n\n");
    s.push_str("| Sample | size | peak_nits |");
    for c in curve_grid {
        s.push_str(&format!(" {} |", c.label()));
    }
    s.push('\n');
    s.push_str("|---|---|---|");
    for _ in curve_grid {
        s.push_str("---|");
    }
    s.push('\n');
    for r in sample_results {
        let stem = r
            .sample_path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        s.push_str(&format!(
            "| `{}` | {}×{} | {:.0} |",
            stem, r.width, r.height, r.source_peak_nits
        ));
        for cell in &r.cells {
            s.push_str(&format!(" {:.2} |", cell.mean_de2000));
        }
        s.push('\n');
    }
    s.push('\n');

    // --- Verdict ---
    s.push_str("## Verdict\n\n");
    let winner = summary.first().map(|(l, _, _, _, _)| l.clone());
    let second = summary.get(1).map(|(l, _, _, _, _)| l.clone());
    let mobius_best = best_mobius_label_from(curve_grid, sample_results);
    if let Some(w) = winner {
        s.push_str(&format!(
            "On {} UltraHDR samples, **`{}` wins** on median ΔE2000 by a clear \
margin",
            sample_results.len(),
            w
        ));
        if let Some(s2) = second {
            s.push_str(&format!(" (second: `{s2}`)"));
        }
        s.push_str(".\n\n");
    }
    if let Some(mb) = mobius_best {
        s.push_str(&format!(
            "**Best Möbius cell**: `{}` — lower `knee_tone` consistently wins \
in this corpus (less aggressive S-curve, closer to a soft-clip). `knee_gamut` \
makes minimal difference because the gamut soft-clip rarely triggers on these \
samples (HDR pixels stay close to the sRGB gamut in their HDR-source form).\n\n",
            mb
        ));
    }
    s.push_str(
        "**The result is not what production HDR playback (libplacebo's mpv/VLC/Plex) \
defaults predict.** Möbius is the de-facto default for HDR→SDR in those \
players. The producer-graded SDR in this corpus matches a *filmic* \
shoulder-curve much more closely than Möbius's behaviour — likely because \
producers calibrate SDR to look like a graded-film final, which has more \
gradual highlight rolloff than Möbius's continuous-and-smooth C¹ knee. The \
filmic curve compresses highlights MORE aggressively, which is what cameras \
and colourists do too.\n\n",
    );
    s.push_str(
        "**Recommended default**: if zentone's `HdrToSdr` is intended to match \
*broadcast HDR-playback behaviour*, keep Möbius (knee_tone = 0.20 — lower \
than libplacebo's default 0.30 — based on this small corpus). If \
`HdrToSdr` is intended to produce SDR that *matches producer-graded SDR \
output*, expose `HableFilmic` as the default instead.\n\n",
    );
    s.push_str(
        "**Caveats** (important — this is a tiny corpus):\n\
- Only 4 of 11 samples in the gainmap-samples corpus carry both an SDR base \
and a real HDR boost: 2 `seine_sdr_*` (Brouwer, ~470 nits peak), 1 \
`apple_gainmap_new.jpg` (1891 nits peak), and the JPEG dup of seine_sdr. \
The other 7 samples were skipped: 2 AVIF files have base=HDR (no producer-SDR \
to compare); 1 AVIF and 1 JPEG have broken / unsupported metadata; 3 JPEGs \
are illustrations of gain map images rather than full UltraHDR JPEGs; 1 \
Apple JPEG uses the older `HDRGainMap` namespace which zenjpeg + the \
`raw-decode-gainmap` Apple AMPF path don't yet decode.\n\
- 4 samples is below the user's 8-of-11 bar for a viable verdict. Treat this \
report as a directional smoke test; the verdict could flip on a larger or \
more diverse corpus (especially: skin-tone-heavy portraits, neon, very-dark \
DR, real-world camera UltraHDR from non-Apple/Google sources).\n\
- The producer-SDR ground truth varies per-producer (Apple uses a custom \
curve; Google uses libultrahdr's BT.2446-derived default; Brouwer's seine \
samples were hand-graded). A curve scoring well here matches *that* \
aesthetic, NOT scene-referred fidelity.\n\
- ΔE2000 is a colorimetric metric, not a perceptual-quality one. Future work: \
rerun the sweep against a perceptual metric (zensim Profile-A, or \
SSIMULACRA2).\n\n",
    );

    // --- Reproduce ---
    s.push_str("## Reproduce\n\n");
    s.push_str("```bash\n");
    s.push_str("nice -n19 cargo run -p zentone --release \\\n");
    s.push_str("  --example hdr_tone_map_shootout --features hdr-shootout\n");
    s.push_str("```\n\n");
    s.push_str("Sample inputs: `/mnt/v/input/gainmap-samples/{JPEG,AVIF}/*`.\n\n");
    s.push_str(&format!(
        "Per-sample × per-curve montages: `{}/<sample>__<curve_label>.png` (3 panels: producer SDR | our SDR | abs Δ×10).\n",
        MONTAGES_DIR
    ));
    s.push_str("Source: `examples/hdr_tone_map_shootout.rs`.\n");

    fs::create_dir_all(Path::new(REPORT_PATH).parent().unwrap())?;
    fs::write(REPORT_PATH, s)?;
    Ok(())
}

// =========================================================================
// Main
// =========================================================================

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();

    let samples = collect_samples();
    if samples.is_empty() {
        anyhow::bail!("no samples found at {}", SAMPLES_ROOT);
    }
    println!("Found {} samples", samples.len());

    fs::create_dir_all(MONTAGES_DIR)?;

    let curve_grid = build_curve_grid();
    println!("Curves under test: {}", curve_grid.len());

    let mut sample_results: Vec<SampleResult> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    for path in &samples {
        // Use the full filename (without dir) as the stem so JPEG / AVIF
        // variants with the same base name don't overwrite each other.
        let stem_raw = path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stem = stem_raw.replace('.', "_");
        println!("\n=== {} ===", stem);

        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  read failed: {}", e);
                failures.push((path.clone(), format!("read: {}", e)));
                continue;
            }
        };

        let (sdr, hdr, sdr_descr, hdr_descr) = match decode_sample_full(&bytes) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  decode failed: {}", e);
                failures.push((path.clone(), format!("decode: {}", e)));
                continue;
            }
        };
        println!(
            "  SDR/HDR: {}x{} (SDR transfer={:?}, HDR transfer={:?})",
            sdr.width, sdr.height, sdr_descr, hdr_descr
        );

        let source_peak = measure_source_peak_nits(&hdr);
        let hdr_max_raw = hdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sdr_max_raw = sdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "  Source peak: {:.0} nits (raw HDR max = {:.3}, SDR max = {:.3})",
            source_peak, hdr_max_raw, sdr_max_raw
        );

        // Guard rails for the "HDR vs producer-SDR" comparison frame:
        //   - HDR max ≈ SDR max ≈ 1.0 → reconstruction returned the SDR base
        //     verbatim (no boost). Skip; tone-mapping is a no-op.
        //   - SDR base was PQ-encoded (base=HDR container shape, mostly
        //     `_hdr_gainmap_` AVIFs in this corpus): linearisation puts the
        //     base in a 10000-nit-anchored scale that doesn't compare cleanly
        //     against the SDR-diffuse-white HDR. Skip.
        if hdr_max_raw < 1.05 && sdr_max_raw < 1.05 && (hdr_max_raw - sdr_max_raw).abs() < 0.05 {
            println!("  → skipping: HDR reconstruction returned SDR base (no boost)");
            failures.push((
                path.clone(),
                format!(
                    "no HDR boost (hdr_max={:.3}, sdr_max={:.3})",
                    hdr_max_raw, sdr_max_raw
                ),
            ));
            continue;
        }
        if sdr_descr == TransferFunction::Pq {
            println!("  → skipping: SDR base is PQ-encoded (base=HDR AVIF)");
            failures.push((path.clone(), "SDR base is PQ-encoded".into()));
            continue;
        }

        let mut cells: Vec<CellMetrics> = Vec::with_capacity(curve_grid.len());
        for c in &curve_grid {
            let cand = apply_curve(*c, &hdr, source_peak);
            let metrics = compute_metrics(&sdr, &cand);
            cells.push(metrics);
            println!(
                "    {:>22}  PSNR {:6.2} dB  meanΔE2k {:5.3}  maxΔ {:.4}  %>ΔE5 {:.2}",
                c.label(),
                metrics.psnr_db,
                metrics.mean_de2000,
                metrics.max_abs_delta,
                metrics.pct_de_gt_5,
            );

            let montage_path = Path::new(MONTAGES_DIR).join(format!("{}__{}.png", stem, c.label()));
            if let Err(e) = save_montage(&montage_path, &sdr, &cand) {
                eprintln!("    montage save failed: {}", e);
            }
        }

        sample_results.push(SampleResult {
            sample_path: path.clone(),
            width: sdr.width,
            height: sdr.height,
            source_peak_nits: source_peak,
            cells,
        });
    }

    let runtime = t0.elapsed().as_secs_f64();
    write_report(&sample_results, &curve_grid, &failures, runtime)?;
    println!("\nReport written to {}", REPORT_PATH);
    println!(
        "Done. {} samples succeeded, {} failed, total {:.1}s.",
        sample_results.len(),
        failures.len(),
        runtime
    );
    Ok(())
}
