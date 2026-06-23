//! HDR → SDR tone-map cross-curve shootout (audited color handling, 2026-06-22).
//!
//! Successor to `hdr_tone_map_shootout_full.rs`. Same corpus & 20-curve grid,
//! but with the color-handling bugs found by `shootout_color_audit_2026-06-22.md`
//! fixed:
//!
//! 1. HDR is converted to **linear BT.2020** before any tone-map curve is
//!    applied (Bt2446A/B/C, BT.2408, BT.2390 all specify BT.2020 input).
//! 2. Candidate output is rotated to match producer SDR primaries before
//!    scoring (BT.709 for iPhone HEIC, DisplayP3 for zfold7 UltraHDR JPEG)
//!    via `convert_to`, so ΔE2000-vs-producer compares pixels in the
//!    producer's chosen gamut.
//! 3. Adds 4th peak method: `measure_percentile @ 0.99999`.
//! 4. CSV gains `color_handling_version` column ("2026-06-22-audited") so
//!    rows merge with the 2026-06-20 baseline.
//! 5. **2026-06-22-v2**: adds tail-aware metrics (ΔE2000 percentiles
//!    p50/p90/p95/p99 + OKLab Euclidean ΔE mean/percentiles + threshold
//!    fraction). OKLab is computed on linear sRGB (BT.709) D65, rotating
//!    from the producer's primaries inside the OKLab subroutine. CSV
//!    `metrics_version="2026-06-22-percentiles-oklab"` distinguishes this
//!    schema; output filename suffixed with `_v2`.
//!
//! Runs 76 samples × 4 peak methods × 20 curves = 6080 cells.
//!
//! Run:
//! ```text
//! nice -n19 cargo run --release \
//!   --example hdr_tone_map_shootout_audited --features hdr-shootout
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use zencodecs::{DecodeRequest, OrientationHint};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_dev::hdr::ContentLightLevel;
use zenpixels_dev::ColorPrimaries;

use zenpixels_convert::hdr::Bt2446A;
use zentone::gamut::soft_clip_knee_strip;
use zentone::{Bt2408Tonemapper, Bt2446B, Bt2446C, ToneMap, ToneMapCurve};

// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const CSV_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-22_v2.csv";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";
const COLOR_HANDLING_VERSION: &str = "2026-06-22-audited";
const METRICS_VERSION: &str = "2026-06-22-percentiles-oklab";

// ΔE2000 histogram: 500 bins of width 0.1 covering [0, 50). Pixels with
// ΔE2000 ≥ 50 are capped to the top bin (rare on tone-mapping; they exist on
// pathological clip/clamp cells but the tail percentiles only need to *see*
// them, not measure them precisely).
const DE2K_HIST_BINS: usize = 500;
const DE2K_HIST_BIN_WIDTH: f64 = 0.1;
const DE2K_DE5_BIN: usize = 50; // ceil(5.0 / 0.1)

// ΔE_OKLab histogram: 500 bins of width 0.001 covering [0, 0.5). OKLab JND
// is around 0.04; 0.1 is "clearly different". Cap above 0.5 (saturated-color
// clip on the worst Möbius-knee combos can exceed 0.3).
const DE_OK_HIST_BINS: usize = 500;
const DE_OK_HIST_BIN_WIDTH: f64 = 0.001;
#[allow(dead_code)]
const DE_OK_THRESHOLD: f64 = 0.04;
const DE_OK_THRESHOLD_BIN: usize = 40; // 0.04 / 0.001
/// New peak method: high percentile (1-in-100,000 pixels).
const PERCENTILE_99999: f32 = 0.99999;

// Möbius knee sweep grid.
const KNEE_TONE_VALUES: &[f32] = &[0.20, 0.30, 0.40, 0.50];
const KNEE_GAMUT_VALUES: &[f32] = &[0.90, 0.95, 0.99];

// Peak-measurement methods. Adds 4th: `measure_percentile @ 0.99999`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PeakMethod {
    Max,
    Robust,
    MaxSmoothed,
    Percentile99999,
}

impl PeakMethod {
    fn label(&self) -> &'static str {
        match self {
            PeakMethod::Max => "measure_max",
            PeakMethod::Robust => "measure_robust",
            PeakMethod::MaxSmoothed => "measure_max_smoothed",
            PeakMethod::Percentile99999 => "measure_percentile_99999",
        }
    }
    fn all() -> &'static [PeakMethod] {
        &[
            PeakMethod::Max,
            PeakMethod::Robust,
            PeakMethod::MaxSmoothed,
            PeakMethod::Percentile99999,
        ]
    }
}

// =========================================================================
// Lock-file refresh
// =========================================================================

fn refresh_lock(activity: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Use shell `date` instead of chrono — we only need ISO-8601.
    let iso = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| format!("ts={}", ts));
    let line = format!("{} claude-shootout-76 {}\n", iso, activity);
    let _ = fs::write(WORKONGOING, line);
}

// =========================================================================
// Sample-finding
// =========================================================================

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk_files(&p, out);
            } else if p.is_file() {
                let ext = p
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "heic") {
                    out.push(p);
                }
            }
        }
    }
}

fn collect_samples() -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_files(Path::new(SAMPLES_ROOT), &mut out);
    out.sort();
    out
}

/// Probe a file for gain-map presence. Returns Some(format_label) if a
/// gain map appears to be present (per probe metadata), None otherwise.
fn detect_gainmap(bytes: &[u8]) -> Option<&'static str> {
    let info = DecodeRequest::new(bytes).probe().ok()?;
    // The reliable check: GainMapPresence::Available OR supplements.gain_map.
    // is_present() returns true only for Available(_), which AVIF/JPEG with
    // parsed metadata report; HEIC frequently reports Unknown at probe time
    // but the codec-side supplements flag may be populated downstream.
    let has = info.gain_map.is_present() || info.supplements.gain_map;
    if has {
        Some(match info.format {
            zencodecs::ImageFormat::Heic => "HEIC",
            zencodecs::ImageFormat::Jpeg => "JPEG",
            zencodecs::ImageFormat::Avif => "AVIF",
            _ => "OTHER",
        })
    } else {
        None
    }
}

// =========================================================================
// Linear-RGB f32 buffer helper
// =========================================================================

#[derive(Clone)]
struct LinearRgb {
    width: u32,
    height: u32,
    /// Tightly packed (no stride) linear-light RGB f32 in `primaries`.
    px: Vec<f32>,
    primaries: ColorPrimaries,
}

impl LinearRgb {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

// =========================================================================
// Decode + linearize  (AUDIT FIX: HDR rotated to BT.2020 before tone-map)
// =========================================================================

/// Returns `(sdr, hdr_bt2020, sdr_transfer, hdr_transfer, hdr_orig_primaries)`.
///
/// SDR is kept in **source primaries** (BT.709 for iPhone HEIC, DisplayP3 for
/// zfold7 UltraHDR JPEG); HDR is converted to **linear BT.2020** because
/// Bt2446A/B/C/Bt2408/Bt2390 specify BT.2020 input.
fn decode_sample_full(
    bytes: &[u8],
) -> anyhow::Result<(
    LinearRgb,
    LinearRgb,
    TransferFunction,
    TransferFunction,
    ColorPrimaries,
)> {
    let sdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR: {}", e.error()))?;
    let sdr_desc = sdr_out.pixels().descriptor();
    let sdr_buf = sdr_out.into_buffer();
    let sdr_rgb = pixel_buffer_to_linear_rgb_preserve_primaries(&sdr_buf)?;

    let hdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .reconstruct_hdr(None)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("reconstruct_hdr: {}", e.error()))?;
    let hdr_desc = hdr_out.pixels().descriptor();
    let hdr_orig_primaries = hdr_desc.primaries;
    let hdr_buf = hdr_out.into_buffer();
    let hdr_rgb = pixel_buffer_to_linear_rgb_in_primaries(&hdr_buf, ColorPrimaries::Bt2020)?;

    if sdr_rgb.width != hdr_rgb.width || sdr_rgb.height != hdr_rgb.height {
        anyhow::bail!(
            "SDR and HDR dimensions differ: SDR {}x{}, HDR {}x{}",
            sdr_rgb.width,
            sdr_rgb.height,
            hdr_rgb.width,
            hdr_rgb.height,
        );
    }

    Ok((
        sdr_rgb,
        hdr_rgb,
        sdr_desc.transfer(),
        hdr_desc.transfer(),
        hdr_orig_primaries,
    ))
}

/// Convert a PixelBuffer to tightly packed linear RGB f32 of the source's primaries.
fn pixel_buffer_to_linear_rgb_preserve_primaries(
    buf: &PixelBuffer,
) -> anyhow::Result<LinearRgb> {
    let src_desc = buf.descriptor();
    pixel_buffer_to_linear_rgb_in_primaries(buf, src_desc.primaries)
}

/// Convert a PixelBuffer to tightly packed linear RGB f32 in `target_primaries`.
/// `convert_to` rotates the gamut via the in-tree `GamutMatrix` if needed.
/// If the source primaries are `Unknown`, falls back to assuming the data is
/// already in `target_primaries` (per the brief's fallback rule).
fn pixel_buffer_to_linear_rgb_in_primaries(
    buf: &PixelBuffer,
    target_primaries: ColorPrimaries,
) -> anyhow::Result<LinearRgb> {
    let src_desc = buf.descriptor();
    let target = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        target_primaries,
    );

    let linear = if src_desc == target {
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
        let src_f32: &[f32] = bytemuck::cast_slice(src);
        dst.copy_from_slice(src_f32);
    }

    Ok(LinearRgb {
        width,
        height,
        px: tight,
        primaries: target_primaries,
    })
}

/// Rotate a `LinearRgb` buffer to a different primaries set, via
/// `PixelBufferConvertExt::convert_to`. Used after a tone curve runs so the
/// (claimed-BT.2020) candidate is rotated to match producer-SDR primaries
/// before scoring (BT.709 or DisplayP3).
fn linear_rgb_change_primaries(
    rgb: &LinearRgb,
    target_primaries: ColorPrimaries,
) -> anyhow::Result<LinearRgb> {
    if rgb.primaries == target_primaries {
        return Ok(rgb.clone());
    }
    let bytes: Vec<u8> = bytemuck::cast_slice(&rgb.px).to_vec();
    let src_desc = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        rgb.primaries,
    );
    let buf = PixelBuffer::from_vec(bytes, rgb.width, rgb.height, src_desc)
        .map_err(|e| anyhow::anyhow!("from_vec src for primary rotation: {:?}", e))?;
    pixel_buffer_to_linear_rgb_in_primaries(&buf, target_primaries)
}

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
// Source-peak measurement (3 methods)
// =========================================================================

fn measure_source_peak_nits(hdr: &LinearRgb, method: PeakMethod) -> f32 {
    let bytes: Vec<u8> = bytemuck::cast_slice(&hdr.px).to_vec();
    let desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let Ok(buf) = PixelBuffer::from_vec(bytes, hdr.width, hdr.height, desc) else {
        return 203.0;
    };
    let cll_opt = match method {
        PeakMethod::Max => ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        ),
        PeakMethod::Robust => ContentLightLevel::measure_robust(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        ),
        PeakMethod::MaxSmoothed => ContentLightLevel::measure_max_smoothed(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        ),
        PeakMethod::Percentile99999 => ContentLightLevel::measure_percentile(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            PERCENTILE_99999,
            LightLevelMethod::MaxRgb,
        )
        .or_else(|| {
            // Fall back to measure_max on degenerate/single-color images.
            ContentLightLevel::measure_max(
                buf.as_slice(),
                DiffuseWhite::BT2408,
                LightLevelMethod::MaxRgb,
            )
        }),
    };
    let cll = cll_opt.unwrap_or_else(|| ContentLightLevel::new(203, 100));
    let max_nits = cll.max_content_light_level as f32;
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
    AcesAp1,
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
            CurveSpec::AcesAp1 => "aces_ap1".into(),
        }
    }

    fn knee_tone(&self) -> Option<f32> {
        match self {
            CurveSpec::Mobius { knee_tone, .. } => Some(*knee_tone),
            _ => None,
        }
    }
    fn knee_gamut(&self) -> Option<f32> {
        match self {
            CurveSpec::Mobius { knee_gamut, .. } => Some(*knee_gamut),
            _ => None,
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
    out.push(CurveSpec::AcesAp1);
    out
}

/// Apply a curve. HDR is anchored 1.0 = 203 nits (SDR diffuse white per
/// ultrahdr-core / heic). Output linear RGB f32 in [0, 1] (target-peak
/// normalized for comparison against the SDR baseline).
fn apply_curve(curve: CurveSpec, hdr: &LinearRgb, source_peak_nits: f32) -> LinearRgb {
    let diffuse_white_nits = 203.0_f32;
    let target_peak_nits = 100.0_f32;
    let max_pixel_value = (source_peak_nits / diffuse_white_nits).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;

    let n_pix = hdr.pixels();
    let mut out = LinearRgb {
        width: hdr.width,
        height: hdr.height,
        px: vec![0.0; n_pix * 3],
        // Output primaries: Bt2446A/B/C, BT.2408, BT.2390 emit BT.2020 per
        // their specs. Mobius/Narkowicz/HableFilmic/AcesAp1 are per-channel
        // ops → output primaries == input primaries (BT.2020 here, since the
        // HDR buffer was rotated to BT.2020 by decode_sample_full). Either
        // way, output is BT.2020.
        primaries: ColorPrimaries::Bt2020,
    };
    let mut scratch = vec![0.0_f32; n_pix * 3];

    match curve {
        CurveSpec::Mobius {
            knee_tone,
            knee_gamut,
        } => {
            // After the Möbius → Bt2446A default swap, HdrToSdr no longer
            // wraps Möbius. The historical Möbius cells in this shootout
            // construct ToneMapCurve::Mobius directly so the grid stays
            // comparable to past runs.
            //
            // ToneMapCurve::Mobius takes target-normalized input in
            // [0, peak] where peak = source / target. Our HDR buffer is
            // anchored 1.0 = 203 nits, so to enter that domain we scale
            // by (203 / target_peak_nits) instead of content_norm_scale.
            let peak = (source_peak_nits / target_peak_nits).max(1.0);
            let s_scale = 203.0_f32 / target_peak_nits;
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * s_scale;
                    }
                });
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            ToneMapCurve::Mobius {
                source_peak: peak,
                knee: knee_tone,
            }
            .map_strip_simd(strip);
            soft_clip_knee_strip(strip, knee_gamut);
            // Suppress unused-variable warning for content_norm_scale in this
            // branch; the other curve branches use it directly.
            let _ = content_norm_scale;
            // Output is target-norm: 1.0 = target_peak_nits = 100 nits.
        }
        CurveSpec::Bt2446A => {
            let tm = Bt2446A::new(source_peak_nits, target_peak_nits);
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * content_norm_scale;
                    }
                });
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2446B => {
            let tm = Bt2446B::new(source_peak_nits, target_peak_nits);
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * content_norm_scale;
                    }
                });
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2446C => {
            let tm = Bt2446C::new(source_peak_nits, target_peak_nits);
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * content_norm_scale;
                    }
                });
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2408 => {
            let tm = Bt2408Tonemapper::new(source_peak_nits, target_peak_nits);
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * content_norm_scale;
                    }
                });
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        CurveSpec::Bt2390 => {
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * content_norm_scale;
                    }
                });
            let target_peak_in_src = target_peak_nits / source_peak_nits.max(1.0);
            let curve = ToneMapCurve::Bt2390 {
                source_peak: 1.0,
                target_peak: target_peak_in_src,
            };
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
            let inv = 1.0 / target_peak_in_src.max(1e-6);
            scratch.par_chunks_mut(8192).for_each(|c| {
                for v in c.iter_mut() {
                    *v *= inv;
                }
            });
        }
        CurveSpec::Narkowicz => {
            let scale_dw = diffuse_white_nits / target_peak_nits;
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * scale_dw;
                    }
                });
            let curve = ToneMapCurve::Narkowicz;
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
        }
        CurveSpec::HableFilmic => {
            let scale_dw = diffuse_white_nits / target_peak_nits;
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * scale_dw;
                    }
                });
            let curve = ToneMapCurve::HableFilmic;
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
        }
        CurveSpec::AcesAp1 => {
            // ACES AP1 expects input normalized so 1.0 = scene-referred 18%
            // grey × ~6 stops headroom — in practice for HDR/SDR rendering
            // we feed it 1.0 = source-display-peak / target-display-peak
            // (mirrors the Narkowicz/HableFilmic input frame).
            let scale_dw = diffuse_white_nits / target_peak_nits;
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * scale_dw;
                    }
                });
            let curve = ToneMapCurve::AcesAp1;
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            curve.map_strip_simd(strip);
        }
    }

    scratch.par_chunks_mut(8192).for_each(|c| {
        for v in c.iter_mut() {
            *v = v.max(0.0).min(1.0);
        }
    });
    out.px.copy_from_slice(&scratch);
    out
}

// =========================================================================
// Metrics
// =========================================================================

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = linear_srgb::tf::linear_to_srgb(v);
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn to_srgb_u8_buffer(lin: &LinearRgb) -> Vec<u8> {
    let mut out = vec![0u8; lin.px.len()];
    out.par_chunks_mut(4096)
        .zip(lin.px.par_chunks(4096))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = linear_to_srgb_u8(s);
            }
        });
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct CellMetrics {
    psnr_db: f32,
    mean_de2000: f32,
    max_abs_delta: f32,
    pct_de_gt_5: f32,
    // New 2026-06-22-v2: tail-aware ΔE percentiles + OKLab Euclidean ΔE.
    de2000_p50: f32,
    de2000_p90: f32,
    de2000_p95: f32,
    de2000_p99: f32,
    de_ok_mean: f32,
    de_ok_p50: f32,
    de_ok_p90: f32,
    de_ok_p95: f32,
    de_ok_p99: f32,
    pct_above_de_ok_0p04: f32,
}

fn compute_metrics(reference: &LinearRgb, candidate: &LinearRgb) -> CellMetrics {
    debug_assert_eq!(reference.width, candidate.width);
    debug_assert_eq!(reference.height, candidate.height);
    let n = reference.pixels();

    // AUDIT FIX: rotate candidate (BT.2020 after the curve) into producer
    // primaries before scoring. This is the brief's directive: "score in
    // whatever gamut Apple chose" — keep producer-SDR in its native gamut
    // (BT.709 for iPhone HEIC, DisplayP3 for zfold7 UltraHDR JPEG), rotate
    // candidate to match. Hand-rolled in-place matrix multiply via the
    // exported `gamut::conversion_matrix` + `apply_matrix_row_f32` —
    // ~4× faster than going through `PixelBuffer::convert_to` per cell
    // (which would allocate ~144 MB per call for a 12 MP buffer).
    let cand_owned;
    let candidate: &LinearRgb = if candidate.primaries == reference.primaries {
        candidate
    } else if let Some(matrix) = zenpixels_convert::gamut::conversion_matrix(
        candidate.primaries,
        reference.primaries,
    ) {
        let mut rotated = LinearRgb {
            width: candidate.width,
            height: candidate.height,
            px: candidate.px.clone(),
            primaries: reference.primaries,
        };
        // Apply the 3×3 matrix row-by-row in parallel.
        let w = candidate.width as usize;
        rotated.px.par_chunks_mut(w * 3).for_each(|row| {
            zenpixels_convert::gamut::apply_matrix_row_f32(row, w, &matrix);
        });
        cand_owned = rotated;
        &cand_owned
    } else {
        // No conversion available (Unknown / identity) — score as-is.
        candidate
    };

    let ref_srgb = to_srgb_u8_buffer(reference);
    let cand_srgb = to_srgb_u8_buffer(candidate);

    // PSNR in 8-bit sRGB byte space — parallel reduce.
    let sq_sum: f64 = ref_srgb
        .par_chunks(8192)
        .zip(cand_srgb.par_chunks(8192))
        .map(|(rc, cc)| {
            let mut s: f64 = 0.0;
            for (&r, &c) in rc.iter().zip(cc.iter()) {
                let d = r as f64 - c as f64;
                s += d * d;
            }
            s
        })
        .sum();
    let n_samples = (n * 3) as f64;
    let mse = sq_sum / n_samples;
    let psnr = if mse <= 0.0 {
        99.0
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };

    // Max |Δ| in normalized linear space — parallel reduce.
    let max_abs: f32 = reference
        .px
        .par_chunks(8192)
        .zip(candidate.px.par_chunks(8192))
        .map(|(rc, cc)| {
            let mut m = 0.0_f32;
            for (&r, &c) in rc.iter().zip(cc.iter()) {
                let d = (r - c).abs();
                if d > m {
                    m = d;
                }
            }
            m
        })
        .reduce(|| 0.0_f32, |a, b| if a > b { a } else { b });

    let HistogramOut {
        sum_de2000,
        de2000_hist,
        sum_de_ok,
        de_ok_hist,
    } = compute_de_histograms(reference, candidate);

    let n_f64 = n as f64;
    let mean_de = (sum_de2000 / n_f64) as f32;
    let count_gt5 = de2000_hist[DE2K_DE5_BIN..].iter().sum::<u64>();
    let pct_gt5 = (count_gt5 as f64 * 100.0 / n_f64) as f32;

    let mean_de_ok = (sum_de_ok / n_f64) as f32;
    let count_de_ok_gt_t = de_ok_hist[DE_OK_THRESHOLD_BIN..].iter().sum::<u64>();
    let pct_de_ok = (count_de_ok_gt_t as f64 * 100.0 / n_f64) as f32;

    let (p50_2k, p90_2k, p95_2k, p99_2k) = percentiles_from_hist_4(
        &de2000_hist,
        n as u64,
        DE2K_HIST_BIN_WIDTH,
        DE2K_HIST_BINS,
    );
    let (p50_ok, p90_ok, p95_ok, p99_ok) = percentiles_from_hist_4(
        &de_ok_hist,
        n as u64,
        DE_OK_HIST_BIN_WIDTH,
        DE_OK_HIST_BINS,
    );

    CellMetrics {
        psnr_db: psnr as f32,
        mean_de2000: mean_de,
        max_abs_delta: max_abs,
        pct_de_gt_5: pct_gt5,
        de2000_p50: p50_2k,
        de2000_p90: p90_2k,
        de2000_p95: p95_2k,
        de2000_p99: p99_2k,
        de_ok_mean: mean_de_ok,
        de_ok_p50: p50_ok,
        de_ok_p90: p90_ok,
        de_ok_p95: p95_ok,
        de_ok_p99: p99_ok,
        pct_above_de_ok_0p04: pct_de_ok,
    }
}

/// CDF walk to find p50/p90/p95/p99 from a histogram. Linear-interpolated
/// inside the target bin (so 0.1-wide ΔE2000 bins resolve to ~0.05 ΔE; 0.001-
/// wide OKLab bins resolve to ~5e-4). Returns the four percentiles as f32.
fn percentiles_from_hist_4(
    hist: &[u64],
    total: u64,
    bin_width: f64,
    nbins: usize,
) -> (f32, f32, f32, f32) {
    if total == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let targets = [0.50_f64, 0.90, 0.95, 0.99];
    let total_f = total as f64;
    let mut cum: u64 = 0;
    let mut out = [0.0_f32; 4];
    let mut next_target = 0usize;
    for (bin, &cnt) in hist.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        let prev = cum;
        cum += cnt;
        while next_target < 4 {
            let target_count = (targets[next_target] * total_f).ceil() as u64;
            if cum >= target_count {
                // Linear interpolation: the target falls inside this bin.
                let into_bin = (target_count - prev) as f64;
                let frac = into_bin / cnt as f64;
                let val = (bin as f64 + frac) * bin_width;
                out[next_target] = val as f32;
                next_target += 1;
            } else {
                break;
            }
        }
        if next_target == 4 {
            break;
        }
    }
    // If we still have unfilled targets (shouldn't happen unless total miscounted),
    // pin them to top bin so it's visible.
    while next_target < 4 {
        out[next_target] = (nbins as f64 * bin_width) as f32;
        next_target += 1;
    }
    (out[0], out[1], out[2], out[3])
}

struct HistogramOut {
    sum_de2000: f64,
    de2000_hist: Vec<u64>,
    sum_de_ok: f64,
    de_ok_hist: Vec<u64>,
}

/// Build per-pixel ΔE2000 + ΔE_OKLab histograms in a single pass over the
/// (ref, candidate) pixel buffers. Each pixel does one Lab convert (D65 XYZ)
/// + one OKLab convert + ΔE2000 + ΔE_OKLab.
///
/// `reference` and `candidate` are in the *producer's primaries* (BT.709 for
/// iPhone HEIC, DisplayP3 for zfold7 UltraHDR JPEG). The current Lab path
/// already passes them straight through to D65 XYZ — which is a producer-
/// primaries-as-BT.709 approximation (DisplayP3 pixels get *slightly* off-axis
/// Lab values, but the saturated-color audit kept it that way). OKLab is
/// specified on BT.709 D65 linear sRGB, so we rotate to BT.709 inside the
/// OKLab subroutine (`oklab_from_linear_in_primaries`).
fn compute_de_histograms(
    reference: &LinearRgb,
    candidate: &LinearRgb,
) -> HistogramOut {
    // Per the brief, both `reference` and `candidate` are already in the same
    // producer-primaries frame (candidate has been rotated to match earlier
    // in compute_metrics). OKLab is defined on linear-sRGB (BT.709) D65 input;
    // rotate per-pixel via the precomputed `<producer> → BT.709` matrix.
    let ok_matrix = if reference.primaries == ColorPrimaries::Bt709 {
        None
    } else {
        zenpixels_convert::gamut::conversion_matrix(reference.primaries, ColorPrimaries::Bt709)
    };

    let (sum2k, hist2k, sum_ok, hist_ok) = reference
        .px
        .par_chunks(3 * 4096)
        .zip(candidate.px.par_chunks(3 * 4096))
        .map(|(rc, cc)| {
            let mut sum_de2k: f64 = 0.0;
            let mut hist_de2k = vec![0u64; DE2K_HIST_BINS];
            let mut sum_de_ok: f64 = 0.0;
            let mut hist_de_ok = vec![0u64; DE_OK_HIST_BINS];
            let n_px = rc.len() / 3;
            for i in 0..n_px {
                let r = [rc[i * 3], rc[i * 3 + 1], rc[i * 3 + 2]];
                let c = [cc[i * 3], cc[i * 3 + 1], cc[i * 3 + 2]];

                // ΔE2000 in Lab (producer-primaries → D65 XYZ → Lab, see
                // linear_rgb_to_lab comment block).
                let r_lab = linear_rgb_to_lab(r);
                let c_lab = linear_rgb_to_lab(c);
                let de2k = delta_e2000(r_lab, c_lab) as f64;
                sum_de2k += de2k;
                {
                    let bin = (de2k / DE2K_HIST_BIN_WIDTH) as usize;
                    let bin = bin.min(DE2K_HIST_BINS - 1);
                    hist_de2k[bin] += 1;
                }

                // ΔE_OKLab — Euclidean distance in OKLab space, computed on
                // linear sRGB (BT.709) D65 input per the OKLab spec.
                let r_lin709 = match ok_matrix {
                    None => r,
                    Some(m) => apply_matrix_pixel(&m, r),
                };
                let c_lin709 = match ok_matrix {
                    None => c,
                    Some(m) => apply_matrix_pixel(&m, c),
                };
                let r_ok = linear_srgb_to_oklab(r_lin709);
                let c_ok = linear_srgb_to_oklab(c_lin709);
                let dl = (r_ok[0] - c_ok[0]) as f64;
                let da = (r_ok[1] - c_ok[1]) as f64;
                let db = (r_ok[2] - c_ok[2]) as f64;
                let de_ok = (dl * dl + da * da + db * db).sqrt();
                sum_de_ok += de_ok;
                {
                    let bin = (de_ok / DE_OK_HIST_BIN_WIDTH) as usize;
                    let bin = bin.min(DE_OK_HIST_BINS - 1);
                    hist_de_ok[bin] += 1;
                }
            }
            (sum_de2k, hist_de2k, sum_de_ok, hist_de_ok)
        })
        .reduce(
            || {
                (
                    0.0_f64,
                    vec![0u64; DE2K_HIST_BINS],
                    0.0_f64,
                    vec![0u64; DE_OK_HIST_BINS],
                )
            },
            |mut a, b| {
                a.0 += b.0;
                for (ai, bi) in a.1.iter_mut().zip(b.1.iter()) {
                    *ai += *bi;
                }
                a.2 += b.2;
                for (ai, bi) in a.3.iter_mut().zip(b.3.iter()) {
                    *ai += *bi;
                }
                a
            },
        );
    HistogramOut {
        sum_de2000: sum2k,
        de2000_hist: hist2k,
        sum_de_ok: sum_ok,
        de_ok_hist: hist_ok,
    }
}

#[inline]
fn apply_matrix_pixel(m: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    [
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    ]
}

/// Björn Ottosson's OKLab transform, on linear sRGB (BT.709) D65 input.
/// Source: https://bottosson.github.io/posts/oklab/. Constants are byte-
/// identical to the reference implementation. OKLab assumes the input is
/// in [0, 1] but stays well-defined for negative or >1 values (cube root
/// of a negative number returns NaN under f32::cbrt only if we pass negative;
/// clamp to keep the metric well-defined for tone-mapped buffers).
#[inline]
fn linear_srgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);
    let l_ = 0.4122214708_f32 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m_ = 0.2119034982_f32 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s_ = 0.0883024619_f32 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l = l_.cbrt();
    let m = m_.cbrt();
    let s = s_.cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

fn linear_rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0].max(0.0).min(1.0) as f64;
    let g = rgb[1].max(0.0).min(1.0) as f64;
    let b = rgb[2].max(0.0).min(1.0) as f64;

    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

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
// Aggregation + report
// =========================================================================

#[derive(Clone)]
struct SampleResult {
    sample_path: PathBuf,
    format_label: String,
    device_class: DeviceClass,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    /// Per peak method: measured source peak nits.
    source_peaks: HashMap<PeakMethod, f32>,
    /// Per peak method × curve index (parallel to global curve_grid): metrics.
    cells: HashMap<PeakMethod, Vec<CellMetrics>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceClass {
    UltraHdrJpeg, // UltraHDR JPEG (Samsung / Pixel / iPhone — anything .jpg/.jpeg)
    Heic,         // HEIC container (Samsung s23u, iPhone 13/16 Pro, ZFold, etc.)
    Other,
}

fn classify_device(path: &Path) -> DeviceClass {
    // Ext-based — both Samsung s23u and iPhone shoot HEIC; UltraHDR JPEG covers
    // Samsung's other phones, Pixels, and the iPhone fallback. The split is
    // "which container", not "which vendor".
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "heic" => DeviceClass::Heic,
        "jpg" | "jpeg" => DeviceClass::UltraHdrJpeg,
        _ => DeviceClass::Other,
    }
}

#[allow(dead_code)]
fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.retain(|v| v.is_finite());
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

// =========================================================================
// Report writer
// =========================================================================

fn write_csv(samples: &[SampleResult], curve_grid: &[CurveSpec]) -> anyhow::Result<()> {
    fs::create_dir_all(Path::new(CSV_PATH).parent().unwrap())?;
    let f = fs::File::create(CSV_PATH)?;
    let mut w = BufWriter::new(f);
    // Schema = 2026-06-22-audited columns + 2026-06-22-v2 tail metrics.
    // New (right of old) columns are tail-aware: ΔE2000 percentiles and
    // OKLab Euclidean ΔE (mean + percentiles + threshold count).
    writeln!(
        w,
        "sample,format,peak_method,curve,knee_tone,knee_gamut,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_above_de5,color_handling_version,\
de2000_p50,de2000_p90,de2000_p95,de2000_p99,\
de_ok_mean,de_ok_p50,de_ok_p90,de_ok_p95,de_ok_p99,\
pct_above_de_ok_0p04,metrics_version"
    )?;
    // Deterministic sort: sample asc, then PeakMethod::all() order, then
    // curve_grid order. PeakMethod::all() is fixed; curve_grid is built
    // once at startup.
    let mut sorted: Vec<&SampleResult> = samples.iter().collect();
    sorted.sort_by(|a, b| a.sample_path.cmp(&b.sample_path));
    for r in sorted {
        let stem = r
            .sample_path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        for &m in PeakMethod::all() {
            let peak = r.source_peaks.get(&m).copied().unwrap_or(0.0);
            let cells = r.cells.get(&m).cloned().unwrap_or_default();
            for (ci, c) in curve_grid.iter().enumerate() {
                if ci >= cells.len() {
                    continue;
                }
                let cell = cells[ci];
                let kt = c
                    .knee_tone()
                    .map(|v| format!("{:.2}", v))
                    .unwrap_or_default();
                let kg = c
                    .knee_gamut()
                    .map(|v| format!("{:.2}", v))
                    .unwrap_or_default();
                writeln!(
                    w,
                    "{},{},{},{},{},{},{:.1},{:.3},{:.4},{:.5},{:.3},{},\
{:.4},{:.4},{:.4},{:.4},\
{:.5},{:.5},{:.5},{:.5},{:.5},\
{:.3},{}",
                    stem,
                    r.format_label,
                    m.label(),
                    c.label(),
                    kt,
                    kg,
                    peak,
                    cell.psnr_db,
                    cell.mean_de2000,
                    cell.max_abs_delta,
                    cell.pct_de_gt_5,
                    COLOR_HANDLING_VERSION,
                    cell.de2000_p50,
                    cell.de2000_p90,
                    cell.de2000_p95,
                    cell.de2000_p99,
                    cell.de_ok_mean,
                    cell.de_ok_p50,
                    cell.de_ok_p90,
                    cell.de_ok_p95,
                    cell.de_ok_p99,
                    cell.pct_above_de_ok_0p04,
                    METRICS_VERSION,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn per_curve_summary(
    samples: &[SampleResult],
    curve_grid: &[CurveSpec],
    method: PeakMethod,
) -> Vec<(String, f32, f32, f32, f32)> {
    let mut out = Vec::new();
    for (ci, c) in curve_grid.iter().enumerate() {
        let psnr: Vec<f32> = samples
            .iter()
            .filter_map(|r| r.cells.get(&method).and_then(|cells| cells.get(ci)))
            .map(|m| m.psnr_db)
            .collect();
        let de: Vec<f32> = samples
            .iter()
            .filter_map(|r| r.cells.get(&method).and_then(|cells| cells.get(ci)))
            .map(|m| m.mean_de2000)
            .collect();
        let max_a: Vec<f32> = samples
            .iter()
            .filter_map(|r| r.cells.get(&method).and_then(|cells| cells.get(ci)))
            .map(|m| m.max_abs_delta)
            .collect();
        let pct5: Vec<f32> = samples
            .iter()
            .filter_map(|r| r.cells.get(&method).and_then(|cells| cells.get(ci)))
            .map(|m| m.pct_de_gt_5)
            .collect();
        out.push((
            c.label(),
            median(psnr),
            median(de),
            median(max_a),
            median(pct5),
        ));
    }
    out.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[allow(dead_code)]
fn best_mobius_for_method(
    samples: &[SampleResult],
    curve_grid: &[CurveSpec],
    method: PeakMethod,
) -> Option<(String, f32, f32)> {
    curve_grid
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c, CurveSpec::Mobius { .. }))
        .map(|(ci, c)| {
            let de = median(
                samples
                    .iter()
                    .filter_map(|r| r.cells.get(&method).and_then(|cells| cells.get(ci)))
                    .map(|m| m.mean_de2000)
                    .collect(),
            );
            (c.label(), de, ci as f32)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}


// =========================================================================
// Main
// =========================================================================

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();

    refresh_lock("scanning-corpus");
    println!("Scanning corpus at {}...", SAMPLES_ROOT);
    let all_files = collect_samples();
    println!(
        "  Found {} candidate files (.jpg/.jpeg/.heic)",
        all_files.len()
    );

    // Probe each file for gain-map presence to filter the work list.
    let mut work_list: Vec<(PathBuf, String)> = Vec::new();
    for path in &all_files {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        if let Some(fmt) = detect_gainmap(&bytes) {
            work_list.push((path.clone(), fmt.to_string()));
        }
    }
    let n_heic = work_list.iter().filter(|(_, f)| f == "HEIC").count();
    let n_jpeg = work_list.iter().filter(|(_, f)| f == "JPEG").count();
    println!(
        "Found {} gain-mapped samples ({} UltraHDR JPEG + {} HEIC) to process",
        work_list.len(),
        n_jpeg,
        n_heic
    );

    if work_list.is_empty() {
        anyhow::bail!("no gain-mapped samples found in {}", SAMPLES_ROOT);
    }

    let curve_grid = build_curve_grid();
    println!(
        "Curves under test: {} (12 Möbius + 8 named)",
        curve_grid.len()
    );
    println!("Peak methods: {:?}", PeakMethod::all());

    let total = work_list.len();

    // Parallelize across samples — each sample fits in one rayon job;
    // the inner curve loop runs sequentially per job. 76 / 16 cores ≈ 5
    // batches of ~20s each = ~100s wall-clock vs. the prior ~38 min when
    // we parallelized curves under a single-sample-at-a-time outer loop.
    // Memory: 16 samples × ~600 MB buffers + scratch ≈ ~15 GB (fits 128 GB).
    type SampleOutcome = Result<SampleResult, (PathBuf, String)>;
    let outcomes: Vec<SampleOutcome> = work_list
        .par_iter()
        .enumerate()
        .map(|(idx, (path, fmt_label))| -> SampleOutcome {
        let stem_raw = path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stem = stem_raw.replace('.', "_");

        // Refresh lock every 10 samples (more often if you want).
        if idx % 5 == 0 {
            refresh_lock(&format!("sweep-{}-of-{}", idx + 1, total));
        }

        println!("\n=== [{}/{}] {} ({}) ===", idx + 1, total, stem, fmt_label);

        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  read failed: {}", e);
                return Err((path.clone(), format!("read: {}", e)));
            }
        };

        let (sdr, hdr, sdr_descr, hdr_descr, hdr_orig_primaries) =
            match decode_sample_full(&bytes) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  decode failed: {}", e);
                    return Err((path.clone(), format!("decode: {}", e)));
                }
            };
        println!(
            "  SDR/HDR: {}x{} (SDR transfer={:?}, primaries={:?}; HDR transfer={:?}, decoded_primaries={:?} → BT.2020)",
            sdr.width, sdr.height, sdr_descr, sdr.primaries, hdr_descr, hdr_orig_primaries,
        );

        let hdr_max_raw = hdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sdr_max_raw = sdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Skip cases.
        if hdr_max_raw < 1.05 && sdr_max_raw < 1.05 && (hdr_max_raw - sdr_max_raw).abs() < 0.05 {
            println!("  → skipping: HDR reconstruction returned SDR base (no boost)");
            return Err((
                path.clone(),
                format!(
                    "no HDR boost (hdr_max={:.3}, sdr_max={:.3})",
                    hdr_max_raw, sdr_max_raw
                ),
            ));
        }
        if sdr_descr == TransferFunction::Pq {
            println!("  → skipping: SDR base is PQ-encoded (base=HDR)");
            return Err((path.clone(), "SDR base is PQ-encoded".into()));
        }
        if !hdr_max_raw.is_finite() || !sdr_max_raw.is_finite() {
            println!("  → skipping: non-finite pixel data");
            return Err((path.clone(), "non-finite pixel data".into()));
        }

        let device_class = classify_device(path);

        let mut source_peaks: HashMap<PeakMethod, f32> = HashMap::new();
        let mut cells_per_method: HashMap<PeakMethod, Vec<CellMetrics>> = HashMap::new();

        for &method in PeakMethod::all() {
            let source_peak = measure_source_peak_nits(&hdr, method);
            source_peaks.insert(method, source_peak);
            println!(
                "  [{}] source_peak={:.0} nits  (raw HDR max={:.3}, SDR max={:.3})",
                method.label(),
                source_peak,
                hdr_max_raw,
                sdr_max_raw
            );

            // Sequential within a sample — outer loop parallelizes across
            // samples (16-wide) so the cores are filled by sample-level
            // parallelism instead of curve-level. par_chunks inside
            // apply_curve / compute_metrics still chew chunks via rayon's
            // work-stealing pool when an outer slot is briefly idle.
            let cells: Vec<CellMetrics> = curve_grid
                .iter()
                .map(|c| {
                    let cand = apply_curve(*c, &hdr, source_peak);
                    compute_metrics(&sdr, &cand)
                })
                .collect();
            let summary_idx_first = 0; // any curve for compact print
            println!(
                "    [{}] {} cells computed, e.g. {} ΔE2000={:.3}",
                method.label(),
                cells.len(),
                curve_grid[summary_idx_first].label(),
                cells[summary_idx_first].mean_de2000
            );
            cells_per_method.insert(method, cells);
        }

        let (w, h) = (sdr.width, sdr.height);
        drop(hdr);
        drop(sdr);

        Ok(SampleResult {
            sample_path: path.clone(),
            format_label: fmt_label.clone(),
            device_class,
            width: w,
            height: h,
            source_peaks,
            cells: cells_per_method,
        })
    })
    .collect();

    let mut sample_results: Vec<SampleResult> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    for outcome in outcomes {
        match outcome {
            Ok(r) => sample_results.push(r),
            Err(e) => failures.push(e),
        }
    }

    let runtime = t0.elapsed().as_secs_f64();
    refresh_lock("writing-csv");
    write_csv(&sample_results, &curve_grid)?;
    println!("CSV written to {}", CSV_PATH);
    let _ = n_jpeg;
    let _ = n_heic;
    println!(
        "Done. {} samples succeeded, {} failed, total {:.1}s.",
        sample_results.len(),
        failures.len(),
        runtime
    );
    Ok(())
}
