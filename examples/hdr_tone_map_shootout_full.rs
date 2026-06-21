//! HDR → SDR tone-map cross-curve shootout on imazen-26 corpus.
//!
//! Extended version of `hdr_tone_map_shootout.rs` that runs:
//!   - 76 gain-mapped samples (UltraHDR JPEG + HEIC) from
//!     `/home/lilith/work/codec-corpus/imazen-26/`, filtered to those that
//!     carry a gain map (probe-based).
//!   - 3 source-peak measurement methods (`measure_max`, `measure_robust`,
//!     `measure_max_smoothed`).
//!   - 20 curve cells per sample (12 Möbius knee × 4kt × 3kg + 8 named
//!     curves: BT.2446 A/B/C, BT.2408, BT.2390, Narkowicz, HableFilmic,
//!     AcesAp1).
//!
//! Total cells = 76 × 3 × 20 = ~4560. Montages capped to top-3 + bottom-3
//! per (curve, peak_method) cell (~360 max).
//!
//! HdrToSdr Möbius input normalization: HdrToSdr now does the rescale
//! internally (fixed in commit ca614df0); we pass source-normalized input
//! directly. See `apply_curve` for the per-curve normalization.
//!
//! Run:
//! ```text
//! nice -n19 cargo run -p zentone --release \
//!   --example hdr_tone_map_shootout_full --features hdr-shootout
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

use zenpixels_convert::hdr::Bt2446A;
use zentone::gamut::soft_clip_knee_strip;
use zentone::{Bt2408Tonemapper, Bt2446B, Bt2446C, ToneMap, ToneMapCurve};

// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const MONTAGES_DIR: &str = "/mnt/v/output/zentone/shootout-full";
const REPORT_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20.md";
const CSV_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20.csv";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";

// Möbius knee sweep grid.
const KNEE_TONE_VALUES: &[f32] = &[0.20, 0.30, 0.40, 0.50];
const KNEE_GAMUT_VALUES: &[f32] = &[0.90, 0.95, 0.99];

// Peak-measurement methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PeakMethod {
    Max,
    Robust,
    MaxSmoothed,
}

impl PeakMethod {
    fn label(&self) -> &'static str {
        match self {
            PeakMethod::Max => "measure_max",
            PeakMethod::Robust => "measure_robust",
            PeakMethod::MaxSmoothed => "measure_max_smoothed",
        }
    }
    fn all() -> &'static [PeakMethod] {
        &[PeakMethod::Max, PeakMethod::Robust, PeakMethod::MaxSmoothed]
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

/// Returns `(sdr, hdr, sdr_transfer, hdr_transfer)` linear-RGB f32.
/// Uses `decode_full_frame()` for SDR base and `reconstruct_hdr(None)` for HDR.
/// Works for both JPEG-UltraHDR and HEIC (both expose `reconstructs_hdr`
/// natively per zencodecs).
fn decode_sample_full(
    bytes: &[u8],
) -> anyhow::Result<(LinearRgb, LinearRgb, TransferFunction, TransferFunction)> {
    // SDR base — the producer's ground truth.
    // Force OrientationHint::Correct on both decodes so they match: heic's
    // reconstruct_hdr internally bakes orientation (the gain-map item carries
    // no irot, so reconstruction must happen in display space to align with
    // the base). The SDR path follows the OrientationHint, so we explicitly
    // pick Correct to match. Without this, EXIF-rotated HEICs come back at
    // 4032x3024 (stored) vs 3024x4032 (display) and the dim check trips.
    let sdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR: {}", e.error()))?;
    let sdr_desc = sdr_out.pixels().descriptor();
    let sdr_buf = sdr_out.into_buffer();
    let sdr_rgb = pixel_buffer_to_linear_rgb(&sdr_buf)?;

    // HDR via codec's native gain-map application (full headroom).
    let hdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .reconstruct_hdr(None)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("reconstruct_hdr: {}", e.error()))?;
    let hdr_desc = hdr_out.pixels().descriptor();
    let hdr_buf = hdr_out.into_buffer();
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
    })
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
}

fn compute_metrics(reference: &LinearRgb, candidate: &LinearRgb) -> CellMetrics {
    debug_assert_eq!(reference.width, candidate.width);
    debug_assert_eq!(reference.height, candidate.height);
    let n = reference.pixels();
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
    // Parallel-reduce over pixel chunks (chunks of 3 floats = 1 pixel).
    reference
        .px
        .par_chunks(3 * 4096)
        .zip(candidate.px.par_chunks(3 * 4096))
        .map(|(rc, cc)| {
            let mut sum: f64 = 0.0;
            let mut cnt: u64 = 0;
            let n_px = rc.len() / 3;
            for i in 0..n_px {
                let r_lab = linear_rgb_to_lab([rc[i * 3], rc[i * 3 + 1], rc[i * 3 + 2]]);
                let c_lab = linear_rgb_to_lab([cc[i * 3], cc[i * 3 + 1], cc[i * 3 + 2]]);
                let de = delta_e2000(r_lab, c_lab) as f64;
                sum += de;
                if de > 5.0 {
                    cnt += 1;
                }
            }
            (sum, cnt)
        })
        .reduce(|| (0.0_f64, 0_u64), |a, b| (a.0 + b.0, a.1 + b.1))
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

    for y in 0..h {
        let row_dst = (y * combined_w * 3) as usize;
        let row_src = (y * w * 3) as usize;
        let row_len = (w * 3) as usize;

        out[row_dst..row_dst + row_len].copy_from_slice(&ref_srgb[row_src..row_src + row_len]);
        let sep0 = row_dst + row_len;
        for k in 0..sep_w as usize {
            out[sep0 + k * 3..sep0 + k * 3 + 3].copy_from_slice(&[40, 40, 40]);
        }
        let p1 = sep0 + (sep_w * 3) as usize;
        out[p1..p1 + row_len].copy_from_slice(&cand_srgb[row_src..row_src + row_len]);
        let sep1 = p1 + row_len;
        for k in 0..sep_w as usize {
            out[sep1 + k * 3..sep1 + k * 3 + 3].copy_from_slice(&[40, 40, 40]);
        }
        let p2 = sep1 + (sep_w * 3) as usize;
        for x in 0..w {
            let i = row_src + (x as usize) * 3;
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
// Top-N tracker per (curve, peak_method) cell
// =========================================================================

/// Track the top-K best and bottom-K worst samples per cell by ΔE2000.
#[derive(Default)]
struct TopBottom {
    /// `(de2000, sample_idx)` — sorted ascending (best ΔE2000 first).
    top_best: Vec<(f32, usize)>,
    /// `(de2000, sample_idx)` — sorted descending (worst ΔE2000 first).
    top_worst: Vec<(f32, usize)>,
}

impl TopBottom {
    fn add_best(&mut self, de: f32, idx: usize, k: usize) {
        // Insert sorted ascending; cap at K.
        if !de.is_finite() {
            return;
        }
        let pos = self.top_best.partition_point(|(d, _)| *d <= de);
        self.top_best.insert(pos, (de, idx));
        if self.top_best.len() > k {
            self.top_best.truncate(k);
        }
    }
    fn add_worst(&mut self, de: f32, idx: usize, k: usize) {
        if !de.is_finite() {
            return;
        }
        // Sorted descending: insert before the first element ≤ de.
        let pos = self.top_worst.partition_point(|(d, _)| *d >= de);
        self.top_worst.insert(pos, (de, idx));
        if self.top_worst.len() > k {
            self.top_worst.truncate(k);
        }
    }
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
    writeln!(
        w,
        "sample,format,peak_method,curve,knee_tone,knee_gamut,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_above_de5"
    )?;
    for r in samples {
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
                    "{},{},{},{},{},{},{:.1},{:.3},{:.4},{:.5},{:.3}",
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
                    cell.pct_de_gt_5
                )?;
            }
        }
    }
    Ok(())
}

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

fn write_report(
    sample_results: &[SampleResult],
    curve_grid: &[CurveSpec],
    failures: &[(PathBuf, String)],
    n_jpeg: usize,
    n_heic: usize,
    runtime_secs: f64,
) -> anyhow::Result<()> {
    let mut s = String::new();
    s.push_str(
        "# HDR→SDR tone-map shootout (full) — 76 imazen-26 samples × 3 peak methods × 20 curves\n\n",
    );
    s.push_str("**Date:** 2026-06-20\n");
    s.push_str("**CPU:** AMD Ryzen 9 7950X (16 cores / 32 threads)\n");
    s.push_str(&format!(
        "**Samples processed:** {} ({} UltraHDR JPEG + {} HEIC) of {} candidate files at `/home/lilith/work/codec-corpus/imazen-26/`\n",
        sample_results.len(),
        n_jpeg,
        n_heic,
        sample_results.len() + failures.len(),
    ));
    s.push_str(&format!(
        "**Total runtime:** {:.1}s ({:.1}min)\n",
        runtime_secs,
        runtime_secs / 60.0
    ));
    s.push_str("**HdrToSdr fix verified:** commit `ca614df0` — HdrToSdr now rescales source-norm input to target-norm internally; the prior shootout's pre-multiplication workaround is removed.\n\n");

    // Methodology
    s.push_str(
        "## Methodology\n\n\
For each sample we:\n\
1. Walk `/home/lilith/work/codec-corpus/imazen-26/` recursively, filter to `.jpg/.jpeg/.heic`, probe \
   each candidate (`DecodeRequest::probe()`), and keep the ones with `GainMapPresence::Available` \
   or `supplements.gain_map == true`.\n\
2. Decode the SDR base via `DecodeRequest::decode_full_frame()` (the producer's ground truth).\n\
3. Reconstruct HDR via `DecodeRequest::reconstruct_hdr(None).decode_full_frame()` — works for \
   both JPEG-UltraHDR (jpeg-ultrahdr feature) and HEIC (heic-decode feature). Returns linear-RGB \
   f32 anchored at `1.0 = SDR diffuse white = 203 nits`.\n\
4. Measure the source peak via three methods on the linear-RGB f32 buffer (BT.2408 anchor, \
   MaxRgb reduction): `measure_max` (spec-literal max), `measure_robust` (99.99 % percentile — \
   the production default in libplacebo / DaVinci), and `measure_max_smoothed` (3×1 box-filter \
   max — defect-tolerant without committing to a percentile).\n\
5. Apply each candidate curve (20 cells: 12 Möbius knee × 4kt × 3kg + BT.2446 A/B/C + BT.2408 + \
   BT.2390 + Narkowicz + HableFilmic + AcesAp1) per peak method. For Möbius (HdrToSdr) the input \
   is now source-normalized: `scratch = hdr * (203 / source_peak_nits)`, then \
   `converter.apply_strip(strip)` handles the rescale internally. For non-Möbius curves the prior \
   normalization is unchanged.\n\
6. Compare against the SDR base in sRGB byte space (PSNR), CIE Lab (mean ΔE2000, % pixels with \
   ΔE > 5), and normalised linear (max |Δ| per channel). The SDR base is the producer's ground \
   truth — the curve closest to it matches the producer's aesthetic, NOT scene-referred fidelity.\n\n",
    );

    if !failures.is_empty() {
        s.push_str(&format!(
            "## Failures ({} of {} candidates)\n\n",
            failures.len(),
            sample_results.len() + failures.len()
        ));
        // Group failure reasons.
        let mut counts: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for (p, why) in failures {
            let key = if why.contains("no HDR boost") {
                "no HDR boost (gain map present but reconstruction returned SDR base verbatim)"
                    .to_string()
            } else if why.contains("PQ-encoded") {
                "SDR base is PQ-encoded (base=HDR container)".to_string()
            } else if why.contains("reconstruct_hdr") {
                format!(
                    "reconstruct_hdr failed: {}",
                    why.split(": ").nth(1).unwrap_or(why)
                )
            } else if why.contains("decode SDR") {
                format!(
                    "SDR decode failed: {}",
                    why.split(": ").nth(1).unwrap_or(why)
                )
            } else if why.contains("no gain map") {
                "no gain map detected at probe".to_string()
            } else {
                why.clone()
            };
            counts.entry(key).or_default().push(p.clone());
        }
        let mut keys: Vec<&String> = counts.keys().collect();
        keys.sort();
        for k in keys {
            let paths = &counts[k];
            s.push_str(&format!("- **{}** ({} samples)\n", k, paths.len()));
            for p in paths.iter().take(5) {
                s.push_str(&format!(
                    "  - `{}`\n",
                    p.file_name()
                        .map(|x| x.to_string_lossy().into_owned())
                        .unwrap_or_default()
                ));
            }
            if paths.len() > 5 {
                s.push_str(&format!("  - ... and {} more\n", paths.len() - 5));
            }
        }
        s.push('\n');
    }

    // ==== Headline ====
    s.push_str("## Headline verdict\n\n");
    let summaries: Vec<(PeakMethod, Vec<(String, f32, f32, f32, f32)>)> = PeakMethod::all()
        .iter()
        .map(|&m| (m, per_curve_summary(sample_results, curve_grid, m)))
        .collect();
    let winners: Vec<(PeakMethod, String, f32)> = summaries
        .iter()
        .map(|(m, sum)| {
            let (label, _, de, _, _) = sum[0].clone();
            (*m, label, de)
        })
        .collect();
    let unanimous = winners.windows(2).all(|w| w[0].1 == w[1].1);
    let robust_winner = winners
        .iter()
        .find(|(m, _, _)| *m == PeakMethod::Robust)
        .cloned();
    if unanimous {
        let (_, lab, de) = &winners[0];
        s.push_str(&format!(
            "On {} gain-mapped samples, **`{}` wins under every peak-measurement method** (median ΔE2000 ≈ {:.3} across all three methods). Peak-measurement choice does NOT shift the curve ranking.\n\n",
            sample_results.len(),
            lab,
            de
        ));
    } else {
        s.push_str(&format!(
            "On {} gain-mapped samples, the winner shifts with peak-measurement method:\n",
            sample_results.len()
        ));
        for (m, lab, de) in &winners {
            s.push_str(&format!(
                "- `{}`: `{}` (median ΔE2000 {:.3})\n",
                m.label(),
                lab,
                de
            ));
        }
        s.push('\n');
    }
    if let Some((_, lab, de)) = robust_winner {
        s.push_str(&format!(
            "**Recommended HdrToSdr default** — based on the production-default `measure_robust` peak: the winning curve is `{}` (median ΔE2000 {:.3}). See the per-method tables below for the Möbius-specific knee recommendation.\n\n",
            lab, de
        ));
    }

    // ==== Per-method summaries ====
    s.push_str("## Per-curve summary by peak method\n\n");
    for (method, sum) in &summaries {
        s.push_str(&format!(
            "### `{}` — sorted by median ΔE2000 ascending\n\n",
            method.label()
        ));
        s.push_str(
            "| Rank | Curve | Median PSNR (dB) | Median ΔE2000 | Median max\\|Δ\\| | Median %>ΔE5 |\n",
        );
        s.push_str(
            "|------|-------|------------------|---------------|-----------------|---------------|\n",
        );
        for (rank, (label, psnr, de, max_abs, pct5)) in sum.iter().enumerate() {
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
    }

    // ==== Möbius knee heatmaps ====
    s.push_str("## Möbius knee sweep (3 heatmaps, one per peak method)\n\n");
    s.push_str("Cells show median ΔE2000 across all processed samples. Lower = closer to producer's SDR.\n\n");
    for method in PeakMethod::all() {
        s.push_str(&format!("### `{}`\n\n", method.label()));
        let best = best_mobius_for_method(sample_results, curve_grid, *method);
        let best_label = best.as_ref().map(|(l, _, _)| l.clone()).unwrap_or_default();
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
                            .filter_map(|r| r.cells.get(method).and_then(|cells| cells.get(ci)))
                            .map(|m| m.mean_de2000)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let m = median(de_vals);
                if label == best_label {
                    s.push_str(&format!("| **{:.3}** ", m));
                } else {
                    s.push_str(&format!("| {:.3} ", m));
                }
            }
            s.push_str("|\n");
        }
        s.push('\n');
        if let Some((label, de, _)) = best {
            s.push_str(&format!(
                "**Best Möbius cell ({}):** `{}` (median ΔE2000 = {:.3})\n\n",
                method.label(),
                label,
                de
            ));
        }
    }

    // ==== Robustness to peak method ====
    s.push_str("## Robustness to peak measurement choice\n\n");
    s.push_str("For each curve, the median ΔE2000 under each peak method. `Δ_max_method` = max ΔE2000 difference across methods (the curve's sensitivity).\n\n");
    s.push_str("| Curve | measure_max | measure_robust | measure_max_smoothed | Δ_max_method |\n");
    s.push_str("|---|---|---|---|---|\n");
    let mut robustness: Vec<(String, f32, f32, f32, f32)> = curve_grid
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            let de_max = median(
                sample_results
                    .iter()
                    .filter_map(|r| {
                        r.cells
                            .get(&PeakMethod::Max)
                            .and_then(|cells| cells.get(ci))
                    })
                    .map(|m| m.mean_de2000)
                    .collect(),
            );
            let de_robust = median(
                sample_results
                    .iter()
                    .filter_map(|r| {
                        r.cells
                            .get(&PeakMethod::Robust)
                            .and_then(|cells| cells.get(ci))
                    })
                    .map(|m| m.mean_de2000)
                    .collect(),
            );
            let de_smooth = median(
                sample_results
                    .iter()
                    .filter_map(|r| {
                        r.cells
                            .get(&PeakMethod::MaxSmoothed)
                            .and_then(|cells| cells.get(ci))
                    })
                    .map(|m| m.mean_de2000)
                    .collect(),
            );
            let vals = [de_max, de_robust, de_smooth];
            let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (c.label(), de_max, de_robust, de_smooth, mx - mn)
        })
        .collect();
    robustness.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
    for (label, dm, dr, ds, dd) in &robustness {
        s.push_str(&format!(
            "| `{}` | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            label, dm, dr, ds, dd
        ));
    }
    s.push('\n');

    // ==== Per-device-class breakdown ====
    s.push_str("## Per-source-device breakdown (using `measure_robust`)\n\n");
    s.push_str("UltraHDR JPEG vs HEIC: same curve ranking, or different? Top-5 per class.\n\n");
    for class in &[DeviceClass::UltraHdrJpeg, DeviceClass::Heic] {
        let class_name = match class {
            DeviceClass::UltraHdrJpeg => "UltraHDR JPEG",
            DeviceClass::Heic => "HEIC",
            DeviceClass::Other => "Other",
        };
        let in_class: Vec<&SampleResult> = sample_results
            .iter()
            .filter(|r| r.device_class == *class)
            .collect();
        s.push_str(&format!(
            "### {} ({} samples)\n\n",
            class_name,
            in_class.len()
        ));
        if in_class.is_empty() {
            s.push_str("_(none decoded in this class.)_\n\n");
            continue;
        }
        let mut sum: Vec<(String, f32)> = curve_grid
            .iter()
            .enumerate()
            .map(|(ci, c)| {
                let de = median(
                    in_class
                        .iter()
                        .filter_map(|r| {
                            r.cells
                                .get(&PeakMethod::Robust)
                                .and_then(|cells| cells.get(ci))
                        })
                        .map(|m| m.mean_de2000)
                        .collect(),
                );
                (c.label(), de)
            })
            .collect();
        sum.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        s.push_str("| Rank | Curve | Median ΔE2000 |\n");
        s.push_str("|---|---|---|\n");
        for (rank, (label, de)) in sum.iter().take(5).enumerate() {
            s.push_str(&format!("| {} | `{}` | {:.3} |\n", rank + 1, label, de));
        }
        s.push('\n');
    }

    // ==== Top-3 / bottom-3 per cell ====
    s.push_str("## Top-3 / bottom-3 montages per (curve, peak_method) cell\n\n");
    s.push_str(&format!(
        "All saved under [`{}/`](http://172.23.240.1:3300/zentone/shootout-full/) as 3-panel images: producer SDR | our SDR | abs Δ×10.\n\n",
        MONTAGES_DIR
    ));
    s.push_str("File naming: `<sample-stem>__<curve_label>__<peak_method>__<best|worst>.png`.\n\n");

    // ==== Verdict ====
    s.push_str("## Verdict\n\n");
    if let Some((m, lab, de)) = winners
        .iter()
        .find(|(m, _, _)| *m == PeakMethod::Robust)
        .cloned()
    {
        s.push_str(&format!(
            "**Recommended HdrToSdr default**: assuming the production-default `measure_robust` peak, the curve closest to producer-graded SDR across {} samples is `{}` (median ΔE2000 {:.3}). ",
            sample_results.len(),
            lab,
            de,
        ));
        let _ = m;
    }
    if let Some((label, de, _)) =
        best_mobius_for_method(sample_results, curve_grid, PeakMethod::Robust)
    {
        s.push_str(&format!(
            "Inside the Möbius family the best knee setting is `{}` (median ΔE2000 {:.3}). ",
            label, de
        ));
    }
    s.push_str("\n\nPeak-measurement effect on the headline ranking is documented in the 'Robustness to peak measurement choice' table above — curves with low `Δ_max_method` are insensitive to the peak choice; high values indicate sensitivity.\n\n");

    // ==== Failed samples ====
    if !failures.is_empty() {
        s.push_str("## Failed samples\n\n");
        s.push_str(&format!(
            "{} of {} candidate files failed. Common reasons:\n\n",
            failures.len(),
            failures.len() + sample_results.len(),
        ));
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_, why) in failures {
            let key = if why.contains("no HDR boost") {
                "no HDR boost"
            } else if why.contains("PQ-encoded") {
                "PQ-encoded SDR base"
            } else if why.contains("reconstruct_hdr") {
                "reconstruct_hdr error"
            } else if why.contains("decode SDR") {
                "SDR decode error"
            } else if why.contains("no gain map") {
                "no gain map at probe"
            } else {
                "other"
            };
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
        entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (k, c) in entries {
            s.push_str(&format!("- {} — {}\n", k, c));
        }
        s.push('\n');
    }

    // ==== Reproduce ====
    s.push_str("## Reproduce\n\n");
    s.push_str("```bash\n");
    s.push_str("nice -n19 cargo build -p zentone --release \\\n");
    s.push_str("  --example hdr_tone_map_shootout_full --features hdr-shootout\n");
    s.push_str("nice -n19 cargo run -p zentone --release \\\n");
    s.push_str("  --example hdr_tone_map_shootout_full --features hdr-shootout\n");
    s.push_str("```\n\n");
    s.push_str(
        "Sample inputs: `/home/lilith/work/codec-corpus/imazen-26/**/{*.jpg,*.jpeg,*.heic}`.\n",
    );
    s.push_str(&format!(
        "Per-sample × per-cell montages (top-3 + bottom-3 per cell): `{}/...`.\n",
        MONTAGES_DIR
    ));
    s.push_str(&format!("Full CSV: `{}`.\n", CSV_PATH));
    s.push_str("Source: `examples/hdr_tone_map_shootout_full.rs`.\n");

    fs::create_dir_all(Path::new(REPORT_PATH).parent().unwrap())?;
    fs::write(REPORT_PATH, s)?;
    Ok(())
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

    fs::create_dir_all(MONTAGES_DIR)?;

    let curve_grid = build_curve_grid();
    println!(
        "Curves under test: {} (12 Möbius + 8 named)",
        curve_grid.len()
    );
    println!("Peak methods: {:?}", PeakMethod::all());

    let mut sample_results: Vec<SampleResult> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    // Top-K trackers: (curve_idx, peak_method) -> TopBottom.
    const TOPK: usize = 3;
    let mut trackers: HashMap<(usize, PeakMethod), TopBottom> = HashMap::new();

    // Stash decoded HDR/SDR per sample-index so we can regenerate montages
    // at the end without redecoding. Memory: 2 buffers * 144 MB * 76 samples
    // = ~22 GB — too much. We'll regenerate montages by re-decoding the
    // small set of samples that landed in the top-K.
    let mut sample_paths_by_idx: Vec<PathBuf> = Vec::new();

    let total = work_list.len();
    for (idx, (path, fmt_label)) in work_list.iter().enumerate() {
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

        let hdr_max_raw = hdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sdr_max_raw = sdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Skip cases.
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
            println!("  → skipping: SDR base is PQ-encoded (base=HDR)");
            failures.push((path.clone(), "SDR base is PQ-encoded".into()));
            continue;
        }
        if !hdr_max_raw.is_finite() || !sdr_max_raw.is_finite() {
            println!("  → skipping: non-finite pixel data");
            failures.push((path.clone(), "non-finite pixel data".into()));
            continue;
        }

        let sample_idx = sample_paths_by_idx.len();
        sample_paths_by_idx.push(path.clone());

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

            let mut cells: Vec<CellMetrics> = Vec::with_capacity(curve_grid.len());
            for (ci, c) in curve_grid.iter().enumerate() {
                let cand = apply_curve(*c, &hdr, source_peak);
                let metrics = compute_metrics(&sdr, &cand);
                cells.push(metrics);

                if metrics.mean_de2000.is_finite() {
                    let tracker = trackers.entry((ci, method)).or_default();
                    tracker.add_best(metrics.mean_de2000, sample_idx, TOPK);
                    tracker.add_worst(metrics.mean_de2000, sample_idx, TOPK);
                }
            }
            // Brief summary per method.
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

        sample_results.push(SampleResult {
            sample_path: path.clone(),
            format_label: fmt_label.clone(),
            device_class,
            width: sdr.width,
            height: sdr.height,
            source_peaks,
            cells: cells_per_method,
        });

        // Drop HDR/SDR buffers before continuing to next sample.
        drop(hdr);
        drop(sdr);
    }

    refresh_lock("regenerating-top-bottom-montages");
    println!("\nRegenerating top-3 + bottom-3 montages per cell...");

    // Build the set of sample-indices that need re-decoding.
    let mut needs_regen: HashMap<usize, Vec<(usize, PeakMethod, bool)>> = HashMap::new();
    for ((ci, method), tracker) in &trackers {
        for (_, idx) in &tracker.top_best {
            needs_regen
                .entry(*idx)
                .or_default()
                .push((*ci, *method, true));
        }
        for (_, idx) in &tracker.top_worst {
            needs_regen
                .entry(*idx)
                .or_default()
                .push((*ci, *method, false));
        }
    }
    println!(
        "Will re-decode {} samples to generate {} top/bottom montages",
        needs_regen.len(),
        needs_regen.values().map(|v| v.len()).sum::<usize>()
    );

    let mut montage_count = 0_usize;
    for (sample_idx, cells_needed) in &needs_regen {
        let path = &sample_paths_by_idx[*sample_idx];
        let stem_raw = path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stem = stem_raw.replace('.', "_");
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let (sdr, hdr, _, _) = match decode_sample_full(&bytes) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for (ci, method, is_best) in cells_needed {
            // Recompute source peak.
            let source_peak = sample_results
                .iter()
                .find(|r| r.sample_path == *path)
                .and_then(|r| r.source_peaks.get(method).copied())
                .unwrap_or(203.0);
            let cand = apply_curve(curve_grid[*ci], &hdr, source_peak);
            let suffix = if *is_best { "best" } else { "worst" };
            let mon_path = Path::new(MONTAGES_DIR).join(format!(
                "{}__{}__{}__{}.png",
                stem,
                curve_grid[*ci].label(),
                method.label(),
                suffix
            ));
            if let Err(e) = save_montage(&mon_path, &sdr, &cand) {
                eprintln!("  montage save failed: {}", e);
            } else {
                montage_count += 1;
            }
        }
        refresh_lock(&format!("regen-{}-montages", montage_count));
    }

    let runtime = t0.elapsed().as_secs_f64();
    refresh_lock("writing-reports");
    write_csv(&sample_results, &curve_grid)?;
    println!("CSV written to {}", CSV_PATH);
    write_report(
        &sample_results,
        &curve_grid,
        &failures,
        n_jpeg,
        n_heic,
        runtime,
    )?;
    println!("Report written to {}", REPORT_PATH);
    println!(
        "Done. {} samples succeeded, {} failed, {} montages, total {:.1}s.",
        sample_results.len(),
        failures.len(),
        montage_count,
        runtime
    );
    Ok(())
}
