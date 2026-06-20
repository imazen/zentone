//! HDR→SDR tone-map shootout addendum: gainforge curves × color spaces.
//!
//! Companion to `hdr_tone_map_shootout_full.rs` (the 20-cell BT.2390-wins
//! shootout at commit `e876ac2`, report
//! `benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`).
//!
//! Tests whether the [`gainforge`] crate (0.5.0) beats zentone's BT.2390
//! winner. Adds 10 gainforge cells per (sample × peak_method) covering the
//! Aces / Filmic / Itu2408 / FilmicSpline curves in RGB, Jzazbz, and Yrg
//! working color spaces.
//!
//! The point isn't the curves themselves (gainforge's Aces, Filmic, etc.
//! mostly overlap with zentone's) — it's the **working color space** axis:
//! does Jzazbz or Yrg mapping move the needle vs RGB per-channel?
//!
//! Pipeline (per cell):
//!   1. Decode SDR base + reconstruct HDR (same as `_full.rs`).
//!   2. PQ-encode HDR f32 → u8 in BT.2020 primaries (gainforge's expected
//!      input frame).
//!   3. Build gainforge `create_tone_mapper_rgb(bt2020_pq, srgb, method,
//!      MappingColorSpace::*)` → run `tonemap_lane(&pq_u8, &mut srgb_u8)`.
//!   4. Decode the sRGB u8 output → linear-sRGB f32.
//!   5. Compare against the SDR base in the same byte-sRGB / Lab spaces as
//!      `_full.rs` (PSNR, mean ΔE2000, max |Δ|, %>ΔE5).
//!
//! Side-car CSV: `benchmarks/hdr_tone_map_shootout_full_2026-06-20_gainforge.csv`.
//! Addendum section appended to `hdr_tone_map_shootout_full_2026-06-20.md`.
//! Top-3 ΔE2000 montages per cell to `/mnt/v/output/zentone/shootout-full-gainforge/`
//! (capped at 30 total).
//!
//! Run:
//! ```text
//! nice -n19 cargo run -p zentone --release \
//!   --example hdr_tone_map_shootout_gainforge --features hdr-shootout
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use gainforge::{
    CommonToneMapperParameters, FilmicSplineParameters, GainHdrMetadata, GamutClipping,
    JzazbzToneMapperParameters, MappingColorSpace, RgbToneMapperParameters, ToneMappingMethod,
    create_tone_mapper_rgb,
};
use moxcms::ColorProfile;
use rayon::prelude::*;
use zencodecs::{DecodeRequest, OrientationHint};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_dev::hdr::ContentLightLevel;

// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const MONTAGES_DIR: &str = "/mnt/v/output/zentone/shootout-full-gainforge";
const REPORT_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20.md";
const CSV_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20_gainforge.csv";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";

// Save every tonemap candidate (76 samples × 30 cells = 2,280 PNGs) to disk
// at a downscaled resolution. Reference SDR saved once per sample. Bounded
// disk usage: ~2.5 GB at 1024-px wide.
const TONEMAP_THUMBNAIL_W: u32 = 1024;

// SDR diffuse-white anchor: our linear HDR buffer is anchored at 1.0 = 203 nits.
const DIFFUSE_WHITE_NITS: f32 = 203.0;
const TARGET_PEAK_NITS: f32 = 100.0;

// =========================================================================
// Peak-method enum (mirror of `_full.rs`)
// =========================================================================

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
    let iso = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "ts=?".to_string());
    let line = format!("{} claude-shootout-gainforge {}\n", iso, activity);
    let _ = fs::write(WORKONGOING, line);
}

// =========================================================================
// Sample-finding + decode (copied from `_full.rs`)
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

fn detect_gainmap(bytes: &[u8]) -> Option<&'static str> {
    let info = DecodeRequest::new(bytes).probe().ok()?;
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

fn decode_sample_full(
    bytes: &[u8],
) -> anyhow::Result<(LinearRgb, LinearRgb, TransferFunction, TransferFunction)> {
    let sdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR: {}", e.error()))?;
    let sdr_desc = sdr_out.pixels().descriptor();
    let sdr_buf = sdr_out.into_buffer();
    let sdr_rgb = pixel_buffer_to_linear_rgb(&sdr_buf)?;

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
            "SDR/HDR dim mismatch: SDR {}x{}, HDR {}x{}",
            sdr_rgb.width,
            sdr_rgb.height,
            hdr_rgb.width,
            hdr_rgb.height,
        );
    }

    Ok((sdr_rgb, hdr_rgb, sdr_desc.transfer(), hdr_desc.transfer()))
}

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
// Gainforge cells
// =========================================================================

#[derive(Clone, Copy, Debug)]
enum CellMethod {
    Aces,
    Filmic,
    Itu2408,
    FilmicSpline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellWcs {
    Rgb,
    Jzazbz,
    Yrg,
}

impl CellWcs {
    fn label(&self) -> &'static str {
        match self {
            CellWcs::Rgb => "rgb",
            CellWcs::Jzazbz => "jzazbz",
            CellWcs::Yrg => "yrg",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GainforgeCell {
    method: CellMethod,
    wcs: CellWcs,
}

impl GainforgeCell {
    fn label(&self) -> String {
        let m = match self.method {
            CellMethod::Aces => "aces",
            CellMethod::Filmic => "filmic",
            CellMethod::Itu2408 => "itu2408",
            CellMethod::FilmicSpline => "filmic_spline",
        };
        format!("gainforge_{}_{}", m, self.wcs.label())
    }
}

fn build_gainforge_grid() -> Vec<GainforgeCell> {
    use CellMethod::*;
    use CellWcs::*;
    vec![
        // Tier 1: Aces × {Rgb, Jzazbz, Yrg}
        GainforgeCell {
            method: Aces,
            wcs: Rgb,
        },
        GainforgeCell {
            method: Aces,
            wcs: Jzazbz,
        },
        GainforgeCell {
            method: Aces,
            wcs: Yrg,
        },
        // Tier 1: Filmic × {Rgb, Jzazbz, Yrg}
        GainforgeCell {
            method: Filmic,
            wcs: Rgb,
        },
        GainforgeCell {
            method: Filmic,
            wcs: Jzazbz,
        },
        GainforgeCell {
            method: Filmic,
            wcs: Yrg,
        },
        // Tier 1: Itu2408 × {Rgb, Jzazbz} (Yrg known slow + redundant for Itu2408)
        GainforgeCell {
            method: Itu2408,
            wcs: Rgb,
        },
        GainforgeCell {
            method: Itu2408,
            wcs: Jzazbz,
        },
        // Tier 1: FilmicSpline × {Rgb, Jzazbz}
        GainforgeCell {
            method: FilmicSpline,
            wcs: Rgb,
        },
        GainforgeCell {
            method: FilmicSpline,
            wcs: Jzazbz,
        },
    ]
}

// =========================================================================
// PQ encode/decode helpers
// =========================================================================

// SMPTE ST.2084 (PQ) constants (BT.2100).
const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = (2523.0 / 4096.0) * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = (2413.0 / 4096.0) * 32.0;
const PQ_C3: f32 = (2392.0 / 4096.0) * 32.0;

/// Linear "PQ-absolute" [0,1] (1.0 = 10000 nits) → PQ-encoded [0,1].
#[inline]
fn pq_from_linear_abs(linear: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }
    let x = linear.clamp(0.0, 1.0);
    let xpo = x.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * xpo;
    let den = 1.0 + PQ_C3 * xpo;
    (num / den).powf(PQ_M2)
}

/// Encode our HDR linear RGB f32 (anchored 1.0 = 203 nits) → PQ-encoded u8
/// for the **standard reference-display path** (Aces, Filmic, FilmicSpline,
/// Reinhard, ReinhardJodie, Agx, Clamp).
///
/// gainforge's u8 factory for non-Itu2408 curves uses
/// `reference_display = 10000 / SDR_WHITE_NITS = 10000 / 203 ≈ 49.26`. This
/// makes the LUT `linear_table_r[u8] = pq_unscaled(u8/255) * 49.26`, so that
/// SDR white (203 nits → `pq_encode(203/10000)` u8) decodes to linear 1.0 in
/// the working color space. We therefore PQ-encode such that 255 u8 = 10000
/// nits standard PQ:
///   1. Convert to PQ-absolute frame: scale by `203/10000 = 0.0203` so
///      1.0 (our SDR white) → 0.0203 (1.0 = 10000 nits).
///   2. Apply PQ OETF.
///   3. Quantize to u8.
fn hdr_to_pq_u8_standard(hdr: &LinearRgb) -> Vec<u8> {
    let abs_scale = DIFFUSE_WHITE_NITS / 10000.0; // 203/10000 ≈ 0.0203
    let mut out = vec![0u8; hdr.px.len()];
    out.par_chunks_mut(4096)
        .zip(hdr.px.par_chunks(4096))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let abs = (s * abs_scale).max(0.0).min(1.0);
                let pq = pq_from_linear_abs(abs);
                *d = (pq * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        });
    out
}

/// Encode our HDR linear RGB f32 (anchored 1.0 = 203 nits) → PQ-encoded u8
/// for the **Itu2408 path**.
///
/// gainforge's u8 factory for `Itu2408` uses `reference_display = 1.0`, so
/// the LUT `linear_table_r[u8] = pq_unscaled(u8/255) * 1.0` (range [0,1]).
/// The curve then multiplies by `content_max_brightness` to recover nits:
/// `luma_nits = pq_unscaled(u8/255) * content_max_brightness`.
///
/// So for our 1000-nit highlight to register correctly as 1000 nits, we need
/// `pq_unscaled(u8/255) = 1000/content_max_brightness`, i.e. PQ-encode in the
/// **content-relative** frame where 255 u8 = `content_max_brightness` nits.
///
/// Steps:
///   1. Convert each pixel to nits: `px * 203`.
///   2. Divide by `content_max_brightness` to get [0,1] content-relative.
///   3. Apply PQ OETF.
///   4. Quantize to u8.
fn hdr_to_pq_u8_itu2408(hdr: &LinearRgb, content_max_brightness: f32) -> Vec<u8> {
    let scale = DIFFUSE_WHITE_NITS / content_max_brightness.max(1.0);
    let mut out = vec![0u8; hdr.px.len()];
    out.par_chunks_mut(4096)
        .zip(hdr.px.par_chunks(4096))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let rel = (s * scale).max(0.0).min(1.0);
                let pq = pq_from_linear_abs(rel);
                *d = (pq * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        });
    out
}

/// Decode gainforge sRGB-u8 output → linear-sRGB f32 anchored at 1.0 = SDR
/// display white (100 nits).
fn srgb_u8_to_linear_rgb(srgb_u8: &[u8], width: u32, height: u32) -> LinearRgb {
    let mut px = vec![0.0_f32; srgb_u8.len()];
    px.par_chunks_mut(4096)
        .zip(srgb_u8.par_chunks(4096))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let e = s as f32 / 255.0;
                *d = linear_srgb::tf::srgb_to_linear(e);
            }
        });
    LinearRgb { width, height, px }
}

// =========================================================================
// Apply a gainforge cell
// =========================================================================

fn make_method(cell: CellMethod, source_peak_nits: f32) -> ToneMappingMethod {
    match cell {
        CellMethod::Aces => ToneMappingMethod::Aces,
        CellMethod::Filmic => ToneMappingMethod::Filmic,
        CellMethod::Itu2408 => {
            ToneMappingMethod::Itu2408(GainHdrMetadata::new(source_peak_nits, TARGET_PEAK_NITS))
        }
        CellMethod::FilmicSpline => {
            ToneMappingMethod::FilmicSpline(FilmicSplineParameters::default())
        }
    }
}

/// Build the working-color-space parameters.
///
/// For Jzazbz, the `content_brightness` parameter is the
/// `display_luminance` in `Jzazbz::from_xyz_with_display_luminance(xyz, ·)`
/// which multiplies `xyz` to get absolute-nit XYZ. So it must be the nit
/// value that chunk=1.0 represents in the post-LUT working space:
///
/// - **Non-Itu2408 curves**: gainforge's LUT scales by
///   `reference_display = 10000/203 ≈ 49.26`, so a PQ u8 representing
///   `n` nits decodes to `chunk = n/203` in the working space — i.e.
///   chunk=1.0 = 203 nits.
/// - **Itu2408 curve**: gainforge's LUT scales by `reference_display = 1.0`,
///   and our `hdr_to_pq_u8_itu2408` encodes such that
///   chunk=1.0 = `content_max_brightness`, which we pin to
///   `source_peak_nits`.
fn make_wcs(wcs: CellWcs, method: CellMethod, source_peak_nits: f32) -> MappingColorSpace {
    let common = CommonToneMapperParameters {
        exposure: 1.0,
        gamut_clipping: GamutClipping::Clip,
    };
    match wcs {
        CellWcs::Rgb => MappingColorSpace::Rgb(RgbToneMapperParameters {
            exposure: 1.0,
            gamut_clipping: GamutClipping::Clip,
        }),
        CellWcs::Yrg => MappingColorSpace::Yrg(common),
        CellWcs::Jzazbz => {
            let content_brightness = match method {
                CellMethod::Itu2408 => source_peak_nits.max(203.0),
                _ => DIFFUSE_WHITE_NITS,
            };
            MappingColorSpace::Jzazbz(JzazbzToneMapperParameters {
                content_brightness,
                exposure: 1.0,
                gamut_clipping: GamutClipping::Clip,
            })
        }
    }
}

fn apply_gainforge_cell(
    cell: GainforgeCell,
    hdr: &LinearRgb,
    pq_u8_standard: &[u8],
    source_peak_nits: f32,
) -> anyhow::Result<LinearRgb> {
    let bt2020_pq = ColorProfile::new_bt2020_pq();
    let srgb = ColorProfile::new_srgb();
    let method = make_method(cell.method, source_peak_nits);
    let wcs = make_wcs(cell.wcs, cell.method, source_peak_nits);
    let mapper = create_tone_mapper_rgb(&bt2020_pq, &srgb, method, wcs)
        .map_err(|e| anyhow::anyhow!("create_tone_mapper_rgb: {:?}", e))?;

    // Itu2408 uses a different reference_display in its u8 LUT (1.0 instead
    // of 10000/203) — its input frame is content-relative PQ, not standard
    // 10000-nit-anchored PQ. Re-encode per call.
    let needs_itu2408_frame = matches!(cell.method, CellMethod::Itu2408);
    let pq_u8_itu2408_storage;
    let pq_u8: &[u8] = if needs_itu2408_frame {
        pq_u8_itu2408_storage = hdr_to_pq_u8_itu2408(hdr, source_peak_nits);
        &pq_u8_itu2408_storage
    } else {
        pq_u8_standard
    };

    let mut dst = vec![0u8; pq_u8.len()];

    // Tonemap one row at a time so we can rayon-parallelize. The mapper is
    // Send+Sync (it's `Arc<dyn ToneMapper + Send + Sync>`).
    let row_bytes = hdr.width as usize * 3;
    let mapper_ref = &mapper;
    let result: Result<(), gainforge::ForgeError> = dst
        .par_chunks_mut(row_bytes)
        .zip(pq_u8.par_chunks(row_bytes))
        .try_for_each(|(d, s)| mapper_ref.tonemap_lane(s, d));
    result.map_err(|e| anyhow::anyhow!("tonemap_lane: {:?}", e))?;

    Ok(srgb_u8_to_linear_rgb(&dst, hdr.width, hdr.height))
}

// =========================================================================
// Metrics (copied from `_full.rs`)
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
// Montage (copied from `_full.rs`)
// =========================================================================

/// Downscale + encode a linear-RGB f32 buffer to an 8-bit sRGB PNG. Saves a
/// 1024-wide preview suitable for visual review without exploding disk usage.
/// Box-filter downscale (cheap; we're not after pixel-perfect resampling here).
fn save_tonemap_thumbnail(path: &Path, lin: &LinearRgb) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let target_w = TONEMAP_THUMBNAIL_W.min(lin.width);
    let scale = lin.width as f32 / target_w as f32;
    let target_h = ((lin.height as f32 / scale).round() as u32).max(1);

    let mut out = vec![0u8; (target_w as usize) * (target_h as usize) * 3];
    let src_w = lin.width as usize;
    out.par_chunks_mut(target_w as usize * 3)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy0 = ((dy as f32) * scale) as usize;
            let sy1 = (((dy as f32) + 1.0) * scale) as usize;
            let sy1 = sy1.min(lin.height as usize).max(sy0 + 1);
            for dx in 0..target_w as usize {
                let sx0 = ((dx as f32) * scale) as usize;
                let sx1 = (((dx as f32) + 1.0) * scale) as usize;
                let sx1 = sx1.min(src_w).max(sx0 + 1);
                let mut acc = [0.0_f32; 3];
                let mut n = 0_f32;
                for sy in sy0..sy1 {
                    let row_base = sy * src_w * 3;
                    for sx in sx0..sx1 {
                        let i = row_base + sx * 3;
                        acc[0] += lin.px[i];
                        acc[1] += lin.px[i + 1];
                        acc[2] += lin.px[i + 2];
                        n += 1.0;
                    }
                }
                let inv_n = if n > 0.0 { 1.0 / n } else { 0.0 };
                let r = acc[0] * inv_n;
                let g = acc[1] * inv_n;
                let b = acc[2] * inv_n;
                row[dx * 3] = linear_to_srgb_u8(r);
                row[dx * 3 + 1] = linear_to_srgb_u8(g);
                row[dx * 3 + 2] = linear_to_srgb_u8(b);
            }
        });

    image::save_buffer(path, &out, target_w, target_h, image::ColorType::Rgb8)?;
    Ok(())
}

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
// Per-sample results
// =========================================================================

#[derive(Clone)]
struct SampleResult {
    sample_path: PathBuf,
    format_label: String,
    /// Per peak method: measured source peak nits.
    source_peaks: HashMap<PeakMethod, f32>,
    /// Per peak method × cell idx (parallel to grid): metrics.
    cells: HashMap<PeakMethod, Vec<CellMetrics>>,
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
// CSV writer
// =========================================================================

fn write_csv(samples: &[SampleResult], grid: &[GainforgeCell]) -> anyhow::Result<()> {
    fs::create_dir_all(Path::new(CSV_PATH).parent().unwrap())?;
    let f = fs::File::create(CSV_PATH)?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "sample,format,peak_method,curve,wcs,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_above_de5"
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
            for (ci, c) in grid.iter().enumerate() {
                if ci >= cells.len() {
                    continue;
                }
                let cell = cells[ci];
                writeln!(
                    w,
                    "{},{},{},{},{},{:.1},{:.3},{:.4},{:.5},{:.3}",
                    stem,
                    r.format_label,
                    m.label(),
                    c.label(),
                    c.wcs.label(),
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

// =========================================================================
// Per-cell summary
// =========================================================================

fn per_cell_summary(
    samples: &[SampleResult],
    grid: &[GainforgeCell],
    method: PeakMethod,
) -> Vec<(String, &'static str, f32, f32, f32, f32)> {
    let mut out = Vec::new();
    for (ci, c) in grid.iter().enumerate() {
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
            c.wcs.label(),
            median(psnr),
            median(de),
            median(max_a),
            median(pct5),
        ));
    }
    out.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// =========================================================================
// Report addendum
// =========================================================================

/// BT.2390 reference numbers from `hdr_tone_map_shootout_full_2026-06-20.md`
/// (commit `e876ac2`). Median ΔE2000 across 76 samples.
const BT2390_DE_MAX: f32 = 8.342;
const BT2390_DE_ROBUST: f32 = 6.089;
const BT2390_DE_SMOOTHED: f32 = 8.334;

fn append_report(
    sample_results: &[SampleResult],
    grid: &[GainforgeCell],
    failures: &[(PathBuf, String)],
    runtime_secs: f64,
) -> anyhow::Result<()> {
    let mut s = String::new();
    s.push_str("\n\n---\n\n");
    s.push_str("## gainforge curve sweep (addendum, 2026-06-20)\n\n");
    s.push_str(&format!(
        "**Tested**: {} cells × {} peak methods × {} samples = {} cell-evaluations. \
Runtime: {:.1}s ({:.1}min). gainforge version: 0.5.0 (`Itu2408` formerly `Rec2408`).\n\n",
        grid.len(),
        PeakMethod::all().len(),
        sample_results.len(),
        grid.len() * PeakMethod::all().len() * sample_results.len(),
        runtime_secs,
        runtime_secs / 60.0,
    ));
    s.push_str("**Goal**: test whether gainforge's `MappingColorSpace::{Jzazbz, Yrg}` color-space mapping beats zentone's per-RGB-channel tone-map (the BT.2390 winner of the main shootout, median ΔE2000 6.089 under `measure_robust`).\n\n");

    s.push_str("**Pipeline**: HDR linear f32 (anchored 1.0 = 203 nits) → PQ-encoded u8 → `gainforge::create_tone_mapper_rgb(BT2020-PQ → sRGB, method, wcs).tonemap_lane(pq_u8, srgb_u8)` → u8 sRGB → linearize → diff against the producer SDR base. PQ encoding uses two frames per gainforge's internal LUT convention:\n");
    s.push_str("- **Aces / Filmic / FilmicSpline**: `reference_display = 10000/203 ≈ 49.26` in gainforge's LUT, so PQ-encode in standard PQ-absolute frame (255 u8 = 10000 nits, our SDR-white u8 ≈ 148).\n");
    s.push_str("- **Itu2408**: `reference_display = 1.0` in gainforge's LUT, so PQ-encode in content-relative frame (255 u8 = `content_max_brightness` nits). Re-encoded per peak-method since `content_max_brightness = source_peak_nits`.\n\n");
    s.push_str("The u8 round-trip is necessary because gainforge's Yrg/Jzazbz factories don't expose a clean f32-linear-RGB entry point; the u8 PQ path is canonical and the quantization (256 PQ-encoded steps) is well below the ΔE differences being measured (>5).\n\n");

    if !failures.is_empty() {
        s.push_str(&format!(
            "**Failures**: {} of {} candidate files failed. ",
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
            } else if why.contains("tonemap") {
                "gainforge tonemap error"
            } else {
                "other"
            };
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
        entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (k, c) in entries {
            s.push_str(&format!("{} = {}; ", k, c));
        }
        s.push_str("\n\n");
    }

    // Per-method tables (cell × method, median ΔE2000 ascending).
    s.push_str("### Per-cell summary by peak method\n\n");
    for method in PeakMethod::all() {
        let bt2390 = match method {
            PeakMethod::Max => BT2390_DE_MAX,
            PeakMethod::Robust => BT2390_DE_ROBUST,
            PeakMethod::MaxSmoothed => BT2390_DE_SMOOTHED,
        };
        s.push_str(&format!(
            "#### `{}` — sorted by median ΔE2000 ascending (BT.2390 reference: {:.3})\n\n",
            method.label(),
            bt2390
        ));
        s.push_str("| Rank | Cell | WCS | Median PSNR (dB) | Median ΔE2000 | Δ vs BT.2390 | Median max\\|Δ\\| | Median %>ΔE5 |\n");
        s.push_str("|------|------|-----|------------------|---------------|--------------|-----------------|---------------|\n");
        let sum = per_cell_summary(sample_results, grid, *method);
        for (rank, (label, wcs, psnr, de, max_abs, pct5)) in sum.iter().enumerate() {
            let delta = de - bt2390;
            let delta_str = if delta < 0.0 {
                format!("**{:+.3} (BEATS)**", delta)
            } else {
                format!("{:+.3}", delta)
            };
            s.push_str(&format!(
                "| {} | `{}` | {} | {:.2} | {:.3} | {} | {:.4} | {:.2} |\n",
                rank + 1,
                label,
                wcs,
                psnr,
                de,
                delta_str,
                max_abs,
                pct5,
            ));
        }
        s.push('\n');
    }

    // Color-space effect analysis
    s.push_str("### Color-space effect: RGB vs Jzazbz vs Yrg\n\n");
    s.push_str("For each curve family, median ΔE2000 under `measure_robust` per working color space (lower is closer to producer SDR).\n\n");
    s.push_str("| Curve | RGB | Jzazbz | Yrg | Best WCS | Δ(RGB→best) |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    let methods_to_show = [
        ("aces", CellMethod::Aces),
        ("filmic", CellMethod::Filmic),
        ("itu2408", CellMethod::Itu2408),
        ("filmic_spline", CellMethod::FilmicSpline),
    ];
    for (m_label, m) in &methods_to_show {
        let de_for = |wcs: CellWcs| -> Option<f32> {
            let ci = grid.iter().position(|c| {
                std::mem::discriminant(&c.method) == std::mem::discriminant(m) && c.wcs == wcs
            })?;
            let de: Vec<f32> = sample_results
                .iter()
                .filter_map(|r| {
                    r.cells
                        .get(&PeakMethod::Robust)
                        .and_then(|cells| cells.get(ci))
                })
                .map(|m| m.mean_de2000)
                .collect();
            let m = median(de);
            if m.is_finite() { Some(m) } else { None }
        };
        let rgb = de_for(CellWcs::Rgb);
        let jzazbz = de_for(CellWcs::Jzazbz);
        let yrg = de_for(CellWcs::Yrg);
        let candidates: Vec<(f32, &str)> = [
            rgb.map(|v| (v, "RGB")),
            jzazbz.map(|v| (v, "Jzazbz")),
            yrg.map(|v| (v, "Yrg")),
        ]
        .into_iter()
        .flatten()
        .collect();
        let best = candidates
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let best_label = best.map(|x| x.1).unwrap_or("?");
        let delta = match (rgb, best) {
            (Some(r), Some((b, _))) => format!("{:+.3}", b - r),
            _ => "n/a".into(),
        };
        let fmt = |v: Option<f32>| -> String {
            v.map(|x| format!("{:.3}", x))
                .unwrap_or_else(|| "n/a".into())
        };
        s.push_str(&format!(
            "| `{}` | {} | {} | {} | **{}** | {} |\n",
            m_label,
            fmt(rgb),
            fmt(jzazbz),
            fmt(yrg),
            best_label,
            delta,
        ));
    }
    s.push('\n');

    // Headline verdict
    s.push_str("### Verdict\n\n");
    let robust_sum = per_cell_summary(sample_results, grid, PeakMethod::Robust);
    if let Some((best_label, best_wcs, _, best_de, _, _)) = robust_sum.first() {
        if *best_de < BT2390_DE_ROBUST {
            s.push_str(&format!(
                "**gainforge winner**: `{}` (WCS={}) at median ΔE2000 {:.3} under `measure_robust` — **beats BT.2390 ({:.3}) by {:.3}**.\n\n",
                best_label,
                best_wcs,
                best_de,
                BT2390_DE_ROBUST,
                BT2390_DE_ROBUST - best_de,
            ));
        } else {
            s.push_str(&format!(
                "**gainforge winner**: `{}` (WCS={}) at median ΔE2000 {:.3} under `measure_robust` — does NOT beat BT.2390 ({:.3}); deficit {:.3}.\n\n",
                best_label,
                best_wcs,
                best_de,
                BT2390_DE_ROBUST,
                best_de - BT2390_DE_ROBUST,
            ));
        }
    }
    // Color-space recommendation
    let mut wcs_wins: HashMap<&'static str, u32> = HashMap::new();
    for (_, m) in &methods_to_show {
        let candidates = [CellWcs::Rgb, CellWcs::Jzazbz, CellWcs::Yrg]
            .iter()
            .filter_map(|wcs| {
                let ci = grid.iter().position(|c| {
                    std::mem::discriminant(&c.method) == std::mem::discriminant(m) && c.wcs == *wcs
                })?;
                let de = median(
                    sample_results
                        .iter()
                        .filter_map(|r| {
                            r.cells
                                .get(&PeakMethod::Robust)
                                .and_then(|cells| cells.get(ci))
                        })
                        .map(|m| m.mean_de2000)
                        .collect::<Vec<_>>(),
                );
                if de.is_finite() {
                    Some((de, wcs.label()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if let Some((_, lab)) = candidates
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        {
            *wcs_wins.entry(*lab).or_insert(0) += 1;
        }
    }
    let mut wcs_rank: Vec<(&&str, &u32)> = wcs_wins.iter().collect();
    wcs_rank.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    if !wcs_rank.is_empty() {
        s.push_str("**Color-space ranking (count of curves where each WCS is best)**: ");
        for (i, (wcs, n)) in wcs_rank.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{} ({})", wcs, n));
        }
        s.push_str(".\n\n");
    }

    s.push_str(&format!(
        "**HdrToSdr default recommendation**: see the `Δ vs BT.2390` column under `measure_robust` above. If no gainforge cell shows a `BEATS` marker, keep the BT.2390 default. If one or more do, the Jzazbz/Yrg color-space path may be worth pulling into zentone (separate conversation — would require zentone to learn the moxcms RGB↔Jzazbz/Yrg conversions, not just adopt a new curve).\n\n",
    ));

    s.push_str(&format!(
        "**Tonemap thumbnails** ({}-pixel-wide PNG previews) saved under [`{}/`](http://172.23.240.1:3300/zentone/shootout-full-gainforge/). Naming: `<sample-stem>__REF.png` (the producer-SDR reference) and `<sample-stem>__<cell_label>__<peak_method>.png` (each tonemap candidate). All cells × samples saved inline during the per-sample sweep (no end-of-run re-decode pass; bounded peak memory).\n\n",
        TONEMAP_THUMBNAIL_W, MONTAGES_DIR
    ));

    s.push_str("### Reproduce\n\n```bash\n");
    s.push_str("nice -n19 cargo run -p zentone --release \\\n");
    s.push_str("  --example hdr_tone_map_shootout_gainforge --features hdr-shootout\n```\n\n");
    s.push_str(&format!(
        "Side-car CSV: `{}`.\nSource: `examples/hdr_tone_map_shootout_gainforge.rs`.\n",
        CSV_PATH
    ));

    // Read the existing report and check if the addendum was already appended.
    let existing = fs::read_to_string(REPORT_PATH)?;
    let marker = "## gainforge curve sweep (addendum,";
    let mut new = if let Some(idx) = existing.find(marker) {
        // Remove the previous addendum: keep everything up to the closest "\n\n---\n\n" before it.
        let pre = &existing[..idx];
        let stripped = pre.trim_end_matches(|c: char| c.is_whitespace() || c == '-');
        format!("{}\n", stripped.trim_end())
    } else {
        existing.trim_end().to_string()
    };
    new.push_str(&s);
    fs::write(REPORT_PATH, new)?;
    Ok(())
}

// =========================================================================
// Top-K per cell tracker
// =========================================================================

#[derive(Default)]
struct TopK {
    /// `(de2000, sample_idx)` — sorted ascending (best ΔE2000 first).
    top: Vec<(f32, usize)>,
}

impl TopK {
    fn add(&mut self, de: f32, idx: usize, k: usize) {
        if !de.is_finite() {
            return;
        }
        let pos = self.top.partition_point(|(d, _)| *d <= de);
        self.top.insert(pos, (de, idx));
        if self.top.len() > k {
            self.top.truncate(k);
        }
    }
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

    let grid = build_gainforge_grid();
    println!("Gainforge cells under test: {}", grid.len());
    for (i, c) in grid.iter().enumerate() {
        println!("  [{}] {}", i + 1, c.label());
    }
    println!("Peak methods: {:?}", PeakMethod::all());

    let mut sample_results: Vec<SampleResult> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    let mut sample_paths_by_idx: Vec<PathBuf> = Vec::new();

    // Streaming CSV: write rows as they're computed so partial results
    // survive crashes / kills. Header first.
    if let Some(parent) = Path::new(CSV_PATH).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut csv = std::io::BufWriter::new(fs::File::create(CSV_PATH)?);
    use std::io::Write as _;
    writeln!(
        csv,
        "sample,format,cell,peak_method,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_de_gt_5"
    )?;
    csv.flush()?;

    let total = work_list.len();
    for (idx, (path, fmt_label)) in work_list.iter().enumerate() {
        let stem_raw = path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stem = stem_raw.replace('.', "_");

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

        let (sdr, hdr, sdr_descr, _hdr_descr) = match decode_sample_full(&bytes) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  decode failed: {}", e);
                failures.push((path.clone(), format!("decode: {}", e)));
                continue;
            }
        };
        println!("  SDR/HDR: {}x{}", sdr.width, sdr.height);

        let hdr_max_raw = hdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sdr_max_raw = sdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if hdr_max_raw < 1.05 && sdr_max_raw < 1.05 && (hdr_max_raw - sdr_max_raw).abs() < 0.05 {
            println!("  → skipping: no HDR boost");
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
            println!("  → skipping: SDR base is PQ-encoded");
            failures.push((path.clone(), "SDR base is PQ-encoded".into()));
            continue;
        }
        if !hdr_max_raw.is_finite() || !sdr_max_raw.is_finite() {
            failures.push((path.clone(), "non-finite pixel data".into()));
            continue;
        }

        let sample_idx = sample_paths_by_idx.len();
        sample_paths_by_idx.push(path.clone());

        // Encode HDR → PQ-u8 once for the standard (non-Itu2408) path;
        // Itu2408 needs its own per-call encoding (different reference_display).
        let pq_u8_standard = hdr_to_pq_u8_standard(&hdr);

        // Save the producer-SDR reference thumbnail once per sample.
        let ref_path = Path::new(MONTAGES_DIR).join(format!("{}__REF.png", stem));
        if let Err(e) = save_tonemap_thumbnail(&ref_path, &sdr) {
            eprintln!("  ref thumbnail save failed: {}", e);
        }

        let mut source_peaks: HashMap<PeakMethod, f32> = HashMap::new();
        let mut cells_per_method: HashMap<PeakMethod, Vec<CellMetrics>> = HashMap::new();

        for &method in PeakMethod::all() {
            let source_peak = measure_source_peak_nits(&hdr, method);
            source_peaks.insert(method, source_peak);
            println!("  [{}] source_peak={:.0} nits", method.label(), source_peak,);

            let mut cells: Vec<CellMetrics> = Vec::with_capacity(grid.len());
            for cell in grid.iter() {
                let metrics = match apply_gainforge_cell(*cell, &hdr, &pq_u8_standard, source_peak)
                {
                    Ok(cand) => {
                        let m = compute_metrics(&sdr, &cand);
                        // Save every candidate tonemap thumbnail inline so we have
                        // the full visual record AND don't have to re-decode at
                        // the end. Failure is non-fatal — log + continue.
                        let mon_path = Path::new(MONTAGES_DIR).join(format!(
                            "{}__{}__{}.png",
                            stem,
                            cell.label(),
                            method.label(),
                        ));
                        if let Err(e) = save_tonemap_thumbnail(&mon_path, &cand) {
                            eprintln!("  thumbnail save failed for {}: {}", mon_path.display(), e);
                        }
                        drop(cand);
                        m
                    }
                    Err(e) => {
                        eprintln!("    cell {} failed: {}", cell.label(), e);
                        CellMetrics {
                            psnr_db: f32::NAN,
                            mean_de2000: f32::NAN,
                            max_abs_delta: f32::NAN,
                            pct_de_gt_5: f32::NAN,
                        }
                    }
                };
                cells.push(metrics);

                // Stream CSV row immediately (durable across crashes).
                writeln!(
                    csv,
                    "{},{},{},{},{:.2},{:.4},{:.4},{:.4},{:.4}",
                    stem_raw,
                    fmt_label,
                    cell.label(),
                    method.label(),
                    source_peak,
                    metrics.psnr_db,
                    metrics.mean_de2000,
                    metrics.max_abs_delta,
                    metrics.pct_de_gt_5,
                )?;
            }
            csv.flush()?;
            println!(
                "    [{}] {} cells computed (e.g. {} ΔE2000={:.3})",
                method.label(),
                cells.len(),
                grid[0].label(),
                cells[0].mean_de2000,
            );
            cells_per_method.insert(method, cells);
        }

        sample_results.push(SampleResult {
            sample_path: path.clone(),
            format_label: fmt_label.clone(),
            source_peaks,
            cells: cells_per_method,
        });

        drop(hdr);
        drop(sdr);
        drop(pq_u8_standard);
    }
    csv.flush()?;

    // All tonemap thumbnails were saved inline during the per-sample loop.
    // Count them for the final summary.
    let thumbnail_count = match fs::read_dir(MONTAGES_DIR) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("png"))
            .count(),
        Err(_) => 0,
    };

    let runtime = t0.elapsed().as_secs_f64();
    refresh_lock("writing-reports");
    println!(
        "Streamed CSV at {} ({} rows × ~9 columns).",
        CSV_PATH,
        sample_results.len() * grid.len() * 3
    );
    append_report(&sample_results, &grid, &failures, runtime)?;
    println!("Addendum appended to {}", REPORT_PATH);
    println!(
        "Done. {} samples succeeded, {} failed, {} thumbnails saved, total {:.1}s.",
        sample_results.len(),
        failures.len(),
        thumbnail_count,
        runtime
    );
    let _ = n_jpeg;
    let _ = n_heic;
    Ok(())
}
