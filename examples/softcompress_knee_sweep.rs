//! SoftCompress `knee` sweep — empirically calibrate the OKLch chroma rolloff
//! default (2026-06-23).
//!
//! Holds the tone-map curve + peak method fixed at the audited shootout's
//! winner (`Bt2446A × measure_max`) and varies only the `SoftCompress` `knee`
//! parameter across 13 values. For each (sample, knee) the driver records:
//!
//!  * `pixels_clipped_pct` — fraction of pixels whose pre-clamp linear BT.709
//!    RGB exceeded `[0, 1]` in any channel. The final `clamp(0, 1)` would
//!    posterize these — the rolloff exists to bring them under that bar.
//!  * `clip_p50 / p90 / p95 / p99` — quantiles of the worst-channel overshoot
//!    magnitude (`max(0, |ch| - 1)`). Tells us how *far* past clip the worst
//!    pixels fell.
//!  * `chroma_compression_pct` — mean OKLch C reduction vs. `knee=1.0`
//!    (no-rolloff baseline). The cost of the rolloff in chroma terms.
//!  * `peak_chroma_compression` — worst-case OKLch C reduction.
//!  * `mean_de2000` / `max_de2000` — Lab D65 ΔE2000 against the same-pipeline-
//!    but-`knee=1.0` output. The perceptual cost.
//!
//! Run:
//! ```text
//! nice -n19 ionice -c3 cargo run --release \
//!   --example softcompress_knee_sweep --features hdr-shootout
//! ```
//!
//! Output: `benchmarks/softcompress_knee_sweep_2026-06-23.csv`.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use zencodecs::{DecodeRequest, OrientationHint};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::gamut as zgamut;
use zenpixels_convert::hdr::{Bt2446A, CllMeasure, LightLevelMethod, SoftCompress};
use zenpixels_convert::oklab as zoklab;
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_dev::hdr::ContentLightLevel;
use zenpixels_dev::ColorPrimaries;

// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const CSV_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/softcompress_knee_sweep_2026-06-23.csv";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";
const SCHEMA_VERSION: &str = "2026-06-23-knee-sweep";

/// 5 GB memory contract (same as audited shootout). 5 outer threads × ~600 MB
/// active sample = ~3 GB sustained + decode spikes. 8 threads was the brief's
/// suggestion but at 24 MP that overshoots; 5 stays in-budget per the prior
/// audited-shootout memory math (see `hdr_tone_map_shootout_audited.rs:73-86`).
const OUTER_THREADS: usize = 5;

/// Knee values to sweep. Dense around the 0.85-0.96 region where the rolloff
/// crossover is expected to live.
const KNEE_GRID: &[f32] = &[
    0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00,
];

/// Clip-magnitude histogram: 1000 bins of width 0.001 covering [0, 1).
/// Worst-case overshoot in production is bounded by the BT.2020→BT.709 matrix
/// — saturated BT.2020 primary can land at -0.8 in BT.709, but post-tone-map
/// chroma is bounded; observed values cluster in 0.0-0.5.
const CLIP_HIST_BINS: usize = 1000;
const CLIP_HIST_BIN_WIDTH: f64 = 0.001;

/// OKLch C delta histogram. Knee=1.0 baseline → knee=N candidate: at extreme
/// rolloff the C reduction can hit 0.3 or so; 0.001 bins resolve ~5e-4.
const CHROMA_HIST_BINS: usize = 500;
const CHROMA_HIST_BIN_WIDTH: f64 = 0.001;

const METRIC_CHUNK_PIXELS: usize = 8192;

// =========================================================================
// Lock-file refresh (same shape as audited shootout)
// =========================================================================

fn refresh_lock(activity: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let iso = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| format!("ts={}", ts));
    let line = format!("{} claude-knee-sweep {}\n", iso, activity);
    let _ = fs::write(WORKONGOING, line);
}

// =========================================================================
// Sample collection (same as audited)
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

// =========================================================================
// Decode helpers (same as audited)
// =========================================================================

#[derive(Clone)]
struct LinearRgb {
    width: u32,
    height: u32,
    px: Vec<f32>,
    #[allow(dead_code)]
    primaries: ColorPrimaries,
}

impl LinearRgb {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

fn decode_sample(bytes: &[u8]) -> anyhow::Result<LinearRgb> {
    let hdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .reconstruct_hdr(None)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("reconstruct_hdr: {}", e.error()))?;
    let hdr_buf = hdr_out.into_buffer();
    let hdr = pixel_buffer_to_linear_rgb_in_primaries(&hdr_buf, ColorPrimaries::Bt2020)?;
    Ok(hdr)
}

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
// measure_max peak (only method we use — the audited shootout's winner)
// =========================================================================

fn measure_max_peak_nits(hdr: &LinearRgb) -> f32 {
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
    let cll = ContentLightLevel::measure_max(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    )
    .unwrap_or_else(|| ContentLightLevel::new(203, 100));
    let max_nits = cll.max_content_light_level as f32;
    max_nits.max(203.0)
}

// =========================================================================
// Pipeline: HDR (BT.2020 linear) → Bt2446A → BT.2020→BT.709 → SoftCompress
// =========================================================================

/// Apply Bt2446A to `hdr` (BT.2020 linear), writing target-normalized linear
/// BT.2020 into `out`. Output: 1.0 = 100 nits = SDR diffuse white.
fn apply_bt2446a(hdr: &LinearRgb, source_peak_nits: f32, out: &mut Vec<f32>) {
    let diffuse_white_nits = 203.0_f32;
    let target_peak_nits = 100.0_f32;
    let max_pixel_value = (source_peak_nits / diffuse_white_nits).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;

    let n_floats = hdr.px.len();
    if out.len() != n_floats {
        out.clear();
        out.resize(n_floats, 0.0);
    }

    let tm = Bt2446A::new(source_peak_nits, target_peak_nits);
    out.par_chunks_mut(8192)
        .zip(hdr.px.par_chunks(8192))
        .for_each(|(sc, hc)| {
            for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                *s = h * content_norm_scale;
            }
        });
    let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(out);
    tm.map_strip_simd(strip);
    // NOTE: do NOT clamp here — the brief requires pre-clamp data so the
    // "clipped" metric sees post-rolloff overshoot.
}

/// In-place BT.2020 → BT.709 matrix rotation. Pre-clamp output.
fn rotate_bt2020_to_bt709(rgb: &mut [f32], width: usize) {
    let m = zgamut::conversion_matrix(ColorPrimaries::Bt2020, ColorPrimaries::Bt709)
        .expect("BT.2020→BT.709 matrix is defined");
    let row_floats = width * 3;
    rgb.par_chunks_mut(row_floats)
        .for_each(|row| zgamut::apply_matrix_row_f32(row, width, &m));
}

/// Apply SoftCompress with `knee` in BT.709 space, in place. Same primaries
/// (BT.709) at input + output (only chroma changes). Pre-clamp.
fn apply_softcompress_bt709(rgb: &mut [f32], knee: f32) {
    let m1 = zoklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
    let m1_inv = zoklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();
    let compressor = SoftCompress::from_matrices(&m1, &m1_inv, knee);
    rgb.par_chunks_mut(METRIC_CHUNK_PIXELS * 3).for_each(|chunk| {
        let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(chunk);
        compressor.apply_strip(strip);
    });
}

// =========================================================================
// Metrics  (clip stats + chroma compression vs baseline + ΔE2000)
// =========================================================================

#[derive(Clone, Copy, Debug, Default)]
struct CellMetrics {
    pixels_clipped_total: u64,
    pixels_clipped_pct: f64,
    clip_p50: f32,
    clip_p90: f32,
    clip_p95: f32,
    clip_p99: f32,
    chroma_compression_pct: f64, // mean (1 - C_cand/C_base) where C_base > eps
    peak_chroma_compression: f64, // max (1 - C_cand/C_base) where C_base > eps
    mean_de2000_vs_baseline: f32,
    max_de2000_vs_baseline: f32,
    runtime_ms: f32,
}

/// Compute clip stats on a pre-clamp linear-BT.709 RGB buffer.
fn compute_clip_stats(rgb: &[f32]) -> (u64, [f32; 4]) {
    let bin_w = CLIP_HIST_BIN_WIDTH;
    let nbins = CLIP_HIST_BINS;
    let chunk_floats = METRIC_CHUNK_PIXELS * 3;

    type Acc = (u64, Vec<u64>); // (clipped_pixels, hist)
    let init = || (0u64, vec![0u64; nbins]);
    let merge = |mut a: Acc, b: Acc| {
        a.0 += b.0;
        for (x, y) in a.1.iter_mut().zip(b.1.iter()) {
            *x += y;
        }
        a
    };

    let (clipped, hist): Acc = rgb
        .par_chunks(chunk_floats)
        .fold(init, |mut acc, c| {
            let n_px = c.len() / 3;
            for i in 0..n_px {
                let r = c[i * 3];
                let g = c[i * 3 + 1];
                let b = c[i * 3 + 2];
                let r_over = (r - 1.0).max(-r).max(0.0);
                let g_over = (g - 1.0).max(-g).max(0.0);
                let b_over = (b - 1.0).max(-b).max(0.0);
                let worst = r_over.max(g_over).max(b_over);
                if worst > 0.0 {
                    acc.0 += 1;
                    let bin = ((worst as f64 / bin_w) as usize).min(nbins - 1);
                    acc.1[bin] += 1;
                }
            }
            acc
        })
        .reduce(init, merge);

    let percentiles = percentiles_from_hist_4(&hist, clipped, bin_w, nbins);
    (clipped, percentiles)
}

/// Compute OKLch C distribution vs. baseline. Returns (mean_compression_pct,
/// peak_compression_pct). Mean is over pixels where baseline C > 1e-4
/// (skipping achromatic).
fn compute_chroma_compression(
    baseline: &[f32],
    candidate: &[f32],
) -> (f64, f64, Vec<u64>, u64) {
    let m1 = zoklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
    let chunk_floats = METRIC_CHUNK_PIXELS * 3;

    type Acc = (f64, f64, u64, Vec<u64>); // (sum_pct, peak_pct, n_counted, hist)
    let init = || (0.0_f64, 0.0_f64, 0u64, vec![0u64; CHROMA_HIST_BINS]);
    let merge = |mut a: Acc, b: Acc| {
        a.0 += b.0;
        if b.1 > a.1 {
            a.1 = b.1;
        }
        a.2 += b.2;
        for (x, y) in a.3.iter_mut().zip(b.3.iter()) {
            *x += y;
        }
        a
    };

    let (sum_pct, peak_pct, n_counted, hist): Acc = baseline
        .par_chunks(chunk_floats)
        .zip(candidate.par_chunks(chunk_floats))
        .fold(init, |mut acc, (bc, cc)| {
            let n_px = bc.len() / 3;
            for i in 0..n_px {
                let (br, bg, bb) = (bc[i * 3], bc[i * 3 + 1], bc[i * 3 + 2]);
                let (cr, cg, cb) = (cc[i * 3], cc[i * 3 + 1], cc[i * 3 + 2]);
                // Clamp negative values to 0 for OKLab (matches what apply_strip
                // does inside compressor — but here we measure the *post-rolloff*
                // values directly). Negative values would yield NaN cbrt.
                let blab = zoklab::rgb_to_oklab(br.max(0.0), bg.max(0.0), bb.max(0.0), &m1);
                let clab = zoklab::rgb_to_oklab(cr.max(0.0), cg.max(0.0), cb.max(0.0), &m1);
                let bc_chroma = (blab[1] * blab[1] + blab[2] * blab[2]).sqrt();
                let cc_chroma = (clab[1] * clab[1] + clab[2] * clab[2]).sqrt();
                if bc_chroma > 1e-4 {
                    let frac = ((bc_chroma - cc_chroma) / bc_chroma).clamp(-1.0, 1.0) as f64;
                    acc.0 += frac;
                    if frac > acc.1 {
                        acc.1 = frac;
                    }
                    acc.2 += 1;
                    // Histogram chroma DELTA (positive when candidate < baseline).
                    let delta = (bc_chroma - cc_chroma).max(0.0) as f64;
                    let bin = ((delta / CHROMA_HIST_BIN_WIDTH) as usize)
                        .min(CHROMA_HIST_BINS - 1);
                    acc.3[bin] += 1;
                }
            }
            acc
        })
        .reduce(init, merge);

    (sum_pct, peak_pct, hist, n_counted)
}

/// Compute mean and max ΔE2000 (Lab D65) between baseline and candidate.
fn compute_de2000(baseline: &[f32], candidate: &[f32]) -> (f64, f32) {
    let chunk_floats = METRIC_CHUNK_PIXELS * 3;
    type Acc = (f64, f32, u64); // (sum, max, n)
    let init = || (0.0_f64, 0.0_f32, 0u64);
    let merge = |a: Acc, b: Acc| (a.0 + b.0, a.1.max(b.1), a.2 + b.2);
    let (sum, max_de, n): Acc = baseline
        .par_chunks(chunk_floats)
        .zip(candidate.par_chunks(chunk_floats))
        .fold(init, |mut acc, (bc, cc)| {
            let n_px = bc.len() / 3;
            for i in 0..n_px {
                let lb = linear_rgb_to_lab_d65([
                    bc[i * 3].clamp(0.0, 1.0),
                    bc[i * 3 + 1].clamp(0.0, 1.0),
                    bc[i * 3 + 2].clamp(0.0, 1.0),
                ]);
                let lc = linear_rgb_to_lab_d65([
                    cc[i * 3].clamp(0.0, 1.0),
                    cc[i * 3 + 1].clamp(0.0, 1.0),
                    cc[i * 3 + 2].clamp(0.0, 1.0),
                ]);
                let de = delta_e2000(lb, lc);
                acc.0 += de as f64;
                if de > acc.1 {
                    acc.1 = de;
                }
                acc.2 += 1;
            }
            acc
        })
        .reduce(init, merge);
    let mean = if n == 0 { 0.0 } else { sum / (n as f64) };
    (mean, max_de)
}

// ---------------------------------------------------------------------------
// Lab D65 + ΔE2000 (verbatim from audited shootout, scalar f32→f64 path)
// ---------------------------------------------------------------------------

use zenpixels_convert::oklab::fast_cbrt;

const LAB_XN: f32 = 0.95047;
const LAB_YN_RECIP: f32 = 1.0;
const LAB_ZN_RECIP: f32 = 1.0 / 1.08883;
const LAB_XN_RECIP: f32 = 1.0 / LAB_XN;
const LAB_DELTA: f32 = 6.0 / 29.0;
const LAB_DELTA_CUBED: f32 = LAB_DELTA * LAB_DELTA * LAB_DELTA;

#[inline]
fn lab_f(t: f32) -> f32 {
    if t > LAB_DELTA_CUBED {
        fast_cbrt(t)
    } else {
        t * (1.0 / (3.0 * LAB_DELTA * LAB_DELTA)) + (4.0 / 29.0)
    }
}

#[inline]
#[allow(clippy::excessive_precision)]
fn linear_rgb_to_lab_d65(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let fx = lab_f(x * LAB_XN_RECIP);
    let fy = lab_f(y * LAB_YN_RECIP);
    let fz = lab_f(z * LAB_ZN_RECIP);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
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
// Histogram → percentile (verbatim from audited shootout)
// =========================================================================

fn percentiles_from_hist_4(
    hist: &[u64],
    total: u64,
    bin_width: f64,
    nbins: usize,
) -> [f32; 4] {
    if total == 0 {
        return [0.0, 0.0, 0.0, 0.0];
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
    while next_target < 4 {
        out[next_target] = (nbins as f64 * bin_width) as f32;
        next_target += 1;
    }
    out
}

// =========================================================================
// Per-sample work
// =========================================================================

#[derive(Clone)]
struct SampleResult {
    sample_path: PathBuf,
    format_label: String,
    width: u32,
    height: u32,
    source_peak_nits: f32,
    /// Per knee value (parallel to KNEE_GRID).
    cells: Vec<CellMetrics>,
}

fn run_sample(
    path: &Path,
    fmt_label: &str,
) -> anyhow::Result<SampleResult> {
    let bytes = fs::read(path)?;
    let hdr = decode_sample(&bytes)?;
    let hdr_max_raw = hdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !hdr_max_raw.is_finite() {
        anyhow::bail!("non-finite hdr pixel data");
    }
    if hdr_max_raw < 1.05 {
        anyhow::bail!("no HDR boost (hdr_max={:.3})", hdr_max_raw);
    }

    let source_peak = measure_max_peak_nits(&hdr);
    println!(
        "  {}: {}x{}, source_peak={:.0} nits",
        path.file_name().unwrap_or_default().to_string_lossy(),
        hdr.width,
        hdr.height,
        source_peak,
    );

    // Build the baseline (knee=1.0, no rolloff) once per sample: HDR → Bt2446A
    // (BT.2020) → BT.2020→BT.709 matrix. Pre-clamp.
    let width = hdr.width as usize;
    let mut tm_out: Vec<f32> = Vec::new();
    apply_bt2446a(&hdr, source_peak, &mut tm_out);
    rotate_bt2020_to_bt709(&mut tm_out, width);
    // Now `tm_out` is the linear-BT.709 baseline (knee=1.0, pre-clamp).

    // For each knee we copy tm_out into a candidate buffer, apply SoftCompress
    // with that knee, then run all three metric passes (clip, chroma, ΔE2000).
    let mut cells = Vec::with_capacity(KNEE_GRID.len());
    let mut cand_buf: Vec<f32> = Vec::with_capacity(tm_out.len());

    for &knee in KNEE_GRID {
        let t0 = std::time::Instant::now();
        cand_buf.clear();
        cand_buf.extend_from_slice(&tm_out);

        if knee < 1.0 {
            apply_softcompress_bt709(&mut cand_buf, knee);
        }
        // knee == 1.0 → no rolloff applied; candidate == baseline.

        let (clipped_total, clip_percentiles) = compute_clip_stats(&cand_buf);
        let n_px = hdr.pixels() as u64;
        let pct_clipped = (clipped_total as f64) * 100.0 / (n_px as f64);

        let (sum_pct, peak_pct, _chroma_hist, n_counted) =
            compute_chroma_compression(&tm_out, &cand_buf);
        let mean_chroma_compression = if n_counted == 0 {
            0.0
        } else {
            sum_pct / (n_counted as f64) * 100.0
        };
        let peak_chroma_compression = peak_pct * 100.0;

        let (mean_de, max_de) = compute_de2000(&tm_out, &cand_buf);

        let runtime_ms = t0.elapsed().as_secs_f32() * 1000.0;
        cells.push(CellMetrics {
            pixels_clipped_total: clipped_total,
            pixels_clipped_pct: pct_clipped,
            clip_p50: clip_percentiles[0],
            clip_p90: clip_percentiles[1],
            clip_p95: clip_percentiles[2],
            clip_p99: clip_percentiles[3],
            chroma_compression_pct: mean_chroma_compression,
            peak_chroma_compression,
            mean_de2000_vs_baseline: mean_de as f32,
            max_de2000_vs_baseline: max_de,
            runtime_ms,
        });
    }

    let (w, h) = (hdr.width, hdr.height);
    Ok(SampleResult {
        sample_path: path.to_path_buf(),
        format_label: fmt_label.to_string(),
        width: w,
        height: h,
        source_peak_nits: source_peak,
        cells,
    })
}

// =========================================================================
// CSV writer
// =========================================================================

fn write_csv(samples: &[SampleResult]) -> anyhow::Result<()> {
    fs::create_dir_all(Path::new(CSV_PATH).parent().unwrap())?;
    let f = fs::File::create(CSV_PATH)?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "sample,format,width,height,source_peak_nits,knee,\
pixels_clipped_total,pixels_clipped_pct,\
clip_p50,clip_p90,clip_p95,clip_p99,\
chroma_compression_pct,peak_chroma_compression,\
mean_de2000_vs_knee_1,max_de2000_vs_knee_1,runtime_ms,schema_version"
    )?;
    let mut sorted: Vec<&SampleResult> = samples.iter().collect();
    sorted.sort_by(|a, b| a.sample_path.cmp(&b.sample_path));
    for r in sorted {
        let stem = r
            .sample_path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        for (ki, &knee) in KNEE_GRID.iter().enumerate() {
            if ki >= r.cells.len() {
                continue;
            }
            let c = r.cells[ki];
            writeln!(
                w,
                "{},{},{},{},{:.1},{:.2},\
{},{:.6},\
{:.6},{:.6},{:.6},{:.6},\
{:.5},{:.5},\
{:.4},{:.4},{:.1},{}",
                stem,
                r.format_label,
                r.width,
                r.height,
                r.source_peak_nits,
                knee,
                c.pixels_clipped_total,
                c.pixels_clipped_pct,
                c.clip_p50,
                c.clip_p90,
                c.clip_p95,
                c.clip_p99,
                c.chroma_compression_pct,
                c.peak_chroma_compression,
                c.mean_de2000_vs_baseline,
                c.max_de2000_vs_baseline,
                c.runtime_ms,
                SCHEMA_VERSION,
            )?;
        }
    }
    Ok(())
}

// =========================================================================
// Main
// =========================================================================

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();

    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(OUTER_THREADS)
        .build_global()
    {
        eprintln!(
            "WARN: rayon pool already initialized ({}); outer concurrency uncapped",
            e
        );
    }

    refresh_lock("scanning-corpus");
    println!("Scanning corpus at {}...", SAMPLES_ROOT);
    let all_files = collect_samples();
    println!("  Found {} candidate files (.jpg/.jpeg/.heic)", all_files.len());

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
    println!("Found {} gain-mapped samples", work_list.len());
    if work_list.is_empty() {
        anyhow::bail!("no gain-mapped samples found in {}", SAMPLES_ROOT);
    }

    let total = work_list.len();
    println!(
        "Sweeping {} knee values × {} samples = {} cells",
        KNEE_GRID.len(),
        total,
        KNEE_GRID.len() * total
    );

    type SampleOutcome = Result<SampleResult, (PathBuf, String)>;
    let outcomes: Vec<SampleOutcome> = work_list
        .par_iter()
        .enumerate()
        .map(|(idx, (path, fmt_label))| -> SampleOutcome {
            if idx % 5 == 0 {
                refresh_lock(&format!("knee-sweep-{}-of-{}", idx + 1, total));
            }
            println!("=== [{}/{}] {} ({}) ===", idx + 1, total,
                     path.file_name().unwrap_or_default().to_string_lossy(),
                     fmt_label);
            match run_sample(path, fmt_label) {
                Ok(r) => Ok(r),
                Err(e) => {
                    eprintln!("  failed: {}", e);
                    Err((path.clone(), e.to_string()))
                }
            }
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
    write_csv(&sample_results)?;
    println!("CSV written to {}", CSV_PATH);
    println!(
        "Done. {} samples succeeded, {} failed, total {:.1}s.",
        sample_results.len(),
        failures.len(),
        runtime
    );
    Ok(())
}
