//! Focused HDR → SDR shootout: Bt2446A under `measure_percentile @ 0.99999`,
//! on the same 76 imazen-26 gain-mapped samples as
//! `hdr_tone_map_shootout_full.rs`.
//!
//! This is a side-program to the full 76 × 3 × 20 shootout. Goal: produce a
//! single CSV row per sample with the new p=0.99999 peak driving Bt2446A,
//! so the result can be compared cell-for-cell against the existing
//! `measure_robust` (p=0.9999) baseline at
//! `benchmarks/hdr_tone_map_shootout_full_2026-06-20.csv`.
//!
//! Same column schema as the parent CSV:
//! `sample,format,peak_method,curve,knee_tone,knee_gamut,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_above_de5`
//!
//! ΔE2000 is computed via the same in-file CIE2000 implementation used by
//! the parent (linear-RGB → CIE Lab D65 → CIE2000), to guarantee a
//! byte-equivalent comparison against the prior baseline numbers.
//!
//! Run:
//! ```text
//! nice -n19 ionice -c3 cargo run -p zentone --release \
//!   --example shootout_p99999_only --features hdr-shootout
//! ```
//!
//! Writes:
//! - `benchmarks/shootout_p99999_2026-06-22.csv`
//!   (76 rows, one per sample, Bt2446A × measure_percentile@0.99999)

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


// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
/// Default output CSV — overridden by `SHOOTOUT_CSV_PATH` env var.
const CSV_PATH: &str = "/home/lilith/work/zen/zentone/benchmarks/shootout_p99999_2026-06-22.csv";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";

/// The new peak method under test. The existing baseline uses
/// `measure_robust` which is equivalent to `measure_percentile @ 0.9999`.
const TARGET_PERCENTILE_DEFAULT: f32 = 0.99999;

/// Read the target percentile from env (`SHOOTOUT_PERCENTILE=0.9999`) for the
/// rerun-the-baseline mode; default 0.99999 (the new candidate).
fn target_percentile() -> f32 {
    std::env::var("SHOOTOUT_PERCENTILE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(TARGET_PERCENTILE_DEFAULT)
}

/// Output CSV path — also env-overridable so a baseline rerun doesn't stomp
/// the candidate CSV.
fn csv_path() -> String {
    std::env::var("SHOOTOUT_CSV_PATH")
        .unwrap_or_else(|_| CSV_PATH.to_string())
}

/// Peak-method label written to the `peak_method` column.
fn peak_method_label(percentile: f32) -> String {
    // `measure_percentile_99999` for p=0.99999, etc. Strip trailing 9s? No —
    // just use a deterministic 5-decimal format so the comparison joiner can
    // route by string match.
    let pct = (percentile * 100_000.0).round() as u64;
    format!("measure_percentile_{}", pct)
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
        .unwrap_or_else(|| "unknown-ts".to_string());
    let line = format!("{} claude-shootout-p99999 {}\n", iso, activity);
    let _ = fs::write(WORKONGOING, line);
}

// =========================================================================
// Sample-finding (mirrors hdr_tone_map_shootout_full.rs)
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
// Decode + linearize (mirrors hdr_tone_map_shootout_full.rs exactly)
// =========================================================================

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
            "SDR and HDR dimensions differ: SDR {}x{}, HDR {}x{}",
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
// Source-peak measurement at percentile p
// =========================================================================

/// Returns `(peak_nits, fallback_used)`. Tries `measure_percentile @ p`;
/// falls back to `measure_max` if the percentile call returns None.
fn measure_source_peak_percentile_nits(hdr: &LinearRgb, percentile: f32) -> (f32, bool) {
    let bytes: Vec<u8> = bytemuck::cast_slice(&hdr.px).to_vec();
    let desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let Ok(buf) = PixelBuffer::from_vec(bytes, hdr.width, hdr.height, desc) else {
        return (203.0, true);
    };
    let cll_opt = ContentLightLevel::measure_percentile(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        percentile,
        LightLevelMethod::MaxRgb,
    );
    if let Some(cll) = cll_opt {
        let max_nits = cll.max_content_light_level as f32;
        (max_nits.max(203.0), false)
    } else {
        let cll = ContentLightLevel::measure_max(
            buf.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .unwrap_or_else(|| ContentLightLevel::new(203, 100));
        ((cll.max_content_light_level as f32).max(203.0), true)
    }
}

// =========================================================================
// Apply Bt2446A — verbatim from hdr_tone_map_shootout_full.rs CurveSpec::Bt2446A
// =========================================================================

fn apply_bt2446a(hdr: &LinearRgb, source_peak_nits: f32) -> LinearRgb {
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

    scratch.par_chunks_mut(8192).for_each(|c| {
        for v in c.iter_mut() {
            *v = v.max(0.0).min(1.0);
        }
    });
    out.px.copy_from_slice(&scratch);
    out
}

// =========================================================================
// Metrics — verbatim from hdr_tone_map_shootout_full.rs
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
// Per-sample result
// =========================================================================

#[derive(Clone)]
struct SampleResult {
    sample_path: PathBuf,
    format_label: String,
    source_peak_nits: f32,
    metrics: CellMetrics,
    fallback_used: bool,
}

// =========================================================================
// CSV writer
// =========================================================================

fn write_csv(samples: &[SampleResult], percentile: f32, out_path: &str) -> anyhow::Result<()> {
    fs::create_dir_all(Path::new(out_path).parent().unwrap())?;
    let f = fs::File::create(out_path)?;
    let mut w = BufWriter::new(f);
    let label = peak_method_label(percentile);
    // Same column schema as the parent shootout CSV, with one extra
    // `fallback_used` column at the end (per the brief — a percentile
    // call that returns None falls back to `measure_max`).
    writeln!(
        w,
        "sample,format,peak_method,curve,knee_tone,knee_gamut,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_above_de5,fallback_used"
    )?;
    for r in samples {
        let stem = r
            .sample_path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        writeln!(
            w,
            "{},{},{},bt2446a,,,{:.1},{:.3},{:.4},{:.5},{:.3},{}",
            stem,
            r.format_label,
            label,
            r.source_peak_nits,
            r.metrics.psnr_db,
            r.metrics.mean_de2000,
            r.metrics.max_abs_delta,
            r.metrics.pct_de_gt_5,
            if r.fallback_used { "true" } else { "false" },
        )?;
    }
    Ok(())
}

// =========================================================================
// Main
// =========================================================================

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();

    let percentile = target_percentile();
    let out_csv = csv_path();
    println!(
        "shootout_p99999_only: percentile={:.5} (label={}) → {}",
        percentile,
        peak_method_label(percentile),
        out_csv
    );

    refresh_lock("scanning-corpus");
    println!("Scanning corpus at {}...", SAMPLES_ROOT);
    let all_files = collect_samples();
    println!(
        "  Found {} candidate files (.jpg/.jpeg/.heic)",
        all_files.len()
    );

    // Probe for gain-map presence.
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

    let mut results: Vec<SampleResult> = Vec::new();
    let mut skipped: Vec<(PathBuf, String)> = Vec::new();
    let total = work_list.len();

    for (idx, (path, fmt_label)) in work_list.iter().enumerate() {
        let stem_raw = path
            .file_name()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        if idx % 5 == 0 {
            refresh_lock(&format!("sweep-{}-of-{}", idx + 1, total));
        }
        println!("\n=== [{}/{}] {} ({}) ===", idx + 1, total, stem_raw, fmt_label);

        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  read failed: {}", e);
                skipped.push((path.clone(), format!("read: {}", e)));
                continue;
            }
        };

        let (sdr, hdr, sdr_descr, _hdr_descr) = match decode_sample_full(&bytes) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  decode failed: {}", e);
                skipped.push((path.clone(), format!("decode: {}", e)));
                continue;
            }
        };

        let hdr_max_raw = hdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sdr_max_raw = sdr.px.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if hdr_max_raw < 1.05 && sdr_max_raw < 1.05 && (hdr_max_raw - sdr_max_raw).abs() < 0.05 {
            println!("  → skipping: HDR reconstruction returned SDR base (no boost)");
            skipped.push((path.clone(), "no HDR boost".into()));
            continue;
        }
        if sdr_descr == TransferFunction::Pq {
            println!("  → skipping: SDR base is PQ-encoded");
            skipped.push((path.clone(), "PQ-encoded SDR base".into()));
            continue;
        }
        if !hdr_max_raw.is_finite() || !sdr_max_raw.is_finite() {
            println!("  → skipping: non-finite pixel data");
            skipped.push((path.clone(), "non-finite pixel data".into()));
            continue;
        }

        let (peak, fallback_used) = measure_source_peak_percentile_nits(&hdr, percentile);
        println!(
            "  peak @ p={:.5}: {:.0} nits  (fallback_used={})  raw HDR max={:.3}",
            percentile, peak, fallback_used, hdr_max_raw
        );

        let cand = apply_bt2446a(&hdr, peak);
        let metrics = compute_metrics(&sdr, &cand);
        println!(
            "  Bt2446A: PSNR={:.2} dB, mean ΔE2000={:.3}, max|Δ|={:.4}, %>ΔE5={:.2}",
            metrics.psnr_db, metrics.mean_de2000, metrics.max_abs_delta, metrics.pct_de_gt_5
        );

        results.push(SampleResult {
            sample_path: path.clone(),
            format_label: fmt_label.clone(),
            source_peak_nits: peak,
            metrics,
            fallback_used,
        });

        drop(hdr);
        drop(sdr);
    }

    refresh_lock("writing-csv");
    write_csv(&results, percentile, &out_csv)?;
    let runtime = t0.elapsed().as_secs_f64();
    println!("\nCSV written to {}", out_csv);
    println!(
        "Done. {} samples succeeded, {} skipped, total {:.1}s ({:.1}min).",
        results.len(),
        skipped.len(),
        runtime,
        runtime / 60.0,
    );

    if !skipped.is_empty() {
        println!("\nSkipped samples:");
        for (p, why) in &skipped {
            println!("  - {} : {}", p.display(), why);
        }
    }
    Ok(())
}
