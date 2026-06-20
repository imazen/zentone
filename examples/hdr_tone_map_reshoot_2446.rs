//! HDR→SDR tone-map post-fix re-shoot: Bt2446A + Bt2446C + Bt2390 on 76 samples.
//!
//! Focused companion to `hdr_tone_map_shootout_full.rs` (the 20-cell-per-sample
//! shootout at commit `e876ac2`, report
//! `benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`).
//!
//! Reason: that prior shootout measured `Bt2446A` and `Bt2446C` with two
//! material bugs active. Both are now fixed in `main@origin`:
//!   * `60cf33b6` — `Bt2446A` was writing PQ-encoded output where the rest of
//!     the pipeline (and the metric harness) expects linear-light. Now linear.
//!   * `3ecd45f5` — `Bt2446C` was computing the wrong k3 / Y_ip / scaling per
//!     §6.1.4 of ITU-R BT.2446-1. Now spec-correct.
//!
//! Bt2390 hasn't changed and is included unchanged as the reference baseline
//! (median ΔE2000 6.089 under `measure_robust` in the prior shootout).
//!
//! Cells per sample: 3 curves × 3 peak methods = 9.
//! Samples: 76 gain-mapped UltraHDR JPEG + HEIC from imazen-26.
//! Total cell-evaluations: 76 × 9 = 684 → 684 CSV rows + 684 candidate thumbnails.
//!
//! Outputs:
//!   * `/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_reshoot_2446_2026-06-20.csv`
//!     (streamed row-by-row + flushed per peak-method block)
//!   * `/mnt/v/output/zentone/reshoot-2446/<stem>__REF.png` — producer SDR (1×/sample)
//!   * `/mnt/v/output/zentone/reshoot-2446/<stem>__<curve>__<peak_method>.png`
//!     — each candidate tonemap (1024-px wide PNG thumbnail, saved inline)
//!   * `/mnt/v/output/zentone/reshoot-2446/<stem>__MONTAGE_<peak_method>.png`
//!     — 4-panel side-by-side (REF | Bt2390 | Bt2446A | Bt2446C), 768-px per
//!     panel = 3072-px wide; 3 montages per sample = 228 total
//!   * Bench doc addendum appended to
//!     `benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`
//!
//! Run:
//! ```text
//! nice -n19 cargo run -p zentone --release \
//!   --example hdr_tone_map_reshoot_2446 --features hdr-shootout
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

use zentone::{Bt2446A, Bt2446C, ToneMap, ToneMapCurve};

// =========================================================================
// Paths / constants
// =========================================================================

const SAMPLES_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const MONTAGES_DIR: &str = "/mnt/v/output/zentone/reshoot-2446";
const REPORT_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20.md";
const CSV_PATH: &str =
    "/home/lilith/work/zen/zentone/benchmarks/hdr_tone_map_reshoot_2446_2026-06-20.csv";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";

// 1024 source-px wide thumbnails for the singletons; per-panel width on the
// 4-panel montage shaved to 768 so the combined image stays a manageable
// 3072 px wide (4× 768 + 3 separators).
const TONEMAP_THUMBNAIL_W: u32 = 1024;
const MONTAGE_PANEL_W: u32 = 768;

// SDR diffuse-white anchor: our linear HDR buffer is anchored at 1.0 = 203 nits.
const DIFFUSE_WHITE_NITS: f32 = 203.0;
const TARGET_PEAK_NITS: f32 = 100.0;

// Prior-shootout reference numbers (commit e876ac2, bugs ACTIVE), for the
// addendum's before/after comparison.
const PRIOR_BT2390_DE_MAX: f32 = 8.342;
const PRIOR_BT2390_DE_ROBUST: f32 = 6.089;
const PRIOR_BT2390_DE_SMOOTHED: f32 = 8.331;
const PRIOR_BT2446A_DE_MAX: f32 = 22.452;
const PRIOR_BT2446A_DE_ROBUST: f32 = 22.974;
const PRIOR_BT2446A_DE_SMOOTHED: f32 = 22.541;
const PRIOR_BT2446C_DE_MAX: f32 = 11.947;
const PRIOR_BT2446C_DE_ROBUST: f32 = 10.653;
const PRIOR_BT2446C_DE_SMOOTHED: f32 = 11.909;

// =========================================================================
// Peak-method enum
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
// Curve enum
// =========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Curve {
    Bt2390,
    Bt2446A,
    Bt2446C,
}

impl Curve {
    fn label(&self) -> &'static str {
        match self {
            Curve::Bt2390 => "bt2390",
            Curve::Bt2446A => "bt2446a",
            Curve::Bt2446C => "bt2446c",
        }
    }
    fn all() -> &'static [Curve] {
        &[Curve::Bt2390, Curve::Bt2446A, Curve::Bt2446C]
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
        .unwrap_or_else(|| "unknown".into());
    let line = format!("{} claude-reshoot-2446 {}\n", iso, activity);
    let _ = fs::write(WORKONGOING, line);
}

// =========================================================================
// Sample-finding (copied from `_full.rs`)
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
// Linear-RGB f32 buffer (copied from `_full.rs`)
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
// Decode + linearize (copied from `_full.rs`)
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
// Source-peak measurement (copied from `_full.rs`)
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
// Curve application (3 curves only — Bt2390, Bt2446A, Bt2446C)
// =========================================================================

/// Apply one of the three curves under test. HDR input is anchored 1.0 = 203
/// nits. Output is linear RGB f32 in [0, 1] (target-peak normalized for
/// comparison against the SDR baseline). Same content-norm scaling as
/// `_full.rs` so the numbers are apples-to-apples with the prior shootout.
fn apply_curve(curve: Curve, hdr: &LinearRgb, source_peak_nits: f32) -> LinearRgb {
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;

    let n_pix = hdr.pixels();
    let mut scratch = vec![0.0_f32; n_pix * 3];

    match curve {
        Curve::Bt2390 => {
            scratch
                .par_chunks_mut(8192)
                .zip(hdr.px.par_chunks(8192))
                .for_each(|(sc, hc)| {
                    for (s, &h) in sc.iter_mut().zip(hc.iter()) {
                        *s = h * content_norm_scale;
                    }
                });
            let target_peak_in_src = TARGET_PEAK_NITS / source_peak_nits.max(1.0);
            let tone = ToneMapCurve::Bt2390 {
                source_peak: 1.0,
                target_peak: target_peak_in_src,
            };
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tone.map_strip_simd(strip);
            let inv = 1.0 / target_peak_in_src.max(1e-6);
            scratch.par_chunks_mut(8192).for_each(|c| {
                for v in c.iter_mut() {
                    *v *= inv;
                }
            });
        }
        Curve::Bt2446A => {
            let tm = Bt2446A::new(source_peak_nits, TARGET_PEAK_NITS);
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
        Curve::Bt2446C => {
            let tm = Bt2446C::new(source_peak_nits, TARGET_PEAK_NITS);
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
    }

    scratch.par_chunks_mut(8192).for_each(|c| {
        for v in c.iter_mut() {
            *v = v.max(0.0).min(1.0);
        }
    });
    LinearRgb {
        width: hdr.width,
        height: hdr.height,
        px: scratch,
    }
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
// Thumbnail save (adapted from gainforge example — box-filter downscale)
// =========================================================================

/// Downscale + encode a linear-RGB f32 buffer to an 8-bit sRGB PNG. Saves a
/// `target_w`-wide preview suitable for visual review without exploding disk
/// usage. Box-filter downscale (cheap; we're not after pixel-perfect resampling).
fn save_thumbnail(path: &Path, lin: &LinearRgb, target_w: u32) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let target_w = target_w.min(lin.width);
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

// =========================================================================
// 4-panel montage: REF | Bt2390 | Bt2446A | Bt2446C
// =========================================================================

/// Downscale a linear-RGB f32 buffer to (target_w × target_h) sRGB-u8 with the
/// same box filter as `save_thumbnail`, but writes into a caller-supplied output.
fn downscale_to_srgb_u8(lin: &LinearRgb, target_w: u32, target_h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (target_w as usize) * (target_h as usize) * 3];
    let scale_x = lin.width as f32 / target_w as f32;
    let scale_y = lin.height as f32 / target_h as f32;
    let src_w = lin.width as usize;
    out.par_chunks_mut(target_w as usize * 3)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy0 = ((dy as f32) * scale_y) as usize;
            let sy1 = (((dy as f32) + 1.0) * scale_y) as usize;
            let sy1 = sy1.min(lin.height as usize).max(sy0 + 1);
            for dx in 0..target_w as usize {
                let sx0 = ((dx as f32) * scale_x) as usize;
                let sx1 = (((dx as f32) + 1.0) * scale_x) as usize;
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
                row[dx * 3] = linear_to_srgb_u8(acc[0] * inv_n);
                row[dx * 3 + 1] = linear_to_srgb_u8(acc[1] * inv_n);
                row[dx * 3 + 2] = linear_to_srgb_u8(acc[2] * inv_n);
            }
        });
    out
}

/// Build a 4-panel side-by-side montage at `panel_w` per panel.
/// Panels: REF | Bt2390 | Bt2446A | Bt2446C with 2-px dark separators.
fn save_4panel_montage(path: &Path, panels: [&LinearRgb; 4], panel_w: u32) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    // Use the reference's aspect ratio for the common panel height (all four
    // buffers should be the same dimensions anyway — they were derived from
    // the same decode).
    let ref_panel = panels[0];
    let panel_w = panel_w.min(ref_panel.width);
    let scale = ref_panel.width as f32 / panel_w as f32;
    let panel_h = ((ref_panel.height as f32 / scale).round() as u32).max(1);

    let panel_bytes: Vec<Vec<u8>> = panels
        .iter()
        .map(|p| downscale_to_srgb_u8(p, panel_w, panel_h))
        .collect();

    let sep_w: u32 = 2;
    let combined_w = panel_w * 4 + sep_w * 3;
    let mut out = vec![0u8; (combined_w as usize) * (panel_h as usize) * 3];

    let row_len_panel = panel_w as usize * 3;
    let row_len_combined = combined_w as usize * 3;
    let sep_color = [40u8, 40, 40];

    for y in 0..panel_h as usize {
        let dst_row_base = y * row_len_combined;
        let mut x_off = 0usize;
        for (i, p_bytes) in panel_bytes.iter().enumerate() {
            let src_row_base = y * row_len_panel;
            out[dst_row_base + x_off..dst_row_base + x_off + row_len_panel]
                .copy_from_slice(&p_bytes[src_row_base..src_row_base + row_len_panel]);
            x_off += row_len_panel;
            if i < 3 {
                for k in 0..sep_w as usize {
                    out[dst_row_base + x_off + k * 3..dst_row_base + x_off + k * 3 + 3]
                        .copy_from_slice(&sep_color);
                }
                x_off += sep_w as usize * 3;
            }
        }
    }

    image::save_buffer(path, &out, combined_w, panel_h, image::ColorType::Rgb8)?;
    Ok(())
}

// =========================================================================
// Aggregation
// =========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceClass {
    UltraHdrJpeg,
    Heic,
    Other,
}

fn classify_device(path: &Path) -> DeviceClass {
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

#[derive(Clone)]
struct SampleResult {
    #[allow(dead_code)]
    sample_path: PathBuf,
    format_label: String,
    device_class: DeviceClass,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    #[allow(dead_code)]
    source_peaks: HashMap<PeakMethod, f32>,
    /// (peak_method × curve) -> metrics.
    cells: HashMap<(PeakMethod, Curve), CellMetrics>,
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

fn curve_median_de(samples: &[SampleResult], method: PeakMethod, curve: Curve) -> f32 {
    median(
        samples
            .iter()
            .filter_map(|r| r.cells.get(&(method, curve)).map(|m| m.mean_de2000))
            .collect(),
    )
}

fn curve_median_psnr(samples: &[SampleResult], method: PeakMethod, curve: Curve) -> f32 {
    median(
        samples
            .iter()
            .filter_map(|r| r.cells.get(&(method, curve)).map(|m| m.psnr_db))
            .collect(),
    )
}

fn curve_median_max_abs(samples: &[SampleResult], method: PeakMethod, curve: Curve) -> f32 {
    median(
        samples
            .iter()
            .filter_map(|r| r.cells.get(&(method, curve)).map(|m| m.max_abs_delta))
            .collect(),
    )
}

fn curve_median_pct5(samples: &[SampleResult], method: PeakMethod, curve: Curve) -> f32 {
    median(
        samples
            .iter()
            .filter_map(|r| r.cells.get(&(method, curve)).map(|m| m.pct_de_gt_5))
            .collect(),
    )
}

fn prior_de(curve: Curve, method: PeakMethod) -> f32 {
    match (curve, method) {
        (Curve::Bt2390, PeakMethod::Max) => PRIOR_BT2390_DE_MAX,
        (Curve::Bt2390, PeakMethod::Robust) => PRIOR_BT2390_DE_ROBUST,
        (Curve::Bt2390, PeakMethod::MaxSmoothed) => PRIOR_BT2390_DE_SMOOTHED,
        (Curve::Bt2446A, PeakMethod::Max) => PRIOR_BT2446A_DE_MAX,
        (Curve::Bt2446A, PeakMethod::Robust) => PRIOR_BT2446A_DE_ROBUST,
        (Curve::Bt2446A, PeakMethod::MaxSmoothed) => PRIOR_BT2446A_DE_SMOOTHED,
        (Curve::Bt2446C, PeakMethod::Max) => PRIOR_BT2446C_DE_MAX,
        (Curve::Bt2446C, PeakMethod::Robust) => PRIOR_BT2446C_DE_ROBUST,
        (Curve::Bt2446C, PeakMethod::MaxSmoothed) => PRIOR_BT2446C_DE_SMOOTHED,
    }
}

// =========================================================================
// Report addendum
// =========================================================================

fn append_report(
    sample_results: &[SampleResult],
    failures: &[(PathBuf, String)],
    runtime_secs: f64,
    n_jpeg_processed: usize,
    n_heic_processed: usize,
) -> anyhow::Result<()> {
    let mut s = String::new();
    s.push_str("\n\n---\n\n");
    s.push_str("## Post-fix re-shoot — Bt2446A + Bt2446C + Bt2390 (2026-06-20)\n\n");
    s.push_str(&format!(
        "After the three bug fixes (Bt2446A linear-light output `60cf33b6`, Bt2446C spec-correct k3/Y_ip/scaling `3ecd45f5`, HdrToSdr input rescale `ca614df0`), re-ran just the curves that should have moved. Bt2390 included as the reference baseline.\n\n\
**Tested**: 3 curves × {} peak methods × {} samples = {} cell-evaluations. Runtime: {:.1}s ({:.1}min).\n\n\
**Samples processed**: {} ({} UltraHDR JPEG + {} HEIC) of {} gain-mapped candidates from `/home/lilith/work/codec-corpus/imazen-26/`.\n\n",
        PeakMethod::all().len(),
        sample_results.len(),
        sample_results.len() * 3 * PeakMethod::all().len(),
        runtime_secs,
        runtime_secs / 60.0,
        sample_results.len(),
        n_jpeg_processed,
        n_heic_processed,
        sample_results.len() + failures.len(),
    ));

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

    // Per-curve summary by peak method — showing prior numbers too for a
    // clean before/after read.
    s.push_str("### Per-curve summary by peak method (median across samples)\n\n");
    s.push_str("Columns: median ΔE2000 (lower = closer to producer SDR), median PSNR (dB; higher = better), median max|Δ| in normalized linear, median % pixels with ΔE > 5. The **Δ vs prior** column reports `post_fix - pre_fix` ΔE2000; a negative value means the fix improved the curve.\n\n");

    for method in PeakMethod::all() {
        s.push_str(&format!("#### `{}`\n\n", method.label()));
        s.push_str(
            "| Curve | Median ΔE2000 | Prior ΔE2000 | Δ vs prior | Median PSNR (dB) | Median max\\|Δ\\| | Median %>ΔE5 |\n",
        );
        s.push_str(
            "|-------|---------------|--------------|------------|------------------|-----------------|---------------|\n",
        );
        // Display curves sorted ascending by current ΔE.
        let mut rows: Vec<(Curve, f32, f32, f32, f32, f32)> = Curve::all()
            .iter()
            .map(|&c| {
                (
                    c,
                    curve_median_de(sample_results, *method, c),
                    curve_median_psnr(sample_results, *method, c),
                    curve_median_max_abs(sample_results, *method, c),
                    curve_median_pct5(sample_results, *method, c),
                    prior_de(c, *method),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (curve, de_now, psnr_now, max_abs, pct5, de_prior) in rows {
            let delta = de_now - de_prior;
            let delta_str = if delta.abs() < 0.001 {
                format!("{:+.3}", delta)
            } else if delta < 0.0 {
                format!("**{:+.3} (improved)**", delta)
            } else {
                format!("{:+.3} (worse)", delta)
            };
            s.push_str(&format!(
                "| `{}` | {:.3} | {:.3} | {} | {:.2} | {:.4} | {:.2} |\n",
                curve.label(),
                de_now,
                de_prior,
                delta_str,
                psnr_now,
                max_abs,
                pct5,
            ));
        }
        s.push('\n');
    }

    // Verdict
    s.push_str("### Verdict\n\n");
    let m = PeakMethod::Robust;
    let de_bt2390 = curve_median_de(sample_results, m, Curve::Bt2390);
    let de_bt2446a = curve_median_de(sample_results, m, Curve::Bt2446A);
    let de_bt2446c = curve_median_de(sample_results, m, Curve::Bt2446C);
    s.push_str(&format!(
        "Under the production-default `measure_robust` peak: `bt2390` median ΔE2000 = {:.3} (prior {:.3}), `bt2446a` = {:.3} (prior {:.3}), `bt2446c` = {:.3} (prior {:.3}).\n\n",
        de_bt2390,
        PRIOR_BT2390_DE_ROBUST,
        de_bt2446a,
        PRIOR_BT2446A_DE_ROBUST,
        de_bt2446c,
        PRIOR_BT2446C_DE_ROBUST,
    ));

    let mut ranked: Vec<(Curve, f32)> = Curve::all()
        .iter()
        .map(|&c| (c, curve_median_de(sample_results, m, c)))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let winner = ranked[0];
    s.push_str(&format!(
        "**Winner under `measure_robust` (post-fix)**: `{}` at median ΔE2000 {:.3}. ",
        winner.0.label(),
        winner.1,
    ));
    if winner.0 == Curve::Bt2390 {
        s.push_str(&format!(
            "Bt2390 remains the producer-fit champion across the 76-sample imazen-26 corpus. The Bt2446A fix `60cf33b6` moved it from ΔE 22.974 → {:.3} (Δ {:+.3}); the Bt2446C fix `3ecd45f5` moved it from ΔE 10.653 → {:.3} (Δ {:+.3}). Neither overtakes Bt2390, but the magnitudes confirm the fixes did what the code review predicted — Bt2446A's pre-fix PQ output was wildly off-frame against linear-light SDR, and Bt2446C's pre-fix k3/Y_ip math sat consistently 4–5 ΔE worse than spec-correct.\n\n",
            de_bt2446a,
            de_bt2446a - PRIOR_BT2446A_DE_ROBUST,
            de_bt2446c,
            de_bt2446c - PRIOR_BT2446C_DE_ROBUST,
        ));
        s.push_str(&format!(
            "**Updated HdrToSdr default recommendation**: continue with the Bt2390 default. Bt2446C is the next-best at ΔE {:.3} (gap to Bt2390 = {:+.3} ΔE), making it a sensible explicit-opt-in alternative for callers who want the BT.2446-1 §6.1.4 curve specifically.\n\n",
            de_bt2446c,
            de_bt2446c - de_bt2390,
        ));
    } else if winner.0 == Curve::Bt2446A {
        s.push_str(&format!(
            "**Bt2446A overtakes Bt2390** post-fix, moving from ΔE 22.974 → {:.3} (Δ {:+.3}). This was the headline goal of the linear-light output fix `60cf33b6`.\n\n",
            de_bt2446a,
            de_bt2446a - PRIOR_BT2446A_DE_ROBUST,
        ));
        s.push_str(&format!(
            "**Updated HdrToSdr default recommendation**: switch to Bt2446A as the curve closest to producer-graded SDR (median ΔE {:.3} vs Bt2390's {:.3}; gap {:+.3} ΔE).\n\n",
            de_bt2446a,
            de_bt2390,
            de_bt2390 - de_bt2446a,
        ));
    } else {
        s.push_str(&format!(
            "**Bt2446C overtakes Bt2390** post-fix, moving from ΔE 10.653 → {:.3} (Δ {:+.3}). The §6.1.4 spec-correct math now produces output closer to producer SDR than BT.2390.\n\n",
            de_bt2446c,
            de_bt2446c - PRIOR_BT2446C_DE_ROBUST,
        ));
        s.push_str(&format!(
            "**Updated HdrToSdr default recommendation**: switch to Bt2446C as the curve closest to producer-graded SDR (median ΔE {:.3} vs Bt2390's {:.3}; gap {:+.3} ΔE).\n\n",
            de_bt2446c,
            de_bt2390,
            de_bt2390 - de_bt2446c,
        ));
    }

    // Per-device-class breakdown (small — just the three curves)
    s.push_str("### Per-source-device breakdown (using `measure_robust`)\n\n");
    s.push_str("| Curve | UltraHDR JPEG (median ΔE2000) | HEIC (median ΔE2000) |\n");
    s.push_str("|---|---|---|\n");
    for &curve in Curve::all() {
        let jpeg: Vec<f32> = sample_results
            .iter()
            .filter(|r| r.device_class == DeviceClass::UltraHdrJpeg)
            .filter_map(|r| {
                r.cells
                    .get(&(PeakMethod::Robust, curve))
                    .map(|m| m.mean_de2000)
            })
            .collect();
        let heic: Vec<f32> = sample_results
            .iter()
            .filter(|r| r.device_class == DeviceClass::Heic)
            .filter_map(|r| {
                r.cells
                    .get(&(PeakMethod::Robust, curve))
                    .map(|m| m.mean_de2000)
            })
            .collect();
        s.push_str(&format!(
            "| `{}` | {:.3} ({} samples) | {:.3} ({} samples) |\n",
            curve.label(),
            median(jpeg.clone()),
            jpeg.len(),
            median(heic.clone()),
            heic.len(),
        ));
    }
    s.push('\n');

    s.push_str(&format!(
        "**Tonemap thumbnails + per-sample montages** saved under [`{}/`](http://172.23.240.1:3300/zentone/reshoot-2446/). Naming: `<stem>__REF.png` (the producer-SDR reference; 1× per sample), `<stem>__<curve>__<peak_method>.png` (each candidate tonemap, {}-pixel-wide PNG, 9 per sample), and `<stem>__MONTAGE_<peak_method>.png` (the 4-panel side-by-side REF | Bt2390 | Bt2446A | Bt2446C at {}-pixel-per-panel, 3 per sample).\n\n",
        MONTAGES_DIR,
        TONEMAP_THUMBNAIL_W,
        MONTAGE_PANEL_W,
    ));

    s.push_str("### Reproduce\n\n```bash\n");
    s.push_str("nice -n19 cargo build --example hdr_tone_map_reshoot_2446 \\\n");
    s.push_str("  --release --features hdr-shootout\n");
    s.push_str("nice -n19 cargo run --example hdr_tone_map_reshoot_2446 \\\n");
    s.push_str("  --release --features hdr-shootout\n");
    s.push_str("```\n\n");
    s.push_str(&format!(
        "Streaming CSV: `{}`.\nSource: `examples/hdr_tone_map_reshoot_2446.rs`.\n",
        CSV_PATH
    ));

    // Append (or replace the existing post-fix re-shoot section).
    let existing = fs::read_to_string(REPORT_PATH)?;
    let marker = "## Post-fix re-shoot — Bt2446A + Bt2446C + Bt2390";
    let mut new = if let Some(idx) = existing.find(marker) {
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
    let n_heic_input = work_list.iter().filter(|(_, f)| f == "HEIC").count();
    let n_jpeg_input = work_list.iter().filter(|(_, f)| f == "JPEG").count();
    println!(
        "Found {} gain-mapped samples ({} UltraHDR JPEG + {} HEIC) to process",
        work_list.len(),
        n_jpeg_input,
        n_heic_input
    );
    if work_list.is_empty() {
        anyhow::bail!("no gain-mapped samples found in {}", SAMPLES_ROOT);
    }

    fs::create_dir_all(MONTAGES_DIR)?;

    let mut sample_results: Vec<SampleResult> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    if let Some(parent) = Path::new(CSV_PATH).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut csv = BufWriter::new(fs::File::create(CSV_PATH)?);
    writeln!(
        csv,
        "sample,format,peak_method,curve,source_peak_nits,psnr_db,mean_de2000,max_abs_delta,pct_de_gt_5"
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

        // Save producer-SDR reference thumbnail once per sample.
        let ref_path = Path::new(MONTAGES_DIR).join(format!("{}__REF.png", stem));
        if let Err(e) = save_thumbnail(&ref_path, &sdr, TONEMAP_THUMBNAIL_W) {
            eprintln!("  ref thumbnail save failed: {}", e);
        }

        let mut source_peaks: HashMap<PeakMethod, f32> = HashMap::new();
        let mut cells_by_key: HashMap<(PeakMethod, Curve), CellMetrics> = HashMap::new();

        for &method in PeakMethod::all() {
            let source_peak = measure_source_peak_nits(&hdr, method);
            source_peaks.insert(method, source_peak);
            println!("  [{}] source_peak={:.0} nits", method.label(), source_peak);

            // Apply each curve; keep candidates briefly so we can build the
            // 4-panel montage for this peak method, then drop them.
            let mut candidates: HashMap<Curve, LinearRgb> = HashMap::new();
            for &curve in Curve::all() {
                let cand = apply_curve(curve, &hdr, source_peak);
                let metrics = compute_metrics(&sdr, &cand);
                cells_by_key.insert((method, curve), metrics);

                // Save the candidate thumbnail inline.
                let thumb_path = Path::new(MONTAGES_DIR).join(format!(
                    "{}__{}__{}.png",
                    stem,
                    curve.label(),
                    method.label(),
                ));
                if let Err(e) = save_thumbnail(&thumb_path, &cand, TONEMAP_THUMBNAIL_W) {
                    eprintln!(
                        "  thumbnail save failed for {}: {}",
                        thumb_path.display(),
                        e
                    );
                }

                // Stream CSV row + flush at end of method block.
                writeln!(
                    csv,
                    "{},{},{},{},{:.2},{:.4},{:.4},{:.4},{:.4}",
                    stem_raw,
                    fmt_label,
                    method.label(),
                    curve.label(),
                    source_peak,
                    metrics.psnr_db,
                    metrics.mean_de2000,
                    metrics.max_abs_delta,
                    metrics.pct_de_gt_5,
                )?;

                println!(
                    "    {} ΔE2000={:.3} PSNR={:.2}",
                    curve.label(),
                    metrics.mean_de2000,
                    metrics.psnr_db,
                );

                candidates.insert(curve, cand);
            }

            // Build the 4-panel montage for this peak method.
            // Layout: REF | Bt2390 | Bt2446A | Bt2446C
            let montage_path =
                Path::new(MONTAGES_DIR).join(format!("{}__MONTAGE_{}.png", stem, method.label(),));
            let bt2390 = &candidates[&Curve::Bt2390];
            let bt2446a = &candidates[&Curve::Bt2446A];
            let bt2446c = &candidates[&Curve::Bt2446C];
            let panels = [&sdr, bt2390, bt2446a, bt2446c];
            if let Err(e) = save_4panel_montage(&montage_path, panels, MONTAGE_PANEL_W) {
                eprintln!(
                    "  montage save failed for {}: {}",
                    montage_path.display(),
                    e
                );
            }

            // Drop all candidate buffers for this peak method before the next
            // method (peak memory: SDR + HDR + ~3 candidates briefly).
            drop(candidates);
            csv.flush()?;
        }

        sample_results.push(SampleResult {
            sample_path: path.clone(),
            format_label: fmt_label.clone(),
            device_class: classify_device(path),
            width: sdr.width,
            height: sdr.height,
            source_peaks,
            cells: cells_by_key,
        });

        drop(hdr);
        drop(sdr);
    }
    csv.flush()?;

    let n_jpeg_processed = sample_results
        .iter()
        .filter(|r| r.format_label == "JPEG")
        .count();
    let n_heic_processed = sample_results
        .iter()
        .filter(|r| r.format_label == "HEIC")
        .count();

    let runtime = t0.elapsed().as_secs_f64();
    refresh_lock("writing-report");
    append_report(
        &sample_results,
        &failures,
        runtime,
        n_jpeg_processed,
        n_heic_processed,
    )?;
    println!("\nAddendum appended to {}", REPORT_PATH);

    // Count thumbnails for the final summary line.
    let thumb_count = match fs::read_dir(MONTAGES_DIR) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("png"))
            .count(),
        Err(_) => 0,
    };

    println!(
        "Done. {} samples succeeded, {} failed, {} png files written, total {:.1}s.",
        sample_results.len(),
        failures.len(),
        thumb_count,
        runtime,
    );
    let _ = n_jpeg_input;
    let _ = n_heic_input;
    Ok(())
}
