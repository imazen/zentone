//! End-to-end Gain-MLP round-trip:
//!
//!   1. Generate (or load) an HDR / SDR image pair.
//!   2. Load a trained Gain-MLP bake (produced by
//!      `tools/train_gain_mlp.py`).
//!   3. Reconstruct HDR via `GainMapMlpDecoder` and report PSNR vs
//!      ground truth.
//!
//! USAGE
//! -----
//!
//! ```text
//! # 1. Train + dump the training pair as .npy:
//! python3 tools/train_gain_mlp.py --synthetic-pair \
//!     --out /tmp/gainmlp.bin --dump-pair /tmp/gainmlp --epochs 5000
//!
//! # 2. Reload the exact pair the trainer saw + reconstruct:
//! cargo run --release --example gainmlp_roundtrip --features gainmap-mlp -- \
//!     /tmp/gainmlp.bin --hdr /tmp/gainmlp_hdr.npy --sdr /tmp/gainmlp_sdr.npy
//! ```
//!
//! Reconstruction PSNR is reported in dB. Canham 2025 hits 48.5 dB on
//! their real-image corpus; the smoke test here on a 128 × 128
//! synthetic chromatic-noise pair achieves 52.7 dB at 5000 epochs.
//!
//! For real HDR/SDR pairs, point `--hdr` / `--sdr` at NumPy `.npy`
//! files (float32 (H, W, 3) row-major) produced by your own
//! pre-processor. The trainer's `--dump-pair` flag exists to make
//! the synthetic case reproducible; real pipelines convert their
//! HDR/SDR sources directly into .npy.

#![cfg(feature = "gainmap-mlp")]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::Model;
use zentone::{GainMapMlpConfig, GainMapMlpDecoder};

fn print_usage_and_exit() -> ExitCode {
    eprintln!("usage: gainmlp_roundtrip <bake.bin> --hdr <hdr.npy> --sdr <sdr.npy>");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return print_usage_and_exit();
    }
    let bake_path = PathBuf::from(&args[0]);
    let mut hdr_path: Option<PathBuf> = None;
    let mut sdr_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hdr" => {
                hdr_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--sdr" => {
                sdr_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => {
                eprintln!("unrecognised flag: {}", args[i]);
                return print_usage_and_exit();
            }
        }
    }

    let bake_bytes = match std::fs::read(&bake_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read bake {:?}: {e}", bake_path);
            return ExitCode::FAILURE;
        }
    };
    let aligned = AlignedBuf::from(bake_bytes);
    let model = match Model::from_bytes(aligned.as_slice()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to parse bake: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let hdr_p = match hdr_path {
        Some(p) => p,
        None => return print_usage_and_exit(),
    };
    let sdr_p = match sdr_path {
        Some(p) => p,
        None => return print_usage_and_exit(),
    };
    let (width, height, hdr, sdr) = match load_pair(&hdr_p, &sdr_p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load_pair failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut decoder = match GainMapMlpDecoder::new(
        &model,
        width as u32,
        height as u32,
        GainMapMlpConfig::default(),
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decoder construction failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut hdr_reconstructed = vec![0.0_f32; sdr.len()];
    if let Err(e) = decoder.apply_image(width as u32, height as u32, &sdr, &mut hdr_reconstructed) {
        eprintln!("apply_image failed: {e}");
        return ExitCode::FAILURE;
    }

    let psnr = peak_signal_noise_ratio(&hdr, &hdr_reconstructed);
    println!("Gain-MLP round-trip:");
    println!("  bake:        {:?} ({} bytes)", bake_path, aligned.len());
    println!("  image:       {width} × {height}");
    println!(
        "  hdr range:   [{:.4}, {:.4}]",
        min_finite(&hdr),
        max_finite(&hdr)
    );
    println!(
        "  recon range: [{:.4}, {:.4}]",
        min_finite(&hdr_reconstructed),
        max_finite(&hdr_reconstructed)
    );
    println!("  PSNR (dB):   {psnr:.2}");
    println!("  Canham 2025 paper PSNR target: 48.5 dB");

    ExitCode::SUCCESS
}

fn load_pair(
    hdr_path: &PathBuf,
    sdr_path: &PathBuf,
) -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
    // Light .npy reader covering the minimum f32 RGB case. Full npy
    // parsing lives in the Python tool; this just handles the trainer's
    // output shape for round-trip validation.
    let hdr_bytes = std::fs::read(hdr_path).map_err(|e| format!("read hdr: {e}"))?;
    let (h, w, hdr) = parse_npy_f32_rgb(&hdr_bytes)?;

    let sdr_bytes = std::fs::read(sdr_path).map_err(|e| format!("read sdr: {e}"))?;
    let (sh, sw, sdr) = parse_npy_f32_rgb(&sdr_bytes)?;
    if (sh, sw) != (h, w) {
        return Err(format!("hdr/sdr shape mismatch: ({h},{w}) vs ({sh},{sw})"));
    }
    Ok((w, h, hdr, sdr))
}

fn parse_npy_f32_rgb(bytes: &[u8]) -> Result<(usize, usize, Vec<f32>), String> {
    // NPY v1.0 header. We accept only the minimal shape this script
    // needs — float32, C-order, shape (H, W, 3).
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a NumPy .npy file".into());
    }
    let major = bytes[6];
    if major != 1 {
        return Err(format!("only NPY v1 supported, got v{major}"));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = core::str::from_utf8(&bytes[10..10 + header_len])
        .map_err(|e| format!("header utf8: {e}"))?;
    if !header.contains("'descr': '<f4'") && !header.contains("'descr': '|f4'") {
        return Err(format!("expected float32 descr, header: {header:?}"));
    }
    let shape_start = header.find('(').ok_or("no shape")? + 1;
    let shape_end = header[shape_start..].find(')').ok_or("no shape end")? + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                s.parse::<usize>().ok()
            }
        })
        .collect();
    if dims.len() != 3 || dims[2] != 3 {
        return Err(format!("expected (H, W, 3), got {dims:?}"));
    }
    let (h, w) = (dims[0], dims[1]);
    let data_start = 10 + header_len;
    let n = h * w * 3;
    let want = n * 4;
    if data_start + want > bytes.len() {
        return Err("truncated array".into());
    }
    let mut data = vec![0.0_f32; n];
    for (i, slot) in data.iter_mut().enumerate() {
        let off = data_start + i * 4;
        *slot = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    }
    Ok((h, w, data))
}

fn peak_signal_noise_ratio(reference: &[f32], reconstructed: &[f32]) -> f32 {
    let n = reference.len().min(reconstructed.len()) as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mut mse = 0.0_f64;
    let mut peak = 0.0_f32;
    for (r, rec) in reference.iter().zip(reconstructed.iter()) {
        let d = r - rec;
        mse += (d * d) as f64;
        if r.abs() > peak {
            peak = r.abs();
        }
    }
    mse /= n as f64;
    if mse == 0.0 {
        return f32::INFINITY;
    }
    let peak = peak.max(1.0); // avoid degenerate
    (10.0 * ((peak as f64).powi(2) / mse).log10()) as f32
}

fn min_finite(xs: &[f32]) -> f32 {
    let mut m = f32::INFINITY;
    for &x in xs {
        if x.is_finite() && x < m {
            m = x;
        }
    }
    m
}

fn max_finite(xs: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &x in xs {
        if x.is_finite() && x > m {
            m = x;
        }
    }
    m
}

#[repr(C, align(16))]
struct AlignedBuf(Vec<u8>);
impl AlignedBuf {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
