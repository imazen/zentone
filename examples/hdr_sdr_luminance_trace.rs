//! Trace HDR and producer-SDR luminance for a single sample so we can
//! compare them to each curve's output. Helps diagnose "why does the
//! ΔE2000 say A wins when the visual review says A is too dark?"
//!
//! Gated behind `hdr-shootout` to share the zencodecs/zenpixels-dev deps
//! with the other shootout examples. Not part of the published API.
//!
//! Run:
//! ```text
//! nice -n19 cargo run --example hdr_sdr_luminance_trace --release \
//!   --features hdr-shootout -- \
//!   /home/lilith/work/codec-corpus/imazen-26/1400-lilith-nature/1542_nature_orange-lily-flower_belltown-seattle-washington_ip16pro_iso80-f1p8_img-9149_5712x4284.heic
//! ```

use std::path::PathBuf;

use zencodecs::{DecodeRequest, OrientationHint};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_dev::hdr::ContentLightLevel;

use zenpixels_convert::hdr::Bt2446A;
use zentone::{Bt2446C, ToneMap, ToneMapCurve};

const DIFFUSE_WHITE_NITS: f32 = 203.0;
const TARGET_PEAK_NITS: f32 = 100.0;

fn pixel_buffer_to_linear_rgb(buf: &PixelBuffer) -> anyhow::Result<(u32, u32, Vec<f32>)> {
    let src_desc = buf.descriptor();
    let target = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        src_desc.primaries,
    );

    let owned;
    let linear: &PixelBuffer = if src_desc == target {
        buf
    } else {
        owned = buf
            .convert_to(target)
            .map_err(|e| anyhow::anyhow!("convert_to linear RGB f32: {:?}", e.error()))?;
        &owned
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

    Ok((width, height, tight))
}

fn stats(label: &str, px: &[f32]) {
    // Luminance via BT.709 weights (matches what most ΔE math implicitly assumes).
    let n_px = px.len() / 3;
    let mut max_l = 0.0_f32;
    let mut min_l = f32::INFINITY;
    let mut sum: f64 = 0.0;
    let mut max_ch: f32 = 0.0;
    for i in 0..n_px {
        let r = px[i * 3];
        let g = px[i * 3 + 1];
        let b = px[i * 3 + 2];
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if y > max_l {
            max_l = y;
        }
        if y < min_l {
            min_l = y;
        }
        sum += y as f64;
        for v in [r, g, b] {
            if v > max_ch {
                max_ch = v;
            }
        }
    }
    let mean = sum / n_px as f64;
    println!(
        "  {:>14}: max-Y = {:.4}, mean-Y = {:.4}, min-Y = {:.4}, max-ch = {:.4}",
        label, max_l, mean as f32, min_l, max_ch
    );
}

fn apply_curve(curve_name: &str, hdr: &[f32], source_peak_nits: f32) -> Vec<f32> {
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;
    let mut scratch: Vec<f32> = hdr.iter().map(|&v| v * content_norm_scale).collect();

    match curve_name {
        "bt2390" => {
            let target_peak_in_src = TARGET_PEAK_NITS / source_peak_nits.max(1.0);
            let tone = ToneMapCurve::Bt2390 {
                source_peak: 1.0,
                target_peak: target_peak_in_src,
            };
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tone.map_strip_simd(strip);
            let inv = 1.0 / target_peak_in_src.max(1e-6);
            for v in scratch.iter_mut() {
                *v *= inv;
            }
        }
        "bt2446a" => {
            let tm = Bt2446A::new(source_peak_nits, TARGET_PEAK_NITS);
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        "bt2446c" => {
            let tm = Bt2446C::new(source_peak_nits, TARGET_PEAK_NITS);
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
        }
        "bt2446c_no_clamp_in_apply" => {
            // Same as bt2446c but DON'T clamp to [0, 1.0] at the end.
            let tm = Bt2446C::new(source_peak_nits, TARGET_PEAK_NITS);
            let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut scratch);
            tm.map_strip_simd(strip);
            // Return un-clamped (i.e. preserve [0, 1.09]).
            return scratch;
        }
        _ => unreachable!(),
    }

    for v in scratch.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    scratch
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("Usage: hdr_sdr_luminance_trace <sample-path>");
    }
    let path = PathBuf::from(&args[1]);
    println!("Tracing {:?}\n", path);

    let bytes = std::fs::read(&path)?;

    // Decode SDR + HDR.
    let sdr_out = DecodeRequest::new(&bytes)
        .with_orientation(OrientationHint::Correct)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR: {}", e.error()))?;
    let sdr_buf = sdr_out.into_buffer();
    let (sw, sh, sdr_px) = pixel_buffer_to_linear_rgb(&sdr_buf)?;
    println!("SDR base buffer: {}x{}", sw, sh);
    stats("producer-SDR", &sdr_px);

    let hdr_out = DecodeRequest::new(&bytes)
        .with_orientation(OrientationHint::Correct)
        .reconstruct_hdr(None)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("reconstruct_hdr: {}", e.error()))?;
    let hdr_buf = hdr_out.into_buffer();
    let (hw, hh, hdr_px) = pixel_buffer_to_linear_rgb(&hdr_buf)?;
    println!("HDR reconstructed buffer: {}x{}", hw, hh);
    stats("reconstructed-HDR", &hdr_px);

    // Measure source peak using same method as production (measure_robust).
    let bytes_hdr: Vec<u8> = bytemuck::cast_slice(&hdr_px).to_vec();
    let desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let buf = PixelBuffer::from_vec(bytes_hdr, hw, hh, desc)?;
    let cll = ContentLightLevel::measure_robust(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    )
    .unwrap_or_else(|| ContentLightLevel::new(203, 100));
    let source_peak_nits = (cll.max_content_light_level as f32).max(203.0);
    println!(
        "\nmeasure_robust source peak: {:.0} nits (= MaxCLL)\n",
        source_peak_nits
    );

    // What does the HDR pixel at the corpus's diffuse-white anchor (203 nits =
    // buffer value 1.0) translate to in nits with the apply_curve scaling?
    let max_pixel_value = source_peak_nits / DIFFUSE_WHITE_NITS;
    println!(
        "apply_curve content_norm_scale {:.4}",
        1.0 / max_pixel_value
    );
    println!(
        "  → buffer 1.0 (= {} nits) → scratch value {:.4} (= 1/{:.2})",
        DIFFUSE_WHITE_NITS,
        1.0 / max_pixel_value,
        max_pixel_value
    );

    // Apply each curve and print stats.
    println!("\nPer-curve output stats (clamped [0,1] as in apply_curve):");
    for &name in &["bt2390", "bt2446a", "bt2446c"] {
        let mapped = apply_curve(name, &hdr_px, source_peak_nits);
        stats(name, &mapped);
    }

    // Hypothesis test: rescale Bt2446C's [0, 1.09] super-white range
    // down to [0, 1.0] by dividing by 1.09 (instead of clamping). This
    // preserves highlight detail at the cost of overall brightness.
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;
    let mut bt2446c_rescale: Vec<f32> = hdr_px.iter().map(|&v| v * content_norm_scale).collect();
    let tm = Bt2446C::new(source_peak_nits, TARGET_PEAK_NITS);
    let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut bt2446c_rescale);
    tm.map_strip_simd(strip);
    for v in bt2446c_rescale.iter_mut() {
        *v = (*v / 1.09).clamp(0.0, 1.0);
    }
    stats("bt2446c/1.09", &bt2446c_rescale);

    let mut bt2446c_rescale2: Vec<f32> = hdr_px.iter().map(|&v| v * content_norm_scale).collect();
    let tm = Bt2446C::new(source_peak_nits, TARGET_PEAK_NITS);
    let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut bt2446c_rescale2);
    tm.map_strip_simd(strip);
    for v in bt2446c_rescale2.iter_mut() {
        *v = (*v / 1.184).clamp(0.0, 1.0);
    }
    stats("bt2446c/1.184", &bt2446c_rescale2);

    // Bt2446C without the apply_curve clamp — shows how much detail is lost
    // in [1.0, 1.09].
    let bt2446c_raw = apply_curve("bt2446c_no_clamp_in_apply", &hdr_px, source_peak_nits);
    let n_super = bt2446c_raw.iter().filter(|&&v| v > 1.0).count();
    let max_super = bt2446c_raw.iter().copied().fold(0.0_f32, f32::max);
    println!(
        "\nBt2446C un-clamped: {} channels in [1.0, 1.09], max value = {:.4} (= {:.1} nits SDR-norm)",
        n_super,
        max_super,
        max_super * 100.0,
    );

    Ok(())
}
