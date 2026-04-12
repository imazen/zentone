//! Visual comparison of all zentone tonemappers on a synthetic HDR gradient.
//!
//! Produces PNG files at `/mnt/v/output/zentone/streaming_visual/`.
//!
//! Run:
//! ```
//! cargo run --example visual_compare --features experimental
//! ```

use std::path::Path;

use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
use zentone::*;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const HDR_PEAK: f32 = 8.0; // 8× SDR white

/// Generate a synthetic HDR test image:
/// - Horizontal axis: luminance ramp from 0 to HDR_PEAK
/// - Vertical axis: hue rotation (top = pure ramp, middle = colored, bottom = saturated)
fn generate_hdr_image() -> Vec<f32> {
    let mut img = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
    for y in 0..HEIGHT {
        let v = y as f32 / HEIGHT as f32;
        for x in 0..WIDTH {
            let t = x as f32 / WIDTH as f32;
            let lum = t * HDR_PEAK;

            // Vary color by vertical position
            let (r, g, b) = if v < 0.33 {
                // Top third: neutral gray ramp
                (lum, lum, lum)
            } else if v < 0.66 {
                // Middle third: warm-to-cool gradient
                let hue = v * 6.0;
                (
                    lum * (0.5 + 0.5 * (hue * 0.7).sin()),
                    lum * (0.5 + 0.3 * (hue * 1.1).cos()),
                    lum * (0.3 + 0.7 * (hue * 0.5).sin()),
                )
            } else {
                // Bottom third: saturated primaries cycling
                let phase = t * 3.0;
                if phase < 1.0 {
                    (lum, lum * 0.1, lum * 0.1)
                } else if phase < 2.0 {
                    (lum * 0.1, lum, lum * 0.1)
                } else {
                    (lum * 0.1, lum * 0.1, lum)
                }
            };

            let idx = ((y * WIDTH + x) * 3) as usize;
            img[idx] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }
    img
}

/// Apply sRGB OETF and convert to u8
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let encoded = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn save_rgb_png(path: &Path, data: &[f32], width: u32, height: u32) {
    let mut bytes = vec![0u8; (width * height * 3) as usize];
    for (i, v) in data.iter().enumerate() {
        bytes[i] = linear_to_srgb_u8(*v);
    }
    image::save_buffer(path, &bytes, width, height, image::ColorType::Rgb8).unwrap();
    println!("  saved {}", path.display());
}

fn apply_curve_to_image(curve: &dyn ToneMap, src: &[f32]) -> Vec<f32> {
    let mut out = src.to_vec();
    // Process row by row
    let row_stride = WIDTH as usize * 3;
    for row in out.chunks_exact_mut(row_stride) {
        curve.map_row(row, 3);
    }
    out
}

fn apply_streaming(src: &[f32]) -> Vec<f32> {
    let cfg = StreamingTonemapConfig::default();
    let mut tm = StreamingTonemapper::new(WIDTH, HEIGHT, 3, cfg).unwrap();
    let row_stride = tm.row_stride();
    let mut out_img = vec![0.0f32; src.len()];
    let mut out_row = vec![0.0f32; row_stride];
    let mut write_offset = 0;

    for row in src.chunks_exact(row_stride) {
        tm.push_row(row).unwrap();
        while let Some(_idx) = tm.pull_row(&mut out_row).unwrap() {
            out_img[write_offset..write_offset + row_stride].copy_from_slice(&out_row);
            write_offset += row_stride;
        }
    }
    tm.finish();
    while let Some(_idx) = tm.pull_row(&mut out_row).unwrap() {
        out_img[write_offset..write_offset + row_stride].copy_from_slice(&out_row);
        write_offset += row_stride;
    }
    out_img
}

fn main() {
    let out_dir = Path::new("/mnt/v/output/zentone/streaming_visual");
    std::fs::create_dir_all(out_dir).unwrap();

    println!("Generating {WIDTH}×{HEIGHT} synthetic HDR image (peak {HDR_PEAK}×)...");
    let hdr = generate_hdr_image();

    // Save a naively-clamped version for reference
    println!("Saving reference (clamped to [0,1])...");
    save_rgb_png(
        &out_dir.join("00_clamped.png"),
        &hdr.iter().map(|v| v.clamp(0.0, 1.0)).collect::<Vec<_>>(),
        WIDTH,
        HEIGHT,
    );

    // Classical curves
    let curves: Vec<(&str, Box<dyn ToneMap>)> = vec![
        ("01_reinhard", Box::new(ToneMapCurve::Reinhard)),
        (
            "02_reinhard_ext",
            Box::new(ToneMapCurve::ExtendedReinhard {
                l_max: HDR_PEAK,
                luma: LUMA_BT709,
            }),
        ),
        ("03_narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("04_uncharted2", Box::new(ToneMapCurve::Uncharted2)),
        ("05_aces_ap1", Box::new(ToneMapCurve::AcesAp1)),
        (
            "06_agx_default",
            Box::new(ToneMapCurve::Agx(AgxLook::Default)),
        ),
        (
            "07_agx_punchy",
            Box::new(ToneMapCurve::Agx(AgxLook::Punchy)),
        ),
        (
            "08_bt2408_4k_1k",
            Box::new(Bt2408Tonemapper::with_luma(4000.0, 1000.0, LUMA_BT709)),
        ),
        (
            "09_bt2408_10k_203",
            Box::new(Bt2408Tonemapper::with_luma(10000.0, 203.0, LUMA_BT709)),
        ),
        (
            "10_filmic_default",
            Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
        ),
    ];

    for (name, curve) in &curves {
        println!("Applying {name}...");
        let out = apply_curve_to_image(curve.as_ref(), &hdr);
        save_rgb_png(&out_dir.join(format!("{name}.png")), &out, WIDTH, HEIGHT);
    }

    // Streaming tonemapper
    println!("Applying streaming tonemapper (local adaptation)...");
    let out = apply_streaming(&hdr);
    save_rgb_png(&out_dir.join("11_streaming.png"), &out, WIDTH, HEIGHT);

    println!(
        "\nDone! {} files in {}",
        curves.len() + 2,
        out_dir.display()
    );
    println!("Open with: feh {}/*.png", out_dir.display());
}
