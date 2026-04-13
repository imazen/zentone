//! Comprehensive visual comparison of all zentone tonemappers.
//!
//! Generates multiple synthetic HDR test scenes, applies every tonemapper,
//! computes zensim scores, saves individual PNGs and a montage.
//!
//! Run:
//! ```
//! cargo run --example visual_compare --features experimental --release
//! ```

use std::path::Path;

use zensim::{RgbSlice, Zensim, ZensimProfile};
use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
use zentone::*;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const OUT_DIR: &str = "/mnt/v/output/zentone/visual_compare";

// ============================================================================
// Synthetic HDR test scenes
// ============================================================================

/// Scene 1: luminance ramp with neutral / colored / saturated bands.
/// Tests: highlight rolloff, black level, saturation handling.
fn scene_ramp() -> Vec<f32> {
    let peak = 8.0;
    let mut img = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
    for y in 0..HEIGHT {
        let v = y as f32 / HEIGHT as f32;
        for x in 0..WIDTH {
            let t = x as f32 / WIDTH as f32;
            let lum = t * peak;
            let (r, g, b) = if v < 0.33 {
                (lum, lum, lum) // neutral gray
            } else if v < 0.66 {
                let hue = v * 6.0;
                (
                    lum * (0.5 + 0.5 * (hue * 0.7).sin()),
                    lum * (0.5 + 0.3 * (hue * 1.1).cos()),
                    lum * (0.3 + 0.7 * (hue * 0.5).sin()),
                )
            } else {
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

/// Scene 2: hue wheel with continuous luminance gradient.
/// Horizontal: hue rotation (0 to 2π). Vertical: luminance (0.1 to 10.0).
/// Tests: hue preservation across the full luminance range.
fn scene_hue_wheel() -> Vec<f32> {
    let mut img = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
    for y in 0..HEIGHT {
        // Continuous log-spaced luminance from 0.1 to 10.0
        let t_y = y as f32 / (HEIGHT - 1) as f32;
        let lum = 0.1 * (100.0_f32).powf(t_y); // 0.1 → 10.0
        for x in 0..WIDTH {
            let hue = x as f32 / WIDTH as f32 * core::f32::consts::TAU;
            let r = lum * (0.5 + 0.5 * hue.cos()).max(0.0);
            let g = lum * (0.5 + 0.5 * (hue - 2.094).cos()).max(0.0);
            let b = lum * (0.5 + 0.5 * (hue + 2.094).cos()).max(0.0);
            let idx = ((y * WIDTH + x) * 3) as usize;
            img[idx] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }
    img
}

/// Scene 3: soft highlight gradient (skin tones + sky-like content).
/// Tests: natural-content behavior, subtle gradients.
fn scene_natural() -> Vec<f32> {
    let mut img = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
    for y in 0..HEIGHT {
        let v = y as f32 / HEIGHT as f32;
        for x in 0..WIDTH {
            let t = x as f32 / WIDTH as f32;
            // Sky-like gradient (top) to skin-like (bottom)
            let sky_r = 0.3 + t * 2.0;
            let sky_g = 0.5 + t * 1.5;
            let sky_b = 0.8 + t * 3.0;
            let skin_r = 0.8 + t * 1.5;
            let skin_g = 0.5 + t * 0.8;
            let skin_b = 0.3 + t * 0.5;
            let r = sky_r * (1.0 - v) + skin_r * v;
            let g = sky_g * (1.0 - v) + skin_g * v;
            let b = sky_b * (1.0 - v) + skin_b * v;
            let idx = ((y * WIDTH + x) * 3) as usize;
            img[idx] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }
    img
}

// ============================================================================
// All tonemappers
// ============================================================================

fn all_tonemappers() -> Vec<(&'static str, Box<dyn ToneMap>)> {
    vec![
        ("Reinhard", Box::new(ToneMapCurve::Reinhard)),
        (
            "ExtReinhard",
            Box::new(ToneMapCurve::ExtendedReinhard {
                l_max: 8.0,
                luma: LUMA_BT709,
            }),
        ),
        (
            "ReinhardJodie",
            Box::new(ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 }),
        ),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic)),
        ("AcesAp1", Box::new(ToneMapCurve::AcesAp1)),
        ("AgxDefault", Box::new(ToneMapCurve::Agx(AgxLook::Default))),
        ("AgxPunchy", Box::new(ToneMapCurve::Agx(AgxLook::Punchy))),
        ("AgxGolden", Box::new(ToneMapCurve::Agx(AgxLook::Golden))),
        ("Bt2408", Box::new(Bt2408Tonemapper::new(8000.0, 1000.0))),
        (
            "Bt2408_MaxRgb",
            Box::new(Bt2408Tonemapper::max_rgb(8000.0, 1000.0)),
        ),
        ("Bt2446A", Box::new(Bt2446A::new(8000.0, 100.0))),
        ("Bt2446B", Box::new(Bt2446B::new(8000.0, 100.0))),
        ("Bt2446C", Box::new(Bt2446C::new(8000.0, 100.0))),
        (
            "FilmicSpline",
            Box::new(CompiledFilmicSpline::for_hdr_peak(10.0)),
        ),
    ]
}

// ============================================================================
// Helpers
// ============================================================================

fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = linear_srgb::tf::linear_to_srgb(v);
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn to_srgb_u8(img: &[f32]) -> Vec<u8> {
    img.iter().map(|v| linear_to_srgb_u8(*v)).collect()
}

fn to_rgb_triples(img: &[u8]) -> Vec<[u8; 3]> {
    img.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn save_png(path: &Path, data: &[u8], w: u32, h: u32) {
    image::save_buffer(path, data, w, h, image::ColorType::Rgb8).unwrap();
}

fn apply_tonemap(tm: &dyn ToneMap, src: &[f32]) -> Vec<f32> {
    let mut out = src.to_vec();
    let stride = WIDTH as usize * 3;
    for row in out.chunks_exact_mut(stride) {
        tm.map_row(row, 3);
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

// ============================================================================
// Main
// ============================================================================

fn main() {
    let out_dir = Path::new(OUT_DIR);
    std::fs::create_dir_all(out_dir).unwrap();

    let scenes: Vec<(&str, Vec<f32>)> = vec![
        ("ramp", scene_ramp()),
        ("hue_wheel", scene_hue_wheel()),
        ("natural", scene_natural()),
    ];

    let zs = Zensim::new(ZensimProfile::latest());
    let tonemappers = all_tonemappers();

    for (scene_name, hdr) in &scenes {
        let scene_dir = out_dir.join(scene_name);
        std::fs::create_dir_all(&scene_dir).unwrap();

        // Save clamped reference
        let clamped: Vec<f32> = hdr.iter().map(|v| v.clamp(0.0, 1.0)).collect();
        let clamped_u8 = to_srgb_u8(&clamped);
        save_png(
            &scene_dir.join("00_clamped.png"),
            &clamped_u8,
            WIDTH,
            HEIGHT,
        );

        // Reinhard as the similarity reference
        let ref_output = apply_tonemap(&ToneMapCurve::Reinhard, hdr);
        let ref_u8 = to_srgb_u8(&ref_output);
        let ref_triples = to_rgb_triples(&ref_u8);
        let ref_src = RgbSlice::new(&ref_triples, WIDTH as usize, HEIGHT as usize);

        println!("\n=== {scene_name} ===");
        println!("{:<16} {:>8} {:>10}", "Curve", "Zensim", "Profile");
        println!("{}", "-".repeat(40));

        let mut results: Vec<(&str, Vec<u8>, f64)> = Vec::new();

        for (name, tm) in &tonemappers {
            let output = apply_tonemap(tm.as_ref(), hdr);
            let out_u8 = to_srgb_u8(&output);
            let out_triples = to_rgb_triples(&out_u8);
            let out_src = RgbSlice::new(&out_triples, WIDTH as usize, HEIGHT as usize);

            let score = zs
                .compute(&ref_src, &out_src)
                .map(|r| r.score())
                .unwrap_or(f64::NAN);

            let profile = if score > 90.0 {
                "similar"
            } else if score > 70.0 {
                "moderate"
            } else if score > 50.0 {
                "different"
            } else {
                "very different"
            };

            println!("{:<16} {:>8.1} {:>10}", name, score, profile);

            // Save individual PNG
            let idx = results.len() + 1;
            save_png(
                &scene_dir.join(format!("{idx:02}_{name}.png")),
                &out_u8,
                WIDTH,
                HEIGHT,
            );

            results.push((name, out_u8, score));
        }

        // Streaming tonemapper
        let streaming_output = apply_streaming(hdr);
        let streaming_u8 = to_srgb_u8(&streaming_output);
        let streaming_triples = to_rgb_triples(&streaming_u8);
        let streaming_src = RgbSlice::new(&streaming_triples, WIDTH as usize, HEIGHT as usize);
        let streaming_score = zs
            .compute(&ref_src, &streaming_src)
            .map(|r| r.score())
            .unwrap_or(f64::NAN);
        println!(
            "{:<16} {:>8.1} {:>10}",
            "Streaming",
            streaming_score,
            if streaming_score > 70.0 {
                "moderate"
            } else {
                "different"
            }
        );
        let idx = results.len() + 1;
        save_png(
            &scene_dir.join(format!("{idx:02}_streaming.png")),
            &streaming_u8,
            WIDTH,
            HEIGHT,
        );

        // Build montage with ImageMagick if available
        let montage_files: Vec<String> = std::fs::read_dir(&scene_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path().display().to_string())
            .filter(|p| p.ends_with(".png") && !p.contains("montage"))
            .collect::<Vec<_>>();

        if !montage_files.is_empty() {
            let mut sorted = montage_files;
            sorted.sort();
            let montage_path = scene_dir.join("montage.png");
            let status = std::process::Command::new("montage")
                .args(&sorted)
                .args([
                    "-geometry",
                    "480x270+2+2",
                    "-tile",
                    "4x",
                    "-background",
                    "black",
                    "-font",
                    "DejaVu-Sans",
                    "-pointsize",
                    "14",
                ])
                .arg(montage_path.to_str().unwrap())
                .status();

            match status {
                Ok(s) if s.success() => println!("  montage → {}", montage_path.display()),
                _ => println!("  (montage skipped — ImageMagick not available)"),
            }
        }
    }

    println!("\nDone! Output in {OUT_DIR}");
    println!("View: feh {OUT_DIR}/*/*.png");
}
