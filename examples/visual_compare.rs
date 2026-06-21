//! Comprehensive visual comparison of all zentone tonemappers.
//!
//! Generates synthetic HDR test scenes plus a real-image set drawn from the
//! imazen-26 PQ-encoded PNG-3.0 HDR corpus, applies every tonemapper, computes
//! zensim scores, saves individual PNGs and a montage. The real-image scene
//! directories also carry an `00_hdr_native.png` — a copy of the source PNG
//! 3.0 (cICP=BT.709 primaries + PQ transfer, 16-bit) that the OS renders as
//! native HDR on a compliant display.
//!
//! Run:
//! ```
//! cargo run --example visual_compare --features experimental --release
//! ```

use std::path::{Path, PathBuf};

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

/// Scene 4: mostly SDR room with bright window highlights (4× peak).
/// Tests: SDR preservation, localized highlight compression.
/// ~80% of pixels are in [0, 0.8], highlights reach 4.0.
fn scene_room_window() -> Vec<f32> {
    let mut img = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
    for y in 0..HEIGHT {
        let v = y as f32 / HEIGHT as f32;
        for x in 0..WIDTH {
            let t = x as f32 / WIDTH as f32;
            let idx = ((y * WIDTH + x) * 3) as usize;

            // Window region: upper-right quadrant with HDR highlights
            let in_window = t > 0.55 && t < 0.95 && v < 0.45;
            // Gradient within window: bright center fading to frame
            let wx = ((t - 0.75) / 0.2).abs();
            let wy = ((v - 0.2) / 0.25).abs();
            let w_dist = (wx * wx + wy * wy).min(1.0);

            if in_window {
                // HDR sky through window: blue-white, up to 4×
                let intensity = 1.0 + 3.0 * (1.0 - w_dist);
                img[idx] = intensity * 0.85;
                img[idx + 1] = intensity * 0.9;
                img[idx + 2] = intensity;
            } else {
                // Room interior: warm SDR lighting, 0.05–0.6
                let base = 0.05 + 0.4 * (1.0 - v) * (0.3 + 0.7 * t);
                // Warm tint (incandescent light)
                img[idx] = base * 1.1;
                img[idx + 1] = base * 0.9;
                img[idx + 2] = base * 0.7;
                // Floor reflection near window
                if v > 0.5 && t > 0.5 {
                    let refl = 0.2 * (1.0 - (v - 0.5) * 2.0).max(0.0) * (t - 0.5) * 2.0;
                    img[idx] += refl * 0.8;
                    img[idx + 1] += refl * 0.85;
                    img[idx + 2] += refl * 0.9;
                }
            }
        }
    }
    img
}

// ============================================================================
// imazen-26 real-image HDR scenes (PQ-encoded 16-bit PNGs, cICP signalled)
// ============================================================================

/// Per-scene HDR mastering peak (cd/m²). The imazen-26 HDR corpus is mastered
/// at 1000 nits; the demo tonemappers are configured at 8000 — so we feed the
/// 1000-nit-bounded linear data normalized so `1.0 ≈ hdr_peak_for_tonemapper`.
const HDR_REAL_PEAK_NITS: f32 = 1000.0;

/// The four representative imazen-26 scenes we ingest. Hand-picked to cover
/// distinct DR/colour regimes the tonemap curves should differentiate on:
/// stained-glass interior (massive intra-frame DR), night-with-highlight,
/// sunset roll-off, controlled food close-up.
const IMAZEN26_REAL_SCENES: &[(&str, &str)] = &[
    (
        "imazen26_stained_glass",
        "/mnt/v/output/imazen-26-hdr-2026-06-14/1200-lilith-interiors/1227_interiors_stained-glass-window_casa-batllo-barcelona_zfold7_iso1600-f1p7_20260315-205230_3000x4000.hdr.png",
    ),
    (
        "imazen26_night_castle",
        "/mnt/v/output/imazen-26-hdr-2026-06-14/1000-lilith-photos-general/1069_general_illuminated-castle-night_cinderella-castle-orange_ip13pro_iso500-f1p5_img-7653_2870x3827.hdr.png",
    ),
    (
        "imazen26_ocean_sunset",
        "/mnt/v/output/imazen-26-hdr-2026-06-14/1400-lilith-nature/1490_nature_boat-wake-at-sunset_unorganized-borough-alaska_s25u_iso200-f1p7_20250804-211018_3000x4000.hdr.png",
    ),
    (
        "imazen26_sashimi",
        "/mnt/v/output/imazen-26-hdr-2026-06-14/1600-lilith-food/1639_food_sashimi-on-plate_colorado_s25u_iso200-f1p7_20250425-214145_4000x3000.hdr.png",
    ),
];

/// SMPTE ST 2084 PQ inverse-EOTF: PQ-encoded `[0, 1]` → absolute cd/m².
fn pq_inverse_eotf(pq_signal: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = (2523.0 / 4096.0) * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = (2413.0 / 4096.0) * 32.0;
    const C3: f32 = (2392.0 / 4096.0) * 32.0;

    if pq_signal <= 0.0 {
        return 0.0;
    }
    let xp = pq_signal.powf(1.0 / M2);
    let num = (xp - C1).max(0.0);
    let den = (C2 - C3 * xp).max(f32::MIN_POSITIVE);
    let y = (num / den).powf(1.0 / M1);
    10_000.0 * y
}

/// Load one imazen-26 HDR PNG and return its native-resolution path plus a
/// linear-light, demo-aspect-resized `Vec<f32>` normalized so that
/// `1.0 = HDR_REAL_PEAK_NITS` (the corpus mastering peak).
///
/// Real images go through PQ → cd/m² → divide-by-peak; the tonemappers we
/// pair them with then see the same `[0, 1]` (+ optional supra-1 specular)
/// range the synthetic scenes use, so the existing apply_tonemap path is
/// reused unchanged.
fn load_imazen26_hdr(path: &Path) -> Option<Vec<f32>> {
    use image::imageops::{FilterType, resize};
    use image::{DynamicImage, ImageReader, Rgb};

    let img = ImageReader::open(path).ok()?.decode().ok()?;
    let rgb16: image::ImageBuffer<Rgb<u16>, Vec<u16>> = match img {
        DynamicImage::ImageRgb16(b) => b,
        DynamicImage::ImageRgba16(b) => DynamicImage::ImageRgba16(b).into_rgb16(),
        DynamicImage::ImageRgb8(b) => DynamicImage::ImageRgb8(b).into_rgb16(),
        other => other.into_rgb16(),
    };
    let resized = resize(&rgb16, WIDTH, HEIGHT, FilterType::Lanczos3);

    let mut out = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for px in resized.pixels() {
        for &c in &px.0 {
            let pq = (c as f32) / 65535.0;
            let nits = pq_inverse_eotf(pq);
            out.push(nits / HDR_REAL_PEAK_NITS);
        }
    }
    Some(out)
}

/// Return the imazen-26 scene set with each entry's source path attached so
/// `main()` can copy the native PNG-3.0 HDR alongside the tonemapped panels.
fn imazen26_scenes() -> Vec<(&'static str, Vec<f32>, PathBuf)> {
    IMAZEN26_REAL_SCENES
        .iter()
        .filter_map(|(name, path_str)| {
            let path = PathBuf::from(path_str);
            if !path.exists() {
                eprintln!("  skip {name} — source missing at {path_str}");
                return None;
            }
            let hdr = load_imazen26_hdr(&path)?;
            Some((*name, hdr, path))
        })
        .collect()
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
        // Bt2446A moved to zenpixels_convert::hdr — no longer in this roster.
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

    // Synthetic scenes: third tuple slot is `None` (no native-HDR source PNG).
    let mut scenes: Vec<(String, Vec<f32>, Option<PathBuf>)> = vec![
        ("ramp".into(), scene_ramp(), None),
        ("hue_wheel".into(), scene_hue_wheel(), None),
        ("natural".into(), scene_natural(), None),
        ("room_window".into(), scene_room_window(), None),
    ];
    // Real-image scenes: drawn from the imazen-26 PQ-encoded PNG-3.0 HDR corpus.
    // The corresponding source PNG path rides along so each scene dir gets an
    // `00_hdr_native.png` copy alongside the SDR tonemap variants — that copy
    // still carries its `cICP=BT.709-primaries + PQ-transfer` chunk, so a modern
    // OS (Windows 11, macOS 14+) renders it as native HDR on an HDR display.
    eprintln!("Loading imazen-26 real-image HDR scenes…");
    scenes.extend(
        imazen26_scenes()
            .into_iter()
            .map(|(n, h, p)| (n.into(), h, Some(p))),
    );

    let zs = Zensim::new(ZensimProfile::latest());
    let tonemappers = all_tonemappers();

    for (scene_name, hdr, source_hdr_png) in &scenes {
        let scene_dir = out_dir.join(scene_name);
        std::fs::create_dir_all(&scene_dir).unwrap();

        // Copy the native PNG-3.0 HDR source next to the SDR panels so an
        // HDR-capable OS can display the actual HDR content (the tonemapped
        // panels next to it are the SDR comparison views).
        if let Some(src) = source_hdr_png {
            let dst = scene_dir.join("00_hdr_native.png");
            if let Err(e) = std::fs::copy(src, &dst) {
                eprintln!(
                    "  warning: could not copy {} → {}: {e}",
                    src.display(),
                    dst.display()
                );
            }
        }

        // Save clamped reference
        let clamped: Vec<f32> = hdr.iter().map(|v| v.clamp(0.0, 1.0)).collect();
        let clamped_u8 = to_srgb_u8(&clamped);
        let clamped_filename = if source_hdr_png.is_some() {
            "00b_clamped.png" // 00_ is reserved for the native-HDR source copy
        } else {
            "00_clamped.png"
        };
        save_png(
            &scene_dir.join(clamped_filename),
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

        // Build montage with ImageMagick if available. Skip the native-HDR
        // source PNG (PQ + cICP) — ImageMagick would tone-shift it into the
        // SDR montage. The native HDR file stays alongside as `00_hdr_native.png`.
        let montage_files: Vec<String> = std::fs::read_dir(&scene_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path().display().to_string())
            .filter(|p| p.ends_with(".png") && !p.contains("montage") && !p.contains("hdr_native"))
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
