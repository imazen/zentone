//! Flicker-test renders for HDR tone-mapping percentile sweep across the
//! full 76-sample imazen-26 shootout corpus (or a user-supplied subset).
//!
//! Per sample, renders 1 REF + 11 Bt2446A variant full-resolution PNGs:
//!   * `<stem>__REF.png`                  — producer SDR base (ground truth)
//!   * `bt2446a__max.png`                 — `measure_max` peak
//!   * `bt2446a__robust.png`              — `measure_robust` peak (=0.9999)
//!   * `bt2446a__smoothed.png`            — `measure_max_smoothed` peak
//!   * `bt2446a__blend25/50/75.png`       — linear blends robust↔max
//!   * `bt2446a__1000nit.png`             — fixed 1000-nit peak
//!   * `bt2446a__p99_995/9_999/9995/9999.png` — percentile sweep readouts
//!
//! Plus, when extractable: `bt2446a__metadata.png` (container-CLL peak).
//!
//! Sample selection:
//!   * `STEMS_FILE` env var → one stem per line, no extension.
//!   * Default: hardcoded 76-stem list (full shootout corpus).
//!   * Set `STEMS_FILE=/path/to/7-outlier.txt` to flip back to the original
//!     7-sample smoke list — useful for debugging the parallel-render path.
//!
//! Source files live under `/home/lilith/work/codec-corpus/imazen-26/`.
//!
//! Outputs are written to `/mnt/v/output/zentone/flicker-test/<stem>/`
//! plus a per-sample `flicker.html` viewer, a top-level `index.html` sorted
//! by `rel_spread = (peak_max - peak_robust) / peak_max` descending, and a
//! flat CSV at `benchmarks/percentile_sweep_2026-06-22.csv` for downstream
//! analysis.
//!
//! Run:
//! ```text
//! nice -n19 ionice -c3 cargo run -p zentone --release \
//!     --example flicker_test_render --features hdr-shootout
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;
use zencodecs::{DecodeRequest, OrientationHint};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::{Bt2446A, CllMeasure, LightLevelHistogram, LightLevelMethod};
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{
    ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_dev::hdr::ContentLightLevel;

// =========================================================================
// Constants
// =========================================================================

const CORPUS_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const OUT_ROOT: &str = "/mnt/v/output/zentone/flicker-test";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";
const CSV_OUT: &str =
    "/home/lilith/work/zen/zentone/benchmarks/percentile_sweep_2026-06-22.csv";

const DIFFUSE_WHITE_NITS: f32 = 203.0;
const TARGET_PEAK_NITS: f32 = 100.0;

/// Full 76-sample shootout corpus stems — extracted from
/// `benchmarks/hdr_tone_map_shootout_full_2026-06-20.csv`.
///
/// Override via `STEMS_FILE=/path/to/file` (one stem per line, no extension)
/// to run on a subset.
const SAMPLE_STEMS_76: &[&str] = &[
    "1064_general_castle-bridge-moat_montjuic-castle-barcelona_zfold7_iso40-f1p7_20260315-175347_4000x3000",
    "1065_general_cathedral-model-workshop_sagrada-familia-barcelona_zfold7_iso64-f1p7_20260317-165822_4000x3000",
    "1066_general_stone-building-facade_park-guell-barcelona_zfold7_iso80-f1p7_20260318-172109_4000x3000",
    "1067_general_architectural-model-display_la-pedrera-barcelona_zfold7_iso500-f1p7_20260319-185645_4000x3000",
    "1069_general_illuminated-castle-night_cinderella-castle-orange_ip13pro_iso500-f1p5_img-7653_2870x3827",
    "1070_general_stone-temple-ruins_ip13pro_iso50-f1p5_img-8852_4032x3024",
    "1227_interiors_stained-glass-window_casa-batllo-barcelona_zfold7_iso1600-f1p7_20260315-205230_4000x3000",
    "1228_interiors_stained-glass-windows_casa-batllo-barcelona_zfold7_iso2500-f1p7_20260315-205328_4000x3000",
    "1229_interiors_lighted-archway-ceiling_casa-batllo-barcelona_zfold7_iso1250-f1p7_20260315-210321_4000x3000",
    "1230_interiors_empty-tiled-pool_casa-batllo-barcelona_zfold7_iso640-f1p7_20260315-210432_4000x3000",
    "1231_interiors_tiled-arched-interior_casa-batllo-barcelona_zfold7_iso640-f2p2_20260315-211323_4000x2252",
    "1232_interiors_stained-glass-windows_sagrada-familia-barcelona_zfold7_iso400-f1p7_20260317-144409_4000x3000",
    "1233_interiors_cathedral-ceiling-interior_sagrada-familia-barcelona_zfold7_iso250-f2p2_20260317-151912_4000x3000",
    "1234_interiors_cathedral-ceiling-columns_sagrada-familia-barcelona_zfold7_iso400-f1p7_20260317-154023_4000x3000",
    "1235_interiors_floral-wall-mural_casa-vicens-barcelona_zfold7_iso1600-f1p7_20260319-165522_4000x3000",
    "1236_interiors_ornate-arched-interior_casa-vicens-barcelona_zfold7_iso400-f2p2_20260319-165702_4000x3000",
    "1237_interiors_person-standing-by-window_casa-vicens-barcelona_zfold7_iso800-f2p2_20260319-170017_4000x3000",
    "1238_interiors_open-doorway-interior_casa-vicens-barcelona_zfold7_iso250-f2p2_20260319-170237_4000x3000",
    "1239_interiors_decorative-coffered-ceiling_casa-vicens-barcelona_zfold7_iso800-f2p2_20260319-170252_4000x3000",
    "1240_interiors_bathroom-with-bathtub_casa-vicens-barcelona_zfold7_iso1000-f2p2_20260319-170336_4000x3000",
    "1241_interiors_living-room-sofa_la-pedrera-barcelona_zfold7_iso1000-f1p7_20260319-183640_4000x3000",
    "1242_interiors_storage-room-clutter_la-pedrera-barcelona_zfold7_iso640-f2p2_20260319-183704_4000x3000",
    "1243_interiors_vintage-living-room_la-pedrera-barcelona_zfold7_iso400-f2p2_20260319-183921_4000x3000",
    "1244_interiors_painted-ceiling-columns_la-pedrera-barcelona_zfold7_iso800-f1p7_20260319-194130_4000x3000",
    "1245_interiors_circular-building-interior_la-pedrera-barcelona_zfold7_iso1000-f2p2_20260319-194139_4000x3000",
    "1246_interiors_woven-roof-structure_azulik-city-arts-francisco-uh-may_ip13pro_iso32-f1p8_img-8944_4032x3024",
    "1487_nature_glacier-over-water_kenai-peninsula-borough-alaska_s25u_iso40-f1p7_20250801-135016_4000x3000",
    "1488_nature_iceberg-ocean-landscape_yakutat-alaska_s25u_iso80-f1p7_20250803-185104_4000x3000",
    "1489_nature_cruise-ship-deck_unorganized-borough-alaska_s25u_iso125-f1p7_20250804-210918_4000x3000",
    "1490_nature_boat-wake-at-sunset_unorganized-borough-alaska_s25u_iso200-f1p7_20250804-211018_4000x3000",
    "1491_nature_ocean-sunset-clouds_unorganized-borough-alaska_s25u_iso100-f3p4_20250804-211059_4000x3000",
    "1492_nature_mountain-lake-landscape_whitehorse-canada_s25u_iso64-f1p7_20250805-162506_4000x3000",
    "1493_nature_sunset-over-sea_lower-davie-vancouver-canada_s25u_iso64-f3p4_20250809-203408_4000x3000",
    "1494_nature_sunset-over-forest_colorado_zfold7_iso250-f1p7_20251110-170409_4000x3000",
    "1495_nature_lighthouse-on-island_york-maine_ip13pro_iso50-f1p5_fullsizerender-1_3855x2891",
    "1496_nature_lake-with-boats_winthrop-maine_ip13pro_iso50-f1p5_fullsizerender-2_3411x2765",
    "1497_nature_pink-flower-on-vine_belltown-seattle-washington_ip16pro_iso80-f1p8_fullsizerender-3_3745x5333",
    "1498_nature_yellow-flowers-garden-bed_colorado_ip13pro_iso50-f1p5_fullsizerender_2773x3709",
    "1502_nature_ice-cave-opening_hornafjordur-iceland_ip13pro_iso32-f1p8_img-0341_4032x3024",
    "1503_nature_passionflower-on-vine_centrum-amsterdam-netherlands_ip16pro_iso50-f1p8_img-0354_4032x3024",
    "1504_nature_cloudy-sky-over-field_colorado_ip13pro_iso50-f1p5_img-0374_4032x3024",
    "1507_nature_ice-cave-waterfall_hornafjordur-iceland_ip13pro_iso200-f1p5_img-0468_4032x3024",
    "1508_nature_ice-cave-interior_hornafjordur-iceland_ip13pro_iso1000-f1p8_img-0474_4032x3024",
    "1509_nature_ice-cave-opening_hornafjordur-iceland_ip13pro_iso1000-f1p8_img-0480_4032x3024",
    "1515_nature_orange-flower-closeup_colorado_ip17pro_iso320-f1p8_img-1420_4032x3024",
    "1516_nature_pink-flower-in-bottle_colorado_ip17pro_iso50-f1p8_img-1509_2355x3308",
    "1519_nature_orange-rose-vase_colorado_ip17pro_iso200-f1p8_img-2700_3961x4825",
    "1520_nature_tulips-in-garden_notre-dame-versailles-france_ip15pro_iso80-f1p8_img-3545_5712x4284",
    "1521_nature_potted-tulips-garden_notre-dame-versailles-france_ip15pro_iso80-f1p8_img-3547_5712x4284",
    "1522_nature_water-lilies-pond_colorado_ip13pro_iso50-f1p5_img-3764_4032x3024",
    "1523_nature_yellow-flowers-garden_colorado_ip13pro_iso50-f1p5_img-3767_4032x3024",
    "1524_nature_yellow-flowers-garden_colorado_ip13pro_iso50-f1p5_img-3841_4032x3024",
    "1525_nature_wildflowers-in-garden_colorado_ip13pro_iso50-f1p5_img-3861_4032x3024",
    "1526_nature_pink-flowers-garden_colorado_ip13pro_iso50-f1p5_img-4543_4032x3024",
    "1527_nature_mountain-wildflowers-landscape_stikine-region-canada_ip16pro_iso100-f1p8_img-4768_5712x4284",
    "1528_nature_pink-blossom-branch_kawazu-japan_ip16pro_iso80-f1p8_img-4931_5712x4284",
    "1529_nature_whale-tail-ocean_ip13pro_iso32-f2p8_img-5573_4032x3024",
    "1530_nature_whale-tail-ocean_ip13pro_iso32-f2p8_img-5574_4032x3024",
    "1531_nature_red-poppy-flowers_taos-new-mexico_ip17pro_iso80-f1p8_img-6189_5712x4284",
    "1532_nature_red-poppy-flowers_taos-new-mexico_ip17pro_iso80-f1p8_img-6194_5712x4284",
    "1533_nature_purple-water-lily_koyama-kamiuchikawaracho-kyoto-japan_ip13pro_iso125-f2p8_img-6205_4032x3024",
    "1534_nature_red-flowers-closeup_ketchikan-alaska_ip16pro_iso125-f1p8_img-6286_4032x3024",
    "1536_nature_purple-flowers-walkway_seward-alaska_ip16pro_iso80-f1p8_img-8763_5712x4284",
    "1537_nature_purple-flowers-closeup_seward-alaska_ip16pro_iso100-f1p8_img-8764_5712x4284",
    "1539_nature_legs-in-water_felipe-carrillo-puerto-mexico_ip13pro_iso32-f1p8_img-8809_4032x3024",
    "1540_nature_purple-flowers-garden_belltown-seattle-washington_ip16pro_iso100-f1p8_img-9109_5712x4284",
    "1541_nature_blue-wildflower-grass_snaefellsbaer-iceland_ip16pro_iso64-f1p8_img-9110_4032x3024",
    "1542_nature_orange-lily-flower_belltown-seattle-washington_ip16pro_iso80-f1p8_img-9149_5712x4284",
    "1543_nature_water-lilies-pond_miyuki-beppu-japan_ip16pro_iso80-f1p8_img-9522_5712x4284",
    "1544_nature_purple-water-lily-flower_miyuki-beppu-japan_ip16pro_iso64-f1p8_img-9550_4032x3024",
    "1545_nature_stalactite-cave-lake_playa-del-carmen-mexico_ip13pro_iso1600-f1p5_img-9591_4032x3024",
    "1546_nature_underground-cave-lake_playa-del-carmen-mexico_ip13pro_iso2000-f1p5_img-9608_4032x3024",
    "1547_nature_orange-flowers-garden_colorado_ip16pro_iso80-f1p8_img-9652_5712x4284",
    "1638_food_sliced-fish-dish_colorado_s25u_iso800-f1p7_20250425-213810_4000x3000",
    "1639_food_sashimi-on-plate_colorado_s25u_iso200-f1p7_20250425-214145_4000x3000",
    "1640_food_birthday-cake-table_colorado_ip16pro_iso125-f1p8_img-2666_5712x4284",
];

/// Load sample stems from the `STEMS_FILE` env var if set, otherwise fall
/// back to the full 76-sample hardcoded list.
fn load_stems() -> Vec<String> {
    if let Ok(path) = std::env::var("STEMS_FILE") {
        let s = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("STEMS_FILE={path}: {e}"));
        return s
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
    }
    SAMPLE_STEMS_76.iter().map(|s| s.to_string()).collect()
}

// =========================================================================
// Lock-file refresh
// =========================================================================

fn refresh_lock(activity: &str) {
    let iso = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let _ = fs::write(WORKONGOING, format!("{iso} claude-flicker-test {activity}\n"));
}

// =========================================================================
// Sample resolution
// =========================================================================

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk_files(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
}

fn resolve_paths(stems: &[String]) -> (Vec<(String, PathBuf)>, Vec<String>) {
    let mut all = Vec::new();
    walk_files(Path::new(CORPUS_ROOT), &mut all);
    let mut out = Vec::new();
    let mut missing = Vec::new();
    for stem in stems {
        match all.iter().find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s == stem.as_str())
                .unwrap_or(false)
        }) {
            Some(p) => out.push((stem.clone(), p.clone())),
            None => missing.push(stem.clone()),
        }
    }
    (out, missing)
}

/// Parse `content_class` (2nd `_`-segment) and `camera` (segment at
/// `NF-3` when 8 segments, `NF-3` for 7 segments too — i.e. always counted
/// from the end so it survives missing-location stems).
fn parse_stem_meta(stem: &str) -> (String, String) {
    let parts: Vec<&str> = stem.split('_').collect();
    let content_class = parts.get(1).copied().unwrap_or("").to_string();
    // Position from end: <id>_<class>_<desc>_<location>_<camera>_<settings>_<imgid>_<wxh>
    //                     0      1       2        3          4         5         6      7
    // Camera = NF-3. For 7-segment stems (missing location), the
    // index NF-3 still lands on the camera token because settings_imgid_res
    // remain at the tail.
    let camera = if parts.len() >= 4 {
        parts[parts.len() - 4].to_string()
    } else {
        String::new()
    };
    (content_class, camera)
}

// =========================================================================
// Linear-RGB f32 helper
// =========================================================================

#[derive(Clone)]
struct LinearRgb {
    width: u32,
    height: u32,
    px: Vec<f32>,
    primaries: ColorPrimaries,
}

impl LinearRgb {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

// =========================================================================
// Decode
// =========================================================================

fn decode_sample(bytes: &[u8]) -> anyhow::Result<(LinearRgb, LinearRgb)> {
    let sdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR: {}", e.error()))?;
    let sdr_buf = sdr_out.into_buffer();
    let sdr_rgb = pixel_buffer_to_linear_rgb(&sdr_buf)?;

    let hdr_out = DecodeRequest::new(bytes)
        .with_orientation(OrientationHint::Correct)
        .reconstruct_hdr(None)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("reconstruct_hdr: {}", e.error()))?;
    let hdr_buf = hdr_out.into_buffer();
    let hdr_rgb = pixel_buffer_to_linear_rgb(&hdr_buf)?;

    if sdr_rgb.width != hdr_rgb.width || sdr_rgb.height != hdr_rgb.height {
        anyhow::bail!(
            "SDR {}x{} vs HDR {}x{} mismatch",
            sdr_rgb.width,
            sdr_rgb.height,
            hdr_rgb.width,
            hdr_rgb.height,
        );
    }
    Ok((sdr_rgb, hdr_rgb))
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
            .map_err(|e| anyhow::anyhow!("convert linear RGB f32: {:?}", e.error()))?
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
        primaries: src_desc.primaries,
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
// Source-peak measurement (histogram-cached + smoothed + container metadata)
// =========================================================================

/// Cached light-level readouts derived from a single histogram scan.
///
/// `measure_robust` / `measure_percentile` both internally call
/// `measure_histogram` (verified in `zenpixels_convert::hdr::measure`), so
/// constructing the histogram once and walking it 5+ times costs O(scan)
/// instead of O(5 × scan).
///
/// `measure_max_smoothed` uses a *separate* spatial-smoothing scan (3-tap
/// box-max over rows) — keep its dedicated call alongside the histogram.
struct PeakCache {
    /// CTA-861.3 literal max (== `histogram.max()`, matches `measure_max`).
    peak_max: f32,
    /// `measure_robust` (== `measure_percentile(DEFAULT_PERCENTILE = 0.9999)`).
    peak_robust: f32,
    /// `measure_max_smoothed` (spatial 3-tap box; not a histogram readout).
    peak_smoothed: f32,
    /// Percentile sweep — same 4 values the old code computed separately.
    peak_p99_995: f32,
    peak_p99_999: f32,
    peak_p99_9995: f32,
    peak_p99_9999: f32,
}

/// Wrap a `LinearRgb` slice in a `PixelBuffer` for the measure_* trait calls.
/// Cloning the bytes here is the price of crossing the
/// `&[f32]` → `PixelBuffer<Vec<u8>>` API boundary; ~70 MB / 24 MP frame is
/// trivial vs. one full-image scan.
fn wrap_for_measure(hdr: &LinearRgb) -> Option<PixelBuffer> {
    let bytes: Vec<u8> = bytemuck::cast_slice(&hdr.px).to_vec();
    let desc = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    PixelBuffer::from_vec(bytes, hdr.width, hdr.height, desc).ok()
}

fn cll_to_peak(cll: Option<ContentLightLevel>) -> f32 {
    let cll = cll.unwrap_or_else(|| ContentLightLevel::new(203, 100));
    (cll.max_content_light_level as f32).max(203.0)
}

/// Match `nits_to_u16` in `zenpixels-convert::hdr::measure` exactly:
/// `(nits + 0.5) as u16` (truncating round, saturating float→int cast,
/// NaN/negative → 0). The histogram returns `f32`; `measure_percentile`
/// upcasts to `f64` before this call, so do the same.
#[inline]
fn nits_f32_to_u16(nits: f32) -> u16 {
    (f64::from(nits) + 0.5) as u16
}

fn peak_from_histogram(h: &LightLevelHistogram, percentile: f32) -> f32 {
    // Bit-exact to `measure_percentile(buf, white, percentile, MaxRgb)`:
    // both go histogram.percentile → nits_to_u16 → (u16 as f32) → max(203).
    let u16_nits = nits_f32_to_u16(h.percentile(percentile));
    (u16_nits as f32).max(203.0)
}

fn peak_max_from_histogram(h: &LightLevelHistogram) -> f32 {
    // Bit-exact to `measure_max` via the
    // `measure_max_and_measure_histogram_max_agree_bit_exact` test path:
    // both reach the same `literal_max_nits` → nits_to_u16.
    let u16_nits = nits_f32_to_u16(h.max());
    (u16_nits as f32).max(203.0)
}

fn measure_peaks(hdr: &LinearRgb) -> PeakCache {
    // One histogram → all 4 percentile peaks + max + robust.
    let Some(buf) = wrap_for_measure(hdr) else {
        // Fall back to 203-nit defaults — unlikely (only fails on
        // invalid descriptor); if it happens, downstream sees identical
        // peaks across all variants which is at least debuggable.
        return PeakCache {
            peak_max: 203.0,
            peak_robust: 203.0,
            peak_smoothed: 203.0,
            peak_p99_995: 203.0,
            peak_p99_999: 203.0,
            peak_p99_9995: 203.0,
            peak_p99_9999: 203.0,
        };
    };

    let hist = ContentLightLevel::measure_histogram(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    );

    let (peak_max, peak_robust, peak_p99_995, peak_p99_999, peak_p99_9995, peak_p99_9999) =
        if let Some(h) = hist {
            (
                peak_max_from_histogram(&h),
                peak_from_histogram(&h, ContentLightLevel::DEFAULT_PERCENTILE),
                peak_from_histogram(&h, 0.99995),
                peak_from_histogram(&h, 0.99999),
                peak_from_histogram(&h, 0.999995),
                peak_from_histogram(&h, 0.999999),
            )
        } else {
            (203.0, 203.0, 203.0, 203.0, 203.0, 203.0)
        };

    // Spatial smoothing has no histogram analog — separate scan.
    let peak_smoothed = cll_to_peak(ContentLightLevel::measure_max_smoothed(
        buf.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    ));

    PeakCache {
        peak_max,
        peak_robust,
        peak_smoothed,
        peak_p99_995,
        peak_p99_999,
        peak_p99_9995,
        peak_p99_9999,
    }
}

/// Container-metadata source-peak extraction.
///
/// Returns `Some((peak_nits, source_label))` if any of the documented paths
/// surface a usable peak, `None` otherwise.
///
/// Search order:
///   * `HDRGainMap:HDRGainMapHeadroom` (Apple HEIC, iP16-class): linear stops
///     above SDR diffuse white. peak_nits = 2^headroom * 203.
///   * `xmpNote:HasExtendedXMP` + Adobe `HDRCapacityMax` (UltraHDR JPEG):
///     same formula. peak_nits = 2^HDRCapacityMax * 203.
///   * `MaxContentLightLevel` (CICP / HEIF `clli` box): raw nits.
fn extract_container_peak(path: &Path) -> Option<(f32, String)> {
    // `-ee` (extractEmbedded) is required for UltraHDR JPEGs: the gain-map
    // XMP (HDRCapacityMax, etc.) is stored inside an embedded JPEG segment
    // that exiftool only descends into with `-ee`. HEIC `HDRGainMapHeadroom`
    // is in the primary XMP and shows up without `-ee`, but `-ee` is
    // harmless for it.
    let out = Command::new("exiftool")
        .args([
            "-ee",
            "-G", "-a", "-s", "-c", "%.6f",
            "-HDRGainMapHeadroom",
            "-HDRCapacityMax",
            "-MaxContentLightLevel",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);

    for line in stdout.lines() {
        // exiftool -G -a -s output: "[XMP]    HDRGainMapHeadroom    : 6.902454"
        let lower = line.to_lowercase();
        let val = line.split(':').nth(1).map(|s| s.trim());
        if lower.contains("hdrgainmapheadroom") {
            if let Some(v) = val.and_then(|s| s.parse::<f32>().ok()) {
                let peak = 2.0_f32.powf(v) * DIFFUSE_WHITE_NITS;
                return Some((peak, format!("HDRGainMapHeadroom={v:.4} stops")));
            }
        } else if lower.contains("hdrcapacitymax") {
            if let Some(v) = val.and_then(|s| s.parse::<f32>().ok()) {
                let peak = 2.0_f32.powf(v) * DIFFUSE_WHITE_NITS;
                return Some((peak, format!("HDRCapacityMax={v:.4} stops")));
            }
        } else if lower.contains("maxcontentlightlevel") {
            if let Some(v) = val.and_then(|s| s.parse::<f32>().ok()) {
                if v > 0.0 {
                    return Some((v, format!("MaxContentLightLevel={v:.0} nits")));
                }
            }
        }
    }
    None
}

// =========================================================================
// Apply Bt2446A
// =========================================================================

fn apply_bt2446a(hdr: &LinearRgb, source_peak_nits: f32) -> LinearRgb {
    let max_pixel_value = (source_peak_nits / DIFFUSE_WHITE_NITS).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;

    let n_pix = hdr.pixels();
    let mut scratch = vec![0.0_f32; n_pix * 3];

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

    scratch.par_chunks_mut(8192).for_each(|c| {
        for v in c.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    });

    LinearRgb {
        width: hdr.width,
        height: hdr.height,
        px: scratch,
        primaries: hdr.primaries,
    }
}

// =========================================================================
// PNG output (truly SDR: convert source primaries → BT.709 + sRGB EOTF)
// =========================================================================
//
// The LinearRgb buffer carries the source's color primaries (DisplayP3 for
// iPhone, BT.709/sRGB for Samsung). To produce a PNG that browsers render
// correctly without an embedded profile, we go through PixelBufferConvertExt
// so zenpixels-convert handles BOTH gamut conversion (src primaries → BT.709)
// AND sRGB EOTF encoding in one fused pass. The output is then bit-for-bit
// what any unprofiled-sRGB PNG should look like.

fn save_png(lin: &LinearRgb, path: &Path) -> anyhow::Result<()> {
    // Wrap the LinearRgb pixels into a PixelBuffer with the source's primaries
    // and linear transfer (which is what the buffer actually carries after
    // pixel_buffer_to_linear_rgb / apply_bt2446a).
    let src_desc = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        lin.primaries,
    );
    let src_bytes: Vec<u8> = bytemuck::cast_slice(&lin.px).to_vec();
    let src_buf = PixelBuffer::from_vec(src_bytes, lin.width, lin.height, src_desc)
        .map_err(|e| anyhow::anyhow!("wrap PixelBuffer: {:?}", e))?;

    // Target: 8-bit sRGB transfer in BT.709 primaries — what a browser
    // assumes when there's no ICC/cICP. This forces a real gamut
    // conversion when src_primaries != BT.709.
    let tgt_desc = PixelDescriptor::new_full(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
    );
    let tgt_buf = src_buf
        .convert_to(tgt_desc)
        .map_err(|e| anyhow::anyhow!("convert to sRGB BT.709: {:?}", e.error()))?;

    // Pack tightly for image::save_buffer (which wants contiguous Rgb8).
    let slice = tgt_buf.as_slice();
    let stride = slice.stride();
    let bytes = slice.as_strided_bytes();
    let row_bytes = lin.width as usize * 3;
    let mut packed = vec![0u8; row_bytes * lin.height as usize];
    for y in 0..lin.height as usize {
        packed[y * row_bytes..(y + 1) * row_bytes]
            .copy_from_slice(&bytes[y * stride..y * stride + row_bytes]);
    }
    image::save_buffer(path, &packed, lin.width, lin.height, image::ColorType::Rgb8)?;
    Ok(())
}

// =========================================================================
// HTML viewer
// =========================================================================

struct SampleReport {
    stem: String,
    path: PathBuf,
    width: u32,
    height: u32,
    peak_max: f32,
    peak_robust: f32,
    peak_smoothed: f32,
    peak_p99_995: f32,
    peak_p99_999: f32,
    peak_p99_9995: f32,
    peak_p99_9999: f32,
    peak_blend25: f32,
    peak_blend50: f32,
    peak_blend75: f32,
    peak_metadata: Option<(f32, String)>,
    variants: Vec<(String, String)>, // (label, relative filename)
    render_seconds: f64,
}

impl SampleReport {
    fn rel_spread(&self) -> f32 {
        if self.peak_max > 0.0 {
            (self.peak_max - self.peak_robust) / self.peak_max
        } else {
            0.0
        }
    }
}

fn write_per_sample_html(out_dir: &Path, rep: &SampleReport) -> anyhow::Result<()> {
    let path = out_dir.join("flicker.html");
    let mut variant_options = String::new();
    for (label, file) in &rep.variants {
        variant_options.push_str(&format!(
            "    <option value=\"{file}\">{label}</option>\n"
        ));
    }
    let meta_line = match &rep.peak_metadata {
        Some((nits, src)) => format!("{nits:.0} nits (from {src})"),
        None => "(no usable container metadata)".to_string(),
    };
    let path_str = rep.path.display();
    let stem = &rep.stem;
    let (w, h) = (rep.width, rep.height);
    let max_n = rep.peak_max;
    let rob_n = rep.peak_robust;
    let sm_n = rep.peak_smoothed;
    let p99_995 = rep.peak_p99_995;
    let p99_999 = rep.peak_p99_999;
    let p99_9995 = rep.peak_p99_9995;
    let p99_9999 = rep.peak_p99_9999;

    // Default selections: variant A = REF, variant B = bt2446a__robust if available.
    let default_a = "REF.png".to_string();
    let default_b = rep
        .variants
        .iter()
        .find(|(_, f)| f.contains("__robust"))
        .map(|(_, f)| f.clone())
        .unwrap_or_else(|| rep.variants[0].1.clone());

    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>flicker: {stem}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #111; color: #eee;
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Consolas, monospace;
    font-size: 13px; }}
  .toolbar {{ position: sticky; top: 0; z-index: 10; background: #1a1a1a;
    border-bottom: 1px solid #333; padding: 10px 14px; display: flex;
    flex-wrap: wrap; gap: 14px; align-items: center; }}
  .toolbar label {{ display: flex; gap: 6px; align-items: center; }}
  .toolbar select, .toolbar input[type=range], .toolbar button {{
    background: #222; color: #eee; border: 1px solid #444; padding: 4px 8px;
    font: inherit; }}
  .toolbar button {{ cursor: pointer; min-width: 64px; }}
  .toolbar .meta {{ flex: 1 1 100%; color: #aaa; font-size: 12px; }}
  .toolbar .meta b {{ color: #fff; }}
  .toolbar .which {{ padding: 3px 8px; border-radius: 3px; font-weight: 600; }}
  .toolbar .which.a {{ background: #2a4d8e; color: #fff; }}
  .toolbar .which.b {{ background: #8e2a4d; color: #fff; }}
  .stage {{ position: relative; }}
  .stage img {{ display: block; width: 100%; height: auto;
    position: absolute; top: 0; left: 0; }}
  .stage .placeholder {{ visibility: hidden; }}
  .legend {{ position: absolute; left: 16px; top: 16px; background: rgba(0,0,0,0.85);
    border: 2px solid #fff; padding: 10px 18px; font-size: 20px; font-weight: 800;
    border-radius: 6px; z-index: 9; pointer-events: none;
    text-shadow: 0 1px 3px rgba(0,0,0,0.9); letter-spacing: 0.02em; }}
</style>
</head>
<body>
<div class="toolbar">
  <label>A
    <select id="varA">
{variant_options}    </select>
    <span class="which a">A</span>
  </label>
  <label>B
    <select id="varB">
{variant_options}    </select>
    <span class="which b">B</span>
  </label>
  <label>speed
    <input id="speed" type="range" min="50" max="2000" value="500" step="50">
    <span id="speedLabel">500 ms</span>
  </label>
  <button id="toggle">pause</button>
  <button id="swap">swap A/B</button>
  <div class="meta">
    <b>{stem}</b><br>
    source: {path_str}<br>
    full-res: {w}×{h} &nbsp;&nbsp;
    max=<b>{max_n:.0}</b> &nbsp; robust=<b>{rob_n:.0}</b> &nbsp;
    smooth=<b>{sm_n:.0}</b> &nbsp;
    p99.995=<b>{p99_995:.0}</b> p99.999=<b>{p99_999:.0}</b>
    p99.9995=<b>{p99_9995:.0}</b> p99.9999=<b>{p99_9999:.0}</b><br>
    metadata = <b>{meta_line}</b>
  </div>
</div>
<div class="stage" id="stage">
  <img id="imgA" class="placeholder" src="{default_a}" alt="A">
  <img id="imgB" class="placeholder" src="{default_b}" alt="B">
  <div class="legend" id="legend">A</div>
</div>
<script>
(function() {{
  var imgA = document.getElementById('imgA');
  var imgB = document.getElementById('imgB');
  var varA = document.getElementById('varA');
  var varB = document.getElementById('varB');
  var legend = document.getElementById('legend');
  var speed = document.getElementById('speed');
  var speedLabel = document.getElementById('speedLabel');
  var toggle = document.getElementById('toggle');
  var swap = document.getElementById('swap');
  var stage = document.getElementById('stage');

  function nameFor(file) {{
    for (var i = 0; i < varA.options.length; i++) {{
      if (varA.options[i].value === file) return varA.options[i].text;
    }}
    return file;
  }}

  function refreshPlaceholder() {{
    // The container needs explicit height equal to the loaded image's
    // natural aspect ratio, since absolutely-positioned <img>s collapse.
    if (imgA.naturalWidth > 0) {{
      var aspect = imgA.naturalHeight / imgA.naturalWidth;
      stage.style.paddingBottom = (aspect * 100).toFixed(4) + '%';
      imgA.classList.remove('placeholder');
      imgB.classList.remove('placeholder');
    }}
  }}
  imgA.addEventListener('load', refreshPlaceholder);

  varA.value = "{default_a}";
  varB.value = "{default_b}";
  varA.addEventListener('change', function() {{ imgA.src = varA.value; }});
  varB.addEventListener('change', function() {{ imgB.src = varB.value; }});

  var showing = 'A';
  imgA.style.visibility = 'visible';
  imgB.style.visibility = 'hidden';

  var running = true;
  var interval = parseInt(speed.value, 10);
  var timer = null;

  function tick() {{
    if (showing === 'A') {{
      imgA.style.visibility = 'hidden';
      imgB.style.visibility = 'visible';
      showing = 'B';
      legend.textContent = 'B: ' + nameFor(varB.value);
      legend.style.background = 'rgba(120, 30, 60, 0.85)';
    }} else {{
      imgA.style.visibility = 'visible';
      imgB.style.visibility = 'hidden';
      showing = 'A';
      legend.textContent = 'A: ' + nameFor(varA.value);
      legend.style.background = 'rgba(40, 70, 130, 0.85)';
    }}
  }}

  function startTimer() {{ if (timer === null && running) timer = setInterval(tick, interval); }}
  function stopTimer() {{ if (timer !== null) {{ clearInterval(timer); timer = null; }} }}

  speed.addEventListener('input', function() {{
    interval = parseInt(speed.value, 10);
    speedLabel.textContent = interval + ' ms';
    if (running) {{ stopTimer(); startTimer(); }}
  }});
  toggle.addEventListener('click', function() {{
    running = !running;
    toggle.textContent = running ? 'pause' : 'play';
    if (running) startTimer(); else stopTimer();
  }});
  swap.addEventListener('click', function() {{
    var aVal = varA.value, bVal = varB.value;
    varA.value = bVal; varB.value = aVal;
    imgA.src = varA.value; imgB.src = varB.value;
  }});

  legend.textContent = 'A: ' + nameFor(varA.value);
  legend.style.background = 'rgba(40, 70, 130, 0.85)';
  startTimer();
}})();
</script>
</body>
</html>
"#
    );
    fs::write(path, html)?;
    Ok(())
}

fn write_index_html(out_root: &Path, reports: &[SampleReport]) -> anyhow::Result<()> {
    let mut rows = String::new();
    let n_samples = reports.len();
    for r in reports {
        let stem = &r.stem;
        let meta_label = match &r.peak_metadata {
            Some((nits, src)) => format!("{nits:.0} nits ({src})"),
            None => "(unavailable)".to_string(),
        };
        let max_n = r.peak_max;
        let rob_n = r.peak_robust;
        let sm_n = r.peak_smoothed;
        let p99_999 = r.peak_p99_999;
        let p99_9999 = r.peak_p99_9999;
        let spread_abs = r.peak_max - r.peak_robust;
        let spread_rel = r.rel_spread() * 100.0;
        let render_s = r.render_seconds;
        rows.push_str(&format!(
            r#"<tr>
  <td><a href="{stem}/flicker.html"><img src="{stem}/REF.png" width="160" loading="lazy" alt="{stem}"></a></td>
  <td>
    <div class="stem"><a href="{stem}/flicker.html">{stem}</a></div>
    <div class="peaks">
      max=<b>{max_n:.0}</b> robust=<b>{rob_n:.0}</b> smooth=<b>{sm_n:.0}</b>
      &middot; p99.999=<b>{p99_999:.0}</b> p99.9999=<b>{p99_9999:.0}</b>
      &middot; meta=<b>{meta_label}</b>
    </div>
    <div class="spread">rel_spread = <b>{spread_rel:.1}%</b> ({spread_abs:.0} nits) &middot; render {render_s:.1}s</div>
  </td>
</tr>
"#
        ));
    }
    let html = format!(
        r#"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>flicker test — {n_samples} samples × 11 percentile variants</title>
<style>
  body {{ margin: 24px; background: #111; color: #eee;
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Consolas, monospace; font-size: 14px; }}
  h1 {{ margin: 0 0 4px; }}
  .sub {{ color: #aaa; margin-bottom: 18px; max-width: 900px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  tr {{ border-bottom: 1px solid #222; }}
  td {{ padding: 14px 8px; vertical-align: top; }}
  td img {{ display: block; border-radius: 4px; }}
  .stem a {{ color: #6cf; font-weight: 700; text-decoration: none; }}
  .stem a:hover {{ text-decoration: underline; }}
  .peaks {{ margin: 6px 0; color: #ccc; }}
  .spread {{ color: #888; font-size: 12px; }}
  b {{ color: #fff; }}
</style></head>
<body>
<h1>HDR tone-map percentile sweep — {n_samples} samples</h1>
<div class="sub">All {n_samples} imazen-26 shootout samples × 11 Bt2446A variants
(max / robust / smoothed / blend25 / blend50 / blend75 / 1000nit /
p99.995 / p99.999 / p99.9995 / p99.9999). Sorted by
<code>rel_spread = (peak_max - peak_robust) / peak_max</code> descending —
top rows are where the percentile choice matters most. Each row links to
the per-sample flicker viewer.</div>
<table>
{rows}
</table>
</body></html>
"#
    );
    fs::write(out_root.join("index.html"), html)?;
    Ok(())
}

// =========================================================================
// CSV writer
// =========================================================================

fn write_csv(reports: &[SampleReport]) -> anyhow::Result<()> {
    let path = Path::new(CSV_OUT);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    let mut out = String::with_capacity(64 * reports.len());
    out.push_str(
        "sample,format,camera,content_class,width,height,peak_max,peak_robust,peak_smoothed,\
         peak_p99_995,peak_p99_999,peak_p99_9995,peak_p99_9999,peak_metadata_nits,\
         peak_metadata_source,peak_blend25,peak_blend50,peak_blend75,rel_spread,abs_spread,\
         render_seconds\n",
    );
    for r in reports {
        let (content_class, camera) = parse_stem_meta(&r.stem);
        let format = r
            .path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let (meta_nits, meta_src) = match &r.peak_metadata {
            Some((n, s)) => (format!("{n:.1}"), s.clone()),
            None => (String::new(), String::new()),
        };
        // Escape any commas / quotes in meta source (it shouldn't have any
        // but `extract_container_peak` formats arbitrary exiftool output).
        let meta_src_q = if meta_src.contains(',') || meta_src.contains('"') {
            format!("\"{}\"", meta_src.replace('"', "\"\""))
        } else {
            meta_src
        };
        let abs_spread = r.peak_max - r.peak_robust;
        let rel_spread = r.rel_spread();
        out.push_str(&format!(
            "{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{:.1},{:.1},{:.1},{:.6},{:.1},{:.2}\n",
            r.stem,
            format,
            camera,
            content_class,
            r.width,
            r.height,
            r.peak_max,
            r.peak_robust,
            r.peak_smoothed,
            r.peak_p99_995,
            r.peak_p99_999,
            r.peak_p99_9995,
            r.peak_p99_9999,
            meta_nits,
            meta_src_q,
            r.peak_blend25,
            r.peak_blend50,
            r.peak_blend75,
            rel_spread,
            abs_spread,
            r.render_seconds,
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

// =========================================================================
// Driver
// =========================================================================

fn process_sample(stem: &str, path: &Path) -> anyhow::Result<SampleReport> {
    let t0 = Instant::now();
    refresh_lock(&format!("decode {stem}"));
    let bytes = fs::read(path)?;
    let (sdr, hdr) = decode_sample(&bytes)?;

    refresh_lock(&format!("measure {stem}"));
    let peaks = measure_peaks(&hdr);
    let peak_metadata = extract_container_peak(path);

    let out_dir = Path::new(OUT_ROOT).join(stem);
    fs::create_dir_all(&out_dir)?;

    refresh_lock(&format!("save REF {stem}"));
    save_png(&sdr, &out_dir.join("REF.png"))?;

    // Linear blends between robust and max — for eyeball-calibrating the
    // "right" percentile when the visual sweet spot sits between the two
    // extremes (sample zfold7 facade was the original prompt:
    // robust=630 was too dark, max=963 was OK, halfway ≈ 800 looked best).
    let blend25 = peaks.peak_robust + 0.25 * (peaks.peak_max - peaks.peak_robust);
    let blend50 = peaks.peak_robust + 0.50 * (peaks.peak_max - peaks.peak_robust);
    let blend75 = peaks.peak_robust + 0.75 * (peaks.peak_max - peaks.peak_robust);

    // The 11 variants — render in parallel via rayon. Each render owns its
    // own f32 scratch + PNG encode, so they don't contend on shared state;
    // bt2446a itself uses rayon internally for the per-pixel scale +
    // clamp, but those inner par_chunks scale down to the available
    // workers when nested under outer par_iter — net win is still large
    // because the SIMD tone-map kernel is the long pole.
    let renders: Vec<(&'static str, &'static str, f32)> = vec![
        ("bt2446a max",         "bt2446a__max.png",       peaks.peak_max),
        ("bt2446a robust",      "bt2446a__robust.png",    peaks.peak_robust),
        ("bt2446a smoothed",    "bt2446a__smoothed.png",  peaks.peak_smoothed),
        ("bt2446a blend25",     "bt2446a__blend25.png",   blend25),
        ("bt2446a blend50",     "bt2446a__blend50.png",   blend50),
        ("bt2446a blend75",     "bt2446a__blend75.png",   blend75),
        ("bt2446a 1000nit",     "bt2446a__1000nit.png",   1000.0),
        ("bt2446a p=99.995%",   "bt2446a__p99_995.png",   peaks.peak_p99_995),
        ("bt2446a p=99.999%",   "bt2446a__p99_999.png",   peaks.peak_p99_999),
        ("bt2446a p=99.9995%",  "bt2446a__p99_9995.png",  peaks.peak_p99_9995),
        ("bt2446a p=99.9999%",  "bt2446a__p99_9999.png",  peaks.peak_p99_9999),
    ];

    refresh_lock(&format!("render 11 variants {stem}"));
    renders.par_iter().try_for_each(|(label, file, peak)| {
        let mapped = apply_bt2446a(&hdr, *peak);
        save_png(&mapped, &out_dir.join(file)).map_err(|e| {
            anyhow::anyhow!("save {label} ({file}) for {stem}: {e}")
        })
    })?;

    let mut variants: Vec<(String, String)> =
        vec![("REF (producer SDR)".to_string(), "REF.png".to_string())];
    for (label, file, peak) in &renders {
        variants.push((format!("{label} ({peak:.0}n)"), file.to_string()));
    }

    match &peak_metadata {
        Some((peak, src)) => {
            refresh_lock(&format!("render metadata {stem}"));
            // Wipe a stale `.missing` marker from a prior run.
            let _ = fs::remove_file(out_dir.join("bt2446a__metadata.png.missing"));
            let mapped = apply_bt2446a(&hdr, *peak);
            save_png(&mapped, &out_dir.join("bt2446a__metadata.png"))?;
            variants.push((
                format!("bt2446a metadata ({peak:.0}n, {src})"),
                "bt2446a__metadata.png".to_string(),
            ));
        }
        None => {
            // Wipe a stale `.png` from a prior run that found metadata.
            let _ = fs::remove_file(out_dir.join("bt2446a__metadata.png"));
            fs::write(
                out_dir.join("bt2446a__metadata.png.missing"),
                "no container MaxCLL / HDRGainMapHeadroom / HDRCapacityMax found\n",
            )?;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let rep = SampleReport {
        stem: stem.to_string(),
        path: path.to_path_buf(),
        width: sdr.width,
        height: sdr.height,
        peak_max: peaks.peak_max,
        peak_robust: peaks.peak_robust,
        peak_smoothed: peaks.peak_smoothed,
        peak_p99_995: peaks.peak_p99_995,
        peak_p99_999: peaks.peak_p99_999,
        peak_p99_9995: peaks.peak_p99_9995,
        peak_p99_9999: peaks.peak_p99_9999,
        peak_blend25: blend25,
        peak_blend50: blend50,
        peak_blend75: blend75,
        peak_metadata,
        variants,
        render_seconds: elapsed,
    };
    write_per_sample_html(&out_dir, &rep)?;
    Ok(rep)
}

/// Median + p10 + p90 of a numeric column, computed once over the whole
/// reports vec. Used for the closing summary table.
fn percentiles<F: Fn(&SampleReport) -> f32>(reports: &[SampleReport], f: F) -> (f32, f32, f32) {
    let mut v: Vec<f32> = reports.iter().map(f).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let p = |q: f32| {
        let idx = ((n as f32 - 1.0) * q).round() as usize;
        v[idx.min(n - 1)]
    };
    (p(0.10), p(0.50), p(0.90))
}

fn main() -> anyhow::Result<()> {
    let stems = load_stems();
    let (samples, missing) = resolve_paths(&stems);
    if !missing.is_empty() {
        eprintln!("warning: {} stems not found under {}:", missing.len(), CORPUS_ROOT);
        for m in &missing {
            eprintln!("  {m}");
        }
    }
    if samples.is_empty() {
        anyhow::bail!("no samples to process");
    }

    fs::create_dir_all(OUT_ROOT)?;

    refresh_lock("start");
    let t_total = Instant::now();
    let failures: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    let reports: Mutex<Vec<SampleReport>> = Mutex::new(Vec::new());
    let total_n = samples.len();

    // Each sample is processed sequentially (decode + 11 parallel renders +
    // PNG encodes saturate the box) — parallelising across samples would
    // multiply peak RAM ~3x, risking OOM at 24 MP frames.
    for (i, (stem, path)) in samples.iter().enumerate() {
        eprintln!("[{}/{}] {stem}", i + 1, total_n);
        match process_sample(stem, path) {
            Ok(rep) => {
                eprintln!(
                    "  {}x{}  max={:.0}n robust={:.0}n smooth={:.0}n p99.999={:.0}n p99.9999={:.0}n  ({:.1}s)",
                    rep.width,
                    rep.height,
                    rep.peak_max,
                    rep.peak_robust,
                    rep.peak_smoothed,
                    rep.peak_p99_999,
                    rep.peak_p99_9999,
                    rep.render_seconds,
                );
                reports.lock().unwrap().push(rep);
            }
            Err(e) => {
                eprintln!("  FAILED: {e}");
                failures.lock().unwrap().push((stem.clone(), e.to_string()));
            }
        }
    }

    let mut reports = reports.into_inner().unwrap();
    let failures = failures.into_inner().unwrap();

    // Sort by rel_spread descending — widest-variance first (the rows
    // where the percentile choice matters most).
    reports.sort_by(|a, b| {
        b.rel_spread()
            .partial_cmp(&a.rel_spread())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    refresh_lock("write index + csv");
    write_index_html(Path::new(OUT_ROOT), &reports)?;
    write_csv(&reports)?;

    let total = t_total.elapsed().as_secs_f64();
    eprintln!(
        "\nTotal: {} samples in {:.1}s ({:.1}s/sample avg)",
        reports.len(),
        total,
        total / reports.len().max(1) as f64,
    );

    let (max_p10, max_p50, max_p90) = percentiles(&reports, |r| r.peak_max);
    let (rob_p10, rob_p50, rob_p90) = percentiles(&reports, |r| r.peak_robust);
    let (sm_p10, sm_p50, sm_p90) = percentiles(&reports, |r| r.peak_smoothed);
    let (p99_999_p10, p99_999_p50, p99_999_p90) = percentiles(&reports, |r| r.peak_p99_999);
    let (p99_9999_p10, p99_9999_p50, p99_9999_p90) = percentiles(&reports, |r| r.peak_p99_9999);
    let (spr_p10, spr_p50, spr_p90) = percentiles(&reports, |r| r.rel_spread() * 100.0);

    eprintln!("\nPeak distribution across {} samples (p10 / p50 / p90, in nits):", reports.len());
    eprintln!("  peak_max       {:6.0} / {:6.0} / {:6.0}", max_p10, max_p50, max_p90);
    eprintln!("  peak_robust    {:6.0} / {:6.0} / {:6.0}", rob_p10, rob_p50, rob_p90);
    eprintln!("  peak_smoothed  {:6.0} / {:6.0} / {:6.0}", sm_p10, sm_p50, sm_p90);
    eprintln!("  p99.999%       {:6.0} / {:6.0} / {:6.0}", p99_999_p10, p99_999_p50, p99_999_p90);
    eprintln!("  p99.9999%      {:6.0} / {:6.0} / {:6.0}", p99_9999_p10, p99_9999_p50, p99_9999_p90);
    eprintln!("  rel_spread %   {:6.1} / {:6.1} / {:6.1}", spr_p10, spr_p50, spr_p90);

    if !failures.is_empty() {
        eprintln!("\n{} sample failures:", failures.len());
        for (stem, err) in &failures {
            eprintln!("  {stem}: {err}");
        }
    }

    eprintln!("\nCSV: {CSV_OUT}");
    eprintln!("URL: http://172.23.240.1:3300/zentone/flicker-test/");
    Ok(())
}
