//! Flicker-test renders for the 7 highest-CLL-variance imazen-26 samples.
//!
//! Per sample, renders 5 full-resolution PNGs:
//!   * `<stem>__REF.png`               — producer SDR base (ground truth)
//!   * `<stem>__bt2446a__max.png`      — Bt2446A with `measure_max` peak
//!   * `<stem>__bt2446a__robust.png`   — Bt2446A with `measure_robust` peak
//!   * `<stem>__bt2446a__smoothed.png` — Bt2446A with `measure_max_smoothed` peak
//!   * `<stem>__bt2446a__metadata.png` — Bt2446A with container-CLL peak
//!     (or a `<stem>__bt2446a__metadata.png.missing` marker file if the
//!     container's MaxCLL / HDRGainMapHeadroom / HDRCapacityMax cannot be
//!     extracted).
//!
//! Source-metadata extraction order:
//!   1. iPhone HEIC — XMP `HDRGainMap:HDRGainMapHeadroom` (linear stops above
//!      SDR diffuse white; `peak_nits = 2^headroom * 203`).
//!   2. UltraHDR JPEG — XMP `HDRCapacityMax` (also linear stops above SDR
//!      diffuse white; same formula).
//!   3. Anything else with `MaxContentLightLevel` (CICP / `clli` box) — use
//!      directly in nits.
//!   4. None of the above — emit the `.missing` marker file.
//!
//! Source files (the user's 7 high-variance picks): see `SAMPLE_STEMS` below.
//!
//! Outputs are written to `/mnt/v/output/zentone/flicker-test/<stem>/`
//! plus a per-sample `flicker.html` viewer that toggles any two of the
//! variants at a configurable interval. The top-level
//! `/mnt/v/output/zentone/flicker-test/index.html` links to all 7.
//!
//! Run:
//! ```text
//! nice -n19 cargo run -p zentone --release \
//!     --example flicker_test_render --features hdr-shootout
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use rayon::prelude::*;
use zencodecs::{DecodeRequest, OrientationHint};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::{Bt2446A, CllMeasure, LightLevelMethod};
use zenpixels_dev::DiffuseWhite;
use zenpixels_dev::buffer::PixelBuffer;
use zenpixels_dev::descriptor::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_dev::hdr::ContentLightLevel;

// =========================================================================
// Constants
// =========================================================================

const CORPUS_ROOT: &str = "/home/lilith/work/codec-corpus/imazen-26";
const OUT_ROOT: &str = "/mnt/v/output/zentone/flicker-test";
const WORKONGOING: &str = "/home/lilith/work/zen/zentone/.workongoing";

const DIFFUSE_WHITE_NITS: f32 = 203.0;
const TARGET_PEAK_NITS: f32 = 100.0;

/// User-picked stems (top 7 by CLL spread across `measure_max` /
/// `measure_robust` / `measure_max_smoothed`).
const SAMPLE_STEMS: &[&str] = &[
    "1526_nature_pink-flowers-garden_colorado_ip13pro_iso50-f1p5_img-4543_4032x3024",
    "1546_nature_underground-cave-lake_playa-del-carmen-mexico_ip13pro_iso2000-f1p5_img-9608_4032x3024",
    "1547_nature_orange-flowers-garden_colorado_ip16pro_iso80-f1p8_img-9652_5712x4284",
    "1066_general_stone-building-facade_park-guell-barcelona_zfold7_iso80-f1p7_20260318-172109_4000x3000",
    "1640_food_birthday-cake-table_colorado_ip16pro_iso125-f1p8_img-2666_5712x4284",
    "1542_nature_orange-lily-flower_belltown-seattle-washington_ip16pro_iso80-f1p8_img-9149_5712x4284",
    "1536_nature_purple-flowers-walkway_seward-alaska_ip16pro_iso80-f1p8_img-8763_5712x4284",
];

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

fn resolve_paths() -> Vec<(String, PathBuf)> {
    let mut all = Vec::new();
    walk_files(Path::new(CORPUS_ROOT), &mut all);
    let mut out = Vec::new();
    for stem in SAMPLE_STEMS {
        match all.iter().find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s == *stem)
                .unwrap_or(false)
        }) {
            Some(p) => out.push((stem.to_string(), p.clone())),
            None => panic!("sample stem not found under {CORPUS_ROOT}: {stem}"),
        }
    }
    out
}

// =========================================================================
// Linear-RGB f32 helper
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
// Source-peak measurement (3 measure_* methods + container metadata)
// =========================================================================

#[derive(Clone, Copy, Debug)]
enum PeakMethod {
    Max,
    Robust,
    MaxSmoothed,
}

// `PeakMethod` doesn't need a `label()` helper here — the variant-key strings
// (`max` / `robust` / `smoothed`) appear inline in the rendering loop's tuple
// table where they actually compose into filenames.

fn measure_peak(hdr: &LinearRgb, method: PeakMethod) -> f32 {
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
    let cll = match method {
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
    let cll = cll.unwrap_or_else(|| ContentLightLevel::new(203, 100));
    (cll.max_content_light_level as f32).max(203.0)
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
    }
}

// =========================================================================
// PNG output
// =========================================================================

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = linear_srgb::tf::linear_to_srgb(v);
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn to_srgb_u8(lin: &LinearRgb) -> Vec<u8> {
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

fn save_png(lin: &LinearRgb, path: &Path) -> anyhow::Result<()> {
    let buf = to_srgb_u8(lin);
    image::save_buffer(path, &buf, lin.width, lin.height, image::ColorType::Rgb8)?;
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
    peak_metadata: Option<(f32, String)>,
    variants: Vec<(String, String)>, // (label, relative filename)
    render_seconds: f64,
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
  .legend {{ position: fixed; right: 16px; top: 64px; background: rgba(0,0,0,0.7);
    border: 1px solid #444; padding: 8px 12px; font-size: 14px; font-weight: 700;
    border-radius: 4px; z-index: 9; }}
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
    measure_max = <b>{max_n:.0}</b> nits &nbsp;
    measure_robust = <b>{rob_n:.0}</b> nits &nbsp;
    measure_max_smoothed = <b>{sm_n:.0}</b> nits &nbsp;
    metadata = <b>{meta_line}</b>
  </div>
</div>
<div class="legend" id="legend">A</div>
<div class="stage" id="stage">
  <img id="imgA" class="placeholder" src="{default_a}" alt="A">
  <img id="imgB" class="placeholder" src="{default_b}" alt="B">
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
    for r in reports {
        let stem = &r.stem;
        let meta_label = match &r.peak_metadata {
            Some((nits, src)) => format!("{nits:.0} nits ({src})"),
            None => "(unavailable)".to_string(),
        };
        let max_n = r.peak_max;
        let rob_n = r.peak_robust;
        let sm_n = r.peak_smoothed;
        let peaks: Vec<f32> = {
            let mut v = vec![r.peak_max, r.peak_robust, r.peak_smoothed];
            if let Some((m, _)) = r.peak_metadata {
                v.push(m);
            }
            v
        };
        let pmax = peaks.iter().cloned().fold(0.0_f32, f32::max);
        let pmin = peaks.iter().cloned().fold(f32::INFINITY, f32::min);
        let spread = if pmin > 0.0 { pmax / pmin } else { 0.0 };
        let render_s = r.render_seconds;
        rows.push_str(&format!(
            r#"<tr>
  <td><a href="{stem}/flicker.html"><img src="{stem}/REF.png" width="160" loading="lazy" alt="{stem}"></a></td>
  <td>
    <div class="stem"><a href="{stem}/flicker.html">{stem}</a></div>
    <div class="peaks">
      max=<b>{max_n:.0}</b> robust=<b>{rob_n:.0}</b> smoothed=<b>{sm_n:.0}</b> meta=<b>{meta_label}</b>
    </div>
    <div class="spread">spread max/min = <b>{spread:.2}×</b> &middot; render {render_s:.1}s</div>
  </td>
</tr>
"#
        ));
    }
    let html = format!(
        r#"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>flicker test — top-7 CLL-variance imazen-26 samples</title>
<style>
  body {{ margin: 24px; background: #111; color: #eee;
    font-family: ui-monospace, "JetBrains Mono", "SF Mono", Consolas, monospace; font-size: 14px; }}
  h1 {{ margin: 0 0 4px; }}
  .sub {{ color: #aaa; margin-bottom: 18px; }}
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
<h1>HDR tone-map flicker test</h1>
<div class="sub">Top-7 highest-CLL-variance samples from imazen-26.
Each row links to a per-sample flicker viewer that toggles between any two of
REF / Bt2446A&times;{{max,robust,smoothed,metadata}} at a configurable interval.</div>
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
// Driver
// =========================================================================

fn process_sample(stem: &str, path: &Path) -> anyhow::Result<SampleReport> {
    let t0 = Instant::now();
    refresh_lock(&format!("decode {stem}"));
    let bytes = fs::read(path)?;
    let (sdr, hdr) = decode_sample(&bytes)?;

    refresh_lock(&format!("measure {stem}"));
    let peak_max = measure_peak(&hdr, PeakMethod::Max);
    let peak_robust = measure_peak(&hdr, PeakMethod::Robust);
    let peak_smoothed = measure_peak(&hdr, PeakMethod::MaxSmoothed);
    let peak_metadata = extract_container_peak(path);

    let out_dir = Path::new(OUT_ROOT).join(stem);
    fs::create_dir_all(&out_dir)?;

    refresh_lock(&format!("save REF {stem}"));
    save_png(&sdr, &out_dir.join("REF.png"))?;

    let mut variants: Vec<(String, String)> = vec![("REF (producer SDR)".to_string(), "REF.png".to_string())];

    for (label, file, peak) in [
        ("bt2446a max",      "bt2446a__max.png",      peak_max),
        ("bt2446a robust",   "bt2446a__robust.png",   peak_robust),
        ("bt2446a smoothed", "bt2446a__smoothed.png", peak_smoothed),
    ] {
        refresh_lock(&format!("render {label} {stem}"));
        let mapped = apply_bt2446a(&hdr, peak);
        save_png(&mapped, &out_dir.join(file))?;
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
        peak_max,
        peak_robust,
        peak_smoothed,
        peak_metadata,
        variants,
        render_seconds: elapsed,
    };
    write_per_sample_html(&out_dir, &rep)?;
    Ok(rep)
}

fn main() -> anyhow::Result<()> {
    let samples = resolve_paths();
    fs::create_dir_all(OUT_ROOT)?;

    refresh_lock("start");
    let t_total = Instant::now();
    let mut reports = Vec::new();
    for (i, (stem, path)) in samples.iter().enumerate() {
        eprintln!("[{}/{}] {stem}", i + 1, samples.len());
        let rep = process_sample(stem, path)?;
        eprintln!(
            "  {}x{}  max={:.0}n robust={:.0}n smoothed={:.0}n meta={}  ({:.1}s)",
            rep.width,
            rep.height,
            rep.peak_max,
            rep.peak_robust,
            rep.peak_smoothed,
            match &rep.peak_metadata {
                Some((p, s)) => format!("{p:.0}n [{s}]"),
                None => "missing".to_string(),
            },
            rep.render_seconds,
        );
        reports.push(rep);
    }
    refresh_lock("write index");
    write_index_html(Path::new(OUT_ROOT), &reports)?;

    let total = t_total.elapsed().as_secs_f64();
    eprintln!(
        "\nTotal: {} samples in {:.1}s ({:.1}s/sample avg)",
        reports.len(),
        total,
        total / reports.len() as f64
    );
    eprintln!("URL: http://172.23.240.1:3300/zentone/flicker-test/");
    Ok(())
}
