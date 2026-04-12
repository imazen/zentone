//! Streaming tonemapper with local adaptation.
//!
//! Single-pass spatially-local HDR→SDR mapping with a lookahead row buffer.
//! The core idea is to track local key (geometric mean luminance) and white
//! point over a coarse grid, then apply a sigmoid roll-off per pixel using
//! those locally-adapted parameters plus AgX-style highlight desaturation.
//!
//! # Memory
//!
//! For 4K (3840×2160) at default config (cell_size 8, lookahead_rows 64):
//! - Grid stats: 480×270 × 16 bytes ≈ 2 MB
//! - Row buffer: 64 × 3840 × 16 bytes ≈ 4 MB
//! - Total ≈ 6 MB vs ~130 MB for a full-frame buffer
//!
//! # Status
//!
//! Lightly tested. API may change.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::math::{floorf, lnf};

/// Configuration for the streaming tonemapper.
///
/// Tunes the local adaptation and per-pixel curve. The channel count is
/// passed separately to [`StreamingTonemapper::new`].
#[derive(Debug, Clone)]
pub struct StreamingTonemapConfig {
    /// Grid cell size in pixels (default: 8).
    pub cell_size: u32,
    /// Number of lookahead rows to buffer (default: 64).
    pub lookahead_rows: u32,
    /// Target mid-gray (key) value in linear (default: 0.18).
    pub target_key: f32,
    /// Contrast boost factor (default: 1.1).
    pub contrast: f32,
    /// Saturation preservation in highlights (default: 0.95).
    pub saturation: f32,
    /// Shadow lift amount (default: 0.02).
    pub shadow_lift: f32,
    /// Highlight desaturation threshold (fraction of white point, default: 0.5).
    pub desat_threshold: f32,
    /// How much of the key adaptation is local vs global (0.0 = fully
    /// global, 1.0 = fully local). Default: 0.3.
    ///
    /// At 0.0, this is a global Reinhard-like tonemapper — the entire
    /// image's geometric-mean key is used, preserving gradients but
    /// losing local shadow/highlight detail. At 1.0, each grid cell
    /// adapts independently, maximizing local contrast but flattening
    /// smooth gradients.
    ///
    /// 0.3 is a reasonable default for photographic content.
    pub locality: f32,
}

impl Default for StreamingTonemapConfig {
    fn default() -> Self {
        Self {
            cell_size: 8,
            lookahead_rows: 64,
            target_key: 0.18,
            contrast: 1.1,
            saturation: 0.95,
            shadow_lift: 0.02,
            desat_threshold: 0.5,
            locality: 0.3,
        }
    }
}

// ============================================================================
// Local adaptation grid
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct CellStats {
    sum_log_lum: f64,
    count: u32,
    max_lum: f32,
    min_lum: f32,
    sum_lum: f64,
}

impl Default for CellStats {
    fn default() -> Self {
        Self {
            sum_log_lum: 0.0,
            count: 0,
            max_lum: 0.0,
            min_lum: f32::MAX,
            sum_lum: 0.0,
        }
    }
}

impl CellStats {
    #[inline]
    fn add(&mut self, lum: f32) {
        if lum > 1e-6 {
            self.sum_log_lum += lnf(lum) as f64;
            self.count += 1;
            self.max_lum = self.max_lum.max(lum);
            self.min_lum = self.min_lum.min(lum);
            self.sum_lum += lum as f64;
        }
    }

    fn compute_params(self, global_key: f32, global_white: f32) -> LocalParams {
        if self.count == 0 {
            return LocalParams {
                key: global_key,
                white: global_white,
                black: 0.0,
            };
        }
        // Geometric mean via sum_log_lum
        let mean_log = self.sum_log_lum / self.count as f64;
        let key = libm::expf(mean_log as f32);
        let mean = (self.sum_lum / self.count as f64) as f32;
        let white = mean + (self.max_lum - mean) * 0.8;

        LocalParams {
            key: key.max(0.001),
            white: white.max(key * 2.0),
            black: if self.min_lum < f32::MAX {
                self.min_lum
            } else {
                0.0
            },
        }
    }
}

/// Local adaptation parameters for a region (internal).
#[derive(Debug, Clone, Copy)]
pub(crate) struct LocalParams {
    pub(crate) key: f32,
    pub(crate) white: f32,
    pub(crate) black: f32,
}

impl Default for LocalParams {
    fn default() -> Self {
        Self {
            key: 0.18,
            white: 4.0,
            black: 0.0,
        }
    }
}

struct AdaptationGrid {
    width: u32,
    height: u32,
    cell_size: u32,
    stats: Vec<CellStats>,
    params: Vec<LocalParams>,
    global_stats: CellStats,
    rows_processed: u32,
}

impl AdaptationGrid {
    fn new(image_width: u32, image_height: u32, cell_size: u32) -> Self {
        let width = image_width.div_ceil(cell_size);
        let height = image_height.div_ceil(cell_size);
        let num_cells = (width * height) as usize;
        Self {
            width,
            height,
            cell_size,
            stats: vec![CellStats::default(); num_cells],
            params: vec![LocalParams::default(); num_cells],
            global_stats: CellStats::default(),
            rows_processed: 0,
        }
    }

    fn add_row(&mut self, row_data: &[f32], y: u32, image_width: u32, channels: usize) {
        let cell_y = y / self.cell_size;
        if cell_y >= self.height {
            return;
        }

        for (x, pixel) in row_data
            .chunks(channels)
            .enumerate()
            .take(image_width as usize)
        {
            let lum = luminance_bt709(pixel[0], pixel[1], pixel[2]);
            self.global_stats.add(lum);
            let cell_x = (x as u32) / self.cell_size;
            if cell_x < self.width {
                let cell_idx = (cell_y * self.width + cell_x) as usize;
                self.stats[cell_idx].add(lum);
            }
        }
    }

    fn finalize_row(&mut self, cell_y: u32) {
        if cell_y >= self.height {
            return;
        }
        let global = self.global_stats.compute_params(0.18, 4.0);
        for cell_x in 0..self.width {
            let idx = (cell_y * self.width + cell_x) as usize;
            self.params[idx] = self.stats[idx].compute_params(global.key, global.white);
        }
        self.rows_processed = self.rows_processed.max(cell_y + 1);
    }

    fn sample(&self, x: f32, y: f32) -> LocalParams {
        let gx = x / self.cell_size as f32;
        let gy = y / self.cell_size as f32;
        let gx_floor = floorf(gx);
        let gy_floor = floorf(gy);

        let x0 = (gx_floor as u32).min(self.width.saturating_sub(1));
        let y0 = (gy_floor as u32).min(self.height.saturating_sub(1));
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));

        // Clamp y1 to rows that have actually been finalized
        let y1 = y1.min(self.rows_processed.saturating_sub(1));
        let y0 = y0.min(y1);

        let fx = gx - gx_floor;
        let fy = gy - gy_floor;

        let p00 = &self.params[(y0 * self.width + x0) as usize];
        let p10 = &self.params[(y0 * self.width + x1) as usize];
        let p01 = &self.params[(y1 * self.width + x0) as usize];
        let p11 = &self.params[(y1 * self.width + x1) as usize];

        LocalParams {
            key: bilinear(p00.key, p10.key, p01.key, p11.key, fx, fy),
            white: bilinear(p00.white, p10.white, p01.white, p11.white, fx, fy),
            black: bilinear(p00.black, p10.black, p01.black, p11.black, fx, fy),
        }
    }

    fn global_params(&self) -> LocalParams {
        self.global_stats.compute_params(0.18, 4.0)
    }

    fn blur_params(&mut self, radius: u32) {
        if radius == 0 || self.width < 3 || self.height < 3 {
            return;
        }
        let mut blurred = vec![LocalParams::default(); self.params.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum_key = 0.0_f32;
                let mut sum_white = 0.0_f32;
                let mut sum_black = 0.0_f32;
                let mut count = 0.0_f32;

                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius + 1).min(self.height);
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius + 1).min(self.width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        let idx = (ny * self.width + nx) as usize;
                        let p = &self.params[idx];
                        sum_key += p.key;
                        sum_white += p.white;
                        sum_black += p.black;
                        count += 1.0;
                    }
                }

                let idx = (y * self.width + x) as usize;
                blurred[idx] = LocalParams {
                    key: sum_key / count,
                    white: sum_white / count,
                    black: sum_black / count,
                };
            }
        }
        self.params = blurred;
    }
}

// ============================================================================
// Streaming tonemapper
// ============================================================================

/// Streaming tonemapper with local adaptation.
///
/// # Flow
///
/// Pull-based; the caller drives push/pull alternation and owns all pixel
/// buffers. Zero allocation in the steady state — the HDR ring buffer and
/// local adaptation grid are pre-allocated in [`StreamingTonemapper::new`].
///
/// ```text
/// loop {
///     tm.push_row(hdr_row)?;
///     while let Some(idx) = tm.pull_row(&mut sdr_row)? {
///         // consume sdr_row (row index = idx)
///     }
/// }
/// tm.finish();
/// while let Some(idx) = tm.pull_row(&mut sdr_row)? { /* flush */ }
/// ```
pub struct StreamingTonemapper {
    config: StreamingTonemapConfig,
    width: u32,
    height: u32,
    /// Channels per pixel (3 or 4).
    channels: u8,
    /// Elements per row (`width * channels`).
    row_stride: usize,
    grid: AdaptationGrid,
    /// Flat HDR ring buffer, `lookahead_rows * row_stride` elements.
    row_buffer: Vec<f32>,
    /// Image-row index of the oldest row currently stored.
    buffer_start_row: u32,
    /// Number of rows currently stored.
    buffer_count: u32,
    /// Next image-row index awaiting output.
    next_output_row: u32,
    input_complete: bool,
}

impl StreamingTonemapper {
    /// Create a new streaming tonemapper.
    ///
    /// Pre-allocates the HDR ring buffer (`lookahead_rows * width * channels`
    /// f32 elements) and the local adaptation grid.
    ///
    /// `channels` must be 3 (RGB) or 4 (RGBA).
    pub fn new(
        width: u32,
        height: u32,
        channels: u8,
        config: StreamingTonemapConfig,
    ) -> Result<Self> {
        if channels != 3 && channels != 4 {
            return Err(Error::InvalidConfig("channels must be 3 or 4"));
        }
        if config.cell_size == 0 {
            return Err(Error::InvalidConfig("cell_size must be >= 1"));
        }
        if config.lookahead_rows == 0 {
            return Err(Error::InvalidConfig("lookahead_rows must be >= 1"));
        }
        let grid = AdaptationGrid::new(width, height, config.cell_size);
        let row_stride = width as usize * channels as usize;
        let buffer_elements = row_stride * config.lookahead_rows as usize;
        Ok(Self {
            config,
            width,
            height,
            channels,
            row_stride,
            grid,
            row_buffer: vec![0.0_f32; buffer_elements],
            buffer_start_row: 0,
            buffer_count: 0,
            next_output_row: 0,
            input_complete: false,
        })
    }

    /// Elements per row: `width * channels`.
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Configured channel count (3 or 4).
    #[inline]
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Push one HDR row.
    ///
    /// `hdr_row.len()` must equal [`row_stride`](Self::row_stride). The row
    /// is copied into the internal ring buffer and used to update local
    /// adaptation statistics — no allocation.
    ///
    /// Returns `Err(BufferTooSmall)` if the slice is shorter than the row
    /// stride, or an error if the ring buffer is full (pull at least one
    /// row first).
    pub fn push_row(&mut self, hdr_row: &[f32]) -> Result<()> {
        if hdr_row.len() < self.row_stride {
            return Err(Error::BufferTooSmall {
                required: self.row_stride,
                actual: hdr_row.len(),
            });
        }
        if self.buffer_count as usize >= self.config.lookahead_rows as usize {
            return Err(Error::RingBufferFull);
        }
        let input_row = self.buffer_start_row + self.buffer_count;
        if input_row >= self.height {
            return Ok(()); // silently drop beyond image height
        }

        let channels = self.channels as usize;
        let src = &hdr_row[..self.row_stride];

        self.grid.add_row(src, input_row, self.width, channels);

        let buffer_idx = (input_row % self.config.lookahead_rows) as usize;
        let slot_start = buffer_idx * self.row_stride;
        self.row_buffer[slot_start..slot_start + self.row_stride].copy_from_slice(src);
        self.buffer_count += 1;

        if input_row % self.config.cell_size == self.config.cell_size - 1 {
            self.grid.finalize_row(input_row / self.config.cell_size);
        }
        Ok(())
    }

    /// Signal that all input has been provided. Call [`pull_row`] repeatedly
    /// afterwards until [`rows_ready`] returns zero to flush remaining rows.
    ///
    /// [`pull_row`]: Self::pull_row
    /// [`rows_ready`]: Self::rows_ready
    pub fn finish(&mut self) {
        self.input_complete = true;
        let last_cell_y = (self.height.saturating_sub(1)) / self.config.cell_size;
        for y in 0..=last_cell_y {
            self.grid.finalize_row(y);
        }
        self.grid.blur_params(1);
    }

    /// Number of rows ready for immediate pull.
    ///
    /// Under steady state we hold `lookahead_rows / 2` rows of context before
    /// emitting; after [`finish`](Self::finish) all buffered rows are emittable.
    pub fn rows_ready(&self) -> u32 {
        let remaining = self.height.saturating_sub(self.next_output_row);
        let required = if self.input_complete {
            0
        } else {
            self.config.lookahead_rows / 2
        };
        self.buffer_count.saturating_sub(required).min(remaining)
    }

    /// Pull the next ready row into `out`.
    ///
    /// `out.len()` must equal [`row_stride`](Self::row_stride).
    ///
    /// Returns:
    /// - `Ok(Some(row_index))` — wrote a row into `out`.
    /// - `Ok(None)` — nothing to pull yet; push more input or call
    ///   [`finish`](Self::finish).
    /// - `Err(BufferTooSmall)` — `out` slice is too short.
    pub fn pull_row(&mut self, out: &mut [f32]) -> Result<Option<u32>> {
        if out.len() < self.row_stride {
            return Err(Error::BufferTooSmall {
                required: self.row_stride,
                actual: out.len(),
            });
        }
        if self.rows_ready() == 0 {
            return Ok(None);
        }

        let row_index = self.next_output_row;
        let buffer_idx = (row_index % self.config.lookahead_rows) as usize;
        let slot_start = buffer_idx * self.row_stride;
        let hdr_slice = &self.row_buffer[slot_start..slot_start + self.row_stride];

        // Dispatch to a const-generic inner loop so the alpha branch and
        // fixed stride are compile-time folded.
        match self.channels {
            3 => self.tonemap_row_impl::<3>(hdr_slice, row_index, &mut out[..self.row_stride]),
            4 => self.tonemap_row_impl::<4>(hdr_slice, row_index, &mut out[..self.row_stride]),
            _ => unreachable!("channels validated in new()"),
        }

        self.next_output_row += 1;
        self.buffer_start_row += 1;
        self.buffer_count -= 1;

        Ok(Some(row_index))
    }

    #[inline]
    fn tonemap_row_impl<const CN: usize>(&self, hdr_row: &[f32], y: u32, out: &mut [f32]) {
        let global = self.grid.global_params();
        for (x, (hdr_pixel, sdr_pixel)) in hdr_row
            .chunks_exact(CN)
            .zip(out.chunks_exact_mut(CN))
            .enumerate()
        {
            let local = self.grid.sample(x as f32, y as f32);
            let loc = self.config.locality;
            let glb = 1.0 - loc;
            let params = LocalParams {
                key: local.key * loc + global.key * glb,
                white: local.white * loc + global.white * glb,
                black: local.black * loc + global.black * glb,
            };

            let rgb = self.tonemap_pixel([hdr_pixel[0], hdr_pixel[1], hdr_pixel[2]], &params);
            sdr_pixel[0] = rgb[0];
            sdr_pixel[1] = rgb[1];
            sdr_pixel[2] = rgb[2];
            if CN == 4 {
                sdr_pixel[3] = hdr_pixel[3];
            }
        }
    }

    fn tonemap_pixel(&self, rgb: [f32; 3], local: &LocalParams) -> [f32; 3] {
        let lum = luminance_bt709(rgb[0], rgb[1], rgb[2]);
        if lum <= 0.0 {
            return [self.config.shadow_lift; 3];
        }

        let key_scale = self.config.target_key / local.key.max(0.001);
        let adapted = lum * key_scale;
        let white_adapted = local.white * key_scale;
        let mapped = self.sigmoid_tonemap(adapted, white_adapted);
        let ratio = mapped / adapted;

        let mut out = [
            (rgb[0] * key_scale * ratio).max(0.0),
            (rgb[1] * key_scale * ratio).max(0.0),
            (rgb[2] * key_scale * ratio).max(0.0),
        ];

        let desat_start = white_adapted * self.config.desat_threshold;
        if adapted > desat_start && white_adapted > desat_start {
            let t = ((adapted - desat_start) / (white_adapted - desat_start)).clamp(0.0, 1.0);
            let desat_factor = t * t * (1.0 - self.config.saturation);
            let out_lum = luminance_bt709(out[0], out[1], out[2]);
            out[0] = out[0] * (1.0 - desat_factor) + out_lum * desat_factor;
            out[1] = out[1] * (1.0 - desat_factor) + out_lum * desat_factor;
            out[2] = out[2] * (1.0 - desat_factor) + out_lum * desat_factor;
        }

        out[0] += self.config.shadow_lift * (1.0 - out[0]);
        out[1] += self.config.shadow_lift * (1.0 - out[1]);
        out[2] += self.config.shadow_lift * (1.0 - out[2]);

        [
            out[0].clamp(0.0, 1.0),
            out[1].clamp(0.0, 1.0),
            out[2].clamp(0.0, 1.0),
        ]
    }

    fn sigmoid_tonemap(&self, x: f32, white: f32) -> f32 {
        // Extended Reinhard with contrast boost and white-point normalization.
        // This is the same formula as the libultrahdr-validated reinhard_extended,
        // with a contrast pre-scale for punch.
        let x_scaled = x * self.config.contrast;
        let w = white * self.config.contrast;
        let w2 = w * w;
        // Extended Reinhard: x * (1 + x/w²) / (1 + x)
        (x_scaled * (1.0 + x_scaled / w2.max(1e-6)) / (1.0 + x_scaled)).min(1.0)
    }

    /// Progress info: (rows_output, total_rows).
    pub fn progress(&self) -> (u32, u32) {
        (self.next_output_row, self.height)
    }
}

#[inline]
fn luminance_bt709(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[inline]
fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bottom = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bottom * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_bad_channels() {
        assert!(StreamingTonemapper::new(64, 64, 2, StreamingTonemapConfig::default()).is_err());
    }

    /// Drive push/pull until the image is consumed. Reuses a single
    /// caller-owned output buffer — zero allocation in the loop.
    fn run(
        tm: &mut StreamingTonemapper,
        row: &[f32],
        height: u32,
        mut consume: impl FnMut(u32, &[f32]),
    ) {
        let mut out = alloc::vec![0.0_f32; tm.row_stride()];
        for _ in 0..height {
            tm.push_row(row).unwrap();
            while let Some(idx) = tm.pull_row(&mut out).unwrap() {
                consume(idx, &out);
            }
        }
        tm.finish();
        while let Some(idx) = tm.pull_row(&mut out).unwrap() {
            consume(idx, &out);
        }
    }

    #[test]
    fn process_uniform_image_rgb() {
        let w = 32_u32;
        let h = 32_u32;
        let mut tm = StreamingTonemapper::new(w, h, 3, StreamingTonemapConfig::default()).unwrap();

        let row = alloc::vec![0.5_f32; tm.row_stride()];
        let mut emitted = 0_u32;
        run(&mut tm, &row, h, |_idx, sdr| {
            assert_eq!(sdr.len(), (w * 3) as usize);
            for &v in sdr {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "out of range in uniform mid-gray: {v}"
                );
            }
            emitted += 1;
        });
        assert_eq!(emitted, h, "all rows should emit");
    }

    #[test]
    fn rgba_alpha_preserved() {
        let w = 16_u32;
        let h = 16_u32;
        let mut tm = StreamingTonemapper::new(w, h, 4, StreamingTonemapConfig::default()).unwrap();

        let row: Vec<f32> = (0..w).flat_map(|_| [0.3_f32, 0.3, 0.3, 0.42]).collect();
        let mut emitted = 0_u32;
        run(&mut tm, &row, h, |_idx, sdr| {
            for pixel in sdr.chunks_exact(4) {
                assert!(
                    (pixel[3] - 0.42).abs() < 1e-6,
                    "alpha not preserved: {}",
                    pixel[3]
                );
            }
            emitted += 1;
        });
        assert_eq!(emitted, h);
    }

    #[test]
    fn push_row_rejects_short_slice() {
        let mut tm = StreamingTonemapper::new(8, 8, 3, StreamingTonemapConfig::default()).unwrap();
        let bad = alloc::vec![0.0_f32; 10]; // needs 24
        let err = tm.push_row(&bad).unwrap_err();
        assert!(matches!(err, Error::BufferTooSmall { .. }));
    }

    #[test]
    fn pull_row_rejects_short_out() {
        let mut tm = StreamingTonemapper::new(8, 8, 3, StreamingTonemapConfig::default()).unwrap();
        let row = alloc::vec![0.5_f32; tm.row_stride()];
        for _ in 0..8 {
            tm.push_row(&row).unwrap();
        }
        tm.finish();
        let mut small = alloc::vec![0.0_f32; 5];
        assert!(matches!(
            tm.pull_row(&mut small),
            Err(Error::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn push_row_refuses_to_overflow_buffer() {
        let cfg = StreamingTonemapConfig {
            lookahead_rows: 4,
            ..Default::default()
        };
        let mut tm = StreamingTonemapper::new(8, 16, 3, cfg).unwrap();
        let row = alloc::vec![0.1_f32; tm.row_stride()];
        for _ in 0..4 {
            tm.push_row(&row).unwrap();
        }
        let err = tm.push_row(&row).unwrap_err();
        assert!(matches!(err, Error::RingBufferFull));
    }

    #[test]
    fn image_smaller_than_lookahead_still_processes() {
        // h = 4 rows, lookahead = 64. Must not deadlock or lose rows.
        let w = 8_u32;
        let h = 4_u32;
        let mut tm = StreamingTonemapper::new(w, h, 3, StreamingTonemapConfig::default()).unwrap();
        let row = alloc::vec![0.5_f32; tm.row_stride()];
        let mut emitted = 0_u32;
        let mut out = alloc::vec![0.0_f32; tm.row_stride()];
        for _ in 0..h {
            tm.push_row(&row).unwrap();
            while tm.pull_row(&mut out).unwrap().is_some() {
                emitted += 1;
            }
        }
        tm.finish();
        while tm.pull_row(&mut out).unwrap().is_some() {
            emitted += 1;
        }
        assert_eq!(emitted, h, "tiny image should emit all rows");
    }

    #[test]
    fn height_not_multiple_of_cell_size() {
        // h = 13, cell_size = 8 → last row-cell is partial.
        let cfg = StreamingTonemapConfig {
            cell_size: 8,
            ..Default::default()
        };
        let w = 16_u32;
        let h = 13_u32;
        let mut tm = StreamingTonemapper::new(w, h, 3, cfg).unwrap();
        let row = alloc::vec![0.3_f32; tm.row_stride()];
        let mut out = alloc::vec![0.0_f32; tm.row_stride()];
        let mut emitted = 0_u32;
        for _ in 0..h {
            tm.push_row(&row).unwrap();
            while tm.pull_row(&mut out).unwrap().is_some() {
                emitted += 1;
            }
        }
        tm.finish();
        while tm.pull_row(&mut out).unwrap().is_some() {
            emitted += 1;
        }
        assert_eq!(emitted, h);
    }

    #[test]
    fn custom_cell_size_and_lookahead() {
        // Non-default cell_size and lookahead_rows. Verify the pipeline
        // still completes and emits every row.
        let cfg = StreamingTonemapConfig {
            cell_size: 4,
            lookahead_rows: 16,
            ..Default::default()
        };
        let w = 64_u32;
        let h = 32_u32;
        let mut tm = StreamingTonemapper::new(w, h, 4, cfg).unwrap();
        let row: Vec<f32> = (0..w).flat_map(|_| [0.4_f32, 0.5, 0.6, 0.9]).collect();
        let mut out = alloc::vec![0.0_f32; tm.row_stride()];
        let mut emitted = 0_u32;
        for _ in 0..h {
            tm.push_row(&row).unwrap();
            while tm.pull_row(&mut out).unwrap().is_some() {
                emitted += 1;
            }
        }
        tm.finish();
        while tm.pull_row(&mut out).unwrap().is_some() {
            emitted += 1;
        }
        assert_eq!(emitted, h);
    }

    #[test]
    fn row_stride_matches_width_times_channels() {
        let tm3 = StreamingTonemapper::new(120, 10, 3, StreamingTonemapConfig::default()).unwrap();
        assert_eq!(tm3.row_stride(), 360);
        assert_eq!(tm3.channels(), 3);
        let tm4 = StreamingTonemapper::new(120, 10, 4, StreamingTonemapConfig::default()).unwrap();
        assert_eq!(tm4.row_stride(), 480);
        assert_eq!(tm4.channels(), 4);
    }

    #[test]
    fn progress_reports_rows_output() {
        let w = 16_u32;
        let h = 8_u32;
        let mut tm = StreamingTonemapper::new(w, h, 3, StreamingTonemapConfig::default()).unwrap();
        let row = alloc::vec![0.5_f32; tm.row_stride()];
        let mut out = alloc::vec![0.0_f32; tm.row_stride()];
        for _ in 0..h {
            tm.push_row(&row).unwrap();
            while tm.pull_row(&mut out).unwrap().is_some() {}
        }
        tm.finish();
        while tm.pull_row(&mut out).unwrap().is_some() {}

        let (done, total) = tm.progress();
        assert_eq!(done, h);
        assert_eq!(total, h);
    }
}
