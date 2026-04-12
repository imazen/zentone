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
    /// Number of channels in input data (3 for RGB, 4 for RGBA). Default: 4.
    pub channels: u8,
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
            channels: 4,
        }
    }
}

impl StreamingTonemapConfig {
    /// Configure for RGB input (3 channels, no alpha).
    pub fn rgb() -> Self {
        Self {
            channels: 3,
            ..Default::default()
        }
    }

    /// Configure for RGBA input (4 channels with alpha).
    pub fn rgba() -> Self {
        Self::default()
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

/// Local adaptation parameters for a region.
#[derive(Debug, Clone, Copy)]
pub struct LocalParams {
    /// Local key (geometric mean luminance).
    pub key: f32,
    /// Local white point (approximate max).
    pub white: f32,
    /// Local black point (approximate min).
    pub black: f32,
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
pub struct StreamingTonemapper {
    config: StreamingTonemapConfig,
    width: u32,
    height: u32,
    grid: AdaptationGrid,
    row_buffer: Vec<Vec<f32>>,
    buffer_start_row: u32,
    buffer_count: u32,
    next_output_row: u32,
    input_complete: bool,
}

/// Output produced by processing a row.
#[derive(Debug, Clone)]
pub struct TonemapOutput {
    /// The tonemapped SDR row (linear RGB/RGBA, ready for an OETF).
    pub sdr_linear: Vec<f32>,
    /// The row index this corresponds to.
    pub row_index: u32,
}

impl StreamingTonemapper {
    /// Create a new streaming tonemapper.
    pub fn new(width: u32, height: u32, config: StreamingTonemapConfig) -> Result<Self> {
        if config.channels != 3 && config.channels != 4 {
            return Err(Error::InvalidConfig("channels must be 3 or 4"));
        }
        if config.cell_size == 0 {
            return Err(Error::InvalidConfig("cell_size must be >= 1"));
        }
        if config.lookahead_rows == 0 {
            return Err(Error::InvalidConfig("lookahead_rows must be >= 1"));
        }
        let grid = AdaptationGrid::new(width, height, config.cell_size);
        let buffer_size = config.lookahead_rows as usize;
        Ok(Self {
            config,
            width,
            height,
            grid,
            row_buffer: vec![Vec::new(); buffer_size],
            buffer_start_row: 0,
            buffer_count: 0,
            next_output_row: 0,
            input_complete: false,
        })
    }

    /// Push HDR rows from a slice with stride.
    ///
    /// - `data`: slice containing row data (linear HDR, f32).
    /// - `stride`: number of f32 elements between row starts
    ///   (≥ `width * channels`).
    /// - `num_rows`: number of rows to process from this slice.
    ///
    /// Returns any tonemapped rows that are ready for output.
    pub fn push_rows(
        &mut self,
        data: &[f32],
        stride: usize,
        num_rows: usize,
    ) -> Result<Vec<TonemapOutput>> {
        let channels = self.config.channels as usize;
        let row_width = self.width as usize * channels;

        for row_idx in 0..num_rows {
            let input_row = self.buffer_start_row + self.buffer_count;
            if input_row >= self.height {
                break;
            }
            let start = row_idx * stride;
            if start + row_width > data.len() {
                break;
            }
            let row_data = &data[start..start + row_width];

            self.grid.add_row(row_data, input_row, self.width, channels);

            let buffer_idx = (input_row % self.config.lookahead_rows) as usize;
            let buffer_slot = &mut self.row_buffer[buffer_idx];
            buffer_slot.clear();
            buffer_slot.extend_from_slice(row_data);
            self.buffer_count += 1;

            let completed_cell_y = input_row / self.config.cell_size;
            if input_row % self.config.cell_size == self.config.cell_size - 1 {
                self.grid.finalize_row(completed_cell_y);
            }
        }

        self.try_output_rows()
    }

    /// Push a single HDR row (convenience wrapper).
    #[inline]
    pub fn push_row(&mut self, row: &[f32]) -> Result<Vec<TonemapOutput>> {
        self.push_rows(row, row.len(), 1)
    }

    /// Signal that all input has been provided. Flushes remaining rows.
    pub fn finish(&mut self) -> Result<Vec<TonemapOutput>> {
        self.input_complete = true;

        let last_cell_y = (self.height.saturating_sub(1)) / self.config.cell_size;
        for y in 0..=last_cell_y {
            self.grid.finalize_row(y);
        }
        self.grid.blur_params(1);

        self.try_output_rows()
    }

    fn try_output_rows(&mut self) -> Result<Vec<TonemapOutput>> {
        let mut outputs = Vec::new();
        let last_input_row = self.buffer_start_row + self.buffer_count;

        let required_ahead = if self.input_complete {
            0
        } else {
            self.config.lookahead_rows / 2
        };

        while self.next_output_row < self.height {
            let rows_ahead = last_input_row.saturating_sub(self.next_output_row);
            if rows_ahead < required_ahead && !self.input_complete {
                break;
            }
            if self.next_output_row < self.buffer_start_row {
                self.next_output_row += 1;
                continue;
            }

            let buffer_idx = (self.next_output_row % self.config.lookahead_rows) as usize;
            let hdr_row = &self.row_buffer[buffer_idx];
            if hdr_row.is_empty() {
                break;
            }

            let sdr_row = self.tonemap_row_inner(hdr_row, self.next_output_row);
            outputs.push(TonemapOutput {
                sdr_linear: sdr_row,
                row_index: self.next_output_row,
            });

            self.next_output_row += 1;
            if self.next_output_row > self.config.lookahead_rows {
                self.buffer_start_row = self.next_output_row - self.config.lookahead_rows;
            }
        }

        Ok(outputs)
    }

    fn tonemap_row_inner(&self, hdr_row: &[f32], y: u32) -> Vec<f32> {
        let channels = self.config.channels as usize;
        let mut sdr_row = vec![0.0_f32; self.width as usize * channels];
        let global = self.grid.global_params();

        for (x, (hdr_pixel, sdr_pixel)) in hdr_row
            .chunks(channels)
            .zip(sdr_row.chunks_mut(channels))
            .enumerate()
            .take(self.width as usize)
        {
            let local = self.grid.sample(x as f32, y as f32);
            // Blend local with global for stability at edges
            let blend = 0.7_f32;
            let params = LocalParams {
                key: local.key * blend + global.key * (1.0 - blend),
                white: local.white * blend + global.white * (1.0 - blend),
                black: local.black * blend + global.black * (1.0 - blend),
            };

            let rgb = self.tonemap_pixel([hdr_pixel[0], hdr_pixel[1], hdr_pixel[2]], &params);
            sdr_pixel[0] = rgb[0];
            sdr_pixel[1] = rgb[1];
            sdr_pixel[2] = rgb[2];
            if channels >= 4 {
                sdr_pixel[3] = hdr_pixel.get(3).copied().unwrap_or(1.0);
            }
        }
        sdr_row
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
        let x_scaled = x * self.config.contrast;
        let w2 = white * white;
        let knee = 0.5_f32;
        if x_scaled < knee {
            x_scaled * 1.05
        } else {
            let base = knee * 1.05;
            let over = x_scaled - knee;
            let compressed = over * (1.0 + over / w2.max(1e-6)) / (1.0 + over);
            (base + compressed * (1.0 - knee)).min(1.0)
        }
    }

    /// Progress info: (rows_output, total_rows).
    pub fn progress(&self) -> (u32, u32) {
        (self.next_output_row, self.height)
    }

    /// Configured number of channels.
    pub fn channels(&self) -> u8 {
        self.config.channels
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
        let cfg = StreamingTonemapConfig {
            channels: 2,
            ..Default::default()
        };
        assert!(StreamingTonemapper::new(64, 64, cfg).is_err());
    }

    #[test]
    fn process_uniform_image() {
        let w = 32_u32;
        let h = 32_u32;
        let cfg = StreamingTonemapConfig::rgb();
        let mut tm = StreamingTonemapper::new(w, h, cfg).unwrap();

        let row: Vec<f32> = vec![0.5_f32; (w * 3) as usize];
        let mut all_out: Vec<TonemapOutput> = Vec::new();
        for _ in 0..h {
            all_out.extend(tm.push_row(&row).unwrap());
        }
        all_out.extend(tm.finish().unwrap());

        assert_eq!(all_out.len() as u32, h, "all rows should emit");
        for output in &all_out {
            assert_eq!(output.sdr_linear.len(), (w * 3) as usize);
            for &v in &output.sdr_linear {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "out of range in uniform mid-gray: {v}"
                );
            }
        }
    }

    #[test]
    fn rgba_alpha_preserved() {
        let w = 16_u32;
        let h = 16_u32;
        let cfg = StreamingTonemapConfig::rgba();
        let mut tm = StreamingTonemapper::new(w, h, cfg).unwrap();

        let row: Vec<f32> = (0..w).flat_map(|_| [0.3_f32, 0.3, 0.3, 0.42]).collect();
        let mut all_out: Vec<TonemapOutput> = Vec::new();
        for _ in 0..h {
            all_out.extend(tm.push_row(&row).unwrap());
        }
        all_out.extend(tm.finish().unwrap());

        assert_eq!(all_out.len() as u32, h);
        for output in &all_out {
            for pixel in output.sdr_linear.chunks_exact(4) {
                assert!(
                    (pixel[3] - 0.42).abs() < 1e-6,
                    "alpha not preserved: {}",
                    pixel[3]
                );
            }
        }
    }
}
