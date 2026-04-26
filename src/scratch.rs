//! Reusable scratch buffers for the SIMD tone-map pipelines.
//!
//! [`TonemapScratch`] owns the per-chunk intermediate buffers (linear-RGB
//! scratch, u8 staging) so the public pipeline functions in [`crate::pipeline`]
//! don't allocate per call. One scratch instance amortizes across an entire
//! image, video stream, or worker thread.
//!
//! The chunk size also caps the working set: each pipeline call processes its
//! input in chunks of at most this many pixels, so even a "whole image as one
//! row" passed in doesn't blow past L2/L3 cache. The default of 4096 keeps
//! `[f32; 3]` × ~3 buffers (~48 KiB) inside most desktop L2 caches.

use alloc::vec::Vec;

/// Reusable scratch buffers + chunk-size policy for the SIMD tone-map pipelines.
///
/// Owns the per-chunk intermediate buffers (linear-RGB scratch, u8 staging) so
/// the public pipeline functions don't allocate per call. One scratch instance
/// amortizes across an entire image / video stream / worker thread.
///
/// The chunk size caps the working set: each pipeline call processes the input
/// in chunks of at most this many pixels, so even a "whole image as one row"
/// passed in doesn't blow past L2/L3 cache. Default 4096 pixels.
///
/// # Example
///
/// ```
/// use zentone::{Bt2408Tonemapper, pipeline::tonemap_pq_row_simd};
/// use zentone::TonemapScratch;
///
/// let mut scratch = TonemapScratch::new();
/// let curve = Bt2408Tonemapper::new(4000.0, 1000.0);
///
/// let pq = vec![[0.58_f32, 0.58, 0.58]; 1024];
/// let mut out = vec![[0.0_f32; 3]; 1024];
/// tonemap_pq_row_simd(&mut scratch, &pq, &mut out, &curve);
/// ```
#[derive(Debug)]
pub struct TonemapScratch {
    chunk_pixels: usize,
    linear_rgb: Vec<[f32; 3]>,
    u8_rgb: Vec<[u8; 3]>,
}

impl Default for TonemapScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl TonemapScratch {
    /// Create with the default chunk size (4096 pixels).
    pub fn new() -> Self {
        Self::with_chunk_size(4096)
    }

    /// Create with a caller-specified chunk cap.
    ///
    /// Larger chunks amortize per-call dispatch overhead better; smaller chunks
    /// fit lower cache levels. The default 4096 keeps an `[f32; 3]` strip × 3
    /// buffers (~48 KiB working set) inside most desktop L2 caches.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_pixels < 8` — SIMD lanes need at least 8 pixels per
    /// chunk to make the kernel dispatch worthwhile.
    pub fn with_chunk_size(chunk_pixels: usize) -> Self {
        assert!(
            chunk_pixels >= 8,
            "chunk size must be at least 8 pixels for SIMD lanes"
        );
        Self {
            chunk_pixels,
            linear_rgb: Vec::new(),
            u8_rgb: Vec::new(),
        }
    }

    /// Number of pixels processed per internal chunk.
    pub fn chunk_size(&self) -> usize {
        self.chunk_pixels
    }

    // ---- pub(crate) accessors for the pipeline's chunk loop ----

    /// Resize-grow the linear-RGB scratch and return a `len`-element slice.
    pub(crate) fn linear_rgb(&mut self, len: usize) -> &mut [[f32; 3]] {
        if self.linear_rgb.len() < len {
            self.linear_rgb.resize(len, [0.0; 3]);
        }
        &mut self.linear_rgb[..len]
    }

    /// Resize-grow both internal buffers and return both as a pair of slices.
    /// Lets a caller hold borrows on both simultaneously without borrow-checker
    /// gymnastics.
    pub(crate) fn linear_and_u8(&mut self, len: usize) -> (&mut [[f32; 3]], &mut [[u8; 3]]) {
        if self.linear_rgb.len() < len {
            self.linear_rgb.resize(len, [0.0; 3]);
        }
        if self.u8_rgb.len() < len {
            self.u8_rgb.resize(len, [0; 3]);
        }
        (&mut self.linear_rgb[..len], &mut self.u8_rgb[..len])
    }

    /// Capacity of the linear-RGB scratch buffer (in pixels).
    ///
    /// Exposed for the memory-bound regression test in `tests/simd_parity.rs`.
    /// Not part of the stable public surface — return values are an
    /// implementation detail.
    #[doc(hidden)]
    pub fn linear_rgb_capacity(&self) -> usize {
        self.linear_rgb.capacity()
    }

    /// Capacity of the u8-RGB staging buffer (in pixels).
    ///
    /// Exposed for the memory-bound regression test in `tests/simd_parity.rs`.
    /// Not part of the stable public surface — return values are an
    /// implementation detail.
    #[doc(hidden)]
    pub fn u8_rgb_capacity(&self) -> usize {
        self.u8_rgb.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_4096() {
        let s = TonemapScratch::new();
        assert_eq!(s.chunk_size(), 4096);
    }

    #[test]
    #[should_panic(expected = "at least 8 pixels")]
    fn rejects_tiny_chunks() {
        let _ = TonemapScratch::with_chunk_size(4);
    }

    #[test]
    fn buffers_grow_only_as_needed() {
        let mut s = TonemapScratch::with_chunk_size(64);
        assert_eq!(s.linear_rgb_capacity(), 0);
        let _ = s.linear_rgb(32);
        assert!(s.linear_rgb_capacity() >= 32);
        let cap_after_first = s.linear_rgb_capacity();
        let _ = s.linear_rgb(8);
        // Smaller request shouldn't shrink.
        assert_eq!(s.linear_rgb_capacity(), cap_after_first);
    }
}
