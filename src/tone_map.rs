//! Core [`ToneMap`] trait.
//!
//! Every tonemapper in zentone implements this trait. The trait is dyn-safe;
//! provided methods ([`map_row`](ToneMap::map_row), [`map_into`](ToneMap::map_into))
//! dispatch on the runtime `channels` value to const-generic inner loops, so
//! callers get compile-time unrolling of the alpha branch and fixed-stride
//! arithmetic without having to parameterize their types.

/// A linear-light RGB tonemapper.
///
/// Implementors provide [`map_rgb`](Self::map_rgb). Row-level methods are
/// provided with defaults that const-unroll the per-pixel loop based on
/// `channels` (3 for RGB, 4 for RGBA — alpha is passed through unchanged).
pub trait ToneMap {
    /// Map a single linear-light RGB triple to linear SDR output.
    ///
    /// Most curves produce output in `[0, 1]`; some (unclamped variants) may
    /// exceed this range. Callers that need a strict display-gamut guarantee
    /// should clamp themselves or apply an OETF that handles over-range input.
    fn map_rgb(&self, rgb: [f32; 3]) -> [f32; 3];

    /// Tonemap a row of interleaved linear f32 pixels in place.
    ///
    /// `channels` must be 3 (RGB) or 4 (RGBA). Row length must be a multiple
    /// of `channels`. Alpha (index 3 for RGBA) is preserved.
    #[inline]
    fn map_row(&self, row: &mut [f32], channels: u8) {
        match channels {
            3 => map_row_cn::<3, Self>(self, row),
            4 => map_row_cn::<4, Self>(self, row),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }

    /// Tonemap `src` into `dst`. Both slices must be the same length and a
    /// multiple of `channels`. Alpha is copied from `src` unchanged.
    #[inline]
    fn map_into(&self, src: &[f32], dst: &mut [f32], channels: u8) {
        match channels {
            3 => map_into_cn::<3, Self>(self, src, dst),
            4 => map_into_cn::<4, Self>(self, src, dst),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }
}

/// Const-generic in-place row tonemapper. Used internally by
/// [`ToneMap::map_row`]; exposed here so callers with a concrete CN can skip
/// the runtime match.
#[inline]
pub fn map_row_cn<const CN: usize, T: ToneMap + ?Sized>(tm: &T, row: &mut [f32]) {
    debug_assert!(CN == 3 || CN == 4);
    for chunk in row.chunks_exact_mut(CN) {
        let mapped = tm.map_rgb([chunk[0], chunk[1], chunk[2]]);
        chunk[0] = mapped[0];
        chunk[1] = mapped[1];
        chunk[2] = mapped[2];
        // Alpha (chunk[3] when CN == 4) is already in place.
    }
}

/// Const-generic copying row tonemapper. See [`ToneMap::map_into`].
#[inline]
pub fn map_into_cn<const CN: usize, T: ToneMap + ?Sized>(tm: &T, src: &[f32], dst: &mut [f32]) {
    debug_assert!(CN == 3 || CN == 4);
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(CN).zip(dst.chunks_exact_mut(CN)) {
        let mapped = tm.map_rgb([s[0], s[1], s[2]]);
        d[0] = mapped[0];
        d[1] = mapped[1];
        d[2] = mapped[2];
        if CN == 4 {
            d[3] = s[3];
        }
    }
}
