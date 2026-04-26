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
///
/// # Contract
///
/// - **Input**: linear-light RGB. Decode any wire-format transfer (PQ, HLG,
///   sRGB) before calling. The valid range depends on the curve — most
///   accept any non-negative finite value; some (e.g.
///   [`Bt2390`](crate::ToneMapCurve::Bt2390)) require pre-normalized input.
///   See per-curve docs.
/// - **Input scale**: the curve decides what `1.0` means. ITU curves
///   ([`Bt2408Tonemapper`](crate::Bt2408Tonemapper),
///   [`Bt2446A`](crate::Bt2446A) /
///   [`B`](crate::Bt2446B) /
///   [`C`](crate::Bt2446C)) use `1.0 = hdr_peak_nits`; classical curves
///   ([`ToneMapCurve`](crate::ToneMapCurve), [`AgX`](crate::AgxLook),
///   filmic) treat input as scene-linear with `1.0 = SDR diffuse white`.
/// - **Output**: linear SDR RGB. Most curves produce values in `[0, 1]`;
///   some unclamped variants may slightly exceed (e.g.
///   [`Bt2446C`](crate::Bt2446C) targets `[0, 1.09]` for super-whites).
/// - **Monotonic** on luminance: doubling the input never decreases the
///   output luminance. Per-channel curves are also per-channel monotonic;
///   matrix-based curves ([`AcesAp1`](crate::ToneMapCurve::AcesAp1),
///   [`AgX`](crate::AgxLook)) are monotonic on neutral input but may
///   redistribute saturation.
/// - **Black at black**: `T(0,0,0) = (0,0,0)` (filmic spline lifts black
///   by `black_point_target`, configurable to 0).
pub trait ToneMap {
    /// Map a single linear-light RGB triple to linear SDR output.
    ///
    /// Most curves produce output in `[0, 1]`; some (unclamped variants) may
    /// exceed this range. Callers that need a strict display-gamut guarantee
    /// should clamp themselves or apply an OETF that handles over-range input.
    #[must_use]
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

    /// Apply this tone map to a strip of pixels in place.
    ///
    /// The default implementation calls [`map_rgb`](Self::map_rgb) per pixel.
    /// Concrete tonemap types can override this with a SIMD kernel for speed —
    /// the fused pipeline kernels in [`crate::pipeline`] (the `_row_simd`
    /// variants) call this method instead of `map_rgb` so the dispatch
    /// indirection is amortized across a whole strip rather than paid per
    /// pixel.
    ///
    /// Overriding is purely a performance opt-in: callers see the same
    /// results either way. The provided default keeps every existing
    /// `impl ToneMap` working unchanged, and the method stays object-safe
    /// for `&dyn ToneMap` use.
    #[inline]
    fn map_strip_simd(&self, strip: &mut [[f32; 3]]) {
        for px in strip.iter_mut() {
            *px = self.map_rgb(*px);
        }
    }
}

/// Const-generic in-place row tonemapper. Called by [`ToneMap::map_row`].
#[inline]
pub(crate) fn map_row_cn<const CN: usize, T: ToneMap + ?Sized>(tm: &T, row: &mut [f32]) {
    debug_assert!(CN == 3 || CN == 4);
    for chunk in row.chunks_exact_mut(CN) {
        let mapped = tm.map_rgb([chunk[0], chunk[1], chunk[2]]);
        chunk[0] = mapped[0];
        chunk[1] = mapped[1];
        chunk[2] = mapped[2];
        // Alpha (chunk[3] when CN == 4) is already in place.
    }
}

/// Const-generic copying row tonemapper. Called by [`ToneMap::map_into`].
#[inline]
pub(crate) fn map_into_cn<const CN: usize, T: ToneMap + ?Sized>(
    tm: &T,
    src: &[f32],
    dst: &mut [f32],
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgxLook, LUMA_BT709, ToneMapCurve};

    /// Build an RGB test row: interleaved triples in `[0, 4]` HDR range.
    fn synth_row_rgb(pixels: usize) -> alloc::vec::Vec<f32> {
        let mut row = alloc::vec::Vec::with_capacity(pixels * 3);
        for i in 0..pixels {
            let t = i as f32 / pixels as f32;
            row.push(t * 4.0);
            row.push((1.0 - t) * 3.5);
            row.push(t * t * 2.0);
        }
        row
    }

    /// Build an RGBA test row with explicit alpha. Pixels match `synth_row_rgb`
    /// so we can cross-check against the 3-channel result.
    fn synth_row_rgba(pixels: usize) -> alloc::vec::Vec<f32> {
        let mut row = alloc::vec::Vec::with_capacity(pixels * 4);
        for i in 0..pixels {
            let t = i as f32 / pixels as f32;
            row.push(t * 4.0);
            row.push((1.0 - t) * 3.5);
            row.push(t * t * 2.0);
            row.push(0.25 + t * 0.5);
        }
        row
    }

    /// A fixed, arbitrary selection of curves that exercise all the internal
    /// dispatch branches.
    fn sample_curves() -> [ToneMapCurve; 7] {
        [
            ToneMapCurve::Reinhard,
            ToneMapCurve::ExtendedReinhard {
                l_max: 4.0,
                luma: LUMA_BT709,
            },
            ToneMapCurve::ReinhardJodie { luma: LUMA_BT709 },
            ToneMapCurve::Narkowicz,
            ToneMapCurve::HableFilmic,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Agx(AgxLook::Default),
        ]
    }

    /// `map_row` must be equivalent to iterating `map_rgb` manually.
    #[test]
    fn map_row_rgb_matches_manual_loop() {
        for curve in sample_curves() {
            let src = synth_row_rgb(17); // prime count, not power of two
            let mut via_row = src.clone();
            curve.map_row(&mut via_row, 3);

            let mut via_manual = src.clone();
            for chunk in via_manual.chunks_exact_mut(3) {
                let out = curve.map_rgb([chunk[0], chunk[1], chunk[2]]);
                chunk[0] = out[0];
                chunk[1] = out[1];
                chunk[2] = out[2];
            }

            for (i, (a, b)) in via_row.iter().zip(via_manual.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "curve {curve:?} diverged RGB at [{i}]: row={a}, manual={b}"
                );
            }
        }
    }

    /// `map_row` on RGBA must match a manual per-pixel loop and must preserve alpha.
    #[test]
    fn map_row_rgba_matches_manual_loop_and_keeps_alpha() {
        for curve in sample_curves() {
            let src = synth_row_rgba(17);
            let mut via_row = src.clone();
            curve.map_row(&mut via_row, 4);

            let mut via_manual = src.clone();
            for chunk in via_manual.chunks_exact_mut(4) {
                let out = curve.map_rgb([chunk[0], chunk[1], chunk[2]]);
                chunk[0] = out[0];
                chunk[1] = out[1];
                chunk[2] = out[2];
                // alpha untouched
            }

            for (i, (a, b)) in via_row.iter().zip(via_manual.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "curve {curve:?} diverged RGBA at [{i}]: row={a}, manual={b}"
                );
            }
            // Sanity: alpha values are the originals
            for (i, pixel) in via_row.chunks_exact(4).enumerate() {
                let expected_alpha = 0.25 + (i as f32 / 17.0) * 0.5;
                assert!(
                    (pixel[3] - expected_alpha).abs() < 1e-6,
                    "curve {curve:?} pixel {i} alpha drift: {}",
                    pixel[3]
                );
            }
        }
    }

    /// `map_into` on RGB must match `copy_from_slice` followed by `map_row`.
    #[test]
    fn map_into_rgb_matches_copy_then_map_row() {
        for curve in sample_curves() {
            let src = synth_row_rgb(17);
            let mut via_copy = src.clone();
            curve.map_row(&mut via_copy, 3);

            let mut via_into = alloc::vec![0.0_f32; src.len()];
            curve.map_into(&src, &mut via_into, 3);

            // map_into uses default trait impl (per-pixel); map_row may use SIMD.
            // FMA can cause 1-ULP differences. Tolerate.
            for (i, (a, b)) in via_copy.iter().zip(via_into.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "curve {curve:?} map_into != map_row at [{i}]: {a} vs {b}"
                );
            }
        }
    }

    /// `map_into` on RGBA must copy alpha from src, not from dst.
    #[test]
    fn map_into_rgba_writes_src_alpha_to_dst() {
        let curve = ToneMapCurve::Reinhard;
        let src = synth_row_rgba(4);
        // Pre-populate dst with a distinctly wrong alpha so we can tell
        // whether map_into overwrites it.
        let mut dst = alloc::vec![0.99_f32; src.len()];
        curve.map_into(&src, &mut dst, 4);
        for (i, (s, d)) in src.chunks_exact(4).zip(dst.chunks_exact(4)).enumerate() {
            assert!(
                (d[3] - s[3]).abs() < 1e-6,
                "pixel {i}: dst alpha {} != src alpha {}",
                d[3],
                s[3]
            );
        }
    }

    /// `map_row_cn::<3>` called directly must agree with the trait dispatch.
    #[test]
    fn map_row_cn_direct_matches_trait_dispatch() {
        let curve = ToneMapCurve::AcesAp1;
        let src = synth_row_rgb(32);

        let mut via_trait = src.clone();
        curve.map_row(&mut via_trait, 3);

        let mut via_direct = src.clone();
        map_row_cn::<3, _>(&curve, &mut via_direct);

        assert_eq!(via_trait, via_direct);
    }

    /// Trait object (dyn ToneMap) round-trip: the same curve as `dyn ToneMap`
    /// must produce the same pixels as the concrete type.
    #[test]
    fn dyn_tone_map_matches_concrete() {
        let curve = ToneMapCurve::HableFilmic;
        let src = synth_row_rgb(8);

        let mut via_concrete = src.clone();
        curve.map_row(&mut via_concrete, 3);

        let obj: &dyn ToneMap = &curve;
        let mut via_dyn = src.clone();
        obj.map_row(&mut via_dyn, 3);

        assert_eq!(via_concrete, via_dyn);
    }

    /// Passing a non-3-non-4 channel count should panic consistently.
    #[test]
    #[should_panic(expected = "channels must be 3 or 4")]
    fn map_row_panics_on_bad_channels() {
        let mut row = [0.1_f32; 12];
        ToneMapCurve::Reinhard.map_row(&mut row, 2);
    }

    #[test]
    #[should_panic(expected = "channels must be 3 or 4")]
    fn map_into_panics_on_bad_channels() {
        let src = [0.1_f32; 12];
        let mut dst = [0.0_f32; 12];
        ToneMapCurve::Reinhard.map_into(&src, &mut dst, 5);
    }
}
