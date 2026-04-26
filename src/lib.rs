//! HDR → SDR tone mapping curves in safe Rust.
//!
//! zentone is a library of tone-mapping **curves** — the math that compresses
//! HDR luminance into an SDR display range. It is **not** a full color
//! management pipeline: linearization, primary conversion, OETF encoding, and
//! ICC handling live in [`linear-srgb`](https://docs.rs/linear-srgb) and
//! [`zenpixels-convert`](https://docs.rs/zenpixels-convert). Use zentone
//! when you have linear-light HDR samples and need to choose a curve.
//!
//! `no_std + alloc`, zero allocation in hot paths, SIMD-accelerated on
//! x86-64 (AVX2+FMA) with scalar fallback everywhere else.
//!
//! # API tiers
//!
//! The public surface splits into three tiers; pick by workload, not by
//! name recognition.
//!
//! - **Hot path — strip / row SIMD.** Use these for any non-trivial
//!   workload (a row, a strip, a whole image). Inputs are packed
//!   `&[[f32; 3]]` / `&[[f32; 4]]` / `&[[u8; 3]]` slices; kernels dispatch
//!   through [`archmage`](https://docs.rs/archmage) at runtime.
//!   Examples: [`pipeline::tonemap_pq_row_simd`],
//!   [`pipeline::tonemap_pq_rgba_row_simd`],
//!   [`pipeline::tonemap_hlg_row_simd`],
//!   [`pipeline::tonemap_pq_to_srgb8_row_simd`],
//!   [`gamut::apply_matrix_row_simd`], [`gamut::soft_clip_row_simd`],
//!   [`hlg::hlg_ootf_row_simd`], and the
//!   [`ToneMap::map_strip_simd`]
//!   trait method (with SIMD overrides on `Bt2408Tonemapper`, `Bt2446A`,
//!   `Bt2446B`, `Bt2446C`, and [`CompiledFilmicSpline`]).
//! - **Reference / per-pixel.** [`ToneMap::map_rgb`], [`gamut::apply_matrix`],
//!   [`gamut::soft_clip`], [`hlg::hlg_ootf`], and the named-curve scalar
//!   functions in [`curves`] (`reinhard_simple`, `bt2390_tonemap`,
//!   `narkowicz_aces`, etc.). These are the parity surface — suitable for
//!   one-off use, doctests, and cross-checks against external reference
//!   implementations. Calls inside an inner loop should usually go through
//!   the row form instead.
//! - **Experimental.** Behind the `experimental` feature, semver-unstable.
//!   Covers [`experimental::LumaToneMap`](experimental/trait.LumaToneMap.html)
//!   and the gain-map splitter for ISO 21496-1 / Apple Ultra HDR,
//!   `Bt2408Yrgb`, the streaming tonemapper, the adaptive LUT fitter, and
//!   the DNG `ProfileToneCurve`. APIs may change without semver bumps until
//!   stabilized.
//!
//! As of 0.2.0, the pipeline ships SIMD strip-form APIs only — the old
//! `&[f32]` + `channels: u8` forms are not present. See `CHANGELOG.md`
//! for the removal record.
//!
//! # What's in the box
//!
//! Curves come in four families. Each implements the [`ToneMap`] trait;
//! pick by use case, not by name.
//!
//! - **Classical, stateless** — [`ToneMapCurve`]: simple Reinhard
//!   (`x/(1+x)`), extended Reinhard with white point, Reinhard-Jodie,
//!   tuned Reinhard (display-aware nits), Narkowicz, Hable, ACES AP1,
//!   AgX (Blender) with [`Default`/`Punchy`/`Golden`](AgxLook) looks,
//!   BT.2390 EETF, and `Clamp`. No state, no allocation, SIMD-accelerated
//!   row paths.
//! - **ITU broadcast standards** — [`Bt2408Tonemapper`] (BT.2408 Annex 5
//!   PQ-domain Hermite spline, YRGB or MaxRGB), [`Bt2446A`] / [`Bt2446B`]
//!   / [`Bt2446C`] (BT.2446 Methods A, B, C). Constructed once with
//!   `(content_max_nits, display_max_nits)`.
//! - **Filmic spline** — [`CompiledFilmicSpline`] / [`FilmicSplineConfig`]:
//!   darktable / Blender-style rational spline with toe / linear / shoulder
//!   regions and per-pixel highlight desaturation. Heavy parameter surface
//!   for calibrated workflows.
//! - **Experimental** (behind the `experimental` feature) — adaptive LUT
//!   fitting, single-pass streaming tonemap with local adaptation, gain-map
//!   splitter for ISO 21496-1 / Apple Ultra HDR, DNG ProfileToneCurve.
//!
//! # Quick start
//!
//! Every tonemapper takes a linear-light RGB triple and returns linear SDR
//! RGB. Wire format decoding (PQ / HLG / sRGB) and primary conversion
//! happen before / after.
//!
//! ```
//! use zentone::{Bt2446C, ToneMap};
//!
//! // 1000 cd/m² HDR content → 203 cd/m² SDR (HDR Reference White).
//! // Input scale: 1.0 = hdr_peak_nits; output scale: 1.0 = sdr_peak_nits.
//! let curve = Bt2446C::new(1000.0, 203.0);
//! let sdr = curve.map_rgb([2.0, 1.0, 0.5]);
//! assert!(sdr.iter().all(|&c| c.is_finite() && c >= 0.0));
//! ```
//!
//! For an entire row, use [`map_row`](ToneMap::map_row) (in place) or
//! [`map_into`](ToneMap::map_into) (separate dst). Both dispatch on
//! `channels` (3 = RGB, 4 = RGBA, alpha preserved):
//!
//! ```
//! use zentone::{ToneMap, ToneMapCurve};
//! let mut row = [0.5_f32, 1.2, 0.3, 0.8, 2.0, 0.1];
//! ToneMapCurve::Narkowicz.map_row(&mut row, 3);
//! ```
//!
//! For a fused PQ→tone-map→sRGB-gamut pipeline on a packed strip
//! (the canonical hot-path shape):
//!
//! ```
//! use zentone::{Bt2446C, TonemapScratch, pipeline::tonemap_pq_row_simd};
//!
//! // 1024 PQ-encoded BT.2020 RGB pixels — 0.58 ≈ HDR Reference White.
//! let pq = vec![[0.58_f32, 0.58, 0.58]; 1024];
//! let mut out = vec![[0.0_f32; 3]; 1024];
//! let curve = Bt2446C::new(1000.0, 203.0);
//! let mut scratch = TonemapScratch::new();
//! tonemap_pq_row_simd(&mut scratch, &pq, &mut out, &curve);
//! ```
//!
//! # Choosing a curve
//!
//! | Need | Pick |
//! |---|---|
//! | "Just give me something cheap and decent" | [`ToneMapCurve::Narkowicz`] or [`HableFilmic`](ToneMapCurve::HableFilmic) |
//! | Game engine / shader port | [`ToneMapCurve::AcesAp1`] or [`Agx`](ToneMapCurve::Agx) |
//! | Broadcast-grade HDR10 / HLG → SDR with display peak nits | [`Bt2408Tonemapper`] (PQ-domain) or [`Bt2446A`] |
//! | Live HLG → SDR, conservative on clipped highlights | [`Bt2446B`] |
//! | HDR → SDR with **mathematical inverse** (round-trip / detection) | [`Bt2446C`] |
//! | Calibrated photo workflow with toe / shoulder control | [`CompiledFilmicSpline`] |
//! | ISO 21496-1 / Ultra HDR gain map encoder | `experimental::LumaGainMapSplitter` |
//! | Re-derive a curve from an HDR/SDR reference pair | `experimental::AdaptiveTonemapper` |
//!
//! Curves that need RGB→Y weights take them at construction time via
//! [`LUMA_BT709`], [`LUMA_BT2020`], or [`LUMA_P3`]. Pick the one that
//! matches the input primaries — passing BT.709 weights for BT.2020 input
//! over-desaturates greens.
//!
//! # Auxiliary trait
//!
//! [`experimental::LumaToneMap`](experimental/trait.LumaToneMap.html) is the
//! scalar Y → Y' interface used by the gain-map splitter. Distinct from
//! [`ToneMap`] because per-channel and matrix-based curves don't have a
//! coherent "luma-only" interpretation.
//!
//! # Utility modules
//!
//! - [`gamut`] — 6 gamut conversion matrices (BT.709 ↔ BT.2020 ↔ Display P3)
//!   plus a hue-preserving [`soft_clip`](gamut::soft_clip).
//! - [`hlg`] — HLG system gamma, OOTF, inverse OOTF, full HLG → display.
//! - [`sdr_hdr`] — reference-white scaling (100 ↔ 203 nits), OOTF gamma
//!   adjustments per BT.2408 §5.1.
//! - [`pipeline`] — fused linearization + tone map + gamut conversion +
//!   soft clip via SIMD strip-form entry points
//!   [`tonemap_pq_row_simd`](pipeline::tonemap_pq_row_simd) /
//!   [`tonemap_pq_to_srgb8_row_simd`](pipeline::tonemap_pq_to_srgb8_row_simd) /
//!   [`tonemap_hlg_row_simd`](pipeline::tonemap_hlg_row_simd) (plus RGBA
//!   variants).
//!
//! # Experimental (`experimental` feature)
//!
//! Behind a feature flag because the APIs are still in flux:
//!
//! - `experimental::AdaptiveTonemapper` — fits a LUT from an HDR/SDR pair.
//! - `experimental::StreamingTonemapper` — single-pass spatially-local
//!   tonemap with bounded-memory pull API.
//! - `experimental::LumaGainMapSplitter` — round-trippable HDR ↔ (SDR, log2 gain)
//!   for ISO 21496-1 / Apple Ultra HDR gain-map encoders.
//! - `experimental::ProfileToneCurve` — DNG camera-profile tone curve;
//!   per-channel or luminance-preserving views via [`ToneMap`].
//! - `experimental::detect::detect_standard` — identifies which standard
//!   curve was applied to a fitted LUT.
//!
//! Lightly tested; API may change without semver bumps until stabilized.
//! See the `experimental` module docs when the feature is enabled.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::excessive_precision)]

extern crate alloc;

mod bt2408;
mod bt2446a;
mod bt2446b;
mod bt2446c;
pub mod curves;
mod error;
mod filmic_spline;
pub mod gamut;
pub mod hlg;
mod math;
pub mod pipeline;
mod scratch;
pub mod sdr_hdr;
mod simd;
mod tone_map;

#[cfg(feature = "experimental")]
pub mod experimental;

pub use bt2408::{Bt2408Tonemapper, EetfSpace};
pub use bt2446a::Bt2446A;
pub use bt2446b::Bt2446B;
pub use bt2446c::Bt2446C;
pub use curves::{AgxLook, ToneMapCurve};
pub use error::{Error, Result};
pub use filmic_spline::{CompiledFilmicSpline, FilmicSplineConfig};
pub use scratch::TonemapScratch;
pub use tone_map::ToneMap;

/// BT.709 / sRGB luminance coefficients `[0.2126, 0.7152, 0.0722]`.
pub const LUMA_BT709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// BT.2020 luminance coefficients `[0.2627, 0.6780, 0.0593]`.
pub const LUMA_BT2020: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// Display-P3 / DCI-P3 luminance coefficients `[0.2289746, 0.6917385, 0.0792869]`.
///
/// Use when input primaries are P3 (e.g. an Apple Ultra HDR base image
/// where the SDR rendition is tagged Display-P3). Matches the weights
/// exposed in `ultrahdr-core` (`color/gamut.rs`).
pub const LUMA_P3: [f32; 3] = [0.2289746, 0.6917385, 0.0792869];
