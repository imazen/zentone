//! HDR → SDR tone mapping in safe Rust.
//!
//! Classical curves (Reinhard, Narkowicz, Hable, ACES, AgX), ITU-R BT.2408/BT.2446
//! standards, darktable filmic spline, plus experimental adaptive and streaming
//! tonemappers. `no_std + alloc`, zero allocation in hot paths, SIMD-accelerated
//! on x86-64 (AVX2+FMA) with scalar fallback everywhere else.
//!
//! # Core trait
//!
//! Every tonemapper implements [`ToneMap`]. Apply it per pixel with
//! [`map_rgb`](ToneMap::map_rgb), in-place on a row with
//! [`map_row`](ToneMap::map_row), or with separate src/dst via
//! [`map_into`](ToneMap::map_into). Row methods dispatch to const-generic
//! inner loops on `channels` (3 or 4); alpha is preserved, stride arithmetic
//! is compile-time folded.
//!
//! ```
//! use zentone::{ToneMap, ToneMapCurve};
//! let mut row = [0.5_f32, 1.2, 0.3, 0.8, 2.0, 0.1];
//! ToneMapCurve::Narkowicz.map_row(&mut row, 3);
//! ```
//!
//! # Implementations
//!
//! - [`ToneMapCurve`] — stateless enum: Reinhard (simple / extended / Jodie /
//!   tuned), Narkowicz, Hable, ACES AP1, AgX (Default / Punchy / Golden),
//!   BT.2390, Clamp. SIMD-accelerated where applicable.
//! - [`Bt2408Tonemapper`] — ITU-R BT.2408 Annex 5 PQ-domain Hermite spline
//!   EETF with YRGB or MaxRGB application space.
//! - [`Bt2446A`] / [`Bt2446B`] / [`Bt2446C`] — ITU-R BT.2446 Methods A, B,
//!   and C (psychophysically-verified, broadcast, and parametric-with-inverse).
//! - [`CompiledFilmicSpline`] — darktable/Blender-style filmic with
//!   configurable latitude, contrast, balance, saturation, and output power.
//!
//! Variants that need luminance weights take them at construction time via
//! [`LUMA_BT709`] or [`LUMA_BT2020`].
//!
//! # Utility modules
//!
//! - [`gamut`] — 6 gamut conversion matrices (BT.709, BT.2020, Display P3).
//! - [`hlg`] — HLG system gamma, OOTF, inverse OOTF, `hlg_to_display`.
//! - [`sdr_hdr`] — reference-white scaling (100↔203 nits), OOTF gamma.
//! - [`pipeline`] — one-call PQ→sRGB and HLG→sRGB with pluggable `&dyn ToneMap`.
//!
//! # Experimental (`experimental` feature)
//!
//! - [`experimental::AdaptiveTonemapper`] — fits a LUT from an HDR/SDR pair.
//! - [`experimental::StreamingTonemapper`] — spatially-local, single-pass,
//!   bounded-memory, pull API.
//! - [`experimental::ProfileToneCurve`] — DNG camera-profile tone curve;
//!   per-channel or luminance-preserving views via [`ToneMap`].
//! - [`experimental::detect::detect_standard`] — identifies which standard
//!   curve was applied to a fitted LUT.
//!
//! Lightly tested; API may change without semver bumps until stabilized.

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
pub use tone_map::ToneMap;

/// BT.709 / sRGB luminance coefficients `[0.2126, 0.7152, 0.0722]`.
pub const LUMA_BT709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// BT.2020 luminance coefficients `[0.2627, 0.6780, 0.0593]`.
pub const LUMA_BT2020: [f32; 3] = [0.2627, 0.6780, 0.0593];
