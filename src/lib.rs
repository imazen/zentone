//! HDR → SDR tone mapping: classical curves, BT.2408, filmic spline,
//! plus experimental adaptive and streaming tonemappers.
//!
//! # Core trait
//!
//! Every tonemapper implements [`ToneMap`]. Once constructed, apply it with
//! [`map_rgb`](ToneMap::map_rgb) per pixel, [`map_row`](ToneMap::map_row) for
//! in-place row work, or [`map_into`](ToneMap::map_into) to copy. The row
//! methods dispatch to const-generic inner loops on `channels` (3 or 4), so
//! the alpha branch and stride arithmetic are compile-time folded.
//!
//! ```
//! use zentone::{ToneMap, ToneMapCurve};
//! let mut row = [0.5_f32, 1.2, 0.3, 0.8, 2.0, 0.1];
//! ToneMapCurve::Narkowicz.map_row(&mut row, 3);
//! ```
//!
//! # Implementations
//!
//! - [`ToneMapCurve`] — classical curve dispatch (Reinhard variants, Hable,
//!   Narkowicz, ACES AP1, AgX, BT.2390, clamp).
//! - [`Bt2408Tonemapper`] — ITU-R BT.2408 PQ-domain Hermite spline.
//! - [`CompiledFilmicSpline`] — darktable/Blender-style filmic spline.
//!
//! Variants / types that need RGB→luminance weights take them at construction
//! time (via [`LUMA_BT709`] or [`LUMA_BT2020`]) so the per-pixel call site
//! doesn't have to thread them through.
//!
//! # Experimental (`experimental` feature)
//!
//! - [`experimental::AdaptiveTonemapper`] — fits a LUT from an HDR/SDR pair.
//! - [`experimental::StreamingTonemapper`] — spatially-local, single-pass,
//!   bounded-memory, pull API.
//! - [`experimental::ProfileToneCurve`] — DNG camera profile tone curve;
//!   expose per-channel or luminance-preserving views via the [`ToneMap`]
//!   trait.
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
mod curves;
mod error;
mod filmic_spline;
pub mod hlg;
mod math;
mod tone_map;

#[cfg(feature = "experimental")]
pub mod experimental;

pub use bt2408::{Bt2408Tonemapper, EetfSpace};
pub use bt2446a::Bt2446A;
pub use curves::{
    AgxLook, ToneMapCurve, aces_ap1, agx_tonemap, bt2390_tonemap, bt2390_tonemap_ext,
    filmic_narkowicz, reinhard_extended, reinhard_jodie, reinhard_simple, uncharted2_filmic,
};
pub use error::{Error, Result};
pub use filmic_spline::{CompiledFilmicSpline, FilmicSplineConfig};
pub use tone_map::ToneMap;

/// BT.709 / sRGB luminance coefficients `[0.2126, 0.7152, 0.0722]`.
pub const LUMA_BT709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// BT.2020 luminance coefficients `[0.2627, 0.6780, 0.0593]`.
pub const LUMA_BT2020: [f32; 3] = [0.2627, 0.6780, 0.0593];
