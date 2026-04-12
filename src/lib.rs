//! HDR → SDR tone mapping: classical curves, BT.2408, filmic spline,
//! plus experimental adaptive and streaming tonemappers.
//!
//! # Stable API
//!
//! The default feature set exposes well-tested classical tone mapping:
//!
//! - [`ToneMapCurve`] — enum dispatch over Reinhard variants, Uncharted 2
//!   (Hable), Narkowicz filmic, ACES AP1, AgX (with Default/Punchy/Golden
//!   looks), BT.2390 EETF, and Clamp.
//! - [`Bt2408Tonemapper`] — ITU-R BT.2408 PQ-domain Hermite spline with
//!   content and display peak nits.
//! - [`CompiledFilmicSpline`] — darktable/Blender Filmic-style parametric
//!   spline (latitude / balance / saturation).
//! - [`tonemap_row`] / [`tonemap_rgb_curve`] — row and per-pixel dispatch.
//!
//! # Experimental API (`experimental` feature)
//!
//! Behind the `experimental` feature flag, zentone also provides primitives
//! with lighter test coverage:
//!
//! - [`experimental::AdaptiveTonemapper`] — learns a LUT from an HDR/SDR
//!   pair, used to preserve artistic intent when re-encoding edited HDR.
//! - [`experimental::StreamingTonemapper`] — single-pass spatially-local
//!   tonemapper with a lookahead row buffer.
//! - [`experimental::ProfileToneCurve`] — DNG camera profile tone curve.
//!
//! These APIs may change without semver bumps until stabilized.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::excessive_precision)]

extern crate alloc;

mod bt2408;
mod curves;
mod error;
mod filmic_spline;
mod math;

#[cfg(feature = "experimental")]
pub mod experimental;

pub use bt2408::Bt2408Tonemapper;
pub use curves::{
    AgxLook, ToneMapCurve, aces_ap1, agx_tonemap, bt2390_tonemap, bt2390_tonemap_ext,
    clamp_tonemap, filmic_narkowicz, reinhard_extended, reinhard_jodie, reinhard_simple,
    tonemap_rgb_curve, tonemap_row, tuned_reinhard, uncharted2_filmic,
};
pub use error::{Error, Result};
pub use filmic_spline::{CompiledFilmicSpline, FilmicSplineConfig};

/// BT.709 luminance coefficients `[0.2126, 0.7152, 0.0722]`.
pub const LUMA_BT709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// BT.2020 luminance coefficients `[0.2627, 0.6780, 0.0593]`.
pub const LUMA_BT2020: [f32; 3] = [0.2627, 0.6780, 0.0593];
