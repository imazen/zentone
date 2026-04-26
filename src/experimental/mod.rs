//! Experimental tonemappers — lightly tested.
//!
//! APIs here may change without semver bumps until stabilized. Turn on with
//! the `experimental` feature.
//!
//! - [`AdaptiveTonemapper`] — fits a LUT from an HDR/SDR pair.
//! - [`StreamingTonemapper`] — single-pass spatially-local with lookahead.
//! - [`ProfileToneCurve`] — DNG camera profile tone curve.
//!
//! The ISO 21496-1 / Apple Ultra HDR gain-map splitter previously hosted
//! here has graduated to the stable [`crate::gainmap`] module — no
//! `experimental` feature gate required.

mod adaptive;
pub mod detect;
mod profile;
mod streaming;

pub use adaptive::{AdaptiveTonemapper, FitConfig, FitStats};
pub use profile::{ProfileLuminance, ProfilePerChannel, ProfileToneCurve};
pub use streaming::{StreamingTonemapConfig, StreamingTonemapper};
