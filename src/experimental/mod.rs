//! Experimental tonemappers — lightly tested.
//!
//! APIs here may change without semver bumps until stabilized. Turn on with
//! the `experimental` feature.
//!
//! - [`AdaptiveTonemapper`] — fits a LUT from an HDR/SDR pair.
//! - [`StreamingTonemapper`] — single-pass spatially-local with lookahead.
//! - [`ProfileToneCurve`] — DNG camera profile tone curve.

mod adaptive;
mod profile;
mod streaming;

pub use adaptive::{
    AdaptiveTonemapper, FitConfig, FitMode, FitStats, LuminanceCurve, PerChannelLut, TonemapMode,
};
pub use profile::ProfileToneCurve;
pub use streaming::{LocalParams, StreamingTonemapConfig, StreamingTonemapper, TonemapOutput};
