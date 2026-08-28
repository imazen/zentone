//! Experimental tonemappers — lightly tested.
//!
//! APIs here may change without semver bumps until stabilized. Turn on with
//! the `experimental` feature.
//!
//! - [`Aces2OutputTransform`] — the ACES 2.0 rendering (ACES2065-1 →
//!   display-linear); [`Aces2DisplayTransform`] adds the display encoding
//!   of the `Output.Academy.*` presets (sRGB / BT.1886 / PQ / HLG / DCDM).
//! - [`aces`] — ACEScct / ACEScc, the ASC CDL, and ACES 1.3 Reference
//!   Gamut Compression.
//! - [`AdaptiveTonemapper`] — fits a LUT from an HDR/SDR pair.
//! - [`StreamingTonemapper`] — single-pass spatially-local with lookahead.
//! - [`ProfileToneCurve`] — DNG camera profile tone curve.
//!
//! The ISO 21496-1 / Apple Ultra HDR gain-map splitter previously hosted
//! here has graduated to the stable [`crate::gainmap`] module — no
//! `experimental` feature gate required.

mod aces2;
mod aces2_display;
mod aces2_simd;
pub mod aces_encodings;
mod adaptive;
pub mod detect;
mod profile;
mod streaming;

pub use aces_encodings as aces;
pub use aces2::{Aces2Config, Aces2OutputTransform, Chromaticities};
pub use aces2_display::{
    ACES_OUTPUT_TRANSFORM_IDS, Aces2DisplayEncoding, Aces2DisplayTransform, DCDM_LINEAR_SCALE,
    DisplayEotf,
};
pub use adaptive::{AdaptiveTonemapper, FitConfig, FitStats};
pub use profile::{ProfileLuminance, ProfilePerChannel, ProfileToneCurve};
pub use streaming::{StreamingTonemapConfig, StreamingTonemapper};
