//! Experimental tonemappers — lightly tested.
//!
//! APIs here may change without semver bumps until stabilized. Turn on with
//! the `experimental` feature.
//!
//! - [`AdaptiveTonemapper`] — fits a LUT from an HDR/SDR pair.
//! - [`StreamingTonemapper`] — single-pass spatially-local with lookahead.
//! - [`ProfileToneCurve`] — DNG camera profile tone curve.
//! - [`LumaGainMapSplitter`] — round-trippable HDR ↔ (SDR, log2 gain map),
//!   using the ISO 21496-1 / Ultra HDR decode form. Wire-compatible with
//!   `zencodec::GainMapParams`.

mod adaptive;
pub mod detect;
mod gain_map;
mod profile;
mod streaming;

pub use adaptive::{AdaptiveTonemapper, FitConfig, FitStats};
pub use gain_map::{
    Bt2408Yrgb, ExtendedReinhardLuma, LumaFn, LumaGainMapSplitter, LumaToneMap, SplitConfig,
    SplitStats, normalized_linear_to_pq_row, pq_to_normalized_linear_row,
};
pub use profile::{ProfileLuminance, ProfilePerChannel, ProfileToneCurve};
pub use streaming::{StreamingTonemapConfig, StreamingTonemapper};
