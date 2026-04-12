//! Error type.

use core::fmt;

/// Result alias for zentone.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors produced by zentone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Input buffer is too small for the declared dimensions and channel count.
    BufferTooSmall {
        /// Required length in elements.
        required: usize,
        /// Actual length in elements.
        actual: usize,
    },
    /// HDR and SDR buffers disagree on dimensions.
    DimensionMismatch {
        /// HDR width.
        hdr_w: u32,
        /// HDR height.
        hdr_h: u32,
        /// SDR width.
        sdr_w: u32,
        /// SDR height.
        sdr_h: u32,
    },
    /// No usable pixel pairs for fitting (all samples were below threshold).
    NoValidSamples,
    /// Invalid configuration parameter.
    InvalidConfig(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "buffer too small: need {required} elements, got {actual}"
                )
            }
            Error::DimensionMismatch {
                hdr_w,
                hdr_h,
                sdr_w,
                sdr_h,
            } => write!(
                f,
                "dimension mismatch: HDR {hdr_w}x{hdr_h} vs SDR {sdr_w}x{sdr_h}"
            ),
            Error::NoValidSamples => write!(f, "no valid pixel pairs for fitting"),
            Error::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl core::error::Error for Error {}
