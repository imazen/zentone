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
    /// HDR and SDR buffers disagree on length.
    LengthMismatch {
        /// HDR buffer length in elements.
        hdr: usize,
        /// SDR buffer length in elements.
        sdr: usize,
    },
    /// Buffer length is not a multiple of the channel count.
    NotAligned {
        /// Buffer length in elements.
        len: usize,
        /// Channel count.
        channels: u8,
    },
    /// No usable pixel pairs for fitting in luminance mode.
    NoValidSamples,
    /// No valid samples in a specific color channel during per-channel fit.
    EmptyChannel {
        /// 0 = R, 1 = G, 2 = B.
        channel: u8,
    },
    /// The streaming ring buffer is full — pull a row before pushing more.
    RingBufferFull,
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
            Error::LengthMismatch { hdr, sdr } => {
                write!(f, "length mismatch: HDR has {hdr} elements, SDR has {sdr}")
            }
            Error::NotAligned { len, channels } => {
                write!(
                    f,
                    "buffer length {len} is not a multiple of channels={channels}"
                )
            }
            Error::NoValidSamples => write!(f, "no valid pixel pairs for fitting"),
            Error::EmptyChannel { channel } => {
                let name = match channel {
                    0 => "red",
                    1 => "green",
                    2 => "blue",
                    _ => "unknown",
                };
                write!(f, "no valid samples in {name} channel (index {channel})")
            }
            Error::RingBufferFull => {
                write!(f, "streaming ring buffer full — pull a row before pushing")
            }
            Error::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl core::error::Error for Error {}
