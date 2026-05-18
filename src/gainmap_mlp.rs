//! Gain-MLP per-pixel HDR reconstruction (Canham et al. 2025).
//!
//! Implements the consumer side of the "Gamma-MLP" / "Gain-MLP" gain
//! map encoding described in Canham, Pieper, Sankaranarayanan, Roghair,
//! Mueller (2025), "Encoding High-Dynamic-Range Gain Maps with Small
//! MLPs". The encoder produces a ~10 KB MLP whose output is the
//! per-pixel log₂ gain residual that converts an SDR base image to its
//! HDR original. The MLP takes
//!
//!   `[x, y, R, G, B]`
//!
//! as input (spatial coordinates in `[0, 1]`, SDR colour in `[0, 1]`)
//! and emits a 3-channel `log₂(gain)` per pixel. The HDR
//! reconstruction is
//!
//! ```text
//! H_c = (S_c + ε) · 2^MLP(x, y, R, G, B)_c − ε     (c ∈ {R, G, B})
//! ```
//!
//! ε is the ISO 21496-1 alternate offset (typical: `1/64 ≈ 0.0156`).
//! See [`GainMapMlpConfig`] for the encoded knob.
//!
//! ## Wire format / bake layout
//!
//! The bake is a standard ZNPR v3 (zenpredict) MLP carrying:
//!
//! - **`scaler_mean` / `scaler_scale`** sized to the expanded input
//!   width (post-sinusoidal embedding), typically 120 floats.
//! - **First-layer `in_dim = 120`** (matching the embedding output).
//! - **Final-layer `out_dim = 3`** (per-channel log₂ gain).
//! - **`zentrain.feature_transforms`** metadata:
//!   `"sinusoidal\nsinusoidal\nsinusoidal\nsinusoidal\nsinusoidal"`
//!   (one per raw input feature x, y, R, G, B).
//! - **`zentrain.feature_transform_params`** metadata: comma-separated
//!   frequencies per feature, e.g. `"1,2,4,8,16,32,64,128,256,512,1024,2048"`
//!   on each line. Each line declares 12 frequencies ⇒ 2·12 = 24
//!   outputs per input ⇒ 5 · 24 = 120 embedded inputs.
//!
//! The Predictor handles the embedding + scaler + forward; this
//! module's job is supplying the `[x, y, R, G, B]` input vector
//! per pixel and applying the reconstruction formula to the MLP
//! output.
//!
//! ## Coordinate convention
//!
//! `x` and `y` are **normalised to `[0, 1]`** (top-left = `(0, 0)`,
//! bottom-right = `(1, 1)`). The trainer must use the same
//! convention; the frequency schedule encoded in
//! `feature_transform_params` is calibrated against this unit-square
//! coordinate space. Pixel-coordinate models would need an entirely
//! different frequency schedule and are not supported by this
//! consumer.
//!
//! ## Bit-depth + colour-space contract
//!
//! Inputs are interleaved `f32` rows with channels = 3 (RGB) in
//! linear or sRGB-encoded `[0, 1]` SDR. The MLP is trained against
//! whatever colour space the encoder produced, so the consumer is
//! colour-space-agnostic — but consumer + encoder MUST agree. Outputs
//! are interleaved `f32` HDR in linear light, no normalisation
//! enforced beyond `H_c ≥ 0` after the ε subtract clamps.

#![cfg(feature = "gainmap-mlp")]

use zenpredict::{Model, Predictor};

use crate::error::{Error, Result};

/// Default alternate offset (`1/64 ≈ 0.015625`), matching the
/// ISO 21496-1 reference implementation and Adobe / Apple Ultra HDR.
pub const DEFAULT_EPSILON: f32 = 1.0 / 64.0;

/// Encoder-side configuration knob carried alongside the bake.
///
/// Today only `epsilon` is decoder-visible; future fields (per-channel
/// ε, log-window override, content-light hints) land here as the spec
/// stabilises.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct GainMapMlpConfig {
    /// Alternate offset in the ISO 21496-1 formula `H = (S + ε) · 2^g − ε`.
    /// Default: [`DEFAULT_EPSILON`].
    pub epsilon: f32,
}

impl Default for GainMapMlpConfig {
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
        }
    }
}

/// Per-pixel HDR reconstruction driver. Holds a [`Predictor`] borrowing
/// the supplied [`Model`] and applies the
/// `H = (S + ε) · 2^MLP − ε` formula to each pixel of an interleaved
/// RGB row.
///
/// One instance per thread; the underlying `Predictor` reuses
/// per-call scratch buffers across pixels.
pub struct GainMapMlpDecoder<'a> {
    predictor: Predictor<'a>,
    cfg: GainMapMlpConfig,
    inv_width: f32,
    inv_height: f32,
    /// Scratch buffer for the 5-D raw input vector `[x, y, R, G, B]`.
    /// One per-pixel allocation amortised across rows.
    raw_input: [f32; 5],
}

impl<'a> GainMapMlpDecoder<'a> {
    /// Construct a decoder for an image of `width × height` pixels.
    ///
    /// The image dimensions are baked into the coord normaliser so
    /// per-row apply only needs the row's `y` index and pixel x indices
    /// — they're normalised internally.
    ///
    /// `width` and `height` must be ≥ 1; the constructor panics
    /// otherwise (decoder construction is a build-time invariant
    /// rather than a per-call check).
    pub fn new(model: &'a Model, width: u32, height: u32, cfg: GainMapMlpConfig) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::gainmap_mlp("width and height must be >= 1"));
        }
        validate_bake_shape(model)?;
        // Normalise coordinates to **[0, 1] inclusive** to match the
        // trainer convention `idx / (n - 1)`. A 1-pixel axis collapses
        // to the constant 0 (no division). This must match the trainer
        // exactly — the sinusoidal frequency schedule is calibrated
        // against this coordinate space and high-frequency terms are
        // wildly sensitive to a single-pixel offset.
        let inv_width = if width > 1 {
            1.0 / ((width - 1) as f32)
        } else {
            0.0
        };
        let inv_height = if height > 1 {
            1.0 / ((height - 1) as f32)
        } else {
            0.0
        };
        Ok(Self {
            predictor: Predictor::new(model),
            cfg,
            inv_width,
            inv_height,
            raw_input: [0.0; 5],
        })
    }

    /// Reconstruct one HDR row from one SDR row. Both rows are
    /// `f32` interleaved RGB, `width` pixels long ⇒ `3 * width` floats.
    ///
    /// `y_pixel` is the row's pixel index in `0..height`. The decoder
    /// normalises to `y_pixel / (height - 1)` so the top row sees
    /// `y_norm = 0.0` and the bottom row sees `y_norm = 1.0`. For a
    /// 1-row image the normaliser degenerates to `0.0` — fine because
    /// the MLP only sees one y value during training.
    pub fn apply_row(&mut self, y_pixel: u32, sdr_row: &[f32], hdr_row: &mut [f32]) -> Result<()> {
        if sdr_row.len() != hdr_row.len() {
            return Err(Error::gainmap_mlp(
                "sdr_row and hdr_row must be the same length",
            ));
        }
        if !sdr_row.len().is_multiple_of(3) {
            return Err(Error::gainmap_mlp(
                "row length must be a multiple of 3 (interleaved RGB)",
            ));
        }
        let width = sdr_row.len() / 3;
        let y_norm = (y_pixel as f32) * self.inv_height;
        let eps = self.cfg.epsilon;
        for px in 0..width {
            let x_norm = (px as f32) * self.inv_width;
            let r = sdr_row[px * 3];
            let g = sdr_row[px * 3 + 1];
            let b = sdr_row[px * 3 + 2];
            self.raw_input = [x_norm, y_norm, r, g, b];
            let mlp_out = self
                .predictor
                .predict_transformed(&self.raw_input)
                .map_err(Error::gainmap_mlp_predict)?;
            if mlp_out.len() != 3 {
                return Err(Error::gainmap_mlp(
                    "bake must produce 3 outputs (per-channel log2 gain)",
                ));
            }
            // H_c = (S_c + ε) · 2^MLP_c − ε
            hdr_row[px * 3] = (r + eps) * exp2(mlp_out[0]) - eps;
            hdr_row[px * 3 + 1] = (g + eps) * exp2(mlp_out[1]) - eps;
            hdr_row[px * 3 + 2] = (b + eps) * exp2(mlp_out[2]) - eps;
        }
        Ok(())
    }

    /// Reconstruct the entire HDR image from an SDR image. Convenience
    /// wrapper over [`Self::apply_row`].
    ///
    /// `width × height` must equal `sdr.len() / 3 == hdr.len() / 3`.
    pub fn apply_image(
        &mut self,
        width: u32,
        height: u32,
        sdr: &[f32],
        hdr: &mut [f32],
    ) -> Result<()> {
        if sdr.len() != hdr.len() {
            return Err(Error::gainmap_mlp("sdr and hdr must be the same length"));
        }
        let expected = (width as usize) * (height as usize) * 3;
        if sdr.len() != expected {
            return Err(Error::gainmap_mlp(
                "buffer length must equal width * height * 3",
            ));
        }
        let row_pixels = width as usize;
        let row_floats = row_pixels * 3;
        for y in 0..height {
            let row_start = (y as usize) * row_floats;
            let row_end = row_start + row_floats;
            self.apply_row(y, &sdr[row_start..row_end], &mut hdr[row_start..row_end])?;
        }
        Ok(())
    }

    /// Reconstruct one pixel — the lowest-level entry point. Mostly
    /// useful for testing and for callers integrating the decoder into
    /// a different row-iteration loop (e.g., tile-based).
    pub fn apply_pixel(&mut self, x_norm: f32, y_norm: f32, sdr: [f32; 3]) -> Result<[f32; 3]> {
        self.raw_input = [x_norm, y_norm, sdr[0], sdr[1], sdr[2]];
        let mlp_out = self
            .predictor
            .predict_transformed(&self.raw_input)
            .map_err(Error::gainmap_mlp_predict)?;
        if mlp_out.len() != 3 {
            return Err(Error::gainmap_mlp(
                "bake must produce 3 outputs (per-channel log2 gain)",
            ));
        }
        let eps = self.cfg.epsilon;
        Ok([
            (sdr[0] + eps) * exp2(mlp_out[0]) - eps,
            (sdr[1] + eps) * exp2(mlp_out[1]) - eps,
            (sdr[2] + eps) * exp2(mlp_out[2]) - eps,
        ])
    }

    /// Return the decoder's configured ε.
    pub fn epsilon(&self) -> f32 {
        self.cfg.epsilon
    }
}

/// Validate that the loaded bake has the shape this consumer expects.
///
/// Specifically:
/// - `n_outputs == 3` (per-channel log₂ gain).
/// - `feature_transforms.len() == 5` (raw inputs x, y, R, G, B).
/// - All five transforms are `Sinusoidal` (the embedding lives in
///   the bake; the consumer only handles unit-coordinate normalisation
///   and the reconstruction formula).
///
/// Returns a descriptive error so a misconfigured bake fails loudly
/// at construction rather than producing wrong pixels per call.
fn validate_bake_shape(model: &Model) -> Result<()> {
    if model.n_outputs() != 3 {
        return Err(Error::gainmap_mlp(
            "bake must have n_outputs = 3 (per-channel log2 gain)",
        ));
    }
    let transforms = model.feature_transforms().ok_or_else(|| {
        Error::gainmap_mlp(
            "bake must declare zentrain.feature_transforms with sinusoidal embeddings",
        )
    })?;
    if transforms.len() != 5 {
        return Err(Error::gainmap_mlp(
            "bake must declare exactly 5 raw input features (x, y, R, G, B)",
        ));
    }
    for t in transforms {
        if !t.is_expander() {
            return Err(Error::gainmap_mlp(
                "each raw input feature must use FeatureTransform::Sinusoidal",
            ));
        }
    }
    Ok(())
}

/// `2^x` portable across std / no_std builds.
#[inline]
fn exp2(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.exp2()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::exp2f(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The bake-load + Predictor end-to-end tests live in
    // `tests/gainmap_mlp_integration.rs` (where `zenpredict-bake` is
    // available as a dev-dep). Unit tests here focus on the math
    // helpers + config defaults.

    #[test]
    fn default_epsilon_matches_iso_21496_1() {
        let cfg = GainMapMlpConfig::default();
        assert_eq!(cfg.epsilon, 1.0 / 64.0);
        assert_eq!(cfg.epsilon, DEFAULT_EPSILON);
    }

    #[test]
    fn exp2_matches_std() {
        for x in [0.0_f32, 1.0, -1.0, 0.5, 2.5, -3.25] {
            let theirs = exp2(x);
            let ours = x.exp2();
            assert!(
                (theirs - ours).abs() < 1e-6,
                "exp2({x}) drift: {theirs} vs {ours}"
            );
        }
    }
}
