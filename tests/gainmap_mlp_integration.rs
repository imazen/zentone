//! End-to-end integration tests for [`zentone::gainmap_mlp`].
//!
//! Builds Gain-MLP-shaped ZNPR v3 bakes (5-D sinusoidal-embedded input
//! → identity-passthrough → 3-D output) and runs them through
//! [`GainMapMlpDecoder`]. The bakes' weights and biases are
//! hand-constructed so the expected per-pixel HDR output is calculable
//! by hand, letting us assert byte-exact reconstruction math.

#![cfg(feature = "gainmap-mlp")]

use zenpredict::{Activation, MetadataType, Model, WeightDtype, keys};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zentone::{GainMapMlpConfig, GainMapMlpDecoder};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// Build a Gain-MLP-shaped bake with:
/// - 5 raw inputs (x, y, R, G, B), all `Sinusoidal` with the supplied
///   frequencies per input (per-input arity = `2 * num_freqs`).
/// - Total expanded dim = 5 * 2 * num_freqs.
/// - Two layers: expanded → `hidden_dim` (ReLU), `hidden_dim` → 3 (Identity).
/// - Weights / biases produce a deterministic 3-output mapping
///   parameterised by the supplied scalars.
///
/// Specifically: `out[c] = constant_bias[c]` regardless of input —
/// the input layer has all-zero weights so the embedded vector is
/// ignored, and the hidden bias drives the output. Use this to test
/// that the decoder's reconstruction math is correct independent of
/// any learned MLP behaviour.
fn make_constant_gain_bake(num_freqs: usize, constant_log2_gain: [f32; 3]) -> Vec<u8> {
    let arity = 2 * num_freqs;
    let expanded_dim = 5 * arity;
    let hidden_dim = 4;
    let scaler_mean = vec![0.0_f32; expanded_dim];
    let scaler_scale = vec![1.0_f32; expanded_dim];

    // Layer 0: expanded_dim → hidden_dim, all zeros (ignores input).
    let l0_weights = vec![0.0_f32; expanded_dim * hidden_dim];
    let l0_biases = vec![0.0_f32; hidden_dim];

    // Layer 1: hidden_dim → 3, zero weights + per-output bias = constant log2 gain.
    let l1_weights = vec![0.0_f32; hidden_dim * 3];
    let l1_biases = constant_log2_gain.to_vec();

    let layers = [
        BakeLayer {
            in_dim: expanded_dim,
            out_dim: hidden_dim,
            activation: Activation::Relu,
            dtype: WeightDtype::F32,
            weights: &l0_weights,
            biases: &l0_biases,
        },
        BakeLayer {
            in_dim: hidden_dim,
            out_dim: 3,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &l1_weights,
            biases: &l1_biases,
        },
    ];

    // 5 lines of "sinusoidal" tokens (one per raw input).
    let transforms_txt = b"sinusoidal\nsinusoidal\nsinusoidal\nsinusoidal\nsinusoidal";
    // 5 lines of identical frequency lists.
    let mut params_txt = Vec::new();
    let freq_line: String = (0..num_freqs)
        .map(|k| (1u32 << k).to_string())
        .collect::<Vec<_>>()
        .join(",");
    for i in 0..5 {
        if i > 0 {
            params_txt.push(b'\n');
        }
        params_txt.extend_from_slice(freq_line.as_bytes());
    }

    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: transforms_txt,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: &params_txt,
        },
    ];

    bake(&BakeRequest {
        schema_hash: 0xfeed_face_dead_beef,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake construction")
}

#[test]
fn decoder_rejects_bake_without_sinusoidal_inputs() {
    // 5 inputs but all `Identity` (not expanders) — must fail at
    // construction. Don't bother with a real bake; we exercise the
    // validate_bake_shape path through a mock.

    // Use a non-expander bake by passing `identity` tokens directly.
    let scaler_mean = vec![0.0_f32; 5];
    let scaler_scale = vec![1.0_f32; 5];
    let l0_w = vec![0.0_f32; 5 * 3];
    let l0_b = vec![0.0_f32; 3];
    let layers = [BakeLayer {
        in_dim: 5,
        out_dim: 3,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &l0_w,
        biases: &l0_b,
    }];
    let metadata = [BakeMetadataEntry {
        key: keys::FEATURE_TRANSFORMS,
        kind: MetadataType::Utf8,
        value: b"identity\nidentity\nidentity\nidentity\nidentity",
    }];
    let bytes = bake(&BakeRequest {
        schema_hash: 1,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake");
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let err = match GainMapMlpDecoder::new(&model, 64, 64, GainMapMlpConfig::default()) {
        Ok(_) => panic!("expected validate_bake_shape to reject this bake"),
        Err(e) => e,
    };
    let s = format!("{err}");
    assert!(
        s.contains("Sinusoidal"),
        "expected validate_bake_shape sinusoidal-check error, got: {s}"
    );
}

#[test]
fn decoder_rejects_bake_with_wrong_output_count() {
    // 5 sinusoidal inputs but output dim = 1 (not 3) — should reject.
    let num_freqs = 1usize;
    let arity = 2 * num_freqs;
    let expanded_dim = 5 * arity;
    let scaler_mean = vec![0.0_f32; expanded_dim];
    let scaler_scale = vec![1.0_f32; expanded_dim];
    let l0_w = vec![0.0_f32; expanded_dim];
    let l0_b = vec![0.0_f32; 1];
    let layers = [BakeLayer {
        in_dim: expanded_dim,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &l0_w,
        biases: &l0_b,
    }];
    let mut params = Vec::new();
    for i in 0..5 {
        if i > 0 {
            params.push(b'\n');
        }
        params.push(b'1');
    }
    let metadata = [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: b"sinusoidal\nsinusoidal\nsinusoidal\nsinusoidal\nsinusoidal",
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: &params,
        },
    ];
    let bytes = bake(&BakeRequest {
        schema_hash: 1,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake");
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let err = match GainMapMlpDecoder::new(&model, 64, 64, GainMapMlpConfig::default()) {
        Ok(_) => panic!("expected validate_bake_shape to reject this bake"),
        Err(e) => e,
    };
    let s = format!("{err}");
    assert!(
        s.contains("n_outputs = 3"),
        "expected n_outputs check error, got: {s}"
    );
}

#[test]
fn apply_pixel_reconstructs_with_constant_gain() {
    // log2_gain = 1.0 ⇒ 2^1 = 2.0 gain ⇒ H = (S + ε) · 2 − ε
    let bytes = make_constant_gain_bake(2, [1.0, 1.0, 1.0]);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut decoder =
        GainMapMlpDecoder::new(&model, 64, 64, GainMapMlpConfig::default()).expect("construct");
    let eps = decoder.epsilon();
    let sdr = [0.5_f32, 0.25, 0.75];
    let hdr = decoder.apply_pixel(0.5, 0.5, sdr).expect("apply");
    let expected = [
        (sdr[0] + eps) * 2.0 - eps,
        (sdr[1] + eps) * 2.0 - eps,
        (sdr[2] + eps) * 2.0 - eps,
    ];
    for c in 0..3 {
        assert!(
            (hdr[c] - expected[c]).abs() < 1e-5,
            "channel {c}: got {} expected {}",
            hdr[c],
            expected[c]
        );
    }
}

#[test]
fn apply_pixel_per_channel_gains_differ() {
    // R doubles, G unchanged (log2_gain=0), B halves (log2_gain=-1).
    let bytes = make_constant_gain_bake(2, [1.0, 0.0, -1.0]);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut decoder =
        GainMapMlpDecoder::new(&model, 64, 64, GainMapMlpConfig::default()).expect("construct");
    let eps = decoder.epsilon();
    let sdr = [0.5_f32, 0.5, 0.5];
    let hdr = decoder.apply_pixel(0.0, 0.0, sdr).expect("apply");
    let expected_r = (sdr[0] + eps) * 2.0 - eps;
    let expected_g = (sdr[1] + eps) * 1.0 - eps; // 2^0 = 1
    let expected_b = (sdr[2] + eps) * 0.5 - eps;
    assert!((hdr[0] - expected_r).abs() < 1e-5);
    assert!((hdr[1] - expected_g).abs() < 1e-5);
    assert!((hdr[2] - expected_b).abs() < 1e-5);
}

#[test]
fn apply_row_runs_full_width() {
    // 8-pixel row through a constant-log2-gain=0 bake: H == S
    // exactly (because 2^0 = 1 ⇒ H = (S + ε) − ε = S).
    let bytes = make_constant_gain_bake(2, [0.0, 0.0, 0.0]);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut decoder =
        GainMapMlpDecoder::new(&model, 8, 1, GainMapMlpConfig::default()).expect("construct");
    let sdr_row: Vec<f32> = (0..8 * 3).map(|i| (i as f32) / 24.0).collect();
    let mut hdr_row = vec![0.0_f32; 8 * 3];
    decoder.apply_row(0, &sdr_row, &mut hdr_row).expect("row");
    for i in 0..8 * 3 {
        // ε cancels exactly only when 2^0 = 1 — verify bit-close.
        assert!(
            (hdr_row[i] - sdr_row[i]).abs() < 1e-5,
            "i={i}: hdr={} sdr={}",
            hdr_row[i],
            sdr_row[i]
        );
    }
}

#[test]
fn apply_image_recovers_constant_doubling() {
    // 4 × 4 image, constant log2_gain = 1.0 ⇒ every channel doubles
    // around the ε offset.
    let bytes = make_constant_gain_bake(2, [1.0, 1.0, 1.0]);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut decoder =
        GainMapMlpDecoder::new(&model, 4, 4, GainMapMlpConfig::default()).expect("construct");
    let eps = decoder.epsilon();
    let sdr: Vec<f32> = (0..4 * 4 * 3).map(|i| (i as f32) / 48.0).collect();
    let mut hdr = vec![0.0_f32; 4 * 4 * 3];
    decoder.apply_image(4, 4, &sdr, &mut hdr).expect("image");
    for i in 0..sdr.len() {
        let expected = (sdr[i] + eps) * 2.0 - eps;
        assert!(
            (hdr[i] - expected).abs() < 1e-5,
            "i={i}: hdr={} expected={}",
            hdr[i],
            expected
        );
    }
}

#[test]
fn apply_image_rejects_mismatched_buffer_size() {
    let bytes = make_constant_gain_bake(2, [0.0, 0.0, 0.0]);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut decoder =
        GainMapMlpDecoder::new(&model, 4, 4, GainMapMlpConfig::default()).expect("construct");
    let sdr = vec![0.0_f32; 4 * 4 * 3];
    // hdr too small
    let mut hdr_short = vec![0.0_f32; 4 * 4 * 3 - 1];
    assert!(decoder.apply_image(4, 4, &sdr, &mut hdr_short).is_err());
    // dimensions don't match buffer length
    let mut hdr_ok = vec![0.0_f32; 4 * 4 * 3];
    assert!(decoder.apply_image(8, 4, &sdr, &mut hdr_ok).is_err());
}

#[test]
fn decoder_rejects_zero_dimensions() {
    let bytes = make_constant_gain_bake(1, [0.0, 0.0, 0.0]);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert!(GainMapMlpDecoder::new(&model, 0, 32, GainMapMlpConfig::default()).is_err());
    assert!(GainMapMlpDecoder::new(&model, 32, 0, GainMapMlpConfig::default()).is_err());
}
