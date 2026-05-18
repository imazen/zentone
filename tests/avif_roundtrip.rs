//! End-to-end Gain-MLP transport test: bake bytes → AVIF muxer
//! ([`zenavif_serialize::Aviffy::set_gain_map_mlp_bake`]) → AVIF parser
//! ([`zenavif_parse::AvifParser::gain_map_mlp_bake`]) → [`GainMapMlpDecoder`].
//!
//! Validates that round-tripping the bake through an ISOBMFF container
//! produces byte-identical bytes AND byte-identical decoder outputs vs
//! running the same bake directly. This is the cross-crate contract
//! test for the experimental `'zmlp'` auxiliary item type.

#![cfg(feature = "gainmap-mlp")]

use zenpredict::{Activation, MetadataType, Model, WeightDtype, keys};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zentone::{GainMapMlpConfig, GainMapMlpDecoder};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// Reproduces the constant-gain bake shape used by
/// [`gainmap_mlp_integration`] so this test is self-contained.
fn make_constant_gain_bake(num_freqs: usize, constant_log2_gain: [f32; 3]) -> Vec<u8> {
    let arity = 2 * num_freqs;
    let expanded_dim = 5 * arity;
    let hidden_dim = 4;
    let scaler_mean = vec![0.0_f32; expanded_dim];
    let scaler_scale = vec![1.0_f32; expanded_dim];
    let l0_weights = vec![0.0_f32; expanded_dim * hidden_dim];
    let l0_biases = vec![0.0_f32; hidden_dim];
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
    let transforms_txt = b"sinusoidal\nsinusoidal\nsinusoidal\nsinusoidal\nsinusoidal";
    let freq_line: String = (0..num_freqs)
        .map(|k| (1u32 << k).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let mut params_txt = Vec::new();
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

/// Minimal tone-map metadata — matches the layout used by both
/// `zenavif-serialize`'s and `zenavif-parse`'s integration tests so
/// the muxed file parses cleanly.
fn make_tmap_metadata() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(0u8);
    buf.extend_from_slice(&0u16.to_be_bytes());
    buf.extend_from_slice(&0u16.to_be_bytes());
    buf.push(1u8 << 6); // use_base_colour_space
    buf.extend_from_slice(&0u32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&0i32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&1i32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&0i32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&0i32.to_be_bytes());
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf
}

#[test]
fn bake_roundtrips_through_avif_container_and_decodes_identically() {
    let bake_bytes = make_constant_gain_bake(2, [1.0, 0.5, -0.25]);

    let primary_data = b"primary_av1_xyz";
    let gain_map_data = b"av01_gain_map_xyz";
    let metadata = make_tmap_metadata();
    let avif_bytes = zenavif_serialize::Aviffy::new()
        .set_gain_map(gain_map_data.to_vec(), 4, 4, 8, metadata)
        .set_gain_map_mlp_bake(bake_bytes.clone())
        .to_vec(primary_data, None, 10, 20, 8);

    let parser = zenavif_parse::AvifParser::from_bytes(&avif_bytes).expect("parse AVIF");
    let recovered_bake = parser
        .gain_map_mlp_bake()
        .expect("zmlp accessor present")
        .expect("resolve zmlp extent")
        .into_owned();
    assert_eq!(
        recovered_bake, bake_bytes,
        "bake bytes must roundtrip byte-identical through AVIF container"
    );

    // Forward both bakes through GainMapMlpDecoder and compare.
    let direct = Aligned(bake_bytes);
    let recovered = Aligned(recovered_bake);
    let direct_model = Model::from_bytes(&direct.0).expect("direct model");
    let recovered_model = Model::from_bytes(&recovered.0).expect("recovered model");

    let width = 4u32;
    let height = 4u32;
    let mut direct_decoder =
        GainMapMlpDecoder::new(&direct_model, width, height, GainMapMlpConfig::default())
            .expect("direct decoder");
    let mut recovered_decoder =
        GainMapMlpDecoder::new(&recovered_model, width, height, GainMapMlpConfig::default())
            .expect("recovered decoder");

    let sdr: Vec<f32> = (0..(width * height * 3) as usize)
        .map(|i| (i as f32) / ((width * height * 3) as f32))
        .collect();
    let mut hdr_direct = vec![0.0_f32; sdr.len()];
    let mut hdr_recovered = vec![0.0_f32; sdr.len()];
    direct_decoder
        .apply_image(width, height, &sdr, &mut hdr_direct)
        .expect("direct apply_image");
    recovered_decoder
        .apply_image(width, height, &sdr, &mut hdr_recovered)
        .expect("recovered apply_image");

    assert_eq!(
        hdr_direct, hdr_recovered,
        "decoder output must be bit-exact between direct and AVIF-recovered bakes"
    );
}

#[test]
fn avif_without_mlp_bake_returns_none() {
    // Same minting flow but without `set_gain_map_mlp_bake`. The
    // parser must report `gain_map_mlp_bake() == None` so consumers
    // can cleanly fall back to the traditional av01 gain map.
    let primary_data = b"primary";
    let gain_map_data = b"av01_only";
    let metadata = make_tmap_metadata();
    let avif_bytes = zenavif_serialize::Aviffy::new()
        .set_gain_map(gain_map_data.to_vec(), 4, 4, 8, metadata)
        .to_vec(primary_data, None, 10, 20, 8);
    let parser = zenavif_parse::AvifParser::from_bytes(&avif_bytes).expect("parse AVIF");
    assert!(parser.gain_map_data().is_some());
    assert!(parser.gain_map_mlp_bake().is_none());
}
