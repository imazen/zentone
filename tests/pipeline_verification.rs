//! End-to-end pipeline verification tests.
//!
//! These tests exercise the full PQ/HLG → sRGB pipeline including
//! transfer function decode, tone mapping, gamut conversion, and
//! soft clipping. They verify:
//!
//! - Every output pixel is in [0, 1] (linear) or [0, 255] (u8)
//! - Neutral gray passes through without hue shift
//! - Saturated BT.2020 primaries don't produce out-of-gamut BT.709
//! - Alpha is preserved in RGBA paths
//! - All three pipeline functions agree on shared pixel values
//! - Channel ordering (hue) is preserved through soft clip
//!
//! This is the test layer we were missing — formula-level parity
//! tests (exhaustive_properties.rs) validate tone curves in isolation
//! but don't catch gamut conversion artifacts.

use zentone::pipeline::*;
use zentone::*;

// ============================================================================
// PQ test data generators
// ============================================================================

/// PQ signal value for a given absolute luminance in nits.
/// Approximation of the ST.2084 OETF sufficient for test data.
fn nits_to_pq(nits: f32) -> f32 {
    linear_srgb::tf::linear_to_pq(nits / 10000.0)
}

/// Generate a ramp of PQ-encoded neutral grays from 0 to `peak_nits`.
fn pq_neutral_ramp(steps: usize, peak_nits: f32) -> Vec<[f32; 3]> {
    let mut row = Vec::with_capacity(steps);
    for i in 0..steps {
        let nits = peak_nits * (i as f32 / (steps - 1) as f32);
        let pq = nits_to_pq(nits);
        row.push([pq, pq, pq]);
    }
    row
}

/// Generate PQ-encoded saturated BT.2020 primaries and secondaries.
/// These are the worst case for gamut conversion — maximum excursion
/// outside BT.709.
fn pq_saturated_bt2020(nits: f32) -> Vec<[f32; 3]> {
    let pq = nits_to_pq(nits);
    vec![
        [pq, 0.0, 0.0], // BT.2020 red
        [0.0, pq, 0.0], // BT.2020 green
        [0.0, 0.0, pq], // BT.2020 blue
        [pq, pq, 0.0],  // BT.2020 yellow
        [0.0, pq, pq],  // BT.2020 cyan
        [pq, 0.0, pq],  // BT.2020 magenta
        [pq, pq, pq],   // BT.2020 white (neutral)
    ]
}

/// All tonemappers we test pipelines with.
fn pipeline_tonemappers() -> Vec<(&'static str, Box<dyn ToneMap>)> {
    vec![
        ("Reinhard", Box::new(ToneMapCurve::Reinhard)),
        ("Narkowicz", Box::new(ToneMapCurve::Narkowicz)),
        ("HableFilmic", Box::new(ToneMapCurve::HableFilmic)),
        ("AcesAp1", Box::new(ToneMapCurve::AcesAp1)),
        ("Agx", Box::new(ToneMapCurve::Agx(AgxLook::Default))),
        ("Bt2408", Box::new(Bt2408Tonemapper::new(4000.0, 1000.0))),
        (
            "Bt2408_MaxRgb",
            Box::new(Bt2408Tonemapper::max_rgb(4000.0, 1000.0)),
        ),
        ("Bt2446A", Box::new(Bt2446A::new(4000.0, 100.0))),
    ]
}

// ============================================================================
// PQ → linear sRGB: all outputs in [0, 1]
// ============================================================================

#[test]
fn pq_to_linear_srgb_all_outputs_in_gamut() {
    let mut scratch = TonemapScratch::new();
    for (name, tm) in pipeline_tonemappers() {
        // Neutral ramp
        let ramp = pq_neutral_ramp(64, 4000.0);
        let mut out = vec![[0.0_f32; 3]; ramp.len()];
        tonemap_pq_row_simd(&mut scratch, &ramp, &mut out, tm.as_ref());

        for (i, px) in out.iter().enumerate() {
            for (ch, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&v),
                    "{name}: neutral ramp pixel {i} ch {ch} = {v} (out of [0,1])"
                );
            }
        }

        // Saturated BT.2020 primaries at 1000 nits
        let sat = pq_saturated_bt2020(1000.0);
        let mut out = vec![[0.0_f32; 3]; sat.len()];
        tonemap_pq_row_simd(&mut scratch, &sat, &mut out, tm.as_ref());

        for (i, px) in out.iter().enumerate() {
            for (ch, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&v),
                    "{name}: saturated pixel {i} ch {ch} = {v} (out of [0,1])"
                );
            }
        }
    }
}

// ============================================================================
// PQ → sRGB u8: all bytes valid, no overflow
// ============================================================================

#[test]
fn pq_to_srgb8_all_bytes_valid() {
    let mut scratch = TonemapScratch::new();
    for (name, tm) in pipeline_tonemappers() {
        let ramp = pq_neutral_ramp(64, 4000.0);
        let mut out = vec![[0u8; 3]; ramp.len()];
        tonemap_pq_to_srgb8_row_simd(&mut scratch, &ramp, &mut out, tm.as_ref());

        // Black should map near 0.
        assert!(
            out[0][0] <= 5 && out[0][1] <= 5 && out[0][2] <= 5,
            "{name}: black not near zero: {:?}",
            out[0]
        );

        // Peak should be bright (exact value varies by curve)
        let last = out[out.len() - 1];
        assert!(
            last[0] > 50 || last[1] > 50 || last[2] > 50,
            "{name}: peak too dark: {last:?}"
        );

        // Saturated primaries — must not panic or produce garbage
        let sat = pq_saturated_bt2020(2000.0);
        let mut out = vec![[0u8; 3]; sat.len()];
        tonemap_pq_to_srgb8_row_simd(&mut scratch, &sat, &mut out, tm.as_ref());
        // Just verify it ran without panic; values are in [0, 255] by type.
    }
}

// ============================================================================
// HLG → linear sRGB: all outputs in [0, 1]
// ============================================================================

#[test]
fn hlg_to_linear_srgb_all_outputs_in_gamut() {
    let mut scratch = TonemapScratch::new();
    for (name, tm) in pipeline_tonemappers() {
        // HLG ramp: 0.0 to 1.0 in HLG signal space
        let mut hlg_row: Vec<[f32; 3]> = Vec::with_capacity(32);
        for i in 0..32 {
            let v = i as f32 / 31.0;
            hlg_row.push([v, v, v]);
        }

        let mut out = vec![[0.0_f32; 3]; hlg_row.len()];
        tonemap_hlg_row_simd(&mut scratch, &hlg_row, &mut out, tm.as_ref(), 1000.0);

        for (i, px) in out.iter().enumerate() {
            for (ch, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&v),
                    "{name}: HLG pixel {i} ch {ch} = {v}"
                );
            }
        }

        // HLG saturated primaries
        let sat: Vec<[f32; 3]> = vec![
            [1.0_f32, 0.0, 0.0], // R
            [0.0, 1.0, 0.0],     // G
            [0.0, 0.0, 1.0],     // B
            [1.0, 1.0, 0.0],     // Y
            [0.0, 1.0, 1.0],     // C
        ];
        let mut out = vec![[0.0_f32; 3]; sat.len()];
        tonemap_hlg_row_simd(&mut scratch, &sat, &mut out, tm.as_ref(), 1000.0);

        for (i, px) in out.iter().enumerate() {
            for (ch, &v) in px.iter().enumerate() {
                assert!(
                    v.is_finite() && (-1e-6..=1.0 + 1e-6).contains(&v),
                    "{name}: HLG saturated pixel {i} ch {ch} = {v}"
                );
            }
        }
    }
}

// ============================================================================
// Neutral gray: no hue shift through pipeline
// ============================================================================

#[test]
fn pq_neutral_gray_stays_neutral() {
    let mut scratch = TonemapScratch::new();
    for (name, tm) in pipeline_tonemappers() {
        // AgX's inset/outset matrices create some per-channel divergence
        // at low luminance. Allow a wider threshold for AgX.
        let is_agx = name.contains("Agx");

        let ramp = pq_neutral_ramp(16, 2000.0);
        let mut out = vec![[0.0_f32; 3]; ramp.len()];
        tonemap_pq_row_simd(&mut scratch, &ramp, &mut out, tm.as_ref());

        for (i, px) in out.iter().enumerate() {
            let max_ch = px[0].max(px[1]).max(px[2]);
            if max_ch < 0.005 {
                continue;
            }
            let min_ch = px[0].min(px[1]).min(px[2]);
            let spread = (max_ch - min_ch) / max_ch;
            // AgX's inset matrix has unequal row sums (1.105, 0.933, 0.960)
            // so neutral input becomes chromatic in the log domain. The
            // nonlinear sigmoid means the outset can't perfectly undo this.
            // At low luminance the spread reaches ~35%.
            let limit = if is_agx { 0.40 } else { 0.02 };
            assert!(
                spread < limit,
                "{name}: neutral gray pixel {i} has hue shift: RGB=[{:.4}, {:.4}, {:.4}], spread={spread:.4}",
                px[0],
                px[1],
                px[2]
            );
        }
    }
}

// ============================================================================
// Channel ordering preserved (hue direction) for saturated colors
// ============================================================================

#[test]
fn pq_saturated_preserves_channel_ordering() {
    let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
    let mut scratch = TonemapScratch::new();

    // BT.2020 "warm red" at 500 nits: R >> G > B
    let pq_r = nits_to_pq(500.0);
    let pq_g = nits_to_pq(100.0);
    let pq_b = nits_to_pq(20.0);
    let src = [[pq_r, pq_g, pq_b]];
    let mut out = [[0.0_f32; 3]];
    tonemap_pq_row_simd(&mut scratch, &src, &mut out, &tm);

    assert!(
        out[0][0] >= out[0][1] && out[0][1] >= out[0][2],
        "warm red should maintain R >= G >= B ordering: {:?}",
        out[0]
    );

    // BT.2020 "teal" at 500 nits: G > B > R
    let src = [[nits_to_pq(30.0), nits_to_pq(400.0), nits_to_pq(300.0)]];
    let mut out = [[0.0_f32; 3]];
    tonemap_pq_row_simd(&mut scratch, &src, &mut out, &tm);

    assert!(
        out[0][1] >= out[0][2] && out[0][2] >= out[0][0],
        "teal should maintain G >= B >= R ordering: {:?}",
        out[0]
    );
}

// ============================================================================
// RGBA alpha preservation for all pipeline paths
// ============================================================================

#[test]
fn all_pipelines_preserve_alpha() {
    let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
    let mut scratch = TonemapScratch::new();
    let alpha = 0.42_f32;

    // PQ → linear sRGB (f32 RGBA)
    let pq = [[0.5_f32, 0.5, 0.5, alpha]];
    let mut out_f32 = [[0.0_f32; 4]];
    tonemap_pq_rgba_row_simd(&mut scratch, &pq, &mut out_f32, &tm);
    assert!(
        (out_f32[0][3] - alpha).abs() < 1e-6,
        "PQ→linear alpha: {} != {alpha}",
        out_f32[0][3]
    );

    // PQ → sRGB u8 (RGBA)
    let mut out_u8 = [[0u8; 4]];
    tonemap_pq_to_srgb8_rgba_row_simd(&mut scratch, &pq, &mut out_u8, &tm);
    let expected_alpha_u8 = (alpha * 255.0 + 0.5) as u8;
    assert_eq!(
        out_u8[0][3], expected_alpha_u8,
        "PQ→sRGB8 alpha: {} != {expected_alpha_u8}",
        out_u8[0][3]
    );

    // HLG → linear sRGB (RGBA)
    let hlg = [[0.75_f32, 0.75, 0.75, alpha]];
    let mut out_f32 = [[0.0_f32; 4]];
    tonemap_hlg_rgba_row_simd(&mut scratch, &hlg, &mut out_f32, &tm, 1000.0);
    assert!(
        (out_f32[0][3] - alpha).abs() < 1e-6,
        "HLG→linear alpha: {} != {alpha}",
        out_f32[0][3]
    );
}

// ============================================================================
// PQ → linear and PQ → u8 pipelines agree
// ============================================================================

#[test]
fn pq_linear_and_srgb8_pipelines_agree() {
    let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
    let mut scratch = TonemapScratch::new();
    let ramp = pq_neutral_ramp(32, 4000.0);

    let mut linear = vec![[0.0_f32; 3]; ramp.len()];
    tonemap_pq_row_simd(&mut scratch, &ramp, &mut linear, &tm);

    let mut u8_out = vec![[0u8; 3]; ramp.len()];
    tonemap_pq_to_srgb8_row_simd(&mut scratch, &ramp, &mut u8_out, &tm);

    // Convert the linear output manually to u8 and compare.
    for (i, (lin_px, u8_px)) in linear.iter().zip(u8_out.iter()).enumerate() {
        for ch in 0..3 {
            let v = lin_px[ch].clamp(0.0, 1.0);
            let srgb = linear_srgb::tf::linear_to_srgb(v);
            let expected = (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            let diff = (u8_px[ch] as i16 - expected as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "pixel {i} ch {ch}: srgb8={} but linear→manual={expected} (linear={:.6})",
                u8_px[ch],
                lin_px[ch]
            );
        }
    }
}

// ============================================================================
// Monotonicity through full pipeline: brighter input → brighter output
// ============================================================================

#[test]
fn pq_pipeline_monotonic_on_neutral_ramp() {
    let mut scratch = TonemapScratch::new();
    for (name, tm) in pipeline_tonemappers() {
        let ramp = pq_neutral_ramp(128, 4000.0);
        let mut out = vec![[0.0_f32; 3]; ramp.len()];
        tonemap_pq_row_simd(&mut scratch, &ramp, &mut out, tm.as_ref());

        // Luminance (all channels equal for neutral) should be monotonic.
        let mut prev_lum = -1.0_f32;
        for (i, px) in out.iter().enumerate() {
            let lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            assert!(
                lum >= prev_lum - 1e-5,
                "{name}: monotonicity broken at pixel {i}: lum={lum:.6} < prev={prev_lum:.6}"
            );
            prev_lum = lum;
        }
    }
}

// ============================================================================
// Wide-gamut stress test: every combination of PQ primary intensities
// ============================================================================

#[test]
fn pq_wide_gamut_grid_all_in_range() {
    let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
    let mut scratch = TonemapScratch::new();

    // 8 intensity levels × 3 channels = 512 test pixels
    let levels: [f32; 8] = [0.0, 50.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 10000.0];
    let mut src: Vec<[f32; 3]> = Vec::with_capacity(512);
    for &r_nits in &levels {
        for &g_nits in &levels {
            for &b_nits in &levels {
                src.push([nits_to_pq(r_nits), nits_to_pq(g_nits), nits_to_pq(b_nits)]);
            }
        }
    }

    let mut out = vec![[0.0_f32; 3]; src.len()];
    tonemap_pq_row_simd(&mut scratch, &src, &mut out, &tm);

    let mut violations = 0;
    for (i, px) in out.iter().enumerate() {
        for (ch, &v) in px.iter().enumerate() {
            if !v.is_finite() || !(-1e-5..=1.0 + 1e-5).contains(&v) {
                violations += 1;
                if violations <= 5 {
                    eprintln!(
                        "violation at pixel {i} ch {ch}: {v:.6} (src PQ: [{:.4}, {:.4}, {:.4}])",
                        src[i][0], src[i][1], src[i][2]
                    );
                }
            }
        }
    }
    assert!(
        violations == 0,
        "{violations} out-of-gamut violations in 512-pixel grid"
    );
}
