// Shared body of the `fuzz_curves` target.
//
// This file is `include!`d by BOTH `fuzz/fuzz_targets/fuzz_curves.rs` (the
// libFuzzer bin) and `tests/fuzz_regression.rs` (the stable-toolchain replay
// harness that runs every seed in `fuzz/regression/`). Single source of truth:
// if the fuzz body and the replay harness drifted apart, a seed that once
// crashed the fuzzer could silently stop exercising the same code path.
//
// Goal: no panic, no NaN, no inf in output for any *finite* input. The output
// SHOULD be in [0, 1] for most curves, but we only assert finite — some
// unclamped variants (BT.2390 with extreme input) can exceed 1.0.
//
// Variant table (`data[0] % FUZZ_CURVE_VARIANTS`). Keep the modulus and the
// index → curve mapping STABLE: the committed regression seeds in
// `fuzz/regression/` encode the curve they target in their first byte, so
// renumbering silently retargets them. When a curve is removed from the
// crate, replace its slot with another curve rather than shrinking the table
// (slot 12 was `Bt2446A` until it moved to `zenpixels-convert` in 0.2.0; it
// now holds `Bt2446B`).

/// Number of curve variants selected by the first input byte.
pub const FUZZ_CURVE_VARIANTS: u8 = 14;

/// Build the curve for a variant index in `0..FUZZ_CURVE_VARIANTS`.
pub fn fuzz_curve_for_variant(variant: u8) -> Box<dyn zentone::ToneMap> {
    use zentone::*;
    let luma = LUMA_BT709;
    match variant {
        0 => Box::new(ToneMapCurve::Reinhard),
        1 => Box::new(ToneMapCurve::ExtendedReinhard { l_max: 4.0, luma }),
        2 => Box::new(ToneMapCurve::ReinhardJodie { luma }),
        3 => Box::new(ToneMapCurve::TunedReinhard {
            content_max_nits: 4000.0,
            display_max_nits: 250.0,
            luma,
        }),
        4 => Box::new(ToneMapCurve::Narkowicz),
        5 => Box::new(ToneMapCurve::HableFilmic),
        6 => Box::new(ToneMapCurve::AcesAp1),
        7 => Box::new(ToneMapCurve::Bt2390 {
            source_peak: 4.0,
            target_peak: 1.0,
        }),
        8 => Box::new(ToneMapCurve::Agx(AgxLook::Default)),
        9 => Box::new(ToneMapCurve::Agx(AgxLook::Punchy)),
        10 => Box::new(ToneMapCurve::Agx(AgxLook::Golden)),
        11 => Box::new(Bt2408Tonemapper::new(4000.0, 1000.0)),
        12 => Box::new(Bt2446B::new(4000.0, 100.0)),
        _ => Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
    }
}

/// Run one fuzz input through `map_rgb`, `map_row`, and `map_into`.
///
/// Panics (the fuzz failure signal) if any output is non-finite.
pub fn run_fuzz_curves(data: &[u8]) {
    if data.len() < 16 {
        return;
    }

    // Use first byte to select curve variant, rest as pixel data
    let variant = data[0] % FUZZ_CURVE_VARIANTS;
    let pixel_bytes = &data[1..];

    // Interpret remaining bytes as raw f32 values. Replace non-finite
    // values with 0 so we can test the full finite range including
    // negatives, subnormals, and large values.
    if pixel_bytes.len() < 12 {
        return;
    }
    let floats: Vec<f32> = pixel_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let v = f32::from_le_bytes(*c);
            if !v.is_finite() { 0.0 } else { v }
        })
        .collect();
    if floats.len() < 3 {
        return;
    }

    let curve = fuzz_curve_for_variant(variant);

    // Exercise map_rgb — assert output is finite for finite input.
    let rgb = [floats[0], floats[1], floats[2]];
    let out = curve.map_rgb(rgb);
    for &v in &out {
        assert!(
            v.is_finite(),
            "map_rgb produced non-finite output: variant {variant}, in {rgb:?}, out {out:?}"
        );
    }

    // Exercise map_row on the whole buffer (trim to multiple of 3)
    let trim = floats.len() - floats.len() % 3;
    if trim >= 3 {
        let mut row: Vec<f32> = floats[..trim].to_vec();
        curve.map_row(&mut row, 3);
        for (i, &v) in row.iter().enumerate() {
            assert!(
                v.is_finite(),
                "map_row produced non-finite output: variant {variant}, index {i}, in {:?}",
                &floats[i - i % 3..i - i % 3 + 3]
            );
        }
    }

    // Exercise map_into
    if trim >= 3 {
        let src = &floats[..trim];
        let mut dst = vec![0.0f32; trim];
        curve.map_into(src, &mut dst, 3);
        for (i, &v) in dst.iter().enumerate() {
            assert!(
                v.is_finite(),
                "map_into produced non-finite output: variant {variant}, index {i}, in {:?}",
                &src[i - i % 3..i - i % 3 + 3]
            );
        }
    }
}
