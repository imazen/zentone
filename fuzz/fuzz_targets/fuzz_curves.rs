#![no_main]
//! Fuzz all ToneMapCurve variants and the ToneMap trait methods.
//!
//! Goal: no panic, no NaN, no inf in output. The output SHOULD be in
//! [0, 1] for most curves, but we only assert finite — some unclamped
//! variants (BT.2390 with extreme input) can exceed 1.0.

use libfuzzer_sys::fuzz_target;
use zentone::*;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    // Use first byte to select curve variant, rest as pixel data
    let variant = data[0] % 14;
    let pixel_bytes = &data[1..];

    // Interpret remaining bytes as raw f32 values. Replace non-finite
    // values with 0 so we can test the full finite range including
    // negatives, subnormals, and large values.
    if pixel_bytes.len() < 12 {
        return;
    }
    let floats: Vec<f32> = pixel_bytes
        .chunks_exact(4)
        .map(|c| {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if !v.is_finite() { 0.0 } else { v }
        })
        .collect();
    if floats.len() < 3 {
        return;
    }

    let luma = LUMA_BT709;
    let curve: Box<dyn ToneMap> = match variant {
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
        12 => Box::new(Bt2446A::new(4000.0, 100.0)),
        _ => Box::new(CompiledFilmicSpline::new(&FilmicSplineConfig::default())),
    };

    // Exercise map_rgb — assert output is finite for finite input.
    let rgb = [floats[0], floats[1], floats[2]];
    let out = curve.map_rgb(rgb);
    for &v in &out {
        assert!(v.is_finite(), "map_rgb produced non-finite output");
    }

    // Exercise map_row on the whole buffer (trim to multiple of 3)
    let trim = floats.len() - floats.len() % 3;
    if trim >= 3 {
        let mut row: Vec<f32> = floats[..trim].to_vec();
        curve.map_row(&mut row, 3);
    }

    // Exercise map_into
    if trim >= 3 {
        let src = &floats[..trim];
        let mut dst = vec![0.0f32; trim];
        curve.map_into(src, &mut dst, 3);
    }
});
