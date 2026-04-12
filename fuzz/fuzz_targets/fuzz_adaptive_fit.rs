#![no_main]
//! Fuzz AdaptiveTonemapper::fit_luminance and fit_per_channel.
//!
//! Goal: no panic on any input pair. Errors are fine (NoValidSamples,
//! EmptyChannel, etc.) — we're testing that the code doesn't crash on
//! degenerate data.

use libfuzzer_sys::fuzz_target;
use zentone::experimental::{AdaptiveTonemapper, FitConfig};

fuzz_target!(|data: &[u8]| {
    if data.len() < 25 {
        return;
    }

    // First byte: config selector
    let mode = data[0] % 2; // 0 = luminance, 1 = per_channel
    let channels = if data[1] % 2 == 0 { 3u8 } else { 4u8 };
    let pixel_bytes = &data[2..];

    // Interpret as f32
    let floats: Vec<f32> = pixel_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let ch = channels as usize;
    let trim = floats.len() - floats.len() % ch;
    if trim < ch {
        return;
    }
    let half = trim / 2;
    let half = half - half % ch;
    if half < ch {
        return;
    }

    let hdr = &floats[..half];
    let sdr = &floats[half..half * 2];

    let cfg = FitConfig {
        max_samples: 1000, // keep fit cheap
        ..Default::default()
    };

    match mode {
        0 => {
            let _ = AdaptiveTonemapper::fit_luminance(hdr, sdr, channels, &cfg);
        }
        _ => {
            let _ = AdaptiveTonemapper::fit_per_channel(hdr, sdr, channels, &cfg);
        }
    }
});
