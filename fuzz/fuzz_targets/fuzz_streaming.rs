#![no_main]
//! Fuzz StreamingTonemapper push/pull cycle.
//!
//! Goal: no panic, no lost rows. We feed random f32 rows through a small
//! tonemapper and verify every pushed row comes back out.

use libfuzzer_sys::fuzz_target;
use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Parse config from first bytes
    let width = (data[0] as u32 % 64).max(1);
    let height = (data[1] as u32 % 64).max(1);
    let channels = if data[2] % 2 == 0 { 3u8 } else { 4u8 };
    let cell_size = (data[3] as u32 % 16).max(1);
    let lookahead = (data[4] as u32 % 32).max(1);

    let cfg = StreamingTonemapConfig {
        cell_size,
        lookahead_rows: lookahead,
        ..Default::default()
    };

    let mut tm = match StreamingTonemapper::new(width, height, channels, cfg) {
        Ok(t) => t,
        Err(_) => return,
    };

    let row_stride = tm.row_stride();
    let mut out = vec![0.0f32; row_stride];

    // Generate row from remaining data or use constant
    let row: Vec<f32> = if data.len() > 5 + row_stride * 4 {
        data[5..]
            .chunks_exact(4)
            .take(row_stride)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        vec![0.5f32; row_stride]
    };

    // Pad if needed
    let row = if row.len() < row_stride {
        let mut padded = row;
        padded.resize(row_stride, 0.5);
        padded
    } else {
        row
    };

    let mut pushed = 0u32;
    let mut pulled = 0u32;

    for _ in 0..height {
        if tm.push_row(&row).is_err() {
            // Ring buffer full — pull first
            while tm.pull_row(&mut out).ok().flatten().is_some() {
                pulled += 1;
            }
            if tm.push_row(&row).is_err() {
                break; // give up
            }
        }
        pushed += 1;
        while tm.pull_row(&mut out).ok().flatten().is_some() {
            pulled += 1;
        }
    }
    tm.finish();
    while tm.pull_row(&mut out).ok().flatten().is_some() {
        pulled += 1;
    }

    // Every pushed row must come out
    assert_eq!(
        pushed, pulled,
        "lost rows: pushed {pushed}, pulled {pulled}"
    );
});
