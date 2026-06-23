//! Phase-1 audit: dump source decode descriptors for 3 spot-check samples
//! (iP13Pro HEIC, zfold7 JPEG, iP16Pro / Samsung HEIC).
//!
//! Prints, for each sample, the (channel_type, layout, transfer, primaries)
//! tuple returned by `decode_full_frame()` (SDR base) AND
//! `reconstruct_hdr(None).decode_full_frame()` (HDR via gain map).
//!
//! Run:
//! ```text
//! nice -n19 cargo run -p zentone --release \
//!   --example audit_primaries_dump --features hdr-shootout
//! ```

use std::fs;
use std::path::Path;

use zencodecs::{DecodeRequest, OrientationHint};

const SPOT_CHECK: &[&str] = &[
    // iP13Pro HEIC (the user's #1526 sample)
    "/home/lilith/work/codec-corpus/imazen-26/1400-lilith-nature/1526_nature_pink-flowers-garden_colorado_ip13pro_iso50-f1p5_img-4543_4032x3024.heic",
    // zfold7 UltraHDR JPEG
    "/home/lilith/work/codec-corpus/imazen-26/1200-lilith-interiors/1227_interiors_stained-glass-window_casa-batllo-barcelona_zfold7_iso1600-f1p7_20260315-205230_4000x3000.jpg",
    // ip13pro HEIC interior
    "/home/lilith/work/codec-corpus/imazen-26/1200-lilith-interiors/1246_interiors_woven-roof-structure_azulik-city-arts-francisco-uh-may_ip13pro_iso32-f1p8_img-8944_4032x3024.heic",
];

fn dump(path: &Path) -> anyhow::Result<()> {
    let bytes = fs::read(path)?;
    let stem = path.file_name().unwrap().to_string_lossy();
    println!("\n========== {} ==========", stem);

    let sdr = DecodeRequest::new(&bytes)
        .with_orientation(OrientationHint::Correct)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("decode SDR: {}", e.error()))?;
    let sdr_desc = sdr.pixels().descriptor();
    println!(
        "SDR base    : channel_type={:?} layout={:?} transfer={:?} primaries={:?}",
        sdr_desc.channel_type(),
        sdr_desc.layout(),
        sdr_desc.transfer(),
        sdr_desc.primaries,
    );

    let hdr = DecodeRequest::new(&bytes)
        .with_orientation(OrientationHint::Correct)
        .reconstruct_hdr(None)
        .decode_full_frame()
        .map_err(|e| anyhow::anyhow!("reconstruct_hdr: {}", e.error()))?;
    let hdr_desc = hdr.pixels().descriptor();
    println!(
        "HDR (gainmap): channel_type={:?} layout={:?} transfer={:?} primaries={:?}",
        hdr_desc.channel_type(),
        hdr_desc.layout(),
        hdr_desc.transfer(),
        hdr_desc.primaries,
    );

    Ok(())
}

fn main() {
    for &p in SPOT_CHECK {
        let path = Path::new(p);
        if !path.exists() {
            eprintln!("SKIP missing: {}", p);
            continue;
        }
        if let Err(e) = dump(path) {
            eprintln!("ERROR {}: {}", p, e);
        }
    }
}
