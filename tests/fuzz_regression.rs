//! Fuzz crash regression suite.
//!
//! Runs every file in `fuzz/regression/` through the `fuzz_curves` fuzz body.
//! Each seed file is a previously-found crash that has been fixed; this test
//! ensures none of them re-introduce a panic or a non-finite output. It runs
//! on the stable toolchain as a plain `cargo test` — no nightly, no
//! `cargo fuzz` — so the seeds gate CI on every platform.
//!
//! The body is shared with the fuzz bin via `include!` of
//! `fuzz/fuzz_targets/fuzz_curves_core.rs` — single source of truth, so a seed
//! that crashed the fuzzer exercises the same code path here.
//!
//! To add a new seed: drop the (preferably `cargo fuzz tmin`-minimized) crash
//! file into `fuzz/regression/` — name it `fuzz_curves_<what>_zentone<issue>`.
//! Seeds are tiny (tens of bytes); anything over 8 KB stays in block storage.
//!
//! To replay a large external crash pile (e.g. a fuzz-farm mirror) in
//! addition to the committed seeds, set `ZENTONE_FUZZ_CRASH_DIR=<dir>`; every
//! regular file under it (recursively) is replayed too. This is opt-in from
//! the caller — the committed seeds always run.

use std::fs;
use std::path::{Path, PathBuf};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fuzz/fuzz_targets/fuzz_curves_core.rs"
));

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

/// Collect every regular file under `dir`, recursively, sorted for
/// deterministic failure messages.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
    out.sort();
}

/// The committed `fuzz/regression/` seeds, exercised through the fuzz body.
///
/// Fails if the regression directory is missing or empty — a harness with
/// nothing to replay is a silent pass, which is worse than a loud failure.
#[test]
fn fuzz_curves_regression_seeds_do_not_panic() {
    let dir = regression_dir();
    assert!(dir.is_dir(), "missing {}", dir.display());
    let mut seeds = Vec::new();
    collect_files(&dir, &mut seeds);
    assert!(!seeds.is_empty(), "no seeds in {}", dir.display());

    let mut replayed = 0usize;
    for seed in &seeds {
        let data = fs::read(seed).expect("read seed");
        assert!(
            data.len() <= 8 * 1024,
            "{} is {} bytes — regression seeds must be ≤ 8 KB (minimize with \
             `cargo fuzz tmin`, or keep it in block storage)",
            seed.display(),
            data.len()
        );
        run_fuzz_curves(&data);
        replayed += 1;
    }
    assert!(replayed > 0);

    // Optional extra pile — see module docs. Caller-controlled via env var.
    if let Ok(extra) = std::env::var("ZENTONE_FUZZ_CRASH_DIR") {
        let extra = PathBuf::from(extra);
        assert!(
            extra.is_dir(),
            "ZENTONE_FUZZ_CRASH_DIR={} is not a directory",
            extra.display()
        );
        let mut files = Vec::new();
        collect_files(&extra, &mut files);
        assert!(
            !files.is_empty(),
            "ZENTONE_FUZZ_CRASH_DIR={} is empty",
            extra.display()
        );
        // A farm pile holds thousands of inputs for a handful of causes. Replay
        // every one (catching the failure instead of stopping at the first) and
        // report a histogram keyed on entry point + curve variant, with one
        // example file per bucket — that is what turns 18,000 artifacts into
        // "Bt2446B, map_rgb and map_row" (zentone#25/#26).
        let quiet: Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync> = Box::new(|_| {});
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(quiet);
        let mut n = 0usize;
        let mut failures: Vec<(String, String)> = Vec::new();
        for f in &files {
            // Skip the farm's sidecar metadata; replay everything else.
            let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.ends_with(".json") || name.ends_with(".txt") {
                continue;
            }
            let data = fs::read(f).expect("read crash file");
            if let Err(payload) = std::panic::catch_unwind(|| run_fuzz_curves(&data)) {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                failures.push((f.display().to_string(), msg));
            }
            n += 1;
        }
        std::panic::set_hook(prev_hook);
        eprintln!(
            "replayed {n} external crash inputs from {} (plus {replayed} committed seeds)",
            extra.display()
        );
        if !failures.is_empty() {
            let mut hist: std::collections::BTreeMap<String, (usize, String, String)> =
                std::collections::BTreeMap::new();
            for (file, msg) in &failures {
                // "map_rgb produced non-finite output: variant 12, in [...]" ->
                // key "map_rgb / variant 12"
                let entry = msg.split_whitespace().next().unwrap_or("?").to_string();
                let variant = msg
                    .split("variant ")
                    .nth(1)
                    .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
                    .unwrap_or("?");
                let key = format!("{entry} / variant {variant}");
                let e = hist
                    .entry(key)
                    .or_insert_with(|| (0, file.clone(), msg.clone()));
                e.0 += 1;
            }
            eprintln!("{} of {n} external inputs failed:", failures.len());
            for (key, (count, file, msg)) in &hist {
                eprintln!("  {count:>7}  {key}\n           e.g. {file}\n           {msg}");
            }
            panic!(
                "{} of {n} external inputs from {} produced a panic / non-finite output (histogram above)",
                failures.len(),
                extra.display()
            );
        }
    }
}

/// The #25/#26 seeds select `Bt2446B` (`data[0] % 14 == 12`) — slot 12 has
/// held Bt2446B since Bt2446A moved to zenpixels-convert — and their first
/// pixel is the exact luminance that overflowed `y / breakpoint` to +Inf.
#[test]
fn zentone25_seeds_target_bt2446b() {
    for name in [
        "fuzz_curves_bt2446b_log_overflow_zentone25",
        "fuzz_curves_bt2446b_log_overflow_row_zentone26",
    ] {
        let data = fs::read(regression_dir().join(name)).expect("read seed");
        assert_eq!(FUZZ_CURVE_VARIANTS, 14, "variant table modulus changed");
        assert_eq!(
            data[0] % FUZZ_CURVE_VARIANTS,
            12,
            "{name} no longer selects Bt2446B"
        );
        // The fuzz body zeroes non-finite lanes before mapping; mirror it so the
        // pin exercises the pixel the farm did (#26's first pixel has NaN lanes
        // in the raw bytes; its overflow sits at index 63 of map_row).
        let curve = fuzz_curve_for_variant(12);
        let px = |i: usize| {
            let v = f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            if v.is_finite() { v } else { 0.0 }
        };
        let rgb = [px(1), px(5), px(9)];
        let out = curve.map_rgb(rgb);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{name}: in {rgb:?} -> out {out:?}"
        );
        run_fuzz_curves(&data);
    }
}

/// The #21 seed selects `ExtendedReinhard` (`data[0] % 14 == 1`). This pins
/// the variant table so a renumbering cannot silently retarget the seed to a
/// different curve and defuse the regression.
#[test]
fn zentone21_seed_targets_extended_reinhard() {
    let seed = regression_dir().join("fuzz_curves_extreme_reinhard_zentone21");
    let data = fs::read(&seed).expect("read #21 seed");
    assert_eq!(
        FUZZ_CURVE_VARIANTS, 14,
        "variant table modulus changed — see core docs"
    );
    assert_eq!(
        data[0] % FUZZ_CURVE_VARIANTS,
        1,
        "#21 seed no longer selects ExtendedReinhard"
    );
    let curve = fuzz_curve_for_variant(1);
    // The seed's own first pixel (huge finite luminance plus a large negative
    // channel) is the exact input that overflowed to ±Inf before 24c0690.
    let px = |i: usize| f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
    let rgb = [px(1), px(5), px(9)];
    assert!(
        rgb.iter().all(|v| v.is_finite()),
        "seed pixel not finite: {rgb:?}"
    );
    let out = curve.map_rgb(rgb);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "in {rgb:?} -> out {out:?}"
    );
}
