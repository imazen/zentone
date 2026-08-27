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
        let mut n = 0usize;
        for f in &files {
            // Skip the farm's sidecar metadata; replay everything else.
            let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.ends_with(".json") || name.ends_with(".txt") {
                continue;
            }
            let data = fs::read(f).expect("read crash file");
            run_fuzz_curves(&data);
            n += 1;
        }
        eprintln!(
            "replayed {n} external crash inputs from {} (plus {replayed} committed seeds)",
            extra.display()
        );
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
