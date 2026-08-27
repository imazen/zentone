#![no_main]
//! Fuzz all ToneMapCurve variants and the ToneMap trait methods.
//!
//! The body lives in `fuzz_curves_core.rs` so that `tests/fuzz_regression.rs`
//! replays `fuzz/regression/` seeds through EXACTLY the same code on the
//! stable toolchain. Edit the core file, not this shim.

use libfuzzer_sys::fuzz_target;

include!("fuzz_curves_core.rs");

fuzz_target!(|data: &[u8]| {
    run_fuzz_curves(data);
});
