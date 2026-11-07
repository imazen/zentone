
# Stable SIMD qualification for the frozen consumer revision.
stable-cohort-check:
    cargo fmt --check
    cargo test --lib --tests
    cargo clippy --lib --tests -- -D warnings
