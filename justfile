# Regenerate the public-API surface snapshots (docs/public-api/)
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo metadata --locked --format-version 1 --manifest-path fuzz/Cargo.toml > /dev/null
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast
