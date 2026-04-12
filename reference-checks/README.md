# Reference-implementation parity checks

This directory contains standalone C/C++ extractions of the reference math
from authoritative sources, compiled into small fixture-producing programs.
The outputs of those programs are committed as **golden files** under
`golden/`; Rust integration tests under `../tests/reference_parity.rs`
assert that zentone's own implementations reproduce those values within a
tight tolerance.

## Why golden files instead of runtime linking

- **Reproducible.** The fixture values are deterministic text files under
  version control. Anyone reviewing a diff can see exactly what zentone is
  being held to.
- **No build-time dependency.** Zentone itself has no C/C++ toolchain
  requirement. Only regenerating the golden files needs `g++` (or `clang++`).
- **Readable.** The C++ files are < 100 lines each and contain verbatim
  excerpts of the reference source with URLs and commit hashes. Anyone can
  audit them without installing libultrahdr etc.

## Layout

```
reference-checks/
├── README.md                       this file
├── build.sh                        regenerates all golden files
├── libultrahdr_reinhard.cpp        libultrahdr extended-Reinhard excerpt
├── libultrahdr_apply_gain.cpp      libultrahdr applyGain excerpt
├── libultrahdr_compute_gain.cpp    libultrahdr computeGain excerpt (encode side)
└── golden/
    ├── libultrahdr_reinhard.csv
    ├── libultrahdr_apply_gain.csv
    └── libultrahdr_compute_gain.csv
```

Each `.cpp` file starts with a header block recording:

1. Source file path (relative to the reference repo root).
2. Upstream commit hash at the time of extraction.
3. Permalink URL.
4. License (libultrahdr is Apache-2.0).

## Sources currently covered

| Reference | Function | Source file | Commit |
|---|---|---|---|
| libultrahdr | `ReinhardMap` / `globalTonemap` | `lib/src/jpegr.cpp:1817-1849` | `8cbc983` |
| libultrahdr | `applyGain(Color e, float gain, metadata)` single-channel | `lib/src/gainmapmath.cpp:797-803` | `8cbc983` |
| libultrahdr | `applyGain(Color e, Color gain, metadata)` per-channel | `lib/src/gainmapmath.cpp:818-834` | `8cbc983` |
| libultrahdr | `computeGain` encode side | `lib/src/gainmapmath.cpp:779-788` | `8cbc983` |

## Regenerating the golden files

```
cd reference-checks
./build.sh
```

After regeneration, run the Rust tests:

```
cargo test --test reference_parity -- --nocapture
```

If a golden file changed, inspect the diff carefully — either upstream
changed its math (rare; worth a CLAUDE.md note and a commit) or you
introduced a regression in the extraction program (most likely — audit
the diff line by line).

## Tolerance

The Rust tests use `abs_err < 1e-6` for f32 comparisons. That is ~8× the
f32 ulp at 1.0 and should absorb one or two fused-multiply-add reorderings
but not a genuine formula change. If a test fails with `abs_err ≈ 2e-7`,
the formula is identical and the difference is floating-point
associativity — tighten or loosen the bound deliberately.

## Non-goals

- **Not a benchmark.** Comparing speed would require linking libultrahdr
  properly and matching build flags, which is way out of scope here.
- **Not a fuzzer.** Test vectors are fixed. If a fuzzed diff turns up,
  capture the inputs and add them to the fixed set.
- **Not the only correctness check.** The `tests/` directory has Rust-only
  unit and property tests for self-consistency. This directory cross-checks
  against external ground truth.
