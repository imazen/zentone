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
├── libultrahdr_luminance.cpp       libultrahdr per-gamut luminance dot products
├── libultrahdr_gamut.cpp           libultrahdr 6 gamut-conversion matrices
├── libultrahdr_hlg_ootf.cpp        libultrahdr HLG OOTF / inverse OOTF
├── libavif_apply_gain.cpp          libavif gain-map weight + apply
├── libplacebo_bt2390.cpp           libplacebo BT.2390 EETF
├── darktable_filmic.cpp            darktable filmic spline
├── aces2_ocio_reference.py         ACES 2.0 stages via OpenColorIO >= 2.4 built-ins (Python)
├── aces_ocio_encodings_reference.py ACES display encodings, ACEScct/cc, RGC, CDL via OpenColorIO (Python)
└── golden/
    ├── libultrahdr_reinhard.csv
    ├── libultrahdr_apply_gain.csv
    ├── libultrahdr_compute_gain.csv
    ├── libultrahdr_luminance.csv
    ├── libultrahdr_gamut.csv
    ├── libultrahdr_hlg_ootf.csv
    ├── libavif_apply_gain.csv
    ├── libplacebo_bt2390.csv
    ├── darktable_filmic.csv
    ├── aces2_ocio_jmh.csv                   RGB -> Hellwig JMh, four input primaries
    ├── aces2_ocio_tonescale_compress.csv    JMh -> tonescaled + chroma-compressed JMh
    ├── aces2_ocio_gamut_compress.csv        JMh -> gamut-compressed JMh
    ├── aces2_ocio_output_transform.csv      AP0 -> display-linear RGB (forward)
    ├── aces2_ocio_output_transform_inv.csv  display-linear RGB -> AP0 (inverse)
    ├── aces2_ocio_display_encoding.csv      AP0 -> display code values, 9 presets (OT + display encoding)
    ├── aces2_ocio_display_encoding_inv.csv  display code values -> AP0 (inverse presets)
    ├── aces_ocio_acescct.csv                ACEScct / ACEScc log encodings
    ├── aces_ocio_rgc13.csv                  ACES 1.3 Reference Gamut Compression fwd/inv
    └── aces_ocio_cdl.csv                    ASC CDL (ASC + no-clamp styles) fwd/inv
```

The ACES 2.0 goldens are the exception to the "C++ excerpt" rule: the
reference there is the Academy's CTL (`aces-core/lib/Lib.Academy.*.ctl`),
and OpenColorIO's ACES 2.0 fixed functions are a maintained port of it that
exposes every stage. `aces2_ocio_reference.py` drives them through a raw
OCIO config (`uv pip install opencolorio numpy`; deterministic sample set).
`aces_ocio_encodings_reference.py` does the same for the pieces around the
rendering: the display-encoding presets (OCIO's `ACES-OUTPUT - … _2.0`
builtins composed with its `DISPLAY - …` builtins), ACEScct / ACEScc,
the ACES 1.3 Reference Gamut Compression fixed function and the ASC CDL.
The consuming tests live in `src/experimental/aces2.rs`,
`aces2_display.rs`, `aces2_simd.rs` and `aces_encodings.rs` (unit tests, so
they can reach the private stage functions) and run with
`cargo test --features experimental`. Known reference limits: OCIO's HLG
display is LUT-based and decodes code values below ~0.005 differently from
the CTL formula (the inverse test skips those three rows), and OCIO's
output-transform fixed function omits the CTL's AP1 clamp (the forward
tests compare in-AP1 rows only).

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
| libultrahdr | `srgbLuminance` / `p3Luminance` / `bt2100Luminance` | `lib/src/gainmapmath.cpp:94/164/195` | `8cbc983` |
| libultrahdr | 6 gamut-conversion matrices (`bt709↔p3`, `bt709↔bt2100`, `p3↔bt2100`) | `lib/src/gainmapmath.cpp:609-633` | `8cbc983` |
| libultrahdr | `hlgOotf` / `hlgInverseOotf` (BT.2100 luminance, parameterized gamma) | `lib/src/gainmapmath.cpp:294/304` | `8cbc983` |
| libavif | `avifGetGainMapWeight` + `avifApplyGainPixel` | upstream extraction | (see `libavif_apply_gain.cpp`) |
| libplacebo | BT.2390 EETF (PQ + scene-linear) | upstream extraction | (see `libplacebo_bt2390.cpp`) |
| darktable | filmic spline | upstream extraction | (see `darktable_filmic.cpp`) |
| ACES 2.0 (via OpenColorIO 2.5.2) | `RGB_to_JMh`, tonescale + chroma compression, gamut compression, full output transform fwd/inv | `aces-core/lib/Lib.Academy.OutputTransform.ctl`, OCIO `ops/fixedfunction/ACES2/` | OCIO tag `v2.5.2` |
| ACES display encodings (via OpenColorIO 2.5.2) | `Output.Academy.*` presets: peak clamp, D60 white scaling, encoding primaries, sRGB / gamma 2.2 / 2.6 / BT.1886 / ST 2084 / HLG inverse EOTFs, fwd/inv | `aces-core/lib/Lib.Academy.DisplayEncoding.ctl`, `aces-output/*.ctl`, OCIO `ACES-OUTPUT - … _2.0` + `DISPLAY - CIE-XYZ-D65_to_…` builtins | OCIO tag `v2.5.2` |
| ACEScct / ACEScc, ACES 1.3 RGC, ASC CDL (via OpenColorIO 2.5.2) | log encode/decode, `FIXED_FUNCTION_ACES_GAMUT_COMP_13`, `CDLTransform` ASC + no-clamp | S-2016-001, S-2014-003, `LMT.Academy.ReferenceGamutCompress.ctl`, ASC CDL v1.2 | OCIO tag `v2.5.2` |

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
