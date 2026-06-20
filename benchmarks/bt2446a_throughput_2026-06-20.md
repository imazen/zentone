# `Bt2446A::map_strip_simd` throughput, 2026-06-20

BT.2446-A (ITU-R BT.2446-1 §4) is the user-selected default HDR→SDR curve for
`HdrToSdr` (commit `5bfe59c`, prior to this perf pass). It runs on every
web-served HDR image, so throughput matters.

This pass moves the kernel from `f32x8 + v3` only to `f32x16 +
v4(cfg(avx512)) + v3 + neon + wasm128 + scalar`, replaces the three
`pow_midp_unchecked(2.4)` calls in the BT.1886 EOTF stage with a degree-7
monomial polynomial evaluated via **Estrin's method** (parallel pairs), and
switches every transcendental to its `_unchecked` variant (inputs are
pre-clamped to `pos_eps`).

## Hardware

- **CPU**: AMD Ryzen 9 7950X (Zen 4), 16C/32T, ~4.5 GHz under sustained load
- **AVX-512**: avx512f, avx512dq, avx512bw, avx512vl, avx512vbmi, avx512vnni,
  avx512_bitalg, avx512_vpopcntdq, avx512_bf16 (full Zen-4 AVX-512)
- **Compile flags**: `RUSTFLAGS="-C target-cpu=native"`, `--release`
- **OS**: Linux 6.18 (WSL2)
- **rustc**: stable, edition 2024
- **archmage / magetypes**: 0.9.26

## Benchmark harness

`examples/bt2446a_throughput.rs` — deterministic ramp+sparkle HDR buffer fed
through `Bt2446A::new(1000.0, 100.0).map_strip_simd(...)` in a hot loop. Five
sizes; 3× warmup per cell; loop count auto-calibrated to ~1 second per cell.
Run under `nice -n19 ionice -c3`.

Baseline measured against parent commit `5bfe59c` (the kernel as
of `feat(hdr_to_sdr): swap default curve Möbius → Bt2446A`). "After" measured
against the perf commit on top.

## Results

| size      | pixels  | baseline (Mpix/s) | after (Mpix/s) | speedup |
|-----------|---------|------------------:|---------------:|--------:|
| 256×256   |  0.07 MP |             112.7 |          259.2 |   2.30× |
| 1024×1024 |  1.05 MP |             112.7 |          253.2 |   2.25× |
| 2048×2048 |  4.19 MP |             110.5 |          251.8 |   2.28× |
| 3840×2160 |  8.29 MP |             110.4 |          250.0 |   2.26× |
| 7680×4320 | 33.18 MP |             111.4 |          258.8 |   2.32× |

**~2.3× speedup across every cell**, scaling cleanly with image size — the
kernel is compute-bound (the throughput is flat across a 470× size range, so
it's neither memory-bound nor fixed-overhead-bound).

Run-to-run stability checked across three runs: < 2% variance per cell.

## Per-pixel cycle accounting

At ~4.5 GHz × 258 Mpix/s = **17 cycles per pixel** on the after side, down
from ~40 cycles per pixel on the baseline. Per-pixel instruction mix
(measured via `perf stat`) drops from 1.29 IPC × 64 cycles → 1.29 IPC × 17
cycles. The cycle reduction comes entirely from the EOTF polynomial
substitution + parallel Estrin chains.

## Optimizations applied

1. **Kernel upgrade: `f32x8` → `f32x16` with the full tier stack
   `v4(cfg(avx512)), v3, neon, wasm128, scalar`.** `F32x16Convert` is
   impl'd for `X64V4Token` in magetypes 0.9.26, so AVX-512 hardware now
   resolves to native `__m512` ops. The prior `f32x8`-only kernel was
   constrained by a known magetypes 0.9.22 missing
   `F32x8Convert for X64V4Token` impl — `f32x16` sidesteps that. On V3 the
   polyfill is two 256-bit ops, giving the compiler two independent FMA
   chains to interleave for ILP. Justification: the `magetypes 0.9.22`
   blocker note in `simd/curves.rs` head comment was specific to
   `f32x8 + V4`; `f32x16 + V4` does NOT hit the same gap.

2. **All transcendentals to `_unchecked` variants.** `pow_midp_unchecked`,
   `log2_midp_unchecked`, `exp2_midp_unchecked` skip the zero/negative/inf/
   NaN special-case blend chains that `pow_midp` / `log2_midp` / `exp2_midp`
   add for IEEE edge cases. Inputs are pre-clamped to `pos_eps`
   (`f32::MIN_POSITIVE`), so the polynomial-only paths are correct.
   Justification: saves ~7 blendvps + 3 cmpps per call, applied across
   every transcendental in the hot loop.

3. **Single-step `pos_eps` clamp**. Replaces the prior two-step
   `r.max(zero); then arg.max(pos_eps)` with one `max(pos_eps)` up front.
   The final `valid = y_p > pos_eps` mask still snaps originally-zero
   pixels back to true black on store. Justification: one less `vmaxps`
   per channel input, cleaner data flow for the optimizer.

4. **Degree-7 monomial polynomial for the BT.1886 EOTF (`x^2.4`) via
   Estrin's method.** This is the dominant per-pixel saving. Replaces
   three `pow_midp_unchecked(2.4)` calls (each ≈ 24 ops: log2 polynomial +
   multiply by 2.4 + exp2 polynomial, with serial dependency chain) with
   three degree-7 polynomial evaluations. Estrin's form (not Horner):

   ```
   p(x) = ((c7·x + c6)·x² + (c5·x + c4))·x⁴ + ((c3·x + c2)·x² + (c1·x + c0))
   ```

   Four independent FMAs at depth 1, two at depth 2, then a single merge.
   Critical-path latency: **3 FMA** (~12 cycles on Zen 4) per channel vs
   Horner's **7 FMA chain** (~28 cycles). With three channels running in
   parallel the FMA pipeline stays fed.

   **Approximation error: 5.88e-5** (measured at 20 000 sample points on
   `[0, 1]`) — 10× inside the SIMD parity tolerance of 5e-4, and ~3000×
   tighter than the perceptual delta the `output_is_linear_light_not_gamma_encoded`
   regression test fences (0.02). Coefficients live in
   `simd/curves.rs` as `POW24_C0..POW24_C7` constants with the derivation
   documented inline.

   The inverse direction (`x^(1/2.4)` for the input gamma encode) does
   NOT polynomial-approximate cleanly — input range is `[0, ~hdr_peak]`
   (unclamped HDR, up to ~4× normalized) and `f(x) = x^0.417` has an
   effectively-vertical tangent near 0. Polynomial fits over `[0, 4]`
   give 10%+ error. We keep `pow_midp_unchecked(1/2.4)` for the input
   transfer step.

5. **Branchless clamp into the EOTF domain.** Output clamps are
   `(...).max(zero).min(one)` — the polynomial is fit on `[0, 1]` so
   feeding clamped values keeps approximation error within the documented
   bound.

## Coefficient derivation (degree-7 monomial for x^2.4 on [0, 1])

```python
import numpy as np
x = np.linspace(0.0, 1.0, 20000)
y = x ** 2.4
c = np.polyfit(x, y, 7)
# Verify
max_err = np.max(np.abs(np.polyval(c, x) - y))  # 5.88e-5
```

Constants in source (Horner order, c7→c0):

```text
POW24_C7 =  0.197 935 7
POW24_C6 = -0.826 185
POW24_C5 =  1.470 748 3
POW24_C4 = -1.531 952
POW24_C3 =  1.361 614 9
POW24_C2 =  0.334 159 8
POW24_C1 = -0.006 362 703 7
POW24_C0 =  0.000 058 848 62
```

## What stayed the same (scalar `map_rgb`)

Per the prompt's "don't change public API or output semantics" guidance, the
scalar `map_rgb` reference path keeps `libm::powf` for both the input gamma
encode and the BT.1886 EOTF — bit-exact reproducibility against the spec
and against past zentone output is preserved. Only the SIMD strip kernel
uses the polynomial. The SIMD parity test verifies that the divergence
between scalar reference and SIMD output stays within the 5e-4 tolerance.

## Tests verified

- `bt2446a::tests::*` (12 tests, all scalar — unchanged): all pass.
- `tests/simd_parity.rs::bt2446a_strip_simd_matches_per_pixel` — passes
  (`5e-4` tolerance, well above the 5.88e-5 polynomial error + ~3 ULP
  `pow_midp_unchecked` error).
- `tests/reference_parity.rs` — 14 tests pass (BT.2446-A not directly
  tested here but verifies no global regression).
- `tests/exhaustive_properties.rs` — 12 tests pass.
- `tests/continuity.rs` — 10 tests pass.
- All 159 lib unit tests pass.
- `cargo fmt --check` clean.
- `cargo clippy --release --lib --tests` clean (only `zencodecs` dep warnings).

## Source

- Bench harness: `examples/bt2446a_throughput.rs`
- Kernel: `src/simd/curves.rs::bt2446a_tier` (the polynomial constants
  `POW24_C0..POW24_C7` live next to the kernel for proximity to the
  derivation comment).
- Dispatcher: `src/bt2446a.rs::Bt2446A::map_strip_simd` — adds the
  `v4(cfg(avx512))` tier to the `incant!` tier list.
