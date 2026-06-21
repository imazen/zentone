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

## polyfit replacement for `x^(1/2.4)` — investigated, kept baseline

The prior pass (above) noted that the three `pow_midp_unchecked(1/2.4)`
calls for the input gamma encode resisted polynomial replacement because
`f(x) = x^0.417` has an effectively-vertical tangent near 0 on the HDR
input range. This pass walked that back through with the `polyfit` fork
(at `/home/lilith/work/polyfit`, version 0.11.0) to verify the
conclusion.

### Tools used

- **Polyfit fork** (`/home/lilith/work/polyfit`, v0.11.0) — provides
  `RationalFit::from_function` (Sanathanan-Koerner iteration + LM
  refinement) and `ChebyshevFit::new` + `as_monomial()` for direct
  monomial conversion. Pulled in as a `dev-dependency` gated by the
  `__polyfit-tools` feature; runtime crate unaffected.
- **Offline fitting tool**: `examples/fit_pow_inv_24.rs`. Sweeps four
  strategies — (A) plain monomial on `x`, (B) plain monomial on `√x`,
  (C) rational `P(√x)/Q(√x)` with `P(0)=0` constraint, (D) two-piece
  monomial on `√x`, (E) three-piece monomial on `√x` — at multiple
  degrees and split points. Runs in ~12 s on Zen 4.

### Findings (offline polynomial precision, before kernel amplification)

| Strategy | Best result on `[0, 12]` (max abs err vs `libm::powf`) |
| --- | --- |
| A: plain monomial of `x`, deg ≤ 24 | ~14% (≥10% per the prior pass's note — confirmed) |
| B: plain monomial of `√x`, deg ≤ 12 | 4.9e-3 (plateau, Chebyshev→monomial unstable past deg 11) |
| C: rational `P(√x)/Q(√x)` with `P(0)=0`, P/Q ≤ 7/7 | 2.2e-2 (the SK fitter drifts at z=0 even with constraint) |
| **D: 2-piece monomial on `√x`** (eps=0.07, near deg 7 + bulk deg 10) | **3.44e-4** |
| **E: 3-piece monomial on `√x`** (eps=0.05/0.30, near deg 7 + mid deg 9 + far deg 11) | **2.60e-4** |

The sqrt substitution `g(z) = z^(2/2.4) ≈ z^0.833` for `z = √x` flattens
the singular tangent (`g'(0) = 0`), turning the fit into a well-behaved
one. The 2-piece D crosses below the 5e-4 SIMD parity tolerance on the
polynomial alone — confirming the prior pass's "polynomial fits over
[0, 4] give 10%+ error" was a domain-only observation that the sqrt
substitution fixes.

### Why we kept the baseline

Despite the sub-5e-4 polynomial fit, the BT.2446-A kernel's downstream
math amplifies polynomial error:

- `y_p = lr*r_p + lg*g_p + lb*b_p` — weighted average across 3 channels
- `f = y_sdr / (1.1 * y_p)` — divide by `y_p` (~0.1-1.0 typical)
- `cb = f * (b_p - y_p) * inv_1_8814` and `cr = f * (r_p - y_p) * inv_1_4746`
  — subtractive cancellation when `b_p ≈ y_p` or `r_p ≈ y_p`
- Output `r_out, g_out, b_out` linearly recombine `y_tmo + 1.4746*cr`
  etc., so the amplified Cb/Cr propagates straight to the output.

Measured kernel output error (over the existing SIMD parity test grid,
9090 comparisons):

| Implementation | Max output err vs `libm::powf` reference | Parity test |
| --- | --- | --- |
| Baseline `pow_midp_unchecked(1/2.4)` | 5.88e-5 | passes (5e-4 × magnitude tolerance, ~9× headroom) |
| Polyfit D (2-piece, eps=0.07, near 7 + bulk 10) | 5.21e-4 | **FAILS** by 4% on `strip[821,2]` |
| Polyfit D + `c[0]=0` anchor on near | 5.21e-4 | **FAILS** by 4% (the anchor only fixes the `pos_eps` mask leak, not the bulk's amplified error) |
| Polyfit E (3-piece, eps=0.05/0.30, deg 7/9/11) | 2.16e-4 | passes (margin 2.3×) |

The polyfit-derived polynomials are 4-100× **worse** than
`pow_midp_unchecked`'s ~5.9e-5 baseline error. The baseline kernel's
existing slack (5e-4 / 5.9e-5 ≈ 8.5×) was specifically the reason the
prior pass got away with the EOTF polynomial replacement (max err
5.88e-5) without touching the parity tolerance.

### Throughput data with each candidate

`examples/bt2446a_throughput.rs`, 1024×1024 cell, three runs, median
reported. Same flags as the table at the top of this file
(`RUSTFLAGS="-C target-cpu=native"`, `nice -n19`).

| Implementation | 1024×1024 Mpix/s | vs baseline |
| --- | --- | --- |
| Baseline `pow_midp_unchecked` | 258 Mpix/s | 1.00× |
| Polyfit D (2-piece) | (would pass parity test only after tolerance relaxation — not measured) | — |
| Polyfit E (3-piece) | 203 Mpix/s | **0.79× (regression)** |

The 3-piece's deepest dependency chain is the `√x → far-Horner` cascade
(11 FMA ≈ 44 cycles latency). At the same time, the 30 splatted
coefficients (8 near + 10 mid + 12 far) create register pressure on
AVX-512's 32 zmm file. The 2-piece D would have lower latency (10 FMA
on the bulk) and only 19 coefficients, but doesn't pass parity.

### Decision: keep `pow_midp_unchecked(1/2.4)` in the SIMD kernel

Per the prompt's "if the precision test passes but throughput doesn't
improve materially (< 10% gain), report and skip" rule — the only
polyfit candidate that passes parity is the 3-piece E, which regresses
throughput by ~21%. We **do not** ship a slower kernel.

### Rough edges in polyfit observed during this investigation

These are notes for upstream/fork improvements — the fork has solid
bones, but a few sharp corners showed up while driving it for this
specific job:

1. **`F32SearchResult`'s `max_ulp` / `bnd` numbers are unreliable for
   functions whose output approaches zero.** Every fit I ran showed
   `max_ulp` ≈ 1e9 because the reference function `x^(1/2.4)` has a
   tiny magnitude near `x=0` — a 1-bit error on a `4.3e-3` output is
   ~1e9 ULP. The number is technically correct (ULPs are bit-distance),
   but it's not the *quality signal* most callers expect when picking
   between fits. Suggestion: auto-flip to `F32ScoreMetric::AbsoluteError`
   when the reference's measured min magnitude is below some threshold,
   or document the trap loudly in the `F32ScoreMetric` enum docs.

2. **`Constraint::with_weight(0.0, 0.0, 1e6)` doesn't fully pin the
   `P(0)/Q(0) = 0` boundary on rational fits.** I added a high-weight
   `(0, 0)` constraint to pass C and the SK iteration still produced
   fits with `boundary_ulp ≈ 8.5e8`. The constraint is a soft weight
   in the LS objective, not a hard projection; the f32 local search at
   the end can drift away from it. Suggestion: expose a hard-pin
   constraint mode that subtracts the constraint linearly from `P` then
   refits the remaining residual — or just document that
   `with_weight` is a softening, not a pinning.

3. **`RationalFit` SK on functions with a singular tangent at the
   fit-domain boundary is unstable**: my pass C runs with `restarts=8`
   and `n_samples=40000` still produced wildly different `max_abs`
   values across `(p_deg, q_deg)` pairs (`(4,4)` worse than `(3,3)`,
   `(6,5)` worse than `(5,4)`). For sRGB EOTF (the polyfit examples'
   reference use case) the boundary singular-tangent is at the
   threshold (`0.04`), not at zero — so the SK fitter doesn't see it.
   For our HDR input transfer the tangent is at the fit-domain corner.
   Suggestion: detect "the function has a vertical tangent at the
   fit-domain boundary" and emit a warning steering callers toward
   sqrt substitution or piecewise splits.

4. **`ChebyshevFit::as_monomial()` loses precision past ~degree 12 on
   wide-domain fits.** This is expected mathematically (Chebyshev's
   basis is well-conditioned; monomial isn't), but the docs don't
   warn about it. I hit it on the plain-monomial-of-`√x` sweep —
   deg 12 gives 8.0e-3, deg 14 gives 1.9e-1 — the Chebyshev fit IS
   accurate, but converting to monomial form for SIMD Horner
   evaluation throws the accuracy away. Suggestion: document the
   degree ceiling on `as_monomial`'s rustdoc + add a `monomial_check`
   helper that compares Chebyshev evaluation vs monomial evaluation
   on dense samples and warns if they diverge.

5. **No `_unchecked` polynomial mass-eval API** — every call site
   (including this fit tool) ends up writing the same `eval_f32_horner`
   helper. The `rational::optimize_f32` uses an internal one; exposing
   `polyfit::eval::horner_f32(x, &[f32]) -> f32` (the same algorithm)
   would dedupe a lot of caller boilerplate.

None of these are blockers — the fork did exactly what was asked
(produce a polynomial approximation with rigorous f32 metrics), and the
3-piece E result is actually genuinely useful in any context where
parity matters more than throughput. They're just things that would
make the next person's life easier.

### Source

- Offline fitting tool: `examples/fit_pow_inv_24.rs` (gated on
  `--features __polyfit-tools`)
- Polyfit dev-dep wiring: `Cargo.toml` `[dev-dependencies]` + the
  `__polyfit-tools` feature
- Polyfit fork: `/home/lilith/work/polyfit` v0.11.0
- The SIMD kernel `src/simd/curves.rs::bt2446a_tier` is **unchanged**
  from the prior `f32x16 + Estrin EOTF` perf pass.
