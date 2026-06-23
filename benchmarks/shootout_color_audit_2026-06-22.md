# HDR→SDR Shootout Color-Handling Audit (2026-06-22)

Audit of `examples/hdr_tone_map_shootout_full.rs` (the source that produced
`hdr_tone_map_shootout_full_2026-06-20.csv`) for color-profile correctness
before regenerating the shootout with the recent Bt2446A fix and the new
percentile-based peak method.

## 1. Producer SDR decode path

`hdr_tone_map_shootout_full.rs:189-195` calls `DecodeRequest::new(bytes)
.decode_full_frame()`. The returned `PixelDescriptor` is then linearized
into f32 RGB **in source primaries** (line 222-260: `pixel_buffer_to_linear_rgb`
sets the target descriptor to `(F32, Rgb, Linear, src_desc.primaries)` —
linearizes the transfer but keeps the gamut as-is).

Observed for 3 spot-check samples (via `examples/audit_primaries_dump.rs`):

| Sample | SDR descriptor returned |
|---|---|
| `1526_…ip13pro_…heic` | `U8 / Rgb / Srgb / **Bt709**` |
| `1227_…zfold7_…jpg` (UltraHDR) | `U8 / Rgb / Srgb / **DisplayP3**` |
| `1246_…ip13pro_…heic` | `U8 / Rgb / Srgb / **Bt709**` |

**Finding**: iPhone HEIC SDR is being decoded into BT.709 primaries (the HEIC
codec normalizes to BT.709 sRGB before returning the base). zfold7 UltraHDR
JPEG returns DisplayP3 sRGB. The shootout linearizes each in its source
primaries (correct), so the SDR-side gamut is preserved.

## 2. HDR reconstruction path

Same file, `decode_sample_full` lines 198-205: `DecodeRequest::new(bytes)
.reconstruct_hdr(None).decode_full_frame()`. Result is then put through
`pixel_buffer_to_linear_rgb` in **source primaries**.

Observed:

| Sample | HDR descriptor returned |
|---|---|
| All 3 spot-checks | `F32 / Rgba / Linear / **Bt709**` |

**Finding**: The HDR-reconstruction path normalizes to **BT.709 linear** for
all three cameras (HEIC + UltraHDR JPEG). This is the codec's choice; the
gain-map reconstruction lands in BT.709 regardless of producer SDR gamut.

## 3. Tone-mapping invocation

`apply_curve` (line 397) currently takes the LinearRgb buffer **as-is** and
passes it directly to `Bt2446A::map_strip_simd`. No primaries conversion is
performed before the curve.

**Bt2446A's contract** (per `zenpixels-convert/src/hdr/bt2446a.rs:33-37`,
quoted verbatim):

> Input is linear-light **BT.2020 RGB** normalized so `1.0 = hdr_peak_nits`.
> Output is linear-light **BT.2020 RGB** normalized so `1.0 = sdr_peak_nits`.

The curve uses BT.2020 luma weights `(0.2627, 0.6780, 0.0593)`.

**Finding (bug)**: The shootout is feeding **BT.709 linear** input into a
curve specified for **BT.2020 linear**. This is the primary correctness gap
the audit was meant to find. Same gap applies to Bt2408 (BT.2020 native),
Bt2446B (BT.2020 native), Bt2446C (BT.2020 native), and Bt2390 (BT.2020 native).
Möbius / Narkowicz / HableFilmic / AcesAp1 are channel-independent so are
unaffected by primaries on input, but their output is in whatever primaries
the input was, so the **output side** of those also needs gamut routing
before scoring.

The numerical impact is modest because BT.2020 and BT.709 luma weights agree
within ~5-6% for typical content, but it is a real systematic bias and is
exactly the kind of thing the audit was tasked to find.

## 4. Post-tone-map handling

After Bt2446A's `map_strip_simd`, the output is treated as already-final
linear RGB in whatever primaries were on input (BT.709 in current runs).
There is no second `convert_to` to put the result into a known frame for
scoring.

**Finding**: For curves that operate in BT.2020 (Bt2446A/B/C, Bt2408, Bt2390)
the output is **claimed to be BT.2020** by the curve's contract, but is
being **treated as BT.709** by the scoring path. The two primaries differ
non-trivially in saturated regions (the user's flagged `1526` pink-flowers
sample is exactly such a region). This compounds with Phase 3's input bug.

`SoftCompress` (the OKLch chroma-rolloff knee gamut step) is wired only for
Möbius via `soft_clip_knee_strip(strip, knee_gamut)` at line 441 — it is
NOT applied to Bt2446A's output for the gamut-rotation step. Out-of-gamut
clamping after the curve is just `v.max(0.0).min(1.0)` (line 572). Channel
clamping in RGB before gamma-encoded scoring throws away saturation
recoverable via OKLch rolloff.

## 5. ΔE2000 computation

Lines 583-722:

1. `to_srgb_u8_buffer(linear)` — applies `linear_srgb::tf::linear_to_srgb`
   gamma to a clamped [0,1] linear buffer and stores u8.
2. `compute_de2000` re-decodes the u8 (`/255.0` then `clamp(0,1)` lol — the
   linear→srgb→linear round trip is lossy in 8-bit but is the same on both
   sides) then matrices into XYZ via the **BT.709→XYZ D65 matrix**
   `(0.4124564, 0.3575761, 0.1804375, …)` (lines 696-698) and into Lab D65
   via the standard `f(t)=t^(1/3)` formula.
3. CIE2000 ΔE in `delta_e2000`.

**Finding**: Both reference (producer SDR) and candidate (our tone-mapped)
are converted to Lab D65 through the **same BT.709→XYZ matrix**. This is
the bug that matters most for cross-sample comparisons:

- Producer SDR from zfold7 is **DisplayP3** linear (the codec returned
  DisplayP3). Treating those DisplayP3 RGB triplets as BT.709 when matrixing
  to XYZ understates Lab `a` (red) and overstates `L` (green-luma) for
  saturated greens/reds — about 6-10 dE2000 of systematic error in a
  fully-saturated red.
- Candidate is BT.709-linear from a BT.709→BT.2020-bug-mismatched curve;
  treating it as BT.709 in the XYZ step matches its primaries label but
  not its true gamut frame.

The two error sources are different (gamut mismatch on the producer side,
curve-spec mismatch on the candidate side). They are scored against each
other through a single matrix, so the ΔE2000 value is a mix of both biases.

## 6. Source primaries identification

Spot-check confirms the codec descriptor is being trusted (no manual
override). The codec is the source of truth for primaries; based on the
3-sample spot-check:

- **iPhone HEIC**: decoder converts to BT.709 sRGB before returning. Apple's
  HEIC container is DisplayP3-tagged on disk, but the in-tree HEIC decoder
  normalizes to BT.709 (likely a transcoder choice at decode time). Result:
  iPhone SDR base lands in BT.709 in our pipeline.
- **zfold7 UltraHDR JPEG**: decoder preserves DisplayP3 tagging — the SDR
  base lands in DisplayP3 linear after `pixel_buffer_to_linear_rgb`.
- **All HDR (gain-map reconstruction)**: lands in BT.709 linear regardless
  of producer (codec normalizes during reconstruction).

This is the codec's behavior, not our pipeline's. We do not need to "fix"
the primaries reported by the codec — they are accurate; we just need to
route the buffers through the right conversions before each algorithm.

---

## Summary of bugs found

1. **HDR input to Bt2446A/B/C/Bt2408/Bt2390 is BT.709**, not BT.2020 as the
   curves specify. *Severity: real systematic bias of a few dE2000 on
   saturated content.*
2. **Curve output for the BT.2020-native curves is BT.2020** (per their
   contract) but is **treated as BT.709** through the rest of the scoring
   pipeline. *Severity: same as #1, compounds.*
3. **ΔE2000 scoring matrix assumes BT.709 → XYZ D65** but producer SDR is
   sometimes DisplayP3 (zfold7) and sometimes BT.709 (iPhone). *Severity:
   high (~6-10 dE2000 systematic on saturated samples from zfold7) —
   different samples are being scored against different ground truths
   without normalization.*
4. **No OKLch chroma rolloff between BT.2020-output curves and BT.709/sRGB
   scoring frame**. Hard `clamp(0,1)` discards saturation that a knee
   could have preserved. *Severity: modest on this corpus (most pixels are
   in-gamut) but matters for the pink-flowers / specular-highlight subset.*

## Phase-2 fix plan (minimal-correctness pipeline)

1. After `pixel_buffer_to_linear_rgb` for HDR, call `.convert_to(target)`
   with `target = PixelDescriptor::new_full(F32, Rgb, None, Linear, Bt2020)`.
   Curve consumes linear BT.2020 throughout.
2. After the curve, the result is linear BT.2020 (target-peak normalized).
   Convert it to linear BT.709 with `convert_to`. Out-of-gamut values get
   `clamp(0,1)` since `SoftCompress` is not wired into this path — note
   this as a TODO and use clamp for now (clip-clamp is the de facto
   shipping default).
3. For comparison frame: convert **both** producer SDR (which may be in
   BT.709 *or* DisplayP3) and our candidate (BT.709 after step 2) to
   **OKLab**, a perceptually uniform space that is primary-agnostic for
   distance computations. ΔE in OKLab (typically `ΔEok` = euclidean
   distance) is used for the comparison; we also report the legacy
   ΔE2000-via-BT.709-XYZ-D65-Lab number for continuity with the
   2026-06-20 CSV.

The fix is surgical: insert one `convert_to(target_2020)` before the curve,
one `convert_to(target_709)` after, and replace the `linear_rgb_to_lab`
function with a primary-aware variant that routes to OKLab.

## Notes & open questions

- `with_orientation(OrientationHint::Correct)` is set for both the SDR and
  HDR decode (line 189-200). Both decodes therefore align in display space.
  Verified — no regression here.
- The `1526` "pink flowers" sample is iP13Pro HEIC → SDR is BT.709, HDR is
  BT.709. The audit findings apply but the gamut-mismatch contribution is
  smaller for this sample than for the zfold7 / Samsung samples (no
  DisplayP3 mismatch on the producer side).
- The 2026-06-20 CSV is being kept on disk and marked stale in the new
  findings file; the new CSV `hdr_tone_map_shootout_full_2026-06-22.csv`
  carries a `color_handling_version = "2026-06-22-audited"` column so the
  two can be merged for ranking-change analysis.
