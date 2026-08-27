# zentone dev/shootout

Dev-only HDR→SDR shootout drivers and offline coefficient-fitting tools for
zentone. **Not published, not a workspace member, not built by CI.**

These binaries need unpublished sibling checkouts (`../zenpixels`,
`../zenpipe`, `../../polyfit` relative to the zentone repo) via `path` deps and
a `[patch.crates-io]` table (see `Cargo.toml`). Keeping them out of the root
manifest is what lets the published crate resolve from crates.io alone.

Run from the zentone repo root:

```bash
nice -n19 cargo run --manifest-path dev/shootout/Cargo.toml --release \
  --bin hdr_tone_map_shootout_audited
```

Binaries (formerly `examples/*.rs` behind the root `hdr-shootout` /
`__polyfit-tools` features):

| bin | purpose |
|---|---|
| `hdr_tone_map_shootout` | cross-curve shootout on real UltraHDR samples (JPEG / AVIF) |
| `hdr_tone_map_shootout_full` | extended shootout on the 76-sample imazen-26 gain-mapped corpus |
| `hdr_tone_map_shootout_audited` | color-audited rerun (2026-06-22) |
| `hdr_tone_map_shootout_gainforge` | addendum: gainforge curves × color spaces |
| `shootout_p99999_only` | percentile 0.99999 vs 0.9999 peak rerun |
| `softcompress_knee_sweep` | SoftCompress `knee` calibration sweep |
| `flicker_test_render` | flicker-test renders for the percentile sweep |
| `hdr_sdr_luminance_trace` | HDR vs producer-SDR luminance trace for one sample |
| `audit_primaries_dump` | phase-1 audit: dump source decode descriptors for spot-check samples |
| `fit_pow_inv_24` | offline polyfit coefficient fitting (see `benchmarks/bt2446a_throughput_2026-06-20.md`) |

Findings and reproduce lines live in `../../benchmarks/*.md`.

## Known state (2026-08-27)

The dependency graph does not currently resolve against the sibling
checkouts: `zenpipe/zencodecs` main now requires `zenjpeg ^0.9.0` (the pinned
patch rev is 0.8.7) and `zenavif ^0.1.7` (yanked on crates.io; zenpipe gets it
through the zen-workspace sibling overlay). Re-pin the patch table against the
current zenpipe lockfile before the next shootout run. This was already the
case while the drivers lived in the root manifest — it is why the crate was
unbuildable locally and red on CI.
