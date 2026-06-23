#!/usr/bin/env python3
"""Analyze the SoftCompress knee sweep CSV and produce the findings report.

Reads `benchmarks/softcompress_knee_sweep_2026-06-23.csv` and emits a markdown
report at `benchmarks/softcompress_knee_findings_2026-06-23.md` with:

    - Aggregate table: per-knee corpus-averaged clip% / chroma-compression /
      ΔE2000.
    - Pareto frontier: knee values where p90 clip% < 0.1% with minimal
      chroma compression.
    - Per-content-class breakdown (general / interiors / nature / food).
    - High-rel_spread subset (saturated-specular tail from the prior
      percentile sweep).
    - Recommendation: the smallest knee that achieves the clip-pct threshold.

Pure stdlib — no pandas.
"""

import csv
import sys
import statistics
from pathlib import Path

CSV_PATH = Path(
    "/home/lilith/work/zen/zentone/benchmarks/softcompress_knee_sweep_2026-06-23.csv"
)
OUT_PATH = Path(
    "/home/lilith/work/zen/zentone/benchmarks/softcompress_knee_findings_2026-06-23.md"
)
PERCENTILE_CSV = Path(
    "/home/lilith/work/zen/zentone/benchmarks/percentile_sweep_2026-06-22.csv"
)

# Production-target clip-rate threshold: a knee is "safe" if its
# corpus-p90 pixels_clipped_pct stays under this fraction.
CLIP_PCT_P90_TARGET = 0.10  # 0.10% of pixels visibly clipped at corpus-p90

# Knee values we trust as candidates for default-update (must be in grid).
KNEE_CANDIDATE_RANGE = (0.50, 1.00)


def percentile(values, p):
    """Linear-interpolated percentile over a list of floats."""
    if not values:
        return float("nan")
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * p
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def content_class(stem):
    """Parse the content class token from the sample stem (token 2 of the
    underscore-split path). Returns the raw token; callers normalize."""
    parts = stem.split("_")
    if len(parts) < 2:
        return "unknown"
    return parts[1]


def load_high_rel_spread_samples():
    """Return the set of sample stems with `rel_spread > 0.25` per the prior
    percentile sweep (the saturated-specular tail)."""
    if not PERCENTILE_CSV.exists():
        return set()
    high = set()
    with PERCENTILE_CSV.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rs = float(row.get("rel_spread", "0"))
            except ValueError:
                continue
            if rs > 0.25:
                high.add(row["sample"])
    return high


def load_sweep_rows():
    rows = []
    with CSV_PATH.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                r["knee_f"] = float(r["knee"])
                r["pixels_clipped_pct_f"] = float(r["pixels_clipped_pct"])
                r["chroma_compression_pct_f"] = float(r["chroma_compression_pct"])
                r["peak_chroma_compression_f"] = float(r["peak_chroma_compression"])
                r["mean_de2000_f"] = float(r["mean_de2000_vs_knee_1"])
                r["max_de2000_f"] = float(r["max_de2000_vs_knee_1"])
                r["clip_p99_f"] = float(r["clip_p99"])
                # Strip the `.heic` / `.jpg` suffix for content-class lookup.
                stem_full = r["sample"]
                r["stem"] = stem_full.rsplit(".", 1)[0]
                r["content_class"] = content_class(r["stem"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def aggregate_by_knee(rows, filter_fn=None):
    """For each knee value, compute corpus-averaged + p90/p99 stats."""
    by_knee = {}
    for r in rows:
        if filter_fn and not filter_fn(r):
            continue
        by_knee.setdefault(r["knee_f"], []).append(r)
    out = []
    for knee in sorted(by_knee):
        bucket = by_knee[knee]
        clip_pcts = [b["pixels_clipped_pct_f"] for b in bucket]
        chroma_cmps = [b["chroma_compression_pct_f"] for b in bucket]
        peak_chromas = [b["peak_chroma_compression_f"] for b in bucket]
        de2k_means = [b["mean_de2000_f"] for b in bucket]
        de2k_maxes = [b["max_de2000_f"] for b in bucket]
        clip_p99s = [b["clip_p99_f"] for b in bucket]
        out.append({
            "knee": knee,
            "n": len(bucket),
            "mean_clip_pct": statistics.mean(clip_pcts),
            "median_clip_pct": statistics.median(clip_pcts),
            "p90_clip_pct": percentile(clip_pcts, 0.90),
            "p99_clip_pct": percentile(clip_pcts, 0.99),
            "max_clip_pct": max(clip_pcts),
            "mean_chroma_compression": statistics.mean(chroma_cmps),
            "median_chroma_compression": statistics.median(chroma_cmps),
            "p90_chroma_compression": percentile(chroma_cmps, 0.90),
            "mean_peak_chroma": statistics.mean(peak_chromas),
            "max_peak_chroma": max(peak_chromas),
            "mean_de2000": statistics.mean(de2k_means),
            "p90_de2000": percentile(de2k_means, 0.90),
            "mean_max_de2000": statistics.mean(de2k_maxes),
            "p99_clip_overshoot": percentile(clip_p99s, 0.90),
        })
    return out


def fmt_pct(v, prec=4):
    return f"{v:.{prec}f}"


def render_aggregate_table(agg):
    lines = [
        "| knee | corpus_p50_clip% | corpus_p90_clip% | corpus_p99_clip% | corpus_max_clip% | mean_chroma_comp% | p90_chroma_comp% | mean_de2000 |",
        "|------|-------------------|-------------------|-------------------|-------------------|--------------------|--------------------|--------------|",
    ]
    for r in agg:
        lines.append(
            f"| {r['knee']:.2f} "
            f"| {r['median_clip_pct']:.6f} "
            f"| {fmt_pct(r['p90_clip_pct'])} "
            f"| {fmt_pct(r['p99_clip_pct'])} "
            f"| {fmt_pct(r['max_clip_pct'])} "
            f"| {fmt_pct(r['mean_chroma_compression'])} "
            f"| {fmt_pct(r['p90_chroma_compression'])} "
            f"| {r['mean_de2000']:.4f} |"
        )
    return "\n".join(lines)


def find_minimal_knee(agg, p90_target):
    """Return the LARGEST knee where p90_clip_pct < p90_target.

    Larger knee = gentler rolloff = less chroma compression. So "the knee
    that achieves the clip target with the LEAST chroma reduction" is the
    largest passing knee. The brief's "minimal knee" is read as "minimal
    deviation from knee=1.0", i.e. minimal rolloff, i.e. largest knee."""
    candidates = [r for r in agg if r["p90_clip_pct"] < p90_target]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["knee"])


def render_per_class_breakdown(rows):
    classes = ["general", "interiors", "nature", "food"]
    other_classes = sorted(
        set(r["content_class"] for r in rows) - set(classes)
    )

    out = ["\n### Per-content-class breakdown\n"]
    for cls in classes + other_classes:
        cls_rows = [r for r in rows if r["content_class"] == cls]
        if not cls_rows:
            continue
        sample_count = len(set(r["stem"] for r in cls_rows))
        agg = aggregate_by_knee(cls_rows)
        if not agg:
            continue
        out.append(f"\n**{cls}** ({sample_count} samples)\n")
        out.append("| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |")
        out.append("|------|------------|---------------------|--------------|")
        for r in agg:
            out.append(
                f"| {r['knee']:.2f} "
                f"| {fmt_pct(r['p90_clip_pct'])} "
                f"| {fmt_pct(r['mean_chroma_compression'])} "
                f"| {r['mean_de2000']:.4f} |"
            )
        min_knee = find_minimal_knee(agg, CLIP_PCT_P90_TARGET)
        if min_knee:
            out.append(
                f"\n_{cls} recommended knee: **{min_knee['knee']:.2f}** "
                f"(p90 clip {fmt_pct(min_knee['p90_clip_pct'])}, "
                f"chroma compression {fmt_pct(min_knee['mean_chroma_compression'])})_\n"
            )
        else:
            out.append(
                f"\n_{cls}: NO knee in grid achieves p90 clip < {CLIP_PCT_P90_TARGET}%_\n"
            )
    return "\n".join(out)


def render_high_rel_spread_section(rows, high_set):
    if not high_set:
        return ""
    sub_rows = [r for r in rows if r["stem"] in high_set]
    if not sub_rows:
        return ""
    agg = aggregate_by_knee(sub_rows)
    if not agg:
        return ""
    sample_count = len(set(r["stem"] for r in sub_rows))
    out = [
        "\n### High-rel_spread subset (saturated specular tail)\n",
        f"Samples from the prior percentile sweep with `rel_spread > 25%` "
        f"(n={sample_count}). These are the cases where the percentile-",
        "based peak measurement diverges sharply from the max-RGB peak —",
        "typically saturated specular highlights on flowers, water, "
        "metal. The question: does this tail need a tighter knee than the",
        "general corpus?\n",
        "| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |",
        "|------|------------|---------------------|--------------|",
    ]
    for r in agg:
        out.append(
            f"| {r['knee']:.2f} "
            f"| {fmt_pct(r['p90_clip_pct'])} "
            f"| {fmt_pct(r['mean_chroma_compression'])} "
            f"| {r['mean_de2000']:.4f} |"
        )
    rec = find_minimal_knee(agg, CLIP_PCT_P90_TARGET)
    if rec:
        out.append(
            f"\n_High-rel_spread recommended knee: **{rec['knee']:.2f}** "
            f"(p90 clip {fmt_pct(rec['p90_clip_pct'])}, "
            f"chroma compression {fmt_pct(rec['mean_chroma_compression'])})_\n"
        )
    return "\n".join(out)


def render_recommendation(agg, agg_per_class, high_subset_rec):
    overall = find_minimal_knee(agg, CLIP_PCT_P90_TARGET)
    current = next((r for r in agg if abs(r["knee"] - 0.90) < 1e-6), None)
    lines = ["\n## Recommendation\n"]
    if overall:
        lines.append(
            f"**Recommended production default: `knee = {overall['knee']:.2f}`**\n"
        )
        lines.append(
            f"Rationale: it is the LARGEST knee value (so the LEAST chroma "
            f"compression / desaturation) in the swept grid where the "
            f"corpus-p90 pixels_clipped_pct stays under {CLIP_PCT_P90_TARGET}% — "
            f"meaning at most 10% of corpus images have more than "
            f"{CLIP_PCT_P90_TARGET}% of their pixels clip-overshoot the "
            f"`[0, 1]` linear-BT.709 range. Tighter knees would cost more "
            f"chroma without reducing clipping below the per-sample noise floor; "
            f"looser knees would let visible clipping leak through.\n"
        )
        lines.append(
            f"At this default, the mean chroma compression is "
            f"{fmt_pct(overall['mean_chroma_compression'])}%, "
            f"the mean ΔE2000 vs. no-rolloff is "
            f"{overall['mean_de2000']:.4f}, and the corpus-p90 ΔE2000 is "
            f"{overall['p90_de2000']:.4f}.\n"
        )
        if current and current["knee"] < overall["knee"]:
            chroma_savings = (
                current["mean_chroma_compression"] - overall["mean_chroma_compression"]
            )
            clip_p90_increase = overall["p90_clip_pct"] - current["p90_clip_pct"]
            lines.append(
                f"\n**Comparison vs. current default `knee = 0.90`:** the "
                f"current default is unnecessarily aggressive. Moving from "
                f"`0.90` → `{overall['knee']:.2f}` saves "
                f"{chroma_savings:.4f}% of chroma compression "
                f"({100.0 * chroma_savings / current['mean_chroma_compression']:.0f}% "
                f"relative reduction in desaturation cost), at the cost of "
                f"a {clip_p90_increase:.4f}% corpus-p90 clip increase "
                f"(still under the {CLIP_PCT_P90_TARGET}% threshold). The "
                f"`max_clip_pct` increases from "
                f"{fmt_pct(current['max_clip_pct'])} → "
                f"{fmt_pct(overall['max_clip_pct'])} on the worst-case sample, "
                f"both well below the no-rolloff `knee = 1.00` baseline of "
                f"{fmt_pct([r for r in agg if abs(r['knee']-1.0)<1e-6][0]['max_clip_pct'])}.\n"
            )
    else:
        lines.append(
            f"**No knee in the swept grid achieves the corpus-p90 clip% "
            f"target of {CLIP_PCT_P90_TARGET}%.** This suggests either the "
            f"corpus has a saturated tail that exceeds the BT.709 gamut "
            f"under any rolloff, or the threshold is too tight. The "
            f"closest passing candidate (lowest p90_clip%) at the gentlest "
            f"rolloff would be:\n"
        )
        best = min(agg, key=lambda r: r["p90_clip_pct"])
        lines.append(
            f"`knee = {best['knee']:.2f}` → p90_clip = {fmt_pct(best['p90_clip_pct'])}%, "
            f"chroma compression = {fmt_pct(best['mean_chroma_compression'])}%.\n"
        )

    return "\n".join(lines)


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}; run the sweep first.")
        return 1
    rows = load_sweep_rows()
    if not rows:
        print("No usable rows in CSV.")
        return 1
    high_set = load_high_rel_spread_samples()

    sample_count = len(set(r["stem"] for r in rows))
    knee_count = len(set(r["knee_f"] for r in rows))
    print(f"Loaded {len(rows)} rows: {sample_count} samples × {knee_count} knees")

    agg_all = aggregate_by_knee(rows)
    agg_table = render_aggregate_table(agg_all)
    per_class = render_per_class_breakdown(rows)
    high_section = render_high_rel_spread_section(rows, high_set)

    recommended = find_minimal_knee(agg_all, CLIP_PCT_P90_TARGET)
    recommendation = render_recommendation(agg_all, None, None)

    # Pareto: knees where p90 clip is at or below the target, ranked by
    # ascending chroma compression.
    pareto = sorted(
        [r for r in agg_all if r["p90_clip_pct"] < CLIP_PCT_P90_TARGET],
        key=lambda r: r["mean_chroma_compression"],
    )
    pareto_lines = ["| knee | p90_clip% | mean_chroma_comp% | mean_de2000 |",
                    "|------|------------|---------------------|--------------|"]
    for r in pareto:
        pareto_lines.append(
            f"| {r['knee']:.2f} "
            f"| {fmt_pct(r['p90_clip_pct'])} "
            f"| {fmt_pct(r['mean_chroma_compression'])} "
            f"| {r['mean_de2000']:.4f} |"
        )

    body = []
    body.append("# SoftCompress `knee` sweep — empirical default calibration (2026-06-23)\n")
    body.append("Sweep design:\n")
    body.append("- Corpus: 76 gain-mapped images from `/home/lilith/work/codec-corpus/imazen-26`.\n")
    body.append("- Pipeline: HDR (BT.2020 linear) → `Bt2446A(measure_max)` → BT.2020→BT.709 matrix → `SoftCompress(knee)`.\n")
    body.append("- Pre-clamp metrics. The final `clamp(0, 1)` is NOT applied before measurement.\n")
    body.append("- 13 knee values: 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00.\n")
    body.append(f"- Total cells: 13 × {sample_count} = {13 * sample_count}.\n")
    body.append(f"- Production-safety target: corpus-p90 `pixels_clipped_pct` < {CLIP_PCT_P90_TARGET}%.\n")
    body.append("\nMetrics:\n")
    body.append("- `pixels_clipped_pct` — fraction of pixels with any pre-clamp linear-BT.709 channel outside `[0, 1]`.\n")
    body.append("- `chroma_compression_pct` — mean OKLch C reduction vs. `knee=1.0` baseline (over chromatic pixels).\n")
    body.append("- `mean_de2000_vs_knee_1` — Lab D65 ΔE2000 against the same pipeline at `knee=1.0`.\n")
    body.append("\n## Aggregate corpus table\n")
    body.append(agg_table)
    body.append("\n\n## Pareto frontier (p90_clip < target, ranked by chroma compression)\n\n")
    body.append("\n".join(pareto_lines))
    body.append("\n")
    body.append(recommendation)
    body.append(per_class)
    body.append(high_section)
    body.append("\n## Method notes\n\n")
    body.append("- SoftCompress is run in BT.709 (target primaries) per the production pipeline order: matrix-rotate first, then OKLch chroma rolloff. Out-of-gamut BT.709 chroma comes from the BT.2020→BT.709 matrix, *not* the tone-map.\n")
    body.append("- At `knee = 1.00` no rolloff is applied (`c <= knee * max_c` always holds in `SoftCompress::compress_planes`), so all clip% / chroma deltas at the baseline are zero by construction. The non-zero clip values at `knee = 1.00` are the pre-rolloff overshoot that the rolloff exists to mitigate.\n")
    body.append("- Clip is BOTH directions: any channel > 1.0 OR any channel < 0.0 counts (the BT.2020→BT.709 matrix can produce negative R/G/B for saturated BT.2020 primaries; the final clamp would zero them).\n")
    body.append("- Driver: `examples/softcompress_knee_sweep.rs`. Outer threads capped at 5 for the ≤5 GB memory contract.\n")
    body.append(f"- Recommended default knee, if any: **{recommended['knee'] if recommended else 'NONE'}**\n")

    OUT_PATH.write_text("\n".join(body))
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
