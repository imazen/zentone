#!/usr/bin/env python3
"""Tail-aware re-analysis of the v2 shootout CSV (2026-06-22-percentiles-oklab).

Reads:
- benchmarks/hdr_tone_map_shootout_full_2026-06-22_v2.csv (the new v2 CSV with
  per-cell ΔE2000 percentiles + OKLab Euclidean ΔE)

Produces:
- a six-criteria top-5 ranking grid
- per-criterion worst-tail sample tables for the top combos
- a recommended production combo using a tail-first tiebreaker policy
- writes the markdown report to benchmarks/shootout_2026-06-22_findings_v2.md
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


ZT = Path("/home/lilith/work/zen/zentone")
V2_CSV = ZT / "benchmarks/hdr_tone_map_shootout_full_2026-06-22_v2.csv"
OUT_MD = ZT / "benchmarks/shootout_2026-06-22_findings_v2.md"


def load_v2() -> list[dict]:
    if not V2_CSV.exists():
        raise SystemExit(f"missing CSV: {V2_CSV}")
    rows: list[dict] = []
    with V2_CSV.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # Convert numeric fields.
            for key in (
                "source_peak_nits",
                "psnr_db",
                "mean_de2000",
                "max_abs_delta",
                "pct_above_de5",
                "de2000_p50",
                "de2000_p90",
                "de2000_p95",
                "de2000_p99",
                "de_ok_mean",
                "de_ok_p50",
                "de_ok_p90",
                "de_ok_p95",
                "de_ok_p99",
                "pct_above_de_ok_0p04",
            ):
                try:
                    r[key] = float(r[key])
                except (KeyError, ValueError):
                    r[key] = float("nan")
            rows.append(r)
    return rows


def aggregate_by_combo(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """Group by (curve, peak_method). Aggregate corpus-wide mean of each per-cell metric."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["curve"], r["peak_method"])].append(r)
    out = {}
    for combo, cells in buckets.items():
        n = len(cells)
        agg = {"n": n}
        for key in (
            "mean_de2000",
            "max_abs_delta",
            "pct_above_de5",
            "de2000_p50",
            "de2000_p90",
            "de2000_p95",
            "de2000_p99",
            "de_ok_mean",
            "de_ok_p50",
            "de_ok_p90",
            "de_ok_p95",
            "de_ok_p99",
            "pct_above_de_ok_0p04",
            "psnr_db",
        ):
            vals = [c[key] for c in cells if c[key] == c[key]]  # filter NaN
            agg[key] = mean(vals) if vals else float("nan")
        out[combo] = agg
    return out


def top_k_by(
    agg: dict[tuple[str, str], dict[str, float]],
    metric: str,
    k: int = 5,
    reverse: bool = False,
) -> list[tuple[tuple[str, str], dict[str, float]]]:
    """Sort (curve, peak) combos by `metric`. `reverse=True` for higher-is-better."""
    items = list(agg.items())
    items.sort(key=lambda x: (x[1][metric], x[0]), reverse=reverse)
    return items[:k]


def fmt_combo(combo: tuple[str, str]) -> str:
    curve, peak = combo
    return f"{curve} × {peak}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def section_top5_grid(agg: dict) -> str:
    """Render the 6-criteria top-5 ranking grid."""
    rankings = [
        ("mean_de2000", False, "Lab ΔE2000 mean (old default)"),
        ("de2000_p95", False, "Lab ΔE2000 p95 (per-image p95 → corpus mean)"),
        ("de2000_p99", False, "Lab ΔE2000 p99 (per-image p99 → corpus mean)"),
        ("pct_above_de5", False, "% pixels per image with ΔE2000 > 5"),
        ("de_ok_mean", False, "OKLab ΔE mean"),
        ("de_ok_p95", False, "OKLab ΔE p95 (per-image p95 → corpus mean)"),
    ]
    out: list[str] = []
    for metric, hib, title in rankings:
        out.append(f"### Top 5 by `{metric}` — {title}\n")
        top = top_k_by(agg, metric, k=5, reverse=hib)
        headers = ["rank", "combo", metric, "de_ok_mean", "de_ok_p95", "mean_de2000", "pct>de5"]
        rows = []
        for i, (combo, a) in enumerate(top, 1):
            rows.append([
                str(i),
                fmt_combo(combo),
                f"{a[metric]:.5f}" if "de_ok" in metric else f"{a[metric]:.4f}",
                f"{a['de_ok_mean']:.5f}",
                f"{a['de_ok_p95']:.5f}",
                f"{a['mean_de2000']:.4f}",
                f"{a['pct_above_de5']:.2f}",
            ])
        out.append(md_table(headers, rows))
        out.append("")
    return "\n".join(out)


def worst_samples_table(
    rows: list[dict], combo: tuple[str, str], metric: str, k: int = 10
) -> str:
    """List the k worst SAMPLES for a given (curve, peak) combo under `metric`."""
    curve, peak = combo
    matching = [r for r in rows if r["curve"] == curve and r["peak_method"] == peak]
    matching.sort(key=lambda r: r[metric], reverse=True)
    top = matching[:k]
    headers = ["sample (stem)", metric, "de_ok_p99", "mean_de2000", "pct>de5"]
    out = []
    for r in top:
        stem = r["sample"]
        # Trim leading nnnn_ prefix for readability.
        out.append([
            stem[:50] + "..." if len(stem) > 53 else stem,
            f"{r[metric]:.4f}" if "de_ok" in metric else f"{r[metric]:.3f}",
            f"{r['de_ok_p99']:.4f}",
            f"{r['mean_de2000']:.3f}",
            f"{r['pct_above_de5']:.2f}",
        ])
    return md_table(headers, out)


def main() -> int:
    rows = load_v2()
    print(f"Loaded {len(rows)} rows from {V2_CSV.name}", file=sys.stderr)
    if len(rows) == 0:
        raise SystemExit("CSV is empty")

    # Spot-check the known cell.
    spot = [
        r for r in rows
        if r["sample"].startswith("1066_general_stone-building-facade")
        and r["curve"] == "bt2446a"
        and r["peak_method"] == "measure_max"
    ]
    if spot:
        s = spot[0]
        print(
            f"SPOT-CHECK 1066_general_stone-building-facade × bt2446a × measure_max:\n"
            f"  mean_de2000 = {s['mean_de2000']:.4f} (was 3.9352 in v1)\n"
            f"  de2000_p95 = {s['de2000_p95']:.4f}\n"
            f"  de2000_p99 = {s['de2000_p99']:.4f}\n"
            f"  de_ok_mean = {s['de_ok_mean']:.5f}\n"
            f"  de_ok_p95 = {s['de_ok_p95']:.5f}",
            file=sys.stderr,
        )

    agg = aggregate_by_combo(rows)
    print(f"  {len(agg)} unique (curve, peak) combos", file=sys.stderr)

    # Top-1 winner under the brief's recommended tiebreaker policy:
    #   Primary: de_ok_p95 (cap perceptual tail)
    #   Secondary: de_ok_mean (overall fidelity)
    #   Tertiary: peak-method cost (max < smoothed/p99999 < robust)
    PEAK_COST = {
        "measure_max": 0,
        "measure_max_smoothed": 1,
        "measure_percentile_99999": 2,
        "measure_robust": 3,
    }

    def production_key(item):
        (curve, peak), a = item
        return (a["de_ok_p95"], a["de_ok_mean"], PEAK_COST.get(peak, 4))

    ranked = sorted(agg.items(), key=production_key)
    winner = ranked[0]

    # Worst-tail tables for the top 4 combos under de2000_p99 and de_ok_p99.
    top_p99 = [c for c, _ in top_k_by(agg, "de2000_p99", k=4)]
    top_ok_p99 = [c for c, _ in top_k_by(agg, "de_ok_p99", k=4)]

    # Detect any combo in top-5 of a NEW metric that wasn't in top-5 of mean_de2000.
    top5_mean = {c for c, _ in top_k_by(agg, "mean_de2000", k=5)}
    surprises: list[tuple[str, str, dict]] = []
    for metric in ("de2000_p95", "de2000_p99", "de_ok_mean", "de_ok_p95"):
        for combo, a in top_k_by(agg, metric, k=5):
            if combo not in top5_mean:
                surprises.append((metric, fmt_combo(combo), a))

    # Build the markdown.
    body_parts = []
    body_parts.append("# HDR → SDR shootout v2 — tail-aware findings (2026-06-22)\n")
    body_parts.append(
        "**Headline.** Re-scoring the 6080-cell audited sweep with per-image ΔE percentiles "
        "and OKLab Euclidean ΔE, to surface tail behavior that `mean_de2000` averaged away.\n"
    )
    body_parts.append(
        f"- CSV: `benchmarks/{V2_CSV.name}` ({len(rows)} rows × {len(rows[0])} cols)\n"
        f"- Unique (curve, peak) combos: {len(agg)}\n"
        f"- metrics_version: `2026-06-22-percentiles-oklab`\n"
    )

    body_parts.append("\n## 1. Six-criteria top-5 grid\n")
    body_parts.append(section_top5_grid(agg))

    body_parts.append("\n## 2. Worst-tail samples under ΔE2000 p99 (top 4 combos)\n")
    for combo in top_p99:
        body_parts.append(f"### {fmt_combo(combo)}\n")
        body_parts.append(worst_samples_table(rows, combo, "de2000_p99", k=10))
        body_parts.append("")

    body_parts.append("\n## 3. Worst-tail samples under OKLab ΔE p99 (top 4 combos)\n")
    for combo in top_ok_p99:
        body_parts.append(f"### {fmt_combo(combo)}\n")
        body_parts.append(worst_samples_table(rows, combo, "de_ok_p99", k=10))
        body_parts.append("")

    if surprises:
        body_parts.append("\n## 4. Surprise findings\n")
        body_parts.append(
            "These (curve, peak) combos appear in a top-5 ranking under a tail-aware metric "
            "but did NOT make the top-5 under `mean_de2000`:\n"
        )
        for metric, combo_label, a in surprises:
            body_parts.append(
                f"- Under `{metric}`: **{combo_label}** "
                f"(de_ok_p95={a['de_ok_p95']:.5f}, mean_de2000={a['mean_de2000']:.4f}, "
                f"pct>de5={a['pct_above_de5']:.2f})"
            )
        body_parts.append("")
    else:
        body_parts.append("\n## 4. Surprise findings\n")
        body_parts.append(
            "Under every new tail-aware metric, the top-5 combos are a subset or "
            "permutation of the top-5 under `mean_de2000`. The ranking *order* "
            "may shift, but no dark-horse combo emerges.\n"
        )

    # Recommended production combo
    body_parts.append("\n## 5. Recommended production combo\n")
    (curve_w, peak_w), a_w = winner
    body_parts.append(
        f"**Recommendation: `{curve_w} × {peak_w}`**\n\n"
        f"- de_ok_p95 = {a_w['de_ok_p95']:.5f}\n"
        f"- de_ok_mean = {a_w['de_ok_mean']:.5f}\n"
        f"- mean_de2000 = {a_w['mean_de2000']:.4f}\n"
        f"- de2000_p95 = {a_w['de2000_p95']:.4f}\n"
        f"- de2000_p99 = {a_w['de2000_p99']:.4f}\n"
        f"- pct_above_de5 = {a_w['pct_above_de5']:.2f}\n"
        f"- pct_above_de_ok_0p04 = {a_w['pct_above_de_ok_0p04']:.2f}\n"
    )
    body_parts.append(
        "\n### Tiebreaker policy\n"
        "1. **Primary: `de_ok_p95`** — caps the perceptual tail. OKLab is more "
        "perceptually uniform than CIE Lab D65, especially on saturated colors "
        "(the kinds of pixels the user could see were wrong in the flicker viewer). "
        "Capping the 95th-percentile pixel ΔE inside each image is the closest "
        "single number to 'no pixel will look obviously wrong'.\n"
        "2. **Secondary: `de_ok_mean`** — overall fidelity once the tail is bounded.\n"
        "3. **Tertiary: peak-method cost** — `measure_max` is the cheapest of the four "
        "(single pass, no smoothing/percentile/robust statistics).\n"
    )

    body_parts.append("\n## 6. Top-5 under the production policy\n")
    body_parts.append(
        "Combining primary + secondary + tertiary into a single sort:\n"
    )
    headers = [
        "rank",
        "combo",
        "de_ok_p95",
        "de_ok_mean",
        "mean_de2000",
        "pct>de5",
        "peak cost",
    ]
    table_rows = []
    for i, (combo, a) in enumerate(ranked[:5], 1):
        table_rows.append([
            str(i),
            fmt_combo(combo),
            f"{a['de_ok_p95']:.5f}",
            f"{a['de_ok_mean']:.5f}",
            f"{a['mean_de2000']:.4f}",
            f"{a['pct_above_de5']:.2f}",
            str(PEAK_COST.get(combo[1], 4)),
        ])
    body_parts.append(md_table(headers, table_rows))
    body_parts.append("")

    # Stability of the curve choice across metrics.
    body_parts.append("\n## 7. Stability of the curve choice\n")
    stable_curves: list[str] = []
    for metric in ("mean_de2000", "de2000_p95", "de2000_p99", "de_ok_mean", "de_ok_p95"):
        top1_curve = top_k_by(agg, metric, k=1)[0][0][0]
        stable_curves.append(f"- `{metric}`: top curve = `{top1_curve}`")
    body_parts.append("\n".join(stable_curves))
    body_parts.append("")

    OUT_MD.write_text("\n".join(body_parts))
    print(f"Wrote {OUT_MD}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
