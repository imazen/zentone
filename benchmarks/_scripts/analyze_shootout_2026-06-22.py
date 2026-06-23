#!/usr/bin/env python3
"""Phase-4 analysis of the audited shootout CSV (2026-06-22).

Reads:
- benchmarks/hdr_tone_map_shootout_full_2026-06-22.csv (the new audited CSV)
- benchmarks/hdr_tone_map_shootout_full_2026-06-20.csv (the prior baseline)
- benchmarks/percentile_sweep_2026-06-22.csv (the rel_spread bucket source)

Produces a 4-section table & ranking dump used to author
benchmarks/shootout_2026-06-22_findings.md.
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from statistics import mean, median
from pathlib import Path


ZT = Path("/home/lilith/work/zen/zentone")
NEW_CSV = ZT / "benchmarks/hdr_tone_map_shootout_full_2026-06-22.csv"
OLD_CSV = ZT / "benchmarks/hdr_tone_map_shootout_full_2026-06-20.csv"
PCT_CSV = ZT / "benchmarks/percentile_sweep_2026-06-22.csv"


def load_csv(p: Path):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def classify_content_class(stem: str) -> str:
    """Map sample stem prefix to content class per imazen-26 layout."""
    # 1000-1199 = general, 1200-1399 = interiors, 1400-1599 = nature,
    # 1600-1799 = food. The sample stem starts with the 4-digit index.
    try:
        n = int(stem.split("_", 1)[0])
    except Exception:
        return "other"
    if 1000 <= n < 1200:
        return "general"
    if 1200 <= n < 1400:
        return "interiors"
    if 1400 <= n < 1600:
        return "nature"
    if 1600 <= n < 1800:
        return "food"
    return "other"


def rank_table(rows, group_by_class: bool = False):
    """Return per (peak_method) → sorted [(curve, mean_de, n)] ascending."""
    by_method_curve = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            de = float(r["mean_de2000"])
        except (KeyError, ValueError):
            continue
        if not (0 <= de < 1000):
            continue
        by_method_curve[r["peak_method"]][r["curve"]].append(de)
    out = {}
    for method, by_curve in by_method_curve.items():
        ranking = sorted(
            [(curve, mean(des), len(des)) for curve, des in by_curve.items()],
            key=lambda t: t[1],
        )
        out[method] = ranking
    return out


def top5_combos(rows):
    """Top-5 (curve, peak_method) combos by mean ΔE2000 across all rows."""
    by_combo = defaultdict(list)
    for r in rows:
        try:
            de = float(r["mean_de2000"])
        except (KeyError, ValueError):
            continue
        if not (0 <= de < 1000):
            continue
        by_combo[(r["curve"], r["peak_method"])].append(de)
    combos = sorted(
        [(c, p, mean(des), len(des)) for (c, p), des in by_combo.items()],
        key=lambda t: t[2],
    )
    return combos


def per_content_class(rows):
    """Best (curve, peak_method) per content class."""
    by_class = defaultdict(list)
    for r in rows:
        cls = classify_content_class(r["sample"])
        by_class[cls].append(r)
    out = {}
    for cls, sub in by_class.items():
        out[cls] = top5_combos(sub)
    return out


def high_rel_spread_subset(rows, pct_rows, threshold: float = 0.25):
    """Filter rows to the samples whose `rel_spread` > threshold (fractional)."""
    if not pct_rows:
        return [], set()
    # `sample` column in percentile_sweep is stem WITHOUT extension; the
    # main CSV's `sample` column is stem WITH extension (e.g. ".heic").
    # Join on stem-without-extension.
    high_stems = set()
    for r in pct_rows:
        sample = r.get("sample")
        if not sample:
            continue
        rs = r.get("rel_spread")
        if rs is None or rs == "":
            continue
        try:
            if float(rs) > threshold:
                high_stems.add(sample)
        except ValueError:
            pass
    def stem_no_ext(s):
        for ext in (".jpg", ".jpeg", ".heic"):
            if s.lower().endswith(ext):
                return s[: -len(ext)]
        return s
    sub = [r for r in rows if stem_no_ext(r["sample"]) in high_stems]
    return sub, high_stems


def main():
    new = load_csv(NEW_CSV)
    if not new:
        print(f"FATAL: {NEW_CSV} missing or empty", file=sys.stderr)
        sys.exit(1)
    old = load_csv(OLD_CSV)
    pct = load_csv(PCT_CSV)
    print(f"new rows={len(new):,}  old rows={len(old):,}  pct rows={len(pct):,}")

    print()
    print("=" * 72)
    print("TOP-5 (curve, peak_method) combos by mean ΔE2000")
    print("=" * 72)
    combos = top5_combos(new)
    print(f"{'rank':>4}  {'curve':<26}  {'peak_method':<28}  {'mean_dE':>8}  {'n':>5}")
    for i, (c, p, m, n) in enumerate(combos[:5], 1):
        print(f"{i:>4}  {c:<26}  {p:<28}  {m:>8.4f}  {n:>5}")
    print()
    print("Worst-5 (sanity check):")
    print(f"{'rank':>4}  {'curve':<26}  {'peak_method':<28}  {'mean_dE':>8}  {'n':>5}")
    for i, (c, p, m, n) in enumerate(reversed(combos[-5:]), 1):
        print(f"{i:>4}  {c:<26}  {p:<28}  {m:>8.4f}  {n:>5}")

    print()
    print("=" * 72)
    print("PER-PEAK-METHOD curve ranking (mean ΔE2000 ascending)")
    print("=" * 72)
    rankings = rank_table(new)
    for method in ["measure_max", "measure_robust", "measure_max_smoothed", "measure_percentile_99999"]:
        if method not in rankings:
            continue
        print(f"\n--- {method} ---")
        print(f"{'rank':>4}  {'curve':<26}  {'mean_dE':>8}  {'n':>5}")
        for i, (c, m, n) in enumerate(rankings[method], 1):
            print(f"{i:>4}  {c:<26}  {m:>8.4f}  {n:>5}")

    print()
    print("=" * 72)
    print("PER-CONTENT-CLASS top-5 (mean ΔE2000)")
    print("=" * 72)
    per_class = per_content_class(new)
    for cls, combos_cls in per_class.items():
        if not combos_cls:
            continue
        print(f"\n--- {cls} ({len(set(r['sample'] for r in new if classify_content_class(r['sample']) == cls))} samples) ---")
        print(f"{'rank':>4}  {'curve':<26}  {'peak_method':<28}  {'mean_dE':>8}")
        for i, (c, p, m, n) in enumerate(combos_cls[:5], 1):
            print(f"{i:>4}  {c:<26}  {p:<28}  {m:>8.4f}")

    print()
    print("=" * 72)
    print("HIGH rel_spread > 25% subset top-5")
    print("=" * 72)
    sub_rows, high_samples = high_rel_spread_subset(new, pct, 0.25)
    print(f"High-rel-spread sample count: {len(high_samples)}  rows: {len(sub_rows)}")
    if sub_rows:
        sub_combos = top5_combos(sub_rows)
        print(f"{'rank':>4}  {'curve':<26}  {'peak_method':<28}  {'mean_dE':>8}  {'n':>5}")
        for i, (c, p, m, n) in enumerate(sub_combos[:5], 1):
            print(f"{i:>4}  {c:<26}  {p:<28}  {m:>8.4f}  {n:>5}")

    print()
    print("=" * 72)
    print("DELTA vs 2026-06-20 baseline (Robust + Bt2446A subset)")
    print("=" * 72)
    if old:
        # Old CSV has same schema minus color_handling_version.
        old_robust_2446a = [
            float(r["mean_de2000"])
            for r in old
            if r["peak_method"] == "measure_robust" and r["curve"] == "bt2446a"
        ]
        new_robust_2446a = [
            float(r["mean_de2000"])
            for r in new
            if r["peak_method"] == "measure_robust" and r["curve"] == "bt2446a"
        ]
        if old_robust_2446a and new_robust_2446a:
            print(f"OLD (2026-06-20) mean ΔE2000 [Robust, bt2446a, n={len(old_robust_2446a)}]: {mean(old_robust_2446a):.4f}")
            print(f"NEW (2026-06-22) mean ΔE2000 [Robust, bt2446a, n={len(new_robust_2446a)}]: {mean(new_robust_2446a):.4f}")
            print(f"Δ (new - old): {mean(new_robust_2446a) - mean(old_robust_2446a):+.4f}")


if __name__ == "__main__":
    main()
