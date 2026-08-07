"""Profile the raw transactions file and write a JSON summary to results/.

Reports the facts the modelling phases depend on: row count, schema, the natural
fraud rate, the time span available for time-based splits, and per-column nulls.
Nothing here rebalances or samples -- the prevalence printed is the real one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "credit_card_transactions-ibm_v2.csv"
OUT = REPO_ROOT / "results" / "data_profile.json"

LABEL = "Is Fraud?"


def main() -> int:
    if not RAW.exists():
        print(f"Missing {RAW}; run scripts/download_data.py first", file=sys.stderr)
        return 1

    lazy = pl.scan_csv(RAW, infer_schema_length=10_000)
    schema = lazy.collect_schema()
    columns = list(schema.names())
    print(f"Columns ({len(columns)}):")
    for name in columns:
        print(f"  {name}: {schema[name]}")

    n_rows = lazy.select(pl.len()).collect(engine="streaming").item()
    print(f"\nRows: {n_rows:,}")

    profile: dict[str, object] = {
        "file": RAW.name,
        "bytes": RAW.stat().st_size,
        "rows": n_rows,
        "columns": {name: str(schema[name]) for name in columns},
    }

    if LABEL in columns:
        counts = (
            lazy.group_by(LABEL)
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .collect(engine="streaming")
        )
        print(f"\nLabel distribution ({LABEL}):")
        dist = {}
        for row in counts.iter_rows(named=True):
            value, n = row[LABEL], row["n"]
            share = n / n_rows
            dist[str(value)] = {"count": n, "share": share}
            print(f"  {value!s:>5}: {n:>12,}  ({share:.4%})")
        profile["label_distribution"] = dist

        positives = sum(v["count"] for k, v in dist.items() if str(k).strip().lower() == "yes")
        if positives:
            print(f"\nFraud prevalence: {positives / n_rows:.4%}  (1 in {n_rows / positives:,.0f})")
            profile["fraud_prevalence"] = positives / n_rows
            profile["fraud_count"] = positives

    if "Year" in columns:
        span = lazy.select(
            pl.col("Year").min().alias("min"), pl.col("Year").max().alias("max")
        ).collect(engine="streaming")
        lo, hi = span["min"][0], span["max"][0]
        print(f"\nYear span: {lo} to {hi}")
        profile["year_min"], profile["year_max"] = lo, hi

        per_year = (
            lazy.group_by("Year").agg(pl.len().alias("n")).sort("Year").collect(engine="streaming")
        )
        print("\nRows per year:")
        by_year = {}
        for row in per_year.iter_rows(named=True):
            by_year[str(row["Year"])] = row["n"]
            print(f"  {row['Year']}: {row['n']:>12,}")
        profile["rows_per_year"] = by_year

    for col, key in (("User", "n_users"), ("Card", "n_cards_per_user")):
        if col in columns:
            n = lazy.select(pl.col(col).n_unique()).collect(engine="streaming").item()
            print(f"\nDistinct {col}: {n:,}")
            profile[key] = n

    if "User" in columns and "Card" in columns:
        n_pairs = (
            lazy.select(pl.struct(["User", "Card"]).n_unique()).collect(engine="streaming").item()
        )
        print(f"Distinct (User, Card) pairs: {n_pairs:,}")
        profile["n_card_accounts"] = n_pairs

    nulls = lazy.select(pl.all().null_count()).collect(engine="streaming")
    nonzero = {c: nulls[c][0] for c in nulls.columns if nulls[c][0] > 0}
    print("\nColumns with nulls:" if nonzero else "\nNo nulls in any column.")
    for c, n in sorted(nonzero.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {n:,} ({n / n_rows:.2%})")
    profile["null_counts"] = nonzero

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profile, indent=2, default=str))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
