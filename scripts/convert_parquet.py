"""Convert the raw CSV to a sorted, typed Parquet file.

This is a format and typing change only -- no filtering, no sampling, no feature
engineering. Three things are normalised because they are lossless and every
downstream step would otherwise repeat them:

  * `Amount` arrives as a string like "$134.09", and negatives as "$-99.00".
  * `Year`/`Month`/`Day`/`Time` are four columns; a real timestamp is needed to
    order transactions and to build strictly-past windows.
  * Column names become snake_case, so downstream code is not littered with
    `pl.col("Is Fraud?")`.

Output is sorted by (user, card_index, timestamp) so that per-account windowing
downstream is a scan rather than a re-sort of 24M rows.

Parsing failures are treated as errors, not as nulls: a silently null Amount would
become a silently wrong feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "raw" / "credit_card_transactions-ibm_v2.csv"
DST = REPO_ROOT / "data" / "processed" / "transactions.parquet"

RENAME = {
    "User": "user",
    "Card": "card_index",
    "Year": "year",
    "Month": "month",
    "Day": "day",
    "Use Chip": "use_chip",
    "Merchant Name": "merchant_id",
    "Merchant City": "merchant_city",
    "Merchant State": "merchant_state",
    "Zip": "zip",
    "MCC": "mcc",
    "Errors?": "errors",
}


def main() -> int:
    if not SRC.exists():
        print(f"Missing {SRC}; run scripts/download_data.py first", file=sys.stderr)
        return 1
    DST.parent.mkdir(parents=True, exist_ok=True)

    timestamp = pl.concat_str(
        [
            pl.col("Year").cast(pl.String),
            pl.lit("-"),
            pl.col("Month").cast(pl.String).str.zfill(2),
            pl.lit("-"),
            pl.col("Day").cast(pl.String).str.zfill(2),
            pl.lit(" "),
            pl.col("Time"),
        ]
    ).str.to_datetime("%Y-%m-%d %H:%M", strict=True)

    # "$134.09" -> 134.09 and "$-99.00" -> -99.0. The sign sits inside the dollar sign.
    amount = pl.col("Amount").str.replace("$", "", literal=True).cast(pl.Float64, strict=True)

    lazy = (
        pl.scan_csv(SRC, infer_schema_length=10_000)
        .with_columns(
            timestamp.alias("timestamp"),
            amount.alias("amount"),
            (pl.col("Is Fraud?") == "Yes").alias("is_fraud"),
        )
        .rename(RENAME)
        .drop(["Time", "Amount", "Is Fraud?"])
        .sort(["user", "card_index", "timestamp"])
    )

    print(f"Converting {SRC.name} -> {DST.name} (sorting by user, card_index, timestamp)")
    lazy.sink_parquet(DST, compression="zstd")

    # Verify nothing was silently lost. A null here means a parse failure that would
    # otherwise surface much later as a mysteriously wrong feature.
    check = pl.scan_parquet(DST)
    n = check.select(pl.len()).collect().item()
    bad = check.select(
        pl.col("timestamp").null_count().alias("ts"),
        pl.col("amount").null_count().alias("amt"),
        pl.col("is_fraud").null_count().alias("lbl"),
    ).collect()

    print(f"Rows written: {n:,}")
    print(f"Size: {DST.stat().st_size:,} bytes  (source {SRC.stat().st_size:,})")
    print(f"Null timestamp={bad['ts'][0]}  amount={bad['amt'][0]}  is_fraud={bad['lbl'][0]}")

    if bad["ts"][0] or bad["amt"][0] or bad["lbl"][0]:
        print("Parse failures detected; refusing to treat this output as usable.", file=sys.stderr)
        return 1

    frauds = check.select(pl.col("is_fraud").sum()).collect().item()
    print(f"Fraud rows: {frauds:,}  ({frauds / n:.4%})")
    if n != 24_386_900 or frauds != 29_757:
        print("WARNING: row or fraud count differs from the Phase 1 profile.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
