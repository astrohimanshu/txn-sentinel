"""Behavioural features computed from an account's own past.

Every feature here is strictly-past: for the transaction being scored, no value is
allowed to depend on that transaction's outcome, on any later transaction, or on any
transaction belonging to a different account.

Two implementation details carry that guarantee, and both are easy to get wrong:

1. Time windows use ``closed="left"``. Polars' rolling windows are right-closed by
   default, which includes the row being scored -- an amount would then appear inside
   its own "past average". ``closed="left"`` makes the window ``[t - w, t)``. Rows
   sharing an identical timestamp are excluded too, which is conservative and correct.

2. Everything is grouped by ``(user, card_index)``. ``card_index`` alone takes only
   nine values across the dataset -- it is a position within a user, not a card id.
   Grouping on it by itself silently merges 2,000 unrelated people into nine
   histories, and the resulting features look entirely plausible.

Callers must pass a frame sorted by ``(user, card_index, timestamp)``.
"""

from __future__ import annotations

import polars as pl

ACCOUNT = ["user", "card_index"]

# Time windows for velocity counts and sums.
WINDOWS = ("1h", "24h", "7d")


def _past_count(window: str) -> pl.Expr:
    """Number of prior transactions on this account within ``window``."""
    # amount is never null, so counting non-null amounts counts transactions.
    return (
        pl.col("amount")
        .is_not_null()
        .cast(pl.Int32)
        .rolling_sum_by("timestamp", window_size=window, closed="left")
        .over(ACCOUNT)
        .fill_null(0)
        .alias(f"txn_count_{window}")
    )


def _past_amount_sum(window: str) -> pl.Expr:
    """Total spend on this account within ``window``, excluding the current row."""
    return (
        pl.col("amount")
        .rolling_sum_by("timestamp", window_size=window, closed="left")
        .over(ACCOUNT)
        .fill_null(0.0)
        .alias(f"amount_sum_{window}")
    )


def add_velocity_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Attach strictly-past behavioural features to a sorted transaction frame."""
    prev_ts = pl.col("timestamp").shift(1).over(ACCOUNT)

    # Count of transactions strictly before this one on the same account. This is the
    # row's 0-based position within its account, which is exactly the prior count.
    n_prior = pl.int_range(pl.len(), dtype=pl.Int64).over(ACCOUNT)

    # Expanding mean of past amounts. cum_sum includes the current row, so shift it.
    past_sum = pl.col("amount").cum_sum().shift(1).over(ACCOUNT)
    past_mean = pl.when(n_prior > 0).then(past_sum / n_prior).otherwise(None)

    features = [
        # Seconds since this account's previous transaction. Null on the first one.
        (pl.col("timestamp") - prev_ts).dt.total_seconds().alias("secs_since_prev"),
        n_prior.alias("txn_count_prior"),
        past_mean.alias("amount_mean_prior"),
        # How unusual is this amount against the account's own history?
        pl.when(past_mean.is_not_null() & (past_mean != 0))
        .then(pl.col("amount") / past_mean)
        .otherwise(None)
        .alias("amount_vs_prior_mean"),
        # First time this account has transacted with this merchant?
        (~pl.col("merchant_id").is_first_distinct().over(ACCOUNT)).alias("merchant_seen_before"),
        # Did the merchant's state change from the previous transaction?
        (pl.col("merchant_state") != pl.col("merchant_state").shift(1).over(ACCOUNT))
        .fill_null(False)
        .alias("merchant_state_changed"),
        # Calendar context of the transaction itself -- not derived from any outcome.
        pl.col("timestamp").dt.hour().alias("hour_of_day"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
    ]
    for window in WINDOWS:
        features.append(_past_count(window))
        features.append(_past_amount_sum(window))

    return lf.with_columns(features)


def feature_columns() -> list[str]:
    """Names of the columns produced by :func:`add_velocity_features`."""
    names = [
        "secs_since_prev",
        "txn_count_prior",
        "amount_mean_prior",
        "amount_vs_prior_mean",
        "merchant_seen_before",
        "merchant_state_changed",
        "hour_of_day",
        "day_of_week",
    ]
    for window in WINDOWS:
        names.append(f"txn_count_{window}")
        names.append(f"amount_sum_{window}")
    return names
