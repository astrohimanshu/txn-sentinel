"""Tests for strictly-past feature construction.

The important test here is `test_features_do_not_change_when_future_rows_are_added`.
Hand-checked window counts can be right while the code still leaks; a feature that is
invariant to appending future transactions cannot be reading the future at all.
"""

from datetime import datetime

import polars as pl

from txn_sentinel.features import ACCOUNT, add_velocity_features

SCHEMA = {
    "user": pl.Int64,
    "card_index": pl.Int64,
    "timestamp": pl.Datetime("us"),
    "amount": pl.Float64,
    "merchant_id": pl.Int64,
    "merchant_state": pl.String,
}


def _frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA, orient="row").sort(ACCOUNT + ["timestamp"])


def _build(rows: list[tuple]) -> pl.DataFrame:
    return add_velocity_features(_frame(rows).lazy()).collect()


# Account (0, 0): three transactions. Account (1, 0) shares card_index 0 on purpose.
ROWS = [
    (0, 0, datetime(2020, 1, 1, 0, 0), 100.0, 1, "CA"),
    (0, 0, datetime(2020, 1, 1, 0, 30), 200.0, 1, "CA"),
    (0, 0, datetime(2020, 1, 1, 2, 0), 300.0, 2, "NY"),
    (1, 0, datetime(2020, 1, 1, 0, 15), 999.0, 1, "TX"),
]


def test_first_transaction_on_an_account_has_no_history():
    out = _build(ROWS).filter((pl.col("user") == 0) & (pl.col("txn_count_prior") == 0))

    row = out.to_dicts()[0]
    assert row["secs_since_prev"] is None
    assert row["txn_count_prior"] == 0
    assert row["amount_mean_prior"] is None
    assert row["amount_vs_prior_mean"] is None
    assert row["merchant_seen_before"] is False
    assert row["txn_count_1h"] == 0
    assert row["amount_sum_1h"] == 0.0


def test_windows_exclude_the_current_transaction():
    out = _build(ROWS).filter(pl.col("user") == 0).sort("timestamp")
    second = out.to_dicts()[1]

    # 00:30. Prior row at 00:00 is 1800s earlier and inside the 1h window.
    assert second["secs_since_prev"] == 1800
    assert second["txn_count_prior"] == 1
    assert second["amount_mean_prior"] == 100.0
    assert second["amount_vs_prior_mean"] == 2.0
    # The current 200.0 must not appear in its own window sum.
    assert second["amount_sum_1h"] == 100.0
    assert second["txn_count_1h"] == 1


def test_window_drops_transactions_that_fall_outside_it():
    out = _build(ROWS).filter(pl.col("user") == 0).sort("timestamp")
    third = out.to_dicts()[2]

    # 02:00. The 1h window [01:00, 02:00) contains neither earlier row.
    assert third["txn_count_1h"] == 0
    assert third["amount_sum_1h"] == 0.0
    # The 24h window contains both.
    assert third["txn_count_24h"] == 2
    assert third["amount_sum_24h"] == 300.0
    assert third["amount_mean_prior"] == 150.0
    assert third["merchant_seen_before"] is False  # merchant 2 is new for this account
    assert third["merchant_state_changed"] is True  # CA -> NY


def test_accounts_sharing_a_card_index_do_not_leak_into_each_other():
    """card_index is an index within a user, not a card id -- (user, card_index) is."""
    out = _build(ROWS)

    other = out.filter(pl.col("user") == 1).to_dicts()[0]
    assert other["txn_count_prior"] == 0
    assert other["secs_since_prev"] is None

    # User 1's 00:15 transaction must not appear in user 0's 00:30 window.
    second = out.filter(pl.col("user") == 0).sort("timestamp").to_dicts()[1]
    assert second["txn_count_1h"] == 1
    assert second["amount_sum_1h"] == 100.0


def test_features_do_not_change_when_future_rows_are_added():
    """The decisive leakage check.

    If any feature could see forward in time, appending later transactions would
    change the values computed for earlier ones.
    """
    future = ROWS + [
        (0, 0, datetime(2020, 1, 1, 3, 0), 5000.0, 9, "TX"),
        (0, 0, datetime(2020, 1, 2, 9, 0), 7500.0, 9, "TX"),
        (1, 0, datetime(2020, 6, 1, 0, 0), 1.0, 3, "TX"),
    ]

    keys = ACCOUNT + ["timestamp"]
    before = _build(ROWS).sort(keys)

    # Select the original transactions back out by key. Taking the first N rows would
    # compare different transactions, since the appended rows re-sort into the middle.
    after = _build(future).join(before.select(keys), on=keys, how="semi").sort(keys)

    assert after.height == before.height
    assert before.equals(after)
