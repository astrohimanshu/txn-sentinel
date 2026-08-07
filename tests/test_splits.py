"""Tests for the frozen split configuration."""

from datetime import datetime

import polars as pl
import pytest

from txn_sentinel.splits import SplitConfig, split_frames

CFG = SplitConfig(
    train_max_year=2017,
    validation_year=2018,
    test_year=2019,
    exclude_years=(2020,),
    account_key=("user", "card_index"),
    label_column="is_fraud",
    timestamp_column="timestamp",
    split_column="year",
)


def _frame() -> pl.LazyFrame:
    years = [2016, 2017, 2018, 2019, 2020]
    return pl.DataFrame(
        {
            "year": years,
            "timestamp": [datetime(y, 6, 1) for y in years],
            "is_fraud": [False] * len(years),
        }
    ).lazy()


def test_config_loads_from_the_repo_file():
    cfg = SplitConfig.load()
    assert cfg.train_max_year == 2017
    assert cfg.validation_year == 2018
    assert cfg.test_year == 2019
    assert 2020 in cfg.exclude_years
    # card_index alone is not a card id; the account key must be the pair.
    assert cfg.account_key == ("user", "card_index")


def test_config_column_names_match_the_processed_schema():
    """The config must name snake_case columns, not the raw CSV headers.

    convert_parquet.py renames every column, so a config still saying "Year" or
    "Is Fraud?" parses fine and then fails only once it meets real data.
    """
    cfg = SplitConfig.load()
    processed_columns = {
        "user",
        "card_index",
        "year",
        "month",
        "day",
        "timestamp",
        "amount",
        "is_fraud",
        "merchant_id",
        "merchant_state",
    }
    assert cfg.split_column in processed_columns
    assert cfg.label_column in processed_columns
    assert cfg.timestamp_column in processed_columns
    for key in cfg.account_key:
        assert key in processed_columns


def test_splits_are_disjoint_and_ordered():
    train, val, test = (df.collect() for df in split_frames(_frame(), CFG))

    assert sorted(train["year"].to_list()) == [2016, 2017]
    assert val["year"].to_list() == [2018]
    assert test["year"].to_list() == [2019]

    assert train["year"].max() < val["year"].min()
    assert val["year"].max() < test["year"].min()


def test_excluded_year_appears_in_no_split():
    """2020 has zero labelled fraud, so it must not reach any split."""
    train, val, test = (df.collect() for df in split_frames(_frame(), CFG))
    for frame in (train, val, test):
        assert 2020 not in frame["year"].to_list()


def test_overlapping_boundaries_are_rejected():
    with pytest.raises(ValueError, match="forward in time"):
        SplitConfig(
            train_max_year=2018,
            validation_year=2018,
            test_year=2019,
            exclude_years=(),
            account_key=("user", "card_index"),
            label_column="is_fraud",
            timestamp_column="timestamp",
            split_column="year",
        ).validate()


def test_evaluation_year_cannot_also_be_excluded():
    with pytest.raises(ValueError, match="exclude_years"):
        SplitConfig(
            train_max_year=2017,
            validation_year=2018,
            test_year=2019,
            exclude_years=(2019,),
            account_key=("user", "card_index"),
            label_column="is_fraud",
            timestamp_column="timestamp",
            split_column="year",
        ).validate()
