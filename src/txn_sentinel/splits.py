"""Frozen time-based splits, loaded from configs/splits.yaml.

Order of operations matters and is easy to invert:

    features FIRST, on the full history, and only THEN split by year.

Splitting first would cut every account off from its own past, so a transaction on
2 January 2018 would look like the account's first ever. The features are strictly
past-only regardless of split, so computing them across the whole file leaks nothing.

The evaluation splits are never resampled or rebalanced. Fraud prevalence in this
data is roughly 0.12%, and every cost figure is meaningless if the test set does not
carry the real rate. Class weighting belongs inside training only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "splits.yaml"


@dataclass(frozen=True)
class SplitConfig:
    """The frozen split boundaries. Do not mutate after the test set is touched."""

    train_max_year: int
    validation_year: int
    test_year: int
    exclude_years: tuple[int, ...]
    account_key: tuple[str, ...]
    label_column: str
    timestamp_column: str
    split_column: str

    @classmethod
    def load(cls, path: Path | str | None = None) -> SplitConfig:
        path = Path(path) if path is not None else DEFAULT_CONFIG
        raw = yaml.safe_load(path.read_text())
        cfg = cls(
            train_max_year=int(raw["train"]["max_year"]),
            validation_year=int(raw["validation"]["year"]),
            test_year=int(raw["test"]["year"]),
            exclude_years=tuple(int(y) for y in raw.get("exclude_years", [])),
            account_key=tuple(raw["account_key"]),
            label_column=raw["label_column"],
            timestamp_column=raw["timestamp_column"],
            split_column=raw.get("split_column", "year"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject boundaries that overlap or run backwards."""
        if not self.train_max_year < self.validation_year < self.test_year:
            raise ValueError(
                "Splits must run forward in time and not overlap: "
                f"train<={self.train_max_year}, val={self.validation_year}, "
                f"test={self.test_year}"
            )
        for name, year in (("validation", self.validation_year), ("test", self.test_year)):
            if year in self.exclude_years:
                raise ValueError(f"{name} year {year} is also listed in exclude_years")


def _year(cfg: SplitConfig) -> pl.Expr:
    return pl.col(cfg.split_column)


def split_frames(
    lf: pl.LazyFrame, cfg: SplitConfig
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Return (train, validation, test) as lazy frames.

    Pass a frame that already carries its features. Excluded years are dropped from
    every split, including train.
    """
    keep = ~_year(cfg).is_in(list(cfg.exclude_years))
    train = lf.filter(keep & (_year(cfg) <= cfg.train_max_year))
    validation = lf.filter(keep & (_year(cfg) == cfg.validation_year))
    test = lf.filter(keep & (_year(cfg) == cfg.test_year))
    return train, validation, test
