"""Model loading and single-transaction scoring.

Features are built by calling the same ``add_velocity_features`` used in training,
over a frame of [history..., transaction], and then taking the last row. Doing it
this way rather than reimplementing the window logic for serving is deliberate: a
hand-written serving path is how train/serve skew gets in, and here the two cannot
drift because they are literally the same function.

The request carries account history because the behaviour-sequence transformer will
need the last 64 transactions. Keeping it in the contract from the first deploy means
swapping LightGBM for the transformer never touches the frontend.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from txn_sentinel.features import add_velocity_features, feature_columns

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "lightgbm_baseline.txt"

# Must match RAW_FEATURES in scripts/train_baseline.py, in order.
RAW_FEATURES = ["amount", "mcc", "use_chip_code", "has_error", "error_code"]

# Physical codes for Polars categoricals depend on encounter order, which is not
# stable across processes. Pin them explicitly so serving matches training intent.
USE_CHIP_LEVELS = ("Chip Transaction", "Online Transaction", "Swipe Transaction")
ERROR_LEVELS = (
    "none",
    "Bad PIN",
    "Insufficient Balance",
    "Technical Glitch",
    "Bad Card Number",
    "Bad Expiration",
    "Bad CVV",
    "Bad Zipcode",
)

SERVING_CONFIG = REPO_ROOT / "models" / "serving_config.json"


def load_threshold() -> float:
    """Operating threshold, picked on VALIDATION at a 0.1% review budget.

    Never chosen on test: tuning the operating point against the split you report is
    the same mistake as moving a split boundary after seeing the score.
    """
    override = os.environ.get("TXN_SENTINEL_THRESHOLD")
    if override:
        return float(override)
    if SERVING_CONFIG.exists():
        return float(json.loads(SERVING_CONFIG.read_text())["threshold"])
    return 0.5


@dataclass(frozen=True)
class Txn:
    """One transaction, as the API receives it."""

    timestamp: datetime
    amount: float
    mcc: int
    use_chip: str
    merchant_id: int
    merchant_state: str | None = None
    errors: str | None = None


def _code(value: str | None, levels: tuple[str, ...], default: str) -> int:
    try:
        return levels.index(value if value is not None else default)
    except ValueError:
        return len(levels)  # unseen level, kept distinct from every known one


def build_feature_row(
    user: int, card_index: int, transaction: Txn, history: list[Txn]
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute the model input for ``transaction`` given the account's past.

    ``history`` is anything known before this transaction; order does not matter as
    the frame is sorted. Entries at or after the transaction's timestamp are dropped,
    so a caller cannot accidentally hand the model its own future.
    """
    past = [h for h in history if h.timestamp < transaction.timestamp]
    rows = [*past, transaction]

    frame = pl.DataFrame(
        {
            "user": [user] * len(rows),
            "card_index": [card_index] * len(rows),
            "timestamp": [r.timestamp for r in rows],
            "amount": [float(r.amount) for r in rows],
            "merchant_id": [int(r.merchant_id) for r in rows],
            "merchant_state": [r.merchant_state for r in rows],
            "mcc": [int(r.mcc) for r in rows],
            "use_chip": [r.use_chip for r in rows],
            "errors": [r.errors for r in rows],
        },
        schema_overrides={"merchant_state": pl.String, "errors": pl.String},
    ).sort(["user", "card_index", "timestamp"])

    featured = (
        add_velocity_features(frame.lazy())
        .with_columns(
            pl.col("use_chip")
            .map_elements(
                lambda v: _code(v, USE_CHIP_LEVELS, "Swipe Transaction"), return_dtype=pl.Int32
            )
            .alias("use_chip_code"),
            pl.col("errors").is_not_null().alias("has_error"),
            pl.col("errors")
            .map_elements(lambda v: _code(v, ERROR_LEVELS, "none"), return_dtype=pl.Int32)
            .alias("error_code"),
        )
        .collect()
    )

    columns = feature_columns() + RAW_FEATURES
    last = featured.tail(1).select(columns)
    values = last.to_numpy().astype(np.float32)
    return values, {c: (None if v is None else float(v)) for c, v in last.to_dicts()[0].items()}


class Scorer:
    """Wraps a trained booster behind the frozen scoring contract."""

    def __init__(self, model_path: Path | str | None = None, threshold: float | None = None):
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}")
        self.booster = lgb.Booster(model_file=str(path))
        self.model_path = path
        self.threshold = load_threshold() if threshold is None else threshold
        self.name = "lightgbm"

    def score(
        self, user: int, card_index: int, transaction: Txn, history: list[Txn]
    ) -> tuple[float, dict[str, float]]:
        x, features = build_feature_row(user, card_index, transaction, history)
        score = float(self.booster.predict(x)[0])
        return score, features
