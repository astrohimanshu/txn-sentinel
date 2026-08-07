"""Export the assets the container needs to serve: model, threshold, replay sample.

These land in models/ and ARE committed, unlike results/. The image is built from a
clean git clone, so anything the service needs at runtime has to be in the repo or
the deploy is not reproducible from a tagged release.

The replay sample deliberately over-represents fraud so the dashboard shows something
happening. That ratio is a demo artefact and is recorded in the file's metadata; it is
not the dataset's prevalence and no metric is computed from it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

from txn_sentinel.evaluation import threshold_for_budget
from txn_sentinel.features import add_velocity_features
from txn_sentinel.splits import SplitConfig, split_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
PARQUET = REPO_ROOT / "data" / "processed" / "transactions.parquet"
RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"

N_FRAUD = 60
N_LEGIT = 140
HISTORY_DEPTH = 20
OPERATING_BUDGET = 0.001
SEED = 7

TXN_FIELDS = ["timestamp", "amount", "mcc", "use_chip", "merchant_id", "merchant_state", "errors"]


def _txn(row: dict) -> dict:
    out = {k: row[k] for k in TXN_FIELDS}
    out["timestamp"] = row["timestamp"].isoformat()
    out["amount"] = float(row["amount"])
    out["mcc"] = int(row["mcc"])
    out["merchant_id"] = int(row["merchant_id"])
    return out


def main() -> int:
    if not PARQUET.exists():
        print(f"Missing {PARQUET}", file=sys.stderr)
        return 1
    model_src = RESULTS / "lightgbm_baseline.txt"
    if not model_src.exists():
        print(f"Missing {model_src}; run scripts/train_baseline.py first", file=sys.stderr)
        return 1
    MODELS.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_src, MODELS / "lightgbm_baseline.txt")
    print(f"Copied model -> {MODELS / 'lightgbm_baseline.txt'}")

    # Operating threshold picked on VALIDATION, never on test.
    val_scores = np.load(RESULTS / "lightgbm_validation_scores.npy")
    threshold = threshold_for_budget(val_scores, OPERATING_BUDGET)
    (MODELS / "serving_config.json").write_text(
        json.dumps(
            {
                "model_file": "lightgbm_baseline.txt",
                "threshold": threshold,
                "operating_budget": OPERATING_BUDGET,
                "chosen_on": "validation (2018)",
            },
            indent=2,
        )
    )
    print(f"Threshold at {OPERATING_BUDGET:.2%} validation budget: {threshold:.6f}")

    cfg = SplitConfig.load()
    _, _, test_lf = split_frames(add_velocity_features(pl.scan_parquet(PARQUET)), cfg)
    test = test_lf.select(
        ["user", "card_index", *TXN_FIELDS, "is_fraud", "txn_count_prior", "amount_mean_prior"]
    ).collect()

    rng = np.random.default_rng(SEED)
    frauds = test.filter(pl.col("is_fraud"))
    legit = test.filter(~pl.col("is_fraud"))
    picked = pl.concat(
        [
            frauds[rng.choice(frauds.height, min(N_FRAUD, frauds.height), replace=False)],
            legit[rng.choice(legit.height, min(N_LEGIT, legit.height), replace=False)],
        ]
    ).sort("timestamp")
    print(f"Sampled {picked.height} cases ({picked['is_fraud'].sum()} fraud)")

    # Pull every transaction for the involved accounts once, then slice per case.
    accounts = picked.select(["user", "card_index"]).unique()
    everything = (
        pl.scan_parquet(PARQUET)
        .select(["user", "card_index", *TXN_FIELDS])
        .join(accounts.lazy(), on=["user", "card_index"], how="semi")
        .collect()
        .sort(["user", "card_index", "timestamp"])
    )

    by_account: dict[tuple[int, int], list[dict]] = {}
    for row in everything.iter_rows(named=True):
        by_account.setdefault((row["user"], row["card_index"]), []).append(row)

    cases = []
    for row in picked.iter_rows(named=True):
        key = (row["user"], row["card_index"])
        prior = [r for r in by_account.get(key, []) if r["timestamp"] < row["timestamp"]]
        cases.append(
            {
                "user": int(row["user"]),
                "card_index": int(row["card_index"]),
                "transaction": _txn(row),
                "history": [_txn(r) for r in prior[-HISTORY_DEPTH:]],
                "is_fraud": bool(row["is_fraud"]),
                # True lifetime aggregates, as a feature store would return.
                "account_stats": {
                    "prior_count": int(row["txn_count_prior"]),
                    "prior_amount_mean": (
                        None
                        if row["amount_mean_prior"] is None
                        else float(row["amount_mean_prior"])
                    ),
                },
            }
        )

    payload = {
        "note": (
            "Held-out 2019 transactions. Fraud is intentionally over-represented so the "
            "dashboard shows activity; this ratio is a demo artefact, not the dataset "
            "prevalence of 0.1211%. Synthetic data (IBM/Altman)."
        ),
        "split": "test (2019)",
        "n_cases": len(cases),
        "n_fraud": int(sum(c["is_fraud"] for c in cases)),
        "history_depth": HISTORY_DEPTH,
        "cases": cases,
    }
    out = MODELS / "replay_sample.json"
    out.write_text(json.dumps(payload))
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
