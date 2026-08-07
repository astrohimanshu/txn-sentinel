"""Train the LightGBM baseline and evaluate it on the frozen splits.

Runs on the cluster CPU. LightGBM is histogram-based and threads well; at 20.6M
training rows this is minutes, and its GPU path is frequently slower at this scale.

Class weighting is applied to TRAINING ONLY. The validation and test splits keep
their natural prevalence -- rebalancing them would make every cost figure fiction.

MLflow tracking: set MLFLOW_TRACKING_URI to the Azure ML workspace URI to log there.
It defaults to a local ./mlruns directory, because Azure credentials live on the
laptop and not on this machine. Either way the run records the git commit, the
config path and the seed, so a result can always be traced back to what produced it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from txn_sentinel.evaluation import evaluate
from txn_sentinel.features import add_velocity_features, feature_columns
from txn_sentinel.splits import DEFAULT_CONFIG, SplitConfig, split_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
PARQUET = REPO_ROOT / "data" / "processed" / "transactions.parquet"
RESULTS = REPO_ROOT / "results"

# Raw transaction attributes that are known at scoring time and leak nothing.
RAW_FEATURES = ["amount", "mcc", "use_chip_code", "has_error", "error_code"]
LABEL = "is_fraud"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_frame() -> pl.LazyFrame:
    lf = add_velocity_features(pl.scan_parquet(PARQUET))
    # LightGBM needs numbers; use_chip has a handful of levels.
    return lf.with_columns(
        pl.col("use_chip").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("use_chip_code"),
        # `errors` is null for 98.4% of rows; the populated 1.6% records
        # authorisation-time failures (bad PIN, insufficient balance, technical
        # glitch) which are known when the transaction is scored.
        pl.col("errors").is_not_null().alias("has_error"),
        pl.col("errors").fill_null("none").cast(pl.Categorical).to_physical().cast(pl.Int32)
        .alias("error_code"),
    )


def to_arrays(frame: pl.LazyFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # amount is both a model feature and the cost weight, so dedupe the projection.
    needed = list(dict.fromkeys([*columns, LABEL, "amount"]))
    df = frame.select(needed).collect()
    x = df.select(columns).to_numpy().astype(np.float32)
    y = df[LABEL].to_numpy().astype(np.int8)
    amounts = df["amount"].to_numpy().astype(np.float64)
    return x, y, amounts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=1.0,
        help="1.0 disables weighting. The full negative/positive ratio (~818) drives "
        "predictions into sigmoid saturation, tying thousands of scores.",
    )
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-rounds", type=int, default=400)
    parser.add_argument("--early-stopping", type=int, default=40)
    parser.add_argument("--experiment", default="txn-sentinel-baseline")
    args = parser.parse_args()

    if not PARQUET.exists():
        print(f"Missing {PARQUET}; run scripts/convert_parquet.py first", file=sys.stderr)
        return 1
    RESULTS.mkdir(parents=True, exist_ok=True)

    cfg = SplitConfig.load()
    columns = feature_columns() + RAW_FEATURES

    print("Building features and splits...")
    t0 = time.time()
    train_lf, val_lf, test_lf = split_frames(build_frame(), cfg)
    x_train, y_train, _ = to_arrays(train_lf, columns)
    x_val, y_val, amt_val = to_arrays(val_lf, columns)
    x_test, y_test, amt_test = to_arrays(test_lf, columns)
    print(f"  train={x_train.shape} val={x_val.shape} test={x_test.shape} in {time.time()-t0:.1f}s")

    for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        print(f"  {name:5} positives={int(y.sum()):>7,}  prevalence={y.mean():.4%}")

    # Any weighting applies to TRAINING ONLY; eval splits keep natural prevalence.
    full_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pos_weight = args.scale_pos_weight
    print(f"  negative/positive ratio={full_ratio:.1f}, using scale_pos_weight={pos_weight}")
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "scale_pos_weight": pos_weight,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 200,
        "num_threads": 0,
        "seed": args.seed,
        "verbosity": -1,
    }

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "num_rounds": args.num_rounds})
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "config_path": str(DEFAULT_CONFIG.relative_to(REPO_ROOT)),
                "seed": str(args.seed),
                "model": "lightgbm",
                "train_max_year": str(cfg.train_max_year),
                "validation_year": str(cfg.validation_year),
                "test_year": str(cfg.test_year),
            }
        )

        print(f"\nTraining LightGBM (scale_pos_weight={pos_weight:.1f})...")
        t0 = time.time()
        booster = lgb.train(
            params,
            lgb.Dataset(x_train, label=y_train, feature_name=columns),
            num_boost_round=args.num_rounds,
            valid_sets=[lgb.Dataset(x_val, label=y_val, feature_name=columns)],
            callbacks=[
                lgb.early_stopping(args.early_stopping, verbose=False),
                lgb.log_evaluation(50),
            ],
        )
        train_seconds = time.time() - t0
        print(f"  done in {train_seconds:.1f}s, best iteration {booster.best_iteration}")

        reports = {}
        for name, x, y, amounts in (
            ("validation", x_val, y_val, amt_val),
            ("test", x_test, y_test, amt_test),
        ):
            scores = booster.predict(x, num_iteration=booster.best_iteration)
            report = evaluate(name, y, scores, amounts, seed=args.seed)
            reports[name] = report
            print("\n" + report.summary())

            mlflow.log_metrics(
                {
                    f"{name}_pr_auc": report.pr_auc,
                    f"{name}_pr_auc_ci_lo": report.pr_auc_ci[0],
                    f"{name}_pr_auc_ci_hi": report.pr_auc_ci[1],
                    f"{name}_roc_auc": report.roc_auc,
                    f"{name}_expected_cost": report.expected_cost,
                    f"{name}_cost_no_model": report.baseline_cost_no_model,
                    **{
                        f"{name}_recall_at_{b.replace('.', '_').replace('%', 'pct')}": v
                        for b, v in report.recall_at_budget.items()
                    },
                }
            )
            np.save(RESULTS / f"lightgbm_{name}_scores.npy", scores)

        mlflow.log_metrics(
            {"train_seconds": train_seconds, "best_iteration": float(booster.best_iteration)}
        )

        importance = sorted(
            zip(columns, booster.feature_importance("gain"), strict=True), key=lambda kv: -kv[1]
        )
        print("\nTop features by gain:")
        for name, gain in importance[:10]:
            print(f"  {name:24} {gain:>14,.0f}")

        model_path = RESULTS / "lightgbm_baseline.txt"
        booster.save_model(str(model_path), num_iteration=booster.best_iteration)
        mlflow.log_artifact(str(model_path))

        payload = {
            "run_id": run.info.run_id,
            "git_commit": git_commit(),
            "seed": args.seed,
            "params": params,
            "best_iteration": booster.best_iteration,
            "train_seconds": train_seconds,
            "feature_importance": [{"feature": n, "gain": float(g)} for n, g in importance],
            "reports": {
                k: {
                    "n": v.n,
                    "n_positive": v.n_positive,
                    "prevalence": v.prevalence,
                    "pr_auc": v.pr_auc,
                    "pr_auc_ci": list(v.pr_auc_ci),
                    "roc_auc": v.roc_auc,
                    "recall_at_budget": v.recall_at_budget,
                    "threshold": v.threshold,
                    "expected_cost": v.expected_cost,
                    "baseline_cost_no_model": v.baseline_cost_no_model,
                    "calibration": v.calibration,
                }
                for k, v in reports.items()
            },
        }
        out = RESULTS / "lightgbm_baseline.json"
        out.write_text(json.dumps(payload, indent=2))
        mlflow.log_artifact(str(out))
        print(f"\nWrote {out}")
        print(f"MLflow run {run.info.run_id} in experiment {args.experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
