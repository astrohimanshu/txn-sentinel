# txn-sentinel

Real-time fraud scoring for card transactions, built on the **synthetic** IBM/Altman
credit-card dataset. The data is machine-generated rather than real cardholder activity,
so nothing measured here describes performance on a live payment network.

**Live: https://ca-txn-sentinel.jollyocean-ec5e02b3.centralindia.azurecontainerapps.io**

The service scales to zero, so the first request after an idle period takes a few seconds
to wake the container. Subsequent requests are fast.

## What it does, and why the hard part is measurement

Card fraud is a needle-in-a-haystack problem: about **one transaction in 820** is fraudulent
here. At that rate the modelling is not the difficult bit — honest evaluation is. A model can
score 0.90 ROC-AUC and still be useless at any review budget a real team could staff, and a
single careless line can leak the future into the past and make everything look excellent.

So this project is built around the measurement discipline:

- **Time-based splits, never random.** Train ≤2017, validate 2018, test 2019. Boundaries live
  in `configs/splits.yaml` and are frozen. A random split leaks a card's future into its own
  past and inflates PR-AUC into meaninglessness.
- **2019 is the test year and 2020 is dropped entirely** — 2020 carries 336,500 rows and
  *zero* labelled frauds, which would make PR-AUC undefined.
- **No feature sees the future.** Every window is strictly-past and keyed on the account. A
  test appends later transactions, recomputes, and asserts earlier rows are unchanged.
- **The evaluation set is never rebalanced.** It carries the real 0.1211% rate, or the cost
  numbers are fiction.
- **PR-AUC is the headline, not ROC-AUC**, alongside recall at a fixed review budget and
  expected cost at the operating threshold.

## Results

LightGBM baseline on the held-out 2019 test split (1,723,938 transactions, 2,087 frauds,
0.1211% prevalence):

| Metric | Value |
| --- | --- |
| PR-AUC | **0.0552**  (95% CI 0.0471–0.0627) |
| recall @ 0.05% review budget | 0.0805 |
| recall @ 0.10% review budget | 0.1145 |
| recall @ 0.50% review budget | 0.2813 |
| expected cost @ 0.10% budget | 159,723 vs 185,097 taking no action |
| ROC-AUC | 0.9010  *(secondary — near-uninformative at this prevalence)* |

PR-AUC of 0.0552 against a random baseline of 0.0012 is roughly **45× lift**. Validation (2018)
scores 0.1056, about double the test figure, because 2018 carries a higher fraud rate
(0.1447%) and different patterns. Both are reported; the test number is the one that counts.

The operating threshold was chosen on **validation**, never on test.

A behaviour-sequence transformer is the next model. It will be compared with a McNemar test
against this baseline, and reported only if the improvement survives it.

## Dataset

[`ealtman2019/credit-card-transactions`](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)
— 24,386,900 synthetic transactions spanning 1991–2020, 29,757 labelled frauds, 2,000 users
and 6,139 distinct card accounts.

One trap worth naming: the `Card` column takes only nine distinct values. It is an index
*within* a user, not a card identifier. The account key is the pair `(User, Card)`. Grouping
per-card features on `Card` alone silently blends 2,000 unrelated people into nine histories,
and the result looks entirely reasonable.

## API

`POST /score` takes a transaction plus the account's recent history and lifetime aggregates:

```jsonc
{
  "user": 1421, "card_index": 0,
  "transaction": { "timestamp": "2019-06-01T12:00:00", "amount": 134.09,
                   "mcc": 5411, "use_chip": "Swipe Transaction",
                   "merchant_id": 3527213246127876953, "merchant_state": "CA",
                   "errors": null },
  "history": [ /* prior transactions, same shape */ ],
  "account_stats": { "prior_count": 5713, "prior_amount_mean": 78.20 }
}
```

```jsonc
{ "score": 0.0123, "decision": "approve", "threshold": 0.062897,
  "model": {"name": "lightgbm", "version": "0.1.0"}, "features": { /* ... */ } }
```

The contract is frozen. `history` exists because the sequence transformer will read the last
64 transactions, so swapping models never changes the request or response shape.

`account_stats` carries lifetime aggregates the way a feature store would. They matter:
features like `prior_count` accumulate over an account's whole life — a median of 5,713 in the
test split — and cannot be rebuilt from a bounded history window. Serving them from `history`
alone pins them at the window length and turns a top-ranked feature into a constant.

Also available: `GET /health`, `GET /replay/sample`, and a replay dashboard at `/`.

## Running it

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn txn_sentinel.api:app --port 8000   # then open http://localhost:8000
uv run pytest
uv run ruff check
```

Reproducing the model from raw data:

```bash
uv run python scripts/download_data.py        # Kaggle -> data/raw/
uv run python scripts/convert_parquet.py      # 2.35 GB CSV -> 277 MB Parquet
uv run python scripts/train_baseline.py       # LightGBM + full eval, logged to MLflow
uv run python scripts/export_serving_assets.py
```

## Layout

| Path | Contents |
| --- | --- |
| `src/txn_sentinel/` | features, splits, evaluation, scoring, API |
| `configs/splits.yaml` | frozen split boundaries |
| `scripts/` | data, training and export entry points |
| `models/` | trained model, operating threshold, replay sample |
| `data/`, `results/` | gitignored |
