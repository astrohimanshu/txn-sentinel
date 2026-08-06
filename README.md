# txn-sentinel

Fraud scoring for card transactions, built on the **synthetic** IBM credit-card transaction dataset released by Altman et al. The data is machine-generated rather than real cardholder activity, so no measurement in this repository describes performance on a live payment network.

The project trains fraud models over a time-ordered transaction history and serves them behind an HTTP API. Models are compared on cost-weighted metrics at the dataset's natural fraud prevalence: PR-AUC, recall at a fixed review budget, and expected cost at the chosen operating threshold.

## Dataset

[`ealtman2019/credit-card-transactions`](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions) on Kaggle — roughly 24M synthetic transactions for a simulated population of cardholders, carrying a labelled fraud flag. The download is not committed; it lands in the gitignored `data/` directory.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync
```

## Run the API

```bash
uv run uvicorn txn_sentinel.api:app --reload --port 8000
```

`GET /health` returns the service status and version.

## Tests and linting

```bash
uv run pytest
uv run ruff check
```

## Docker

```bash
docker build -t txn-sentinel .
docker run --rm -p 8000:8000 txn-sentinel
```

The container listens on `$PORT` (default `8000`) and runs as a non-root user.

## Layout

| Path                | Contents                                  |
| ------------------- | ----------------------------------------- |
| `src/txn_sentinel/` | Package source                            |
| `tests/`            | Test suite                                |
| `configs/`          | Split boundaries and model configuration  |
| `scripts/`          | Data and training entry points            |
| `data/`             | Local dataset copy (gitignored)           |
| `results/`          | Run outputs and evaluation tables (gitignored) |
