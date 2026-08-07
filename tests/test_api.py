"""Tests for the HTTP contract.

The /score request and response shapes are frozen: the transformer must be able to
replace LightGBM behind them without a frontend change. These tests pin that shape.
"""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from txn_sentinel import __version__
from txn_sentinel.api import app
from txn_sentinel.scoring import DEFAULT_MODEL

client = TestClient(app)

BASE = datetime(2019, 6, 1, 12, 0)


def _txn(offset_minutes: int, amount: float, merchant: int = 111) -> dict:
    return {
        "timestamp": (BASE + timedelta(minutes=offset_minutes)).isoformat(),
        "amount": amount,
        "mcc": 5411,
        "use_chip": "Swipe Transaction",
        "merchant_id": merchant,
        "merchant_state": "CA",
        "errors": None,
    }


def _request(history: list[dict] | None = None) -> dict:
    return {
        "user": 0,
        "card_index": 0,
        "transaction": _txn(0, 250.0),
        "history": history if history is not None else [],
    }


def test_health_reports_status_and_version():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert "model_loaded" in body


def test_dashboard_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "txn-sentinel" in response.text


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not exported yet")
def test_score_returns_the_frozen_contract():
    history = [_txn(-600, 40.0), _txn(-300, 55.0), _txn(-90, 32.0)]
    response = client.post("/score", json=_request(history))
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"score", "decision", "threshold", "model", "features"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"review", "approve"}
    assert set(body["model"]) == {"name", "version"}
    assert body["features"]["txn_count_prior"] == 3


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not exported yet")
def test_scoring_works_with_no_history():
    response = client.post("/score", json=_request([]))
    assert response.status_code == 200
    assert response.json()["features"]["txn_count_prior"] == 0


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not exported yet")
def test_history_at_or_after_the_transaction_is_ignored():
    """A caller must not be able to hand the model its own future."""
    past_only = client.post("/score", json=_request([_txn(-60, 40.0)])).json()
    with_future = client.post(
        "/score", json=_request([_txn(-60, 40.0), _txn(+60, 9999.0), _txn(0, 5000.0)])
    ).json()

    assert past_only["score"] == with_future["score"]
    assert past_only["features"] == with_future["features"]


def test_score_rejects_a_malformed_request():
    response = client.post("/score", json={"user": 0})
    assert response.status_code == 422
