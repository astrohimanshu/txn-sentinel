"""Tests for the cost-weighted evaluation metrics."""

import numpy as np
import pytest

from txn_sentinel.evaluation import (
    bootstrap_ci,
    expected_cost,
    flag_top_k,
    mcnemar,
    pr_auc,
    recall_at_budget,
    threshold_for_budget,
)


def test_perfect_ranking_scores_pr_auc_of_one():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_recall_at_budget_finds_the_top_ranked_fraud():
    y_true = np.zeros(100, dtype=int)
    y_true[99] = 1
    y_score = np.arange(100, dtype=float)  # the single fraud scores highest

    # Reviewing the top 1% catches it.
    assert recall_at_budget(y_true, y_score, 0.01) == pytest.approx(1.0)


def test_recall_at_budget_misses_a_bottom_ranked_fraud():
    y_true = np.zeros(100, dtype=int)
    y_true[0] = 1
    y_score = np.arange(100, dtype=float)  # the fraud scores lowest
    assert recall_at_budget(y_true, y_score, 0.01) == pytest.approx(0.0)


def test_tied_scores_still_honour_the_budget():
    """Regression: an aggressive scale_pos_weight saturates the sigmoid and ties
    thousands of scores. Threshold-based selection then flags far more than the
    budget, which made three different budgets report identical recall."""
    n = 10_000
    y_score = np.ones(n)  # every score identical -- the pathological case
    y_true = np.zeros(n, dtype=int)
    y_true[:100] = 1

    for budget in (0.001, 0.01, 0.1):
        flagged = flag_top_k(y_score, budget)
        assert flagged.sum() == round(budget * n)


def test_different_budgets_give_different_recall():
    rng = np.random.default_rng(0)
    y_true = np.zeros(20_000, dtype=int)
    y_true[:200] = 1
    # Positives rank higher on average but overlap the negatives.
    y_score = rng.random(20_000) + y_true * 0.3

    r_small = recall_at_budget(y_true, y_score, 0.001)
    r_large = recall_at_budget(y_true, y_score, 0.05)
    assert r_small < r_large


def test_budget_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="budget"):
        threshold_for_budget(np.arange(10, dtype=float), 0.0)
    with pytest.raises(ValueError, match="budget"):
        threshold_for_budget(np.arange(10, dtype=float), 1.5)


def test_expected_cost_charges_missed_fraud_and_reviews():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([100.0, 200.0, 5.0, 5.0])

    # At 0.5: index 0 caught, index 1 missed (200), index 2 false alarm (1 review).
    cost = expected_cost(y_true, y_score, 0.5, amounts, review_cost=3.0)
    assert cost == pytest.approx(200.0 + 3.0)


def test_expected_cost_uses_absolute_amounts():
    """Refunds carry negative amounts; a missed one still represents real exposure."""
    y_true = np.array([1])
    y_score = np.array([0.0])
    amounts = np.array([-99.0])
    assert expected_cost(y_true, y_score, 0.5, amounts, review_cost=3.0) == pytest.approx(99.0)


def test_recall_is_undefined_without_positives():
    with pytest.raises(ValueError, match="no positives"):
        recall_at_budget(np.zeros(10, dtype=int), np.arange(10, dtype=float), 0.1)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = (rng.random(2000) < 0.05).astype(int)
    y_score = rng.random(2000) + y_true * 0.5  # informative but noisy

    point = pr_auc(y_true, y_score)
    lo, hi = bootstrap_ci(pr_auc, y_true, y_score, n_boot=50, seed=1)

    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_mcnemar_counts_only_discordant_pairs():
    y_true = np.array([1, 1, 1, 1, 0, 0])
    pred_a = np.array([1, 1, 0, 0, 0, 0])
    pred_b = np.array([1, 0, 1, 1, 0, 0])

    out = mcnemar(y_true, pred_a, pred_b)
    assert out["a_correct_b_wrong"] == 1.0
    assert out["b_correct_a_wrong"] == 2.0
    assert out["n_discordant"] == 3.0
    assert 0.0 <= out["p_value"] <= 1.0


def test_mcnemar_returns_p_one_when_models_agree_everywhere():
    y_true = np.array([1, 0, 1, 0])
    pred = np.array([1, 0, 0, 0])
    out = mcnemar(y_true, pred, pred.copy())
    assert out["n_discordant"] == 0.0
    assert out["p_value"] == 1.0
