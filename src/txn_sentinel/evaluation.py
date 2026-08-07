"""Cost-weighted evaluation at the natural fraud rate.

At roughly 0.12% positives, ROC-AUC is close to uninformative: a model can score
0.95 there and still be useless at any review budget an operations team could
staff. The headline numbers are therefore

  * PR-AUC (average precision),
  * recall at a fixed review budget,
  * expected cost at the chosen operating threshold,

with ROC-AUC relegated to a secondary table.

Nothing here resamples or rebalances. Every metric is computed on the split exactly
as it arrives, carrying the real prevalence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

# Cost of manually reviewing one flagged transaction, in the dataset's currency
# units. A missed fraud costs the transaction amount instead -- see expected_cost.
DEFAULT_REVIEW_COST = 3.0


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision. The headline ranking metric at this prevalence."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Secondary only. Near-uninformative at 0.1% positives."""
    return float(roc_auc_score(y_true, y_score))


def threshold_for_budget(y_score: np.ndarray, budget: float) -> float:
    """Score threshold that flags exactly the top ``budget`` fraction of traffic.

    ``budget`` is the share of transactions a review team can look at, e.g. 0.001
    for one in a thousand.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    return float(np.quantile(y_score, 1.0 - budget))


def recall_at_budget(y_true: np.ndarray, y_score: np.ndarray, budget: float) -> float:
    """Share of frauds caught when only the top ``budget`` fraction is reviewed."""
    positives = y_true.sum()
    if positives == 0:
        raise ValueError("no positives in y_true; recall is undefined")
    threshold = threshold_for_budget(y_score, budget)
    return float(y_true[y_score >= threshold].sum() / positives)


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    review_cost: float = DEFAULT_REVIEW_COST,
) -> float:
    """Total cost of operating at ``threshold``.

    A missed fraud costs its transaction amount; a false alarm costs one review.
    Flagged frauds are assumed to be stopped and so cost nothing. This is the
    number that decides which model is actually better.
    """
    flagged = y_score >= threshold
    missed_fraud = (~flagged) & (y_true == 1)
    false_alarm = flagged & (y_true == 0)
    return float(np.abs(amounts[missed_fraud]).sum() + false_alarm.sum() * review_cost)


def cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    review_cost: float = DEFAULT_REVIEW_COST,
    n_points: int = 200,
) -> dict[str, np.ndarray]:
    """Expected cost across candidate thresholds, for picking an operating point."""
    quantiles = np.linspace(0.90, 0.9999, n_points)
    thresholds = np.quantile(y_score, quantiles)
    costs = np.array(
        [expected_cost(y_true, y_score, t, amounts, review_cost) for t in thresholds]
    )
    return {
        "review_fraction": 1.0 - quantiles,
        "threshold": thresholds,
        "cost": costs,
    }


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Predicted vs observed fraud rate, in equal-count score bins."""
    order = np.argsort(y_score)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        if idx.size == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(idx.size),
                "mean_predicted": float(y_score[idx].mean()),
                "observed_rate": float(y_true[idx].mean()),
            }
        )
    return rows


def bootstrap_ci(
    metric,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric taking (y_true, y_score).

    ``n_boot`` defaults to 200 rather than 1000: each resample recomputes the metric
    over the full split, and at 1.7M rows the extra precision is not worth the wait.
    Resampling is over all rows, so the prevalence varies naturally between draws.
    """
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        if sample_true.sum() == 0:
            continue
        values.append(metric(sample_true, y_score[idx]))
    if not values:
        raise ValueError("every bootstrap resample was empty of positives")
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired binary predictions from two models.

    Only the discordant pairs carry information: cases where exactly one model is
    right. The exact binomial test is used rather than the chi-square approximation
    because at this prevalence the discordant counts are often small.
    """
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b_only = int((~a_right & b_right).sum())
    a_only = int((a_right & ~b_right).sum())

    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {
        "a_correct_b_wrong": float(a_only),
        "b_correct_a_wrong": float(b_only),
        "n_discordant": float(n),
        "p_value": p,
    }


@dataclass
class EvalReport:
    """Everything reported for one model on one split."""

    split: str
    n: int
    n_positive: int
    prevalence: float
    pr_auc: float
    pr_auc_ci: tuple[float, float]
    roc_auc: float
    recall_at_budget: dict[str, float]
    threshold: float
    expected_cost: float
    baseline_cost_no_model: float
    calibration: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{self.split}: n={self.n:,} positives={self.n_positive:,} "
            f"prevalence={self.prevalence:.4%}",
            f"  PR-AUC     {self.pr_auc:.4f}  95% CI [{self.pr_auc_ci[0]:.4f}, "
            f"{self.pr_auc_ci[1]:.4f}]",
            f"  ROC-AUC    {self.roc_auc:.4f}   (secondary; near-uninformative here)",
        ]
        for budget, recall in sorted(self.recall_at_budget.items()):
            lines.append(f"  recall@{budget:<7} {recall:.4f}")
        saved = self.baseline_cost_no_model - self.expected_cost
        lines.append(
            f"  expected cost {self.expected_cost:,.0f} vs {self.baseline_cost_no_model:,.0f} "
            f"with no model ({saved:,.0f} saved)"
        )
        return "\n".join(lines)


def evaluate(
    split: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    budgets: tuple[float, ...] = (0.0005, 0.001, 0.005),
    operating_budget: float = 0.001,
    review_cost: float = DEFAULT_REVIEW_COST,
    seed: int = 0,
) -> EvalReport:
    """Full report for one model on one split."""
    threshold = threshold_for_budget(y_score, operating_budget)
    return EvalReport(
        split=split,
        n=int(y_true.shape[0]),
        n_positive=int(y_true.sum()),
        prevalence=float(y_true.mean()),
        pr_auc=pr_auc(y_true, y_score),
        pr_auc_ci=bootstrap_ci(pr_auc, y_true, y_score, seed=seed),
        roc_auc=roc_auc(y_true, y_score),
        recall_at_budget={f"{b:.2%}": recall_at_budget(y_true, y_score, b) for b in budgets},
        threshold=threshold,
        expected_cost=expected_cost(y_true, y_score, threshold, amounts, review_cost),
        # Flagging nothing: every fraud is missed, no reviews are paid for.
        baseline_cost_no_model=float(np.abs(amounts[y_true == 1]).sum()),
        calibration=calibration_table(y_true, y_score),
    )
