"""Model evaluation: cross-validation on the training set and metrics on the test set.

The metric used to rank the models is ``config.PRIMARY_METRIC``.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate

from test_house_prediction.core import config

PRIMARY_METRIC: str = config.PRIMARY_METRIC  # used to rank the models
LOWER_IS_BETTER: bool = PRIMARY_METRIC in {"rmse", "mae"}  # errors: lower is better


def scoring() -> dict[str, Any]:
    """Return the metrics computed by cross-validation: metric name → scikit-learn scorer.

    The "neg_" scorers are turned back into positive errors by ``to_metric``.

    Returns:
        dict[str, Any]: Scorer names.

    Raises:
        ValueError: If ``PRIMARY_METRIC`` is not one of the available metrics.

    """
    metrics: dict[str, Any] = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    if PRIMARY_METRIC not in metrics:
        raise ValueError(f"PRIMARY_METRIC {PRIMARY_METRIC!r} unavailable, expected one of {list(metrics)}")
    return metrics


def to_metric(name: str, score: float) -> float:
    """Turn a scikit-learn score back into the metric (the "neg_" scorers are negated errors).

    Args:
        name: Metric name, key of ``scoring()``.
        score: Score returned by its scorer.

    Returns:
        float: The metric value.

    """
    scorer = scoring()[name]
    return -score if isinstance(scorer, str) and scorer.startswith("neg_") else score


def get_cv(n_splits: int | None = None) -> KFold:
    """Cross-validation splitter: shuffled folds.

    Args:
        n_splits: Number of folds (defaults to ``config.CV_FOLDS``).

    Returns:
        KFold: The splitter.

    """
    return KFold(n_splits=n_splits or config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def cross_validate_model(
    estimator: BaseEstimator, features: pd.DataFrame, target: pd.Series, n_splits: int | None = None
) -> dict[str, float]:
    """Cross-validate an estimator (the whole pipeline, so the preprocessing is refitted on each fold).

    Args:
        estimator: Pipeline or model to evaluate.
        features: Training features.
        target: Training target.
        n_splits: Number of folds (defaults to ``config.CV_FOLDS``).

    Returns:
        dict[str, float]: ``<metric>_mean`` and ``<metric>_std`` for each metric of ``scoring()``,
        plus ``fit_time`` (mean, in seconds).

    """
    metrics = scoring()
    scores = cross_validate(estimator, features, target, cv=get_cv(n_splits), scoring=metrics, n_jobs=config.N_JOBS)
    results: dict[str, float] = {}
    for name in metrics:
        values = np.asarray([to_metric(name, float(score)) for score in scores[f"test_{name}"]], dtype=float)
        results[f"{name}_mean"] = float(values.mean())
        results[f"{name}_std"] = float(values.std())
    results["fit_time"] = float(np.mean(scores["fit_time"]))
    return results


def rank_models(cv_results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Gather the cross-validation results of several models, best model first.

    Args:
        cv_results: Model name → result of ``cross_validate_model``.

    Returns:
        pd.DataFrame: One row per model (index: model name), sorted on ``PRIMARY_METRIC``.

    """
    table = pd.DataFrame.from_dict(cv_results, orient="index")
    return table.sort_values(f"{PRIMARY_METRIC}_mean", ascending=LOWER_IS_BETTER)


def evaluate_on_test(estimator: BaseEstimator, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Compute the metrics of a fitted estimator on the test set.

    Args:
        estimator: Fitted pipeline or model.
        features: Test features.
        target: Test target.

    Returns:
        dict[str, float]: Metric name → value.

    """
    predictions = estimator.predict(features)
    return {
        "rmse": float(root_mean_squared_error(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }
