"""Hyperparameter search: tune the best models of the comparison with RandomizedSearchCV.

The search uses the same pipeline, cross-validation splitter and metric as ``train.py``, so the
preprocessing is refitted on each fold (no leakage). The search spaces are in ``train.PARAM_SPACES``.
Workflow: ``make train`` (default hyperparameters) → ``make tune`` → best tuned model evaluated once
on the test set and saved in models/.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from test_house_prediction.core import config
from test_house_prediction.core.features.preprocess import build_preprocessor, infer_feature_types
from test_house_prediction.core.models.evaluate import (
    LOWER_IS_BETTER,
    PRIMARY_METRIC,
    evaluate_on_test,
    get_cv,
    scoring,
    to_metric,
)
from test_house_prediction.core.models.train import (
    MODELS,
    PARAM_SPACES,
    build_pipeline,
    compare_models,
    log_pipeline,
    prepare_data,
    save_model,
)
from test_house_prediction.core.tracking import setup_mlflow


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a hyperparameter search."""

    model_name: str
    best_params: dict[str, Any]
    pipeline: Pipeline  # best tuned pipeline, refitted on the whole training set
    search_results: pd.DataFrame  # one row per tuned model, best first
    test_metrics: dict[str, float]
    model_path: Path | None
    run_id: str


def tune_model(name: str, features: pd.DataFrame, target: pd.Series, n_iter: int | None = None) -> RandomizedSearchCV:
    """Search the hyperparameters of one model by cross-validation on the training set.

    Args:
        name: Model of ``MODELS`` with a search space in ``PARAM_SPACES``.
        features: Training features.
        target: Training target.
        n_iter: Random combinations to try (defaults to ``config.N_ITER``, capped to the size of the space).

    Returns:
        RandomizedSearchCV: The fitted search; ``best_estimator_`` is refitted on the whole training set.

    Raises:
        KeyError: If the model has no search space.

    """
    if name not in PARAM_SPACES:
        raise KeyError(f"no search space for {name!r} in PARAM_SPACES, tunable models: {list(PARAM_SPACES)}")
    space = PARAM_SPACES[name]
    n_iter = min(n_iter or config.N_ITER, len(ParameterGrid(space)))
    pipeline = build_pipeline(MODELS[name](), build_preprocessor(*infer_feature_types(features)))
    search = RandomizedSearchCV(
        pipeline,
        space,
        n_iter=n_iter,
        scoring=scoring()[PRIMARY_METRIC],
        cv=get_cv(),
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        error_score=np.nan,  # an invalid combination is skipped instead of stopping the search
    )
    search.fit(features, target)
    score = to_metric(PRIMARY_METRIC, search.best_score_)
    logger.info(f"{name}: {PRIMARY_METRIC} = {score:.4f} with {search.best_params_} ({n_iter} combinations)")
    return search


def tune(
    data: pd.DataFrame | None = None,
    model_names: list[str] | None = None,
    top_k: int | None = None,
    n_iter: int | None = None,
    model_path: Path | None = None,
    save: bool = True,
    registered_model_name: str | None = None,
    model_alias: str | None = None,
) -> TuningResult:
    """Tune the given models (or the best ones of the comparison), then evaluate and save the best one.

    Args:
        data: Processed data (defaults to ``load_processed()``).
        model_names: Models to tune; defaults to the ``top_k`` best tunable models by cross-validation.
        top_k: Number of models to tune when ``model_names`` is not given (defaults to ``config.TUNE_TOP_K``).
        n_iter: Random combinations per model (defaults to ``config.N_ITER``).
        model_path: Where to save the pipeline (defaults to ``models/<config.MODEL_FILE>``).
        save: Save the best tuned pipeline.
        registered_model_name: If set, register the model under this name in the MLflow Model Registry.
        model_alias: With ``registered_model_name``: alias given to the new version.

    Returns:
        TuningResult: Best tuned model, its hyperparameters, the search summary and the test metrics.

    """
    x_train, x_test, y_train, y_test = prepare_data(data)
    if model_names is None:
        ranking = compare_models(x_train, y_train, [name for name in MODELS if name in PARAM_SPACES])
        model_names = [str(name) for name in ranking.index[: top_k or config.TUNE_TOP_K]]
        logger.info(f"models to tune (best by cross-validation): {model_names}")

    searches = {name: tune_model(name, x_train, y_train, n_iter) for name in model_names}
    search_results = pd.DataFrame(
        {
            "model": list(searches),
            PRIMARY_METRIC: [to_metric(PRIMARY_METRIC, search.best_score_) for search in searches.values()],
            "best_params": [search.best_params_ for search in searches.values()],
        }
    ).sort_values(PRIMARY_METRIC, ascending=LOWER_IS_BETTER, ignore_index=True)
    best_name = str(search_results.loc[0, "model"])
    best = searches[best_name]
    pipeline: Pipeline = best.best_estimator_
    test_metrics = evaluate_on_test(pipeline, x_test, y_test)
    logger.info(f"best tuned model: {best_name} {best.best_params_}, test set: {test_metrics}")
    path = save_model(pipeline, model_path) if save else None

    setup_mlflow()
    with mlflow.start_run(run_name=f"tune-{best_name}") as run:
        for name, search in searches.items():  # one child run per tried combination
            for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], strict=True):
                if np.isnan(score):
                    continue
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_params({"model": name, **params})
                    mlflow.log_metric(f"cv_{PRIMARY_METRIC}", to_metric(PRIMARY_METRIC, float(score)))
        mlflow.log_params({"model": best_name, "problem_type": config.PROBLEM_TYPE, **best.best_params_})
        mlflow.log_metric(f"cv_{PRIMARY_METRIC}", to_metric(PRIMARY_METRIC, best.best_score_))
        mlflow.log_metrics({f"test_{metric}": value for metric, value in test_metrics.items()})
        mlflow.log_text(search_results.to_csv(index=False), "tuning_results.csv")
        log_pipeline(pipeline, x_test, registered_model_name, model_alias)
    return TuningResult(
        model_name=best_name,
        best_params=best.best_params_,
        pipeline=pipeline,
        search_results=search_results,
        test_metrics=test_metrics,
        model_path=path,
        run_id=run.info.run_id,
    )


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Arguments to parse (defaults to ``sys.argv``).

    """
    parser = argparse.ArgumentParser(description="Tune the hyperparameters of the best models and save the best one.")
    parser.add_argument("--models", nargs="+", metavar="NAME", help=f"models to tune: {list(PARAM_SPACES)}")
    parser.add_argument("--top-k", type=int, default=None, help="tune the K best models of the comparison")
    parser.add_argument("--n-iter", type=int, default=None, help="random combinations tried per model")
    parser.add_argument("--no-save", action="store_true", help="do not save the tuned model")
    parser.add_argument("--register", metavar="NAME", default=None, help="register the model under NAME in MLflow")
    parser.add_argument(
        "--alias",
        metavar="ALIAS",
        default=None,
        help="with --register: alias of the new version (e.g. production), served via MLFLOW_MODEL_URI",
    )
    args = parser.parse_args(argv)
    if args.alias and not args.register:
        parser.error("--alias needs --register NAME")
    result = tune(
        model_names=args.models,
        top_k=args.top_k,
        n_iter=args.n_iter,
        save=not args.no_save,
        registered_model_name=args.register,
        model_alias=args.alias,
    )
    logger.info(f"best tuned model: {result.model_name}, test metrics: {result.test_metrics}")


if __name__ == "__main__":
    main()
