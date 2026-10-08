"""Model training: classic regression models compared by cross-validation.

Each model is built by a function: add or tune them, then register them in ``MODELS``, and their
hyperparameter search space in ``PARAM_SPACES`` (used by ``tune.py``).
``train`` compares the models, refits the best one on the training set, evaluates it on the
test set and saves the whole pipeline (preprocessing + model) in models/.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from test_house_prediction.core import config
from test_house_prediction.core.data_io.load import load_processed
from test_house_prediction.core.features.preprocess import (
    build_preprocessor,
    infer_feature_types,
    split_features_target,
    split_train_test,
)
from test_house_prediction.core.models.evaluate import (
    PRIMARY_METRIC,
    cross_validate_model,
    evaluate_on_test,
    rank_models,
)
from test_house_prediction.core.tracking import setup_mlflow
from test_house_prediction.core.utils.paths import MODELS_DIR


# --- Models ------------------------------------------------------------------------------------
def dummy_baseline() -> BaseEstimator:
    """Baseline that always predicts the mean of the target: every model must beat it."""
    return DummyRegressor(strategy="mean")


def linear_regression() -> BaseEstimator:
    """Ordinary least squares linear regression: the reference interpretable model."""
    return LinearRegression()


def ridge() -> BaseEstimator:
    """Ridge regression: linear model with L2 regularization (correlated features)."""
    return Ridge(alpha=1.0)


def lasso() -> BaseEstimator:
    """Lasso regression: linear model with L1 regularization (sets useless coefficients to 0)."""
    return Lasso(alpha=0.1, max_iter=10_000)


def k_nearest_neighbors() -> BaseEstimator:
    """K nearest neighbors: mean target of the closest training rows."""
    return KNeighborsRegressor(n_neighbors=5)


def decision_tree() -> BaseEstimator:
    """Decision tree: readable rules, but overfits easily (limit max_depth)."""
    return DecisionTreeRegressor(max_depth=8, random_state=config.RANDOM_STATE)


def random_forest() -> BaseEstimator:
    """Random forest: robust ensemble of trees, a good default choice."""
    return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=config.RANDOM_STATE)


def gradient_boosting() -> BaseEstimator:
    """Gradient boosting (histogram-based, like LightGBM): often the best on tabular data."""
    return HistGradientBoostingRegressor(random_state=config.RANDOM_STATE)


def svr() -> BaseEstimator:
    """Support vector regression (RBF kernel): good on small datasets, slow beyond ~50,000 rows."""
    return SVR(kernel="rbf", C=1.0)


MODELS: dict[str, Callable[[], BaseEstimator]] = {
    "dummy_baseline": dummy_baseline,
    "linear_regression": linear_regression,
    "ridge": ridge,
    "lasso": lasso,
    "k_nearest_neighbors": k_nearest_neighbors,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
    "svr": svr,
}

# --- Hyperparameter search spaces (tune.py) -----------------------------------------------------
# For each model: a list of sub-spaces, each one a dict "model__<parameter>" → candidate values.
# Several sub-spaces describe parameters that depend on each other (e.g. a solver and its options).
# Models without a useful hyperparameter (baselines, plain linear regression) have no entry.
ParamSpace = dict[str, list[Any]]

PARAM_SPACES: dict[str, list[ParamSpace]] = {
    "ridge": [{"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}],
    "lasso": [{"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}],
    "k_nearest_neighbors": [{"model__n_neighbors": [3, 5, 7, 11, 15, 25], "model__weights": ["uniform", "distance"]}],
    "decision_tree": [{"model__max_depth": [3, 5, 8, 12, None], "model__min_samples_leaf": [1, 2, 5, 10, 20]}],
    "random_forest": [
        {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", 0.5, 1.0],
        }
    ],
    "gradient_boosting": [
        {
            "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "model__max_iter": [100, 200, 400],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }
    ],
    "svr": [
        {"model__C": [0.1, 1.0, 10.0, 100.0], "model__epsilon": [0.01, 0.1, 0.5], "model__gamma": ["scale", 0.01, 0.1]}
    ],
}


# --- Training ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a training run."""

    model_name: str
    pipeline: Pipeline
    cv_results: pd.DataFrame  # one row per compared model, best first
    test_metrics: dict[str, float]
    model_path: Path | None
    run_id: str


def build_pipeline(model: BaseEstimator, preprocessor: ColumnTransformer) -> Pipeline:
    """Chain the preprocessing and the model: a single object to fit, save and serve.

    Args:
        model: Unfitted model.
        preprocessor: Unfitted preprocessor (``build_preprocessor``).

    Returns:
        Pipeline: ``preprocess`` then ``model``.

    """
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def compare_models(features: pd.DataFrame, target: pd.Series, model_names: list[str] | None = None) -> pd.DataFrame:
    """Cross-validate several models on the training set.

    Args:
        features: Training features.
        target: Training target.
        model_names: Models of ``MODELS`` to compare (defaults to all of them).

    Returns:
        pd.DataFrame: Cross-validation results, best model first.

    Raises:
        KeyError: If a model name is unknown.

    """
    names = model_names or list(MODELS)
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        raise KeyError(f"unknown models {unknown}, available: {list(MODELS)}")
    numeric, categorical = infer_feature_types(features)
    results: dict[str, dict[str, float]] = {}
    for name in names:
        pipeline = build_pipeline(MODELS[name](), build_preprocessor(numeric, categorical))
        results[name] = cross_validate_model(pipeline, features, target)
        logger.info(f"{name}: {PRIMARY_METRIC} = {results[name][f'{PRIMARY_METRIC}_mean']:.4f}")
    return rank_models(results)


def prepare_data(
    data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the processed data and split it into training and test sets.

    Args:
        data: Processed data (defaults to ``load_processed()``).

    Returns:
        tuple: ``X_train, X_test, y_train, y_test``.

    """
    data = load_processed() if data is None else data
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = split_train_test(features, target)
    logger.info(f"train: {len(x_train)} rows, test: {len(x_test)} rows, features: {list(features.columns)}")
    return x_train, x_test, y_train, y_test


def save_model(pipeline: Pipeline, path: Path | None = None) -> Path:
    """Save the fitted pipeline with joblib.

    Args:
        pipeline: Fitted pipeline.
        path: Destination (defaults to ``models/<config.MODEL_FILE>``).

    Returns:
        Path: The written file.

    """
    path = path or MODELS_DIR / config.MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"model saved to {path}")
    return path


def log_pipeline(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    registered_model_name: str | None = None,
    alias: str | None = None,
) -> None:
    """Log the fitted pipeline in the active MLflow run, with its signature and an input example.

    Args:
        pipeline: Fitted pipeline.
        x_test: Test features (signature and input example).
        registered_model_name: If set, register the model under this name in the MLflow Model Registry.
        alias: With ``registered_model_name``: alias given to the new version (e.g. ``staging``,
            ``production``); the API serves ``models:/<name>@<alias>`` (``MLFLOW_MODEL_URI``).

    """
    info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        signature=infer_signature(x_test, pipeline.predict(x_test)),
        input_example=x_test.head(5),
        registered_model_name=registered_model_name,
        serialization_format="cloudpickle",
    )
    if registered_model_name and alias:
        version = info.registered_model_version
        mlflow.MlflowClient().set_registered_model_alias(registered_model_name, alias, str(version))
        logger.info(f"model {registered_model_name} version {version} has the alias {alias!r}")


def train(
    data: pd.DataFrame | None = None,
    model_names: list[str] | None = None,
    model_path: Path | None = None,
    save: bool = True,
    registered_model_name: str | None = None,
    model_alias: str | None = None,
) -> TrainingResult:
    """Compare the models, refit the best one, evaluate it on the test set and save it.

    Args:
        data: Processed data (defaults to ``load_processed()``).
        model_names: Models of ``MODELS`` to compare (defaults to all of them).
        model_path: Where to save the pipeline (defaults to ``models/<config.MODEL_FILE>``).
        save: Save the fitted pipeline.
        registered_model_name: If set, register the model under this name in the MLflow Model Registry.
        model_alias: With ``registered_model_name``: alias given to the new version.

    Returns:
        TrainingResult: Best model, comparison table, test metrics and saved file.

    """
    x_train, x_test, y_train, y_test = prepare_data(data)
    cv_results = compare_models(x_train, y_train, model_names)
    best_name = str(cv_results.index[0])
    logger.info(f"best model by cross-validation: {best_name}\n{cv_results.round(4).to_string()}")

    pipeline = build_pipeline(MODELS[best_name](), build_preprocessor(*infer_feature_types(x_train)))
    pipeline.fit(x_train, y_train)
    test_metrics = evaluate_on_test(pipeline, x_test, y_test)
    logger.info(f"{best_name} on the test set: {test_metrics}")
    path = save_model(pipeline, model_path) if save else None

    setup_mlflow()
    with mlflow.start_run(run_name=best_name) as run:
        for name, row in cv_results.iterrows():  # one child run per compared model
            with mlflow.start_run(run_name=str(name), nested=True):
                mlflow.log_param("model", name)
                mlflow.log_metrics({f"cv_{metric}": float(value) for metric, value in row.items()})
        mlflow.log_params({"model": best_name, "problem_type": config.PROBLEM_TYPE, "cv_folds": config.CV_FOLDS})
        mlflow.log_metrics({f"cv_{metric}": float(value) for metric, value in cv_results.loc[best_name].items()})
        mlflow.log_metrics({f"test_{metric}": value for metric, value in test_metrics.items()})
        mlflow.log_text(cv_results.to_csv(), "model_comparison.csv")
        log_pipeline(pipeline, x_test, registered_model_name, model_alias)
    return TrainingResult(
        model_name=best_name,
        pipeline=pipeline,
        cv_results=cv_results,
        test_metrics=test_metrics,
        model_path=path,
        run_id=run.info.run_id,
    )


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Arguments to parse (defaults to ``sys.argv``).

    """
    parser = argparse.ArgumentParser(description="Compare the models, then train and save the best one.")
    parser.add_argument("--models", nargs="+", metavar="NAME", help=f"models to compare (default: all): {list(MODELS)}")
    parser.add_argument("--list", action="store_true", help="list the available models and exit")
    parser.add_argument("--no-save", action="store_true", help="do not save the trained model")
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
    if args.list:
        print("\n".join(MODELS))
        return
    result = train(
        model_names=args.models,
        save=not args.no_save,
        registered_model_name=args.register,
        model_alias=args.alias,
    )
    logger.info(f"best model: {result.model_name}, test metrics: {result.test_metrics}")
