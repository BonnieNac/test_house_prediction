"""Inference: load the saved pipeline and predict on new data.

The Docker image holds code only: the model comes from ``models/<config.MODEL_FILE>`` (make train,
mounted in the API container).
With ``MLFLOW_MODEL_URI`` set (e.g. ``models:/<name>@production``), it comes from the MLflow Model Registry.
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from test_house_prediction.core import config
from test_house_prediction.core.tracking import setup_mlflow
from test_house_prediction.core.utils.paths import MODELS_DIR


@lru_cache(maxsize=4)
def load_model(path: Path | None = None) -> Pipeline:
    """Load the pipeline saved by the training (cached: read from disk only once).

    Args:
        path: Saved pipeline (defaults to ``models/<config.MODEL_FILE>``).

    Returns:
        Pipeline: The fitted preprocessing + model pipeline.

    Raises:
        FileNotFoundError: If no model has been trained yet.

    """
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    if path is None and model_uri:
        setup_mlflow()
        try:
            registered: Pipeline = mlflow.sklearn.load_model(model_uri)
        except mlflow.exceptions.MlflowException as error:
            raise FileNotFoundError(f"model {model_uri} unavailable in MLflow: {error.message}") from error
        return registered
    path = path or MODELS_DIR / config.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}: run 'make train' first")
    pipeline: Pipeline = joblib.load(path)
    return pipeline


def predict(features: pd.DataFrame, model: Pipeline | None = None) -> np.ndarray:
    """Predict the target for new rows.

    The rows must contain the same columns as the training features (after ``build_features``);
    extra columns are ignored by the preprocessing.

    Args:
        features: Rows to predict.
        model: Fitted pipeline (defaults to ``load_model()``).

    Returns:
        np.ndarray: One prediction per row.

    """
    if model is None:
        model = load_model()
    return np.asarray(model.predict(features))
