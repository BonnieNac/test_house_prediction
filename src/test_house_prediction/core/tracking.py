"""MLflow tracking configuration shared by every training or evaluation script."""

import os

import mlflow
from dotenv import load_dotenv

from test_house_prediction.core.utils.paths import PROJECT_ROOT

DEFAULT_TRACKING_URI: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_EXPERIMENT_NAME: str = "test_house_prediction"


def setup_mlflow(experiment_name: str | None = None) -> str:
    """Point MLflow to the tracking server and select the experiment.

    ``MLFLOW_TRACKING_URI`` and ``MLFLOW_EXPERIMENT_NAME`` (environment or ``.env``) take
    precedence over the defaults: a local SQLite store at the project root.

    Args:
        experiment_name: Experiment to use; overrides ``MLFLOW_EXPERIMENT_NAME``.

    Returns:
        str: The tracking URI in use.

    """
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME)
    return tracking_uri
