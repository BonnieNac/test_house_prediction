"""Tests for the MLflow tracking setup (the training is tested in test_ml_pipeline.py)."""

import pytest
from pytest_mock import MockerFixture

from test_house_prediction.core import tracking


def test_setup_mlflow_defaults(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without environment variables, the local SQLite store and the package experiment are used."""
    mocker.patch.object(tracking, "load_dotenv")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    set_uri = mocker.patch.object(tracking.mlflow, "set_tracking_uri")
    set_experiment = mocker.patch.object(tracking.mlflow, "set_experiment")

    assert tracking.setup_mlflow() == tracking.DEFAULT_TRACKING_URI
    set_uri.assert_called_once_with(tracking.DEFAULT_TRACKING_URI)
    set_experiment.assert_called_once_with(tracking.DEFAULT_EXPERIMENT_NAME)


def test_setup_mlflow_from_environment(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME override the defaults."""
    mocker.patch.object(tracking, "load_dotenv")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp")
    mocker.patch.object(tracking.mlflow, "set_tracking_uri")
    set_experiment = mocker.patch.object(tracking.mlflow, "set_experiment")

    assert tracking.setup_mlflow() == "http://localhost:5000"
    set_experiment.assert_called_once_with("exp")
