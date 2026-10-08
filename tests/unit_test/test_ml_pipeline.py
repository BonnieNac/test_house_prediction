"""Tests of the ML pipeline on a synthetic dataset: adapt them once the real data is known."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from test_house_prediction.api.main import app
from test_house_prediction.core import config, pipelines
from test_house_prediction.core.data_io import load
from test_house_prediction.core.data_io.clean import clean, drop_missing_target, parse_dates
from test_house_prediction.core.features.build_features import build_features
from test_house_prediction.core.features.preprocess import (
    build_preprocessor,
    infer_feature_types,
    split_features_target,
)
from test_house_prediction.core.models import evaluate
from test_house_prediction.core.models import predict as predict_module
from test_house_prediction.core.models import train as train_module
from test_house_prediction.core.models import tune as tune_module
from test_house_prediction.core.models.evaluate import PRIMARY_METRIC, cross_validate_model

FAST_MODELS = ["dummy_baseline", "linear_regression"]
FAST_TUNABLE = ["ridge", "decision_tree"]
N_ROWS = 200

client = TestClient(app)


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """Return a synthetic raw dataset, with one duplicated row and one row without target."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Num Feature": rng.normal(size=N_ROWS),
            "other-num": rng.uniform(0, 10, N_ROWS),
            "Category": rng.choice(["a", "b", "c"], N_ROWS),
        }
    )
    signal = df["Num Feature"] + 0.3 * df["other-num"] + (df["Category"] == "a")
    df[config.TARGET] = 3 * signal + rng.normal(scale=0.1, size=N_ROWS)
    without_target = df.head(1).assign(**{config.TARGET: None})
    return pd.concat([df, df.head(1), without_target], ignore_index=True)


@pytest.fixture
def processed(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Return the processed dataset, as written by the preprocessing pipeline."""
    return build_features(clean(raw_data))


@pytest.fixture(autouse=True)
def synthetic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the tests on the synthetic data whatever the dataset settings written in core/config.py."""
    monkeypatch.setattr(config, "RAW_DATA_FILE", "dataset.csv")
    monkeypatch.setattr(config, "PROCESSED_DATA_FILE", "dataset_processed.csv")
    monkeypatch.setattr(config, "ID_COLUMNS", [])
    monkeypatch.setattr(config, "NUMERIC_FEATURES", None)
    monkeypatch.setattr(config, "CATEGORICAL_FEATURES", None)


@pytest.fixture(autouse=True)
def fast_and_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the cross-validation in-process (small data) and forget the model loaded by a previous test."""
    monkeypatch.setattr(config, "N_JOBS", 1)
    predict_module.load_model.cache_clear()


@pytest.fixture(scope="session")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return one temporary SQLite store for the whole session (creating a store takes seconds)."""
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


@pytest.fixture(autouse=True)
def local_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking_uri: str) -> None:
    """Isolate MLflow in the temporary store; the tests read their own runs by run_id."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "tests")
    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)  # the API reads models/ unless a test sets it


def test_clean(raw_data: pd.DataFrame) -> None:
    """Cleaning standardizes the names and removes duplicates and rows without target."""
    df = clean(raw_data)

    assert all(c == c.lower() and " " not in c and "-" not in c for c in df.columns)
    assert len(df) == N_ROWS
    assert df[config.TARGET].notna().all()


def test_parse_dates_day_first() -> None:
    """Dates written day first are read as such; an invalid date raises unless strict is off."""
    dates = parse_dates(pd.Series(["31/12/2024", "01/02/2025"]))
    assert dates.tolist() == [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-02-01")]
    with pytest.raises(ValueError):
        parse_dates(pd.Series(["2024-01-01", "not a date"]))
    assert parse_dates(pd.Series(["2024-01-01", "not a date"]), strict=False).isna().tolist() == [False, True]


def test_drop_missing_target_requires_target() -> None:
    """A clear error is raised when the target column is missing."""
    with pytest.raises(KeyError, match="TARGET"):
        drop_missing_target(pd.DataFrame({"x": [1]}))


def test_save_and_load(tmp_path: Path, raw_data: pd.DataFrame, processed: pd.DataFrame) -> None:
    """Raw and processed files are read back with the same shape."""
    raw_data.to_csv(tmp_path / "raw.csv", index=False)
    assert load.load_raw("raw.csv", directory=tmp_path).shape == raw_data.shape

    path = load.save_processed(processed, "processed.csv", directory=tmp_path)
    assert load.load_processed(path.name, directory=tmp_path).shape == processed.shape


def test_load_raw_latest_snapshot(tmp_path: Path, raw_data: pd.DataFrame) -> None:
    """A file pattern reads the latest dated snapshot, and reports when none exists."""
    raw_data.head(3).to_csv(tmp_path / "api_2026-01-01.csv", index=False)
    raw_data.head(5).to_csv(tmp_path / "api_2026-02-01.csv", index=False)

    assert len(load.load_raw("api_*.csv", directory=tmp_path)) == 5
    with pytest.raises(FileNotFoundError, match="make fetch"):
        load.load_raw("sql_*.parquet", directory=tmp_path)


def test_read_table_errors(tmp_path: Path) -> None:
    """Missing files and unsupported extensions are rejected."""
    with pytest.raises(FileNotFoundError):
        load.read_table(tmp_path / "missing.csv")
    (tmp_path / "data.txt").write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        load.read_table(tmp_path / "data.txt")


def test_preprocessor_outputs_no_missing_values(processed: pd.DataFrame) -> None:
    """The preprocessor imputes and encodes every feature."""
    features, _ = split_features_target(processed)
    features.iloc[0, 0] = None

    transformed = build_preprocessor(*infer_feature_types(features)).fit_transform(features)

    assert not np.isnan(transformed).any()


def test_cross_validate_model(processed: pd.DataFrame) -> None:
    """Cross-validation returns the mean and standard deviation of each metric."""
    features, target = split_features_target(processed)
    pipeline = train_module.build_pipeline(
        train_module.MODELS[FAST_MODELS[1]](), build_preprocessor(*infer_feature_types(features))
    )

    scores = cross_validate_model(pipeline, features, target, n_splits=3)

    assert {f"{PRIMARY_METRIC}_mean", f"{PRIMARY_METRIC}_std", "fit_time"} <= scores.keys()


def test_unknown_metric_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PRIMARY_METRIC that is not computed is reported with the available ones."""
    monkeypatch.setattr(evaluate, "PRIMARY_METRIC", "mape")
    with pytest.raises(ValueError, match="unavailable"):
        evaluate.scoring()


def test_every_model_fits(processed: pd.DataFrame) -> None:
    """Every model of the catalog trains and predicts inside the pipeline."""
    features, target = split_features_target(processed)
    preprocessor = build_preprocessor(*infer_feature_types(features))
    for name, make_model in train_module.MODELS.items():
        pipeline = train_module.build_pipeline(make_model(), preprocessor).fit(features, target)
        assert len(pipeline.predict(features.head(3))) == 3, name


def test_compare_models_rejects_unknown_model(processed: pd.DataFrame) -> None:
    """An unknown model name is reported with the available ones."""
    features, target = split_features_target(processed)
    with pytest.raises(KeyError, match="unknown"):
        train_module.compare_models(features, target, ["not_a_model"])


def test_train_saves_a_loadable_model(tmp_path: Path, processed: pd.DataFrame) -> None:
    """Training ranks the models, evaluates the best one and saves the pipeline."""
    result = train_module.train(data=processed, model_names=FAST_MODELS, model_path=tmp_path / "model.joblib")

    assert set(result.cv_results.index) == set(FAST_MODELS)
    assert result.model_name == result.cv_results.index[0]
    assert PRIMARY_METRIC in result.test_metrics
    assert result.model_path is not None
    features, _ = split_features_target(processed)
    assert len(predict_module.predict(features.head(3), predict_module.load_model(result.model_path))) == 3
    run = mlflow.get_run(result.run_id)
    assert run.data.params["model"] == result.model_name
    assert f"test_{PRIMARY_METRIC}" in run.data.metrics


def test_main(tmp_path: Path, processed: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command line trains on the processed data and saves the model in models/."""
    monkeypatch.setattr(train_module, "load_processed", lambda: processed)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    train_module.main(["--models", *FAST_MODELS, "--register", "test-model"])
    assert len(mlflow.MlflowClient().search_model_versions("name='test-model'")) == 1
    assert (tmp_path / config.MODEL_FILE).exists()


def test_main_lists_models(capsys: pytest.CaptureFixture[str]) -> None:
    """--list prints the model catalog."""
    train_module.main(["--list"])
    assert capsys.readouterr().out.split() == list(train_module.MODELS)


def test_param_spaces_match_the_models(processed: pd.DataFrame) -> None:
    """Every search space targets a model of the catalog and only parameters that the pipeline accepts."""
    features, _ = split_features_target(processed)
    preprocessor = build_preprocessor(*infer_feature_types(features))
    for name, space in train_module.PARAM_SPACES.items():
        assert name in train_module.MODELS, name
        for sub_space in space:
            pipeline = train_module.build_pipeline(train_module.MODELS[name](), preprocessor)
            pipeline.set_params(**{parameter: values[0] for parameter, values in sub_space.items()})


def test_tune_given_models(tmp_path: Path, processed: pd.DataFrame) -> None:
    """Tuning the given models returns the best one with its hyperparameters, evaluated and saved."""
    result = tune_module.tune(data=processed, model_names=FAST_TUNABLE, n_iter=2, model_path=tmp_path / "tuned.joblib")

    assert result.model_name == result.search_results.loc[0, "model"]
    assert set(result.search_results["model"]) == set(FAST_TUNABLE)
    assert result.best_params and all(key.startswith("model__") for key in result.best_params)
    assert PRIMARY_METRIC in result.test_metrics
    assert result.model_path is not None and result.model_path.exists()
    run = mlflow.get_run(result.run_id)
    assert run.data.params["model"] == result.model_name
    assert f"test_{PRIMARY_METRIC}" in run.data.metrics


def test_tune_best_models_of_the_comparison(processed: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without model names, the top_k best tunable models of the comparison are tuned."""
    spaces = {name: train_module.PARAM_SPACES[name] for name in FAST_TUNABLE}
    monkeypatch.setattr(tune_module, "PARAM_SPACES", spaces)

    result = tune_module.tune(data=processed, top_k=1, n_iter=1, save=False)

    assert len(result.search_results) == 1
    assert result.model_name in FAST_TUNABLE
    assert result.model_path is None


def test_tune_rejects_a_model_without_search_space(processed: pd.DataFrame) -> None:
    """A model without search space (a baseline) is reported with the tunable models."""
    features, target = split_features_target(processed)
    with pytest.raises(KeyError, match="no search space"):
        tune_module.tune_model(FAST_MODELS[0], features, target)


def test_tune_main(tmp_path: Path, processed: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command line tunes the given model and saves it in models/."""
    monkeypatch.setattr(train_module, "load_processed", lambda: processed)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    tune_module.main(["--models", FAST_TUNABLE[0], "--n-iter", "2"])

    assert (tmp_path / config.MODEL_FILE).exists()


def test_run_preprocessing(tmp_path: Path, raw_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """The preprocessing pipeline turns data/raw into data/processed."""
    monkeypatch.setattr(load, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(load, "PROCESSED_DATA_DIR", tmp_path / "processed")
    (tmp_path / "raw").mkdir()
    raw_data.to_csv(tmp_path / "raw" / config.RAW_DATA_FILE, index=False)

    path = pipelines.run_preprocessing()

    assert path == tmp_path / "processed" / config.PROCESSED_DATA_FILE
    assert len(load.load_processed()) > 0


def test_run_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training pipeline delegates to train()."""
    monkeypatch.setattr(pipelines, "train", lambda: "trained")
    assert pipelines.run_training() == "trained"


def test_predict_endpoint(tmp_path: Path, processed: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /predict returns one prediction per row, and 422 when columns are missing."""
    train_module.train(data=processed, model_names=FAST_MODELS, model_path=tmp_path / config.MODEL_FILE)
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path)
    features, _ = split_features_target(processed)
    records = json.loads(features.head(2).to_json(orient="records"))

    response = client.post("/predict", json={"records": records})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2

    assert client.post("/predict", json={"records": [{"unknown": 1}]}).status_code == 422


def test_registered_model_served_from_mlflow(
    tmp_path: Path, processed: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--alias marks the registered version, and the API serves it through MLFLOW_MODEL_URI."""
    monkeypatch.setattr(train_module, "load_processed", lambda: processed)
    train_module.main(["--models", *FAST_MODELS, "--no-save", "--register", "served-model", "--alias", "production"])
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/served-model@production")
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path / "no-local-model")
    features, _ = split_features_target(processed)

    response = client.post("/predict", json={"records": json.loads(features.head(2).to_json(orient="records"))})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_unknown_registered_model_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model missing from MLflow is reported as unavailable (503), like a missing file."""
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/missing-model@production")
    assert client.post("/predict", json={"records": [{"x": 1}]}).status_code == 503


def test_alias_needs_register() -> None:
    """--alias without --register is rejected."""
    with pytest.raises(SystemExit):
        train_module.main(["--alias", "production"])


def test_predict_endpoint_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /predict answers 503 until a model has been trained."""
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path)
    response = client.post("/predict", json={"records": [{"x": 1}]})
    assert response.status_code == 503
