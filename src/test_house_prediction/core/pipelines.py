"""End-to-end pipelines: preprocessing (raw → processed) and training."""

from pathlib import Path

from loguru import logger

from test_house_prediction.core.data_io.clean import clean
from test_house_prediction.core.data_io.load import load_raw, save_processed
from test_house_prediction.core.features.build_features import build_features
from test_house_prediction.core.models.train import TrainingResult, train


def run_preprocessing() -> Path:
    """Load the raw data, clean it, build the features and write the processed dataset.

    Returns:
        Path: The processed dataset.

    """
    processed = build_features(clean(load_raw()))
    path = save_processed(processed)
    logger.info(f"preprocessing done: {processed.shape[0]} rows, {processed.shape[1]} columns")
    return path


def run_training() -> TrainingResult:
    """Train on the processed dataset (``run_preprocessing`` must have run before).

    Returns:
        TrainingResult: Best model and its metrics.

    """
    return train()
