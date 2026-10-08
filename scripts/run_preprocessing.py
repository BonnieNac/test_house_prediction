"""Preprocessing: data/raw/<RAW_DATA_FILE> → data/processed/<PROCESSED_DATA_FILE> (make preprocess)."""

from test_house_prediction.core.pipelines import run_preprocessing
from test_house_prediction.core.utils import setup_file_logging

if __name__ == "__main__":
    setup_file_logging("preprocess")
    run_preprocessing()
