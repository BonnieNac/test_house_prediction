"""Preprocessing: data/raw/<RAW_DATA_FILE> → data/processed/<PROCESSED_DATA_FILE> (make preprocess)."""

from test_house_prediction.core.pipelines import run_preprocessing

if __name__ == "__main__":
    run_preprocessing()
