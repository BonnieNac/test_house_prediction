"""Read and write the datasets of data/raw and data/processed."""

from pathlib import Path

import pandas as pd
from loguru import logger

from test_house_prediction.core import config
from test_house_prediction.core.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

READERS = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
}


def read_table(path: Path) -> pd.DataFrame:
    """Read a tabular file, the reader being chosen from its extension.

    Args:
        path: File to read (.csv, .xlsx, .xls, .json or .parquet).

    Returns:
        pd.DataFrame: The file content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not supported.

    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: put the dataset there or change the file name in core/config.py")
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"unsupported extension {path.suffix!r}, expected one of {sorted(READERS)}")
    df: pd.DataFrame = reader(path)
    logger.info(f"{path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def resolve(directory: Path, filename: str) -> Path:
    """Return the path of a data file; a pattern such as ``api_*.parquet`` selects the latest match.

    Dated snapshots (``make fetch``) sort chronologically, so the last match is the most recent one.

    Args:
        directory: Folder of the file.
        filename: File name or glob pattern.

    Returns:
        Path: The file.

    Raises:
        FileNotFoundError: If no file matches the pattern.

    """
    if not any(char in filename for char in "*?["):
        return directory / filename
    matches = sorted(directory.glob(filename))
    if not matches:
        raise FileNotFoundError(f"no file matching {filename!r} in {directory}: run 'make fetch' first")
    return matches[-1]


def load_raw(filename: str | None = None, directory: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset.

    Args:
        filename: File name or pattern (defaults to ``config.RAW_DATA_FILE``).
        directory: Folder of the file (defaults to data/raw).

    Returns:
        pd.DataFrame: The raw data.

    """
    return read_table(resolve(directory or RAW_DATA_DIR, filename or config.RAW_DATA_FILE))


def load_processed(filename: str | None = None, directory: Path | None = None) -> pd.DataFrame:
    """Load the processed dataset written by ``save_processed``.

    Args:
        filename: File name (defaults to ``config.PROCESSED_DATA_FILE``).
        directory: Folder of the file (defaults to data/processed).

    Returns:
        pd.DataFrame: The processed data.

    """
    df = read_table((directory or PROCESSED_DATA_DIR) / (filename or config.PROCESSED_DATA_FILE))
    return df


def save_processed(df: pd.DataFrame, filename: str | None = None, directory: Path | None = None) -> Path:
    """Write the processed dataset (CSV or Parquet, according to the extension).

    Args:
        df: Data to write.
        filename: File name (defaults to ``config.PROCESSED_DATA_FILE``).
        directory: Destination folder (defaults to data/processed).

    Returns:
        Path: The written file.

    """
    path = (directory or PROCESSED_DATA_DIR) / (filename or config.PROCESSED_DATA_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    logger.info(f"processed data written to {path}")
    return path
