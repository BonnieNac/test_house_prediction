"""Generic cleaning of the raw dataset: complete ``clean`` with the project's own rules."""

import re

import pandas as pd
from loguru import logger

from test_house_prediction.core import config


def standardize_name(name: object) -> str:
    """Lower-case a column name and replace spaces, dashes and dots by ``_``.

    Args:
        name: Raw column name.

    Returns:
        str: Standardized name, as used in core/config.py.

    """
    return re.sub(r"[\s\-.]+", "_", str(name).strip().lower())


def parse_dates(values: pd.Series, strict: bool = True) -> pd.Series:
    """Convert dates written as text (ISO, 31/12/2024, 2024-12-31 08:00…) to datetime.

    The format may vary from one row to the next; dates are read day first when some of them
    start with a day (31/12/2024, 31.12.2024).

    Args:
        values: Dates (text or already datetime).
        strict: Raise on an unparsable value; otherwise it becomes missing.

    Returns:
        pd.Series: The dates.

    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = values.astype(str)
    day_first = bool(text.str.match(r"^\d{1,2}[/.]").any())
    return pd.to_datetime(text, errors="raise" if strict else "coerce", format="mixed", dayfirst=day_first)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the column names and replace spaces, dashes and dots by ``_``.

    Args:
        df: Data to clean.

    Returns:
        pd.DataFrame: Data with standardized column names.

    """
    return df.rename(columns=standardize_name)


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading and trailing spaces in text columns; empty strings become missing values.

    Args:
        df: Data to clean.

    Returns:
        pd.DataFrame: Data with stripped text values.

    """
    df = df.copy()
    for column in df.select_dtypes(include=["object", "string"]).columns:
        stripped = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        df[column] = stripped.replace("", pd.NA)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows.

    Args:
        df: Data to clean.

    Returns:
        pd.DataFrame: Data without duplicates.

    """
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        logger.info(f"{n_duplicates} duplicated rows removed")
    return df.drop_duplicates().reset_index(drop=True)


def drop_mostly_empty_columns(df: pd.DataFrame, max_missing_ratio: float = 0.9) -> pd.DataFrame:
    """Remove the columns whose share of missing values exceeds ``max_missing_ratio``.

    Args:
        df: Data to clean.
        max_missing_ratio: Maximum share of missing values allowed in a column.

    Returns:
        pd.DataFrame: Data without the mostly empty columns.

    """
    missing_ratio = df.isna().mean()
    to_drop = [str(c) for c in missing_ratio[missing_ratio > max_missing_ratio].index if c != config.TARGET]
    if to_drop:
        logger.info(f"mostly empty columns removed: {to_drop}")
    return df.drop(columns=to_drop)


def drop_missing_target(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Remove the rows without a target value (they cannot be used to train).

    Args:
        df: Data to clean.
        target: Target column (defaults to ``config.TARGET``).

    Returns:
        pd.DataFrame: Data with a known target on every row.

    Raises:
        KeyError: If the target column is missing.

    """
    target = target or config.TARGET
    if target not in df.columns:
        raise KeyError(
            f"target column {target!r} not found: set TARGET in core/config.py (columns: {list(df.columns)})"
        )
    return df.dropna(subset=[target]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning step to the raw data.

    Add the project-specific rules here: business filters, outliers, type fixes…

    Args:
        df: Raw data.

    Returns:
        pd.DataFrame: Clean data.

    """
    df = standardize_column_names(df)
    df = strip_strings(df)
    df = drop_duplicates(df)
    df = drop_mostly_empty_columns(df)
    df = drop_missing_target(df)
    df = df.infer_objects()  # text columns that only held numbers become numeric again
    logger.info(f"clean data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
