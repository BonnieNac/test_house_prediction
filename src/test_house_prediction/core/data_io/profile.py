"""Statistical profile of a dataset, used to adapt core/config.py (make profile, /adapt-config).

The profile holds statistics only, never rows: types, missing values, number of distinct values,
suspicious values and candidate columns (date, identifier, series, numbers stored as text).
The row count and the target distribution are computed on the whole file (one column at a time);
the rest on the first ``SAMPLE_ROWS`` rows of a CSV, or on a random sample of the other formats.
"""

import argparse
import json
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from test_house_prediction.core.data_io.clean import parse_dates, standardize_name
from test_house_prediction.core.data_io.load import READERS

SAMPLE_ROWS = 100_000
CHUNK_ROWS = 200_000
# Values often used in place of a missing value (pandas already reads "", "NA", "N/A", "null"… as missing)
NUMERIC_SENTINELS = {-1, -99, -999, -9999, 999, 9999, 99999, 999999}
TEXT_SENTINELS = {"?", "-", "--", ".", "missing", "unknown", "inconnu", "none", "nan", "nd", "n.d.", "#n/a", "#value!"}
MIN_SENTINEL_SHARE = 0.01  # a value counts as a sentinel above 1 % of the non-missing values
PARSE_SHARE = 0.9  # share of non-missing values that must parse (dates, numbers stored as text)
MAX_CLASSES = 20  # above, a numeric target is treated as continuous
IMBALANCED, RARE = 0.20, 0.05  # metric rule thresholds (share of the minority class, share of outliers)
FREQUENCIES = {1: "daily", 7: "weekly"}


def check_file(path: Path) -> str:
    """Check that the file exists and has a supported extension.

    Args:
        path: Data file.

    Returns:
        str: The lower-case extension.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not supported.

    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    suffix = path.suffix.lower()
    if suffix not in READERS:
        raise ValueError(f"unsupported extension {suffix!r}, expected one of {sorted(READERS)}")
    return suffix


def read_sample(path: Path, rows: int | None = None) -> tuple[pd.DataFrame, int]:
    """Read a sample of the file and count its rows.

    Args:
        path: Data file (.csv, .xlsx, .xls, .json or .parquet).
        rows: Maximum number of rows of the sample (defaults to ``SAMPLE_ROWS``).

    Returns:
        tuple[pd.DataFrame, int]: The sample (first rows of a CSV, random rows otherwise) and the row count.

    """
    rows = rows or SAMPLE_ROWS
    if check_file(path) == ".csv":
        sample = pd.read_csv(path, nrows=rows, low_memory=False)
        return sample, sum(len(part) for part in iter_column(path, str(sample.columns[0])))
    df: pd.DataFrame = READERS[path.suffix.lower()](path)
    return (df.sample(n=rows, random_state=0) if len(df) > rows else df), len(df)


def iter_column(path: Path, column: str) -> Iterator[pd.Series]:
    """Yield the whole column, by chunks for a CSV (one column in memory at a time).

    Args:
        path: Data file.
        column: Column to read.

    Yields:
        pd.Series: Parts of the column.

    """
    if path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, usecols=[column], chunksize=CHUNK_ROWS, low_memory=False):
            yield chunk[column]
    else:
        yield READERS[path.suffix.lower()](path)[column]


def is_text(values: pd.Series) -> bool:
    """Tell whether a column holds text (object dtype, or the string dtype of pandas 3).

    Args:
        values: Column.

    Returns:
        bool: True for a text column.

    """
    return bool(values.dtype == object or pd.api.types.is_string_dtype(values))


def parse_numbers(values: pd.Series) -> pd.Series:
    """Parse text such as ``"1 234,5"``, ``"1.234,5"`` or ``"12.5 %"`` into numbers (missing when impossible).

    Args:
        values: Text values.

    Returns:
        pd.Series: The parsed numbers.

    """
    text = values.astype(str).str.strip().str.replace(r"[\s %€$£]", "", regex=True)
    decimal_comma = text.str.match(r"^-?\d{1,3}(\.\d{3})*,\d+$|^-?\d+,\d+$")
    text = text.where(~decimal_comma, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def to_dates(values: pd.Series) -> pd.Series:
    """Convert a date column to datetime as ``clean`` does, unparsable values becoming missing.

    Args:
        values: Date column.

    Returns:
        pd.Series: The dates.

    """
    return parse_dates(values, strict=False)


def parses_as_dates(values: pd.Series) -> bool:
    """Tell whether most text values are dates.

    Args:
        values: Non-missing text values.

    Returns:
        bool: True if at least ``PARSE_SHARE`` of the values parse as dates.

    """
    if values.astype(str).str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() < PARSE_SHARE:
        return False
    return bool(to_dates(values).notna().mean() >= PARSE_SHARE)


def sentinels(values: pd.Series) -> list[Any]:
    """Return the values that look like placeholders for missing data (-999, "?", "unknown"…).

    Args:
        values: Non-missing values of a column.

    Returns:
        list: ``{"value": …, "pct": …}`` for each suspicious value present in at least ``MIN_SENTINEL_SHARE``
        of the values.

    """
    found: list[Any] = []
    for value, share in values.value_counts(normalize=True).items():
        if share < MIN_SENTINEL_SHARE:
            break
        if isinstance(value, str):
            suspicious = value.strip().lower() in TEXT_SENTINELS
        else:
            suspicious = not isinstance(value, bool) and value in NUMERIC_SENTINELS
        if suspicious:
            found.append({"value": value.item() if hasattr(value, "item") else value, "pct": round(100 * share, 2)})
    return found


def profile_column(values: pd.Series) -> dict[str, Any]:
    """Describe one column: type, missing values, distinct values and what looks wrong.

    Args:
        values: The column (sample).

    Returns:
        dict[str, Any]: Statistics and ``flags`` of the column.

    """
    present = values.dropna()
    n_unique = int(present.nunique())
    info: dict[str, Any] = {
        "dtype": str(values.dtype),
        "missing_pct": round(100 * float(values.isna().mean()), 2),
        "n_unique": n_unique,
    }
    flags: list[str] = []
    if n_unique <= 1:
        flags.append("constant")
    if pd.api.types.is_datetime64_any_dtype(values):
        flags.append("date")
    elif is_text(values) and not present.empty:
        if parses_as_dates(present):
            flags.append("date_as_text")
        elif parse_numbers(present).notna().mean() >= PARSE_SHARE:
            flags.append("number_as_text")
        elif n_unique > 50:
            flags.append("high_cardinality")
    parsed = {"date", "date_as_text", "number_as_text"} & set(flags)  # unique values, but not identifiers
    if len(present) > 1 and n_unique >= 0.95 * len(present) and not pd.api.types.is_float_dtype(values) and not parsed:
        flags.append("identifier")
    if found := sentinels(present):
        info["suspicious_values"] = found
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and not present.empty:
        info["min"], info["max"] = float(present.min()), float(present.max())
    if flags:
        info["flags"] = flags
    return info


def frequency_name(days: int) -> str:
    """Name the step between two dates.

    Args:
        days: Median number of days between two consecutive dates.

    Returns:
        str: daily, weekly, monthly, quarterly, yearly or "<n> days".

    """
    if days in FREQUENCIES:
        return FREQUENCIES[days]
    if 28 <= days <= 31:
        return "monthly"
    if 89 <= days <= 92:
        return "quarterly"
    return "yearly" if days >= 365 else f"{days} days"


def profile_dates(
    df: pd.DataFrame, date_column: str, columns: dict[str, dict[str, Any]], target: str | None = None
) -> dict[str, Any]:
    """Describe the time axis: span, frequency, gaps and the columns that identify several series.

    Args:
        df: Sample.
        date_column: Date column.
        columns: Column profiles (to pick the series candidates).
        target: Target column, never a series candidate.

    Returns:
        dict[str, Any]: Time series statistics.

    """
    dates = to_dates(df[date_column])
    unique = pd.Series(dates.dropna().unique()).sort_values(ignore_index=True)
    info: dict[str, Any] = {"column": date_column, "n_dates": len(unique)}
    if len(unique) < 3:
        return info
    step = unique.diff().dropna().median()
    info["start"], info["end"] = str(unique.iloc[0].date()), str(unique.iloc[-1].date())
    if step.days >= 1:
        periods = int((unique.iloc[-1] - unique.iloc[0]) / step) + 1
        info.update({"frequency": frequency_name(step.days), "history_periods": periods})
        info["missing_dates"] = max(periods - len(unique), 0)
    else:
        info["frequency"] = str(step)
    if dates.duplicated().any():  # several rows per date: look for the columns identifying a series
        candidates = [
            name
            for name, column in columns.items()
            if name not in {date_column, target}
            and 1 < column["n_unique"] <= 1000
            and not {"identifier", "number_as_text"} & set(column.get("flags", []))
            and not column["dtype"].startswith("float")  # a series is never identified by a measure
        ]
        keyed = df.assign(_date=dates)
        groups = [[c] for c in candidates if not keyed.duplicated(["_date", c]).any()]
        if not groups:
            pairs = combinations(candidates[:15], 2)
            groups = [list(pair) for pair in pairs if not keyed.duplicated(["_date", *pair]).any()]
        info["several_rows_per_date"] = True
        info["series_candidates"] = groups[:5]
    return info


def profile_target(path: Path, target: str, problem_type: str | None = None) -> dict[str, Any]:
    """Describe the target on the whole file: class distribution and, if numeric, distribution and outliers.

    Args:
        path: Data file.
        target: Target column.
        problem_type: classification, regression or timeseries (defaults to a guess from the values).

    Returns:
        dict[str, Any]: Target statistics, ``kind`` being classification or continuous.

    """
    counts = pd.Series(dtype="float64")
    numeric_parts: list[pd.Series] = []
    missing = 0
    for part in iter_column(path, target):
        missing += int(part.isna().sum())
        counts = counts.add(part.value_counts(dropna=True), fill_value=0)
        if pd.api.types.is_numeric_dtype(part) and not pd.api.types.is_bool_dtype(part):
            numeric_parts.append(part.dropna())
    numeric = pd.concat(numeric_parts) if numeric_parts else pd.Series(dtype="float64")
    if problem_type is None:
        problem_type = "classification" if numeric.empty or len(counts) <= MAX_CLASSES else "regression"
    info: dict[str, Any] = {"column": target, "missing": missing, "n_unique": len(counts)}
    if problem_type == "classification":
        shares = (counts / counts.sum()).sort_values(ascending=False)
        minority = shares.index[-1]
        info["kind"] = "classification"
        info["classes"] = {str(k): {"count": int(counts[k]), "share": round(float(v), 4)} for k, v in shares.items()}
        info["minority_class"] = minority.item() if hasattr(minority, "item") else minority
        info["minority_share"] = round(float(shares.iloc[-1]), 4)
    elif not numeric.empty:
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        outliers = (numeric < q1 - 1.5 * (q3 - q1)) | (numeric > q3 + 1.5 * (q3 - q1))
        info["kind"] = "continuous"
        info.update(
            {
                "mean": float(numeric.mean()),
                "std": float(numeric.std()),
                "min": float(numeric.min()),
                "median": float(numeric.median()),
                "max": float(numeric.max()),
                "skew": round(float(numeric.skew()), 3),
                "outlier_share": round(float(outliers.mean()), 4),
            }
        )
    else:
        info["kind"] = "not_numeric"  # a regression target stored as text: see number_as_text
    return info


def suggest_metric(problem_type: str, target: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed metric rule to the target profile.

    Classification, by share of the minority class: from 20 % → f1, 5 to 20 % → balanced_accuracy,
    below 5 % → average_precision of the minority class (binary) or balanced_accuracy (more classes).
    Regression and time series: rmse, or mae when more than 5 % of the values are outliers.

    Args:
        problem_type: classification, regression or timeseries.
        target: Result of ``profile_target``.

    Returns:
        dict[str, Any]: ``PRIMARY_METRIC``, ``POSITIVE_CLASS`` (classification) and the ``reason``.

    """
    if problem_type == "classification":
        share = target["minority_share"]
        minority = f"minority class {target['minority_class']!r} = {100 * share:.1f} %"
        if share >= IMBALANCED:
            return {"PRIMARY_METRIC": "f1", "POSITIVE_CLASS": None, "reason": f"{minority}: balanced enough"}
        if share >= RARE or target["n_unique"] > 2:
            return {"PRIMARY_METRIC": "balanced_accuracy", "POSITIVE_CLASS": None, "reason": f"{minority}: imbalanced"}
        return {
            "PRIMARY_METRIC": "average_precision",
            "POSITIVE_CLASS": target["minority_class"],
            "reason": f"{minority}: rare class, accuracy would be misleading",
        }
    outlier_share = target.get("outlier_share", 0.0)
    if outlier_share > RARE:
        return {"PRIMARY_METRIC": "mae", "reason": f"{100 * outlier_share:.1f} % outliers: mae is less sensitive"}
    return {"PRIMARY_METRIC": "rmse", "reason": f"{100 * outlier_share:.1f} % outliers"}


def profile_dataset(path: Path, target: str | None = None, problem_type: str | None = None) -> dict[str, Any]:
    """Profile a dataset without exposing any row.

    Args:
        path: Data file.
        target: Target column (raw name): adds its distribution on the whole file and the metric rule.
        problem_type: classification, regression or timeseries (defaults to a guess from the target).

    Returns:
        dict[str, Any]: The profile; ``config_names`` gives the column names used in core/config.py.

    Raises:
        KeyError: If the target column does not exist.

    """
    sample, n_rows = read_sample(path)
    columns = {str(name): profile_column(sample[name]) for name in sample.columns}
    profile: dict[str, Any] = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "n_rows": n_rows,
        "sample_rows": len(sample),
        "n_columns": len(columns),
        "columns": columns,
        "config_names": {name: standardize_name(name) for name in columns},
    }
    dates = [name for name, column in columns.items() if {"date", "date_as_text"} & set(column.get("flags", []))]
    if dates:
        profile["time"] = profile_dates(sample, dates[0], columns, target) | {"other_date_columns": dates[1:]}
    if target is not None:
        if target not in columns:
            raise KeyError(f"target {target!r} not found, columns: {list(columns)}")
        profile["target"] = profile_target(path, target, problem_type)
        kind = profile["target"]["kind"]
        if kind != "not_numeric":
            profile["metric"] = suggest_metric(
                problem_type or kind.replace("continuous", "regression"), profile["target"]
            )
    return profile


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: print the profile as JSON.

    Args:
        argv: Arguments to parse (defaults to ``sys.argv``).

    """
    parser = argparse.ArgumentParser(description="Statistical profile of a dataset (no rows are printed).")
    parser.add_argument("path", type=Path, help="data file (.csv, .xlsx, .xls, .json, .parquet)")
    parser.add_argument("--target", help="target column, raw name")
    parser.add_argument("--problem-type", choices=["classification", "regression", "timeseries"])
    args = parser.parse_args(argv)
    print(
        json.dumps(
            profile_dataset(args.path, args.target, args.problem_type), ensure_ascii=False, indent=1, default=str
        )
    )
