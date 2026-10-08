"""Tests of the dataset profile on a file full of traps (dates as text, decimal commas, sentinels…)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_house_prediction.core.data_io import profile

N_DAYS = 200


@pytest.fixture
def trap_csv(tmp_path: Path) -> Path:
    """Write two daily series (stores A and B) with a rare target and the usual data quality traps."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    n = 2 * N_DAYS
    df = pd.DataFrame(
        {
            "Date": [day.strftime("%d/%m/%Y") for day in dates for _ in range(2)],
            "Store": ["A", "B"] * N_DAYS,
            "Client ID": range(n),
            "Price": [f"{value:,.2f}".replace(",", " ").replace(".", ",") for value in rng.uniform(1000, 5000, n)],
            "Age": np.where(rng.random(n) < 0.05, -999, rng.integers(18, 80, n)),
            "Comment": np.where(rng.random(n) < 0.1, "?", "ok"),
            "Constant": 1,
            "Amount": np.r_[rng.normal(100, 10, n - 40), rng.normal(1000, 10, 40)],
            "Fraud": np.where(np.arange(n) % 25 == 0, "yes", "no"),
        }
    )
    path = tmp_path / "trap.csv"
    df.to_csv(path, index=False)
    return path


def test_columns_flags(trap_csv: Path) -> None:
    """Dates and numbers stored as text, identifiers, sentinels and constant columns are reported."""
    columns = profile.profile_dataset(trap_csv)["columns"]

    assert "date_as_text" in columns["Date"]["flags"]
    assert columns["Price"]["flags"] == ["number_as_text"]
    assert "identifier" in columns["Client ID"]["flags"]
    assert [s["value"] for s in columns["Age"]["suspicious_values"]] == [-999]
    assert 3 < columns["Age"]["suspicious_values"][0]["pct"] < 7
    assert [s["value"] for s in columns["Comment"]["suspicious_values"]] == ["?"]
    assert "constant" in columns["Constant"]["flags"]
    assert "flags" not in columns["Store"]


def test_time_axis_and_series(trap_csv: Path) -> None:
    """The day-first dates give a daily axis, and Store identifies the series."""
    time = profile.profile_dataset(trap_csv)["time"]

    assert (time["start"], time["end"]) == ("2024-01-01", "2024-07-18")
    assert time["frequency"] == "daily"
    assert time["missing_dates"] == 0
    assert time["series_candidates"] == [["Store"]]


def test_target_is_never_a_series_candidate(tmp_path: Path) -> None:
    """A target with one value per date and series does not identify a series."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame([{"day": day, "store": store} for day in dates for store in (1, 2)])
    df["units"] = [i % 25 for i in range(len(df))]  # repeated values: not an identifier
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)

    assert profile.profile_dataset(path, target="units")["time"]["series_candidates"] == [["store"]]


def test_target_on_the_whole_file(trap_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The target distribution covers every row, even beyond the sample."""
    monkeypatch.setattr(profile, "SAMPLE_ROWS", 50)
    monkeypatch.setattr(profile, "CHUNK_ROWS", 64)

    result = profile.profile_dataset(trap_csv, target="Fraud")

    assert result["sample_rows"] == 50
    assert result["n_rows"] == 2 * N_DAYS
    assert result["target"]["minority_class"] == "yes"
    assert result["target"]["minority_share"] == 0.04
    assert result["metric"] == {
        "PRIMARY_METRIC": "average_precision",
        "POSITIVE_CLASS": "yes",
        "reason": "minority class 'yes' = 4.0 %: rare class, accuracy would be misleading",
    }
    assert result["config_names"]["Client ID"] == "client_id"


def test_continuous_target(trap_csv: Path) -> None:
    """A numeric target with outliers gets mae."""
    result = profile.profile_dataset(trap_csv, target="Amount", problem_type="regression")

    assert result["target"]["kind"] == "continuous"
    assert 0.1 <= result["target"]["outlier_share"] < 0.12  # the 40 shifted rows, plus normal tails
    assert result["metric"]["PRIMARY_METRIC"] == "mae"


def test_regression_target_stored_as_text(trap_csv: Path) -> None:
    """A regression target stored as text gets no metric (it must be converted first)."""
    result = profile.profile_dataset(trap_csv, target="Price", problem_type="regression")

    assert result["target"]["kind"] == "not_numeric"
    assert "metric" not in result


@pytest.mark.parametrize(
    ("minority_share", "n_classes", "expected"),
    [
        (0.4, 2, "f1"),
        (0.2, 2, "f1"),
        (0.1, 2, "balanced_accuracy"),
        (0.03, 3, "balanced_accuracy"),
        (0.03, 2, "average_precision"),
    ],
)
def test_metric_rule(minority_share: float, n_classes: int, expected: str) -> None:
    """The classification rule follows the share of the minority class."""
    target = {"minority_share": minority_share, "minority_class": "x", "n_unique": n_classes}
    assert profile.suggest_metric("classification", target)["PRIMARY_METRIC"] == expected


def test_metric_rule_without_outliers() -> None:
    """A continuous target without outliers keeps rmse."""
    assert profile.suggest_metric("timeseries", {"outlier_share": 0.01})["PRIMARY_METRIC"] == "rmse"


@pytest.mark.parametrize(
    ("days", "name"),
    [(1, "daily"), (7, "weekly"), (30, "monthly"), (91, "quarterly"), (365, "yearly"), (14, "14 days")],
)
def test_frequency_name(days: int, name: str) -> None:
    """The median step between dates is named."""
    assert profile.frequency_name(days) == name


def test_parquet_sample_and_pairs_of_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Other formats are sampled; two columns can be needed to identify a series."""
    monkeypatch.setattr(profile, "SAMPLE_ROWS", 30)
    dates = pd.date_range("2024-01-01", periods=10, freq="MS")
    df = pd.DataFrame(
        [{"month": day, "store": store, "product": product} for day in dates for store in "AB" for product in "XY"]
    )
    path = tmp_path / "sales.parquet"
    df.to_parquet(path)

    result = profile.profile_dataset(path)

    assert (result["n_rows"], result["sample_rows"]) == (40, 30)
    assert result["time"]["frequency"] == "monthly"
    assert result["time"]["series_candidates"] == [["store", "product"]]


def test_errors(tmp_path: Path, trap_csv: Path) -> None:
    """Missing files, unsupported formats and unknown targets are rejected."""
    with pytest.raises(FileNotFoundError):
        profile.profile_dataset(tmp_path / "missing.csv")
    (tmp_path / "data.txt").write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        profile.profile_dataset(tmp_path / "data.txt")
    with pytest.raises(KeyError, match="not found"):
        profile.profile_dataset(trap_csv, target="nope")


def test_main_prints_json(trap_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line prints the profile as JSON, without any row."""
    profile.main([str(trap_csv), "--target", "Fraud"])
    output = capsys.readouterr().out

    assert json.loads(output)["metric"]["POSITIVE_CLASS"] == "yes"
    assert "2024-01-05" not in output and "01/02/2024" not in output
