"""Preprocessing pipeline (imputation, scaling, encoding) and train/test split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from test_house_prediction.core import config


def split_features_target(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features from the target and drop the columns never used as features.

    Args:
        df: Processed data.
        target: Target column (defaults to ``config.TARGET``).

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features ``X`` and target ``y``.

    """
    target = target or config.TARGET
    excluded = [target, *config.ID_COLUMNS]
    features = df.drop(columns=[c for c in excluded if c in df.columns])
    return features, df[target]


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical columns (from core/config.py, otherwise from the dtypes).

    Args:
        features: Feature columns.

    Returns:
        tuple[list[str], list[str]]: Numeric columns, categorical columns.

    """
    numeric = config.NUMERIC_FEATURES
    if numeric is None:
        numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = config.CATEGORICAL_FEATURES
    if categorical is None:
        categorical = [c for c in features.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build the preprocessing pipeline, fitted on the training data only.

    - numeric columns: median imputation, then standardization;
    - categorical columns: imputation with the most frequent value, then one-hot encoding
      (categories unseen during training are ignored).

    Args:
        numeric: Numeric columns.
        categorical: Categorical columns.

    Returns:
        ColumnTransformer: The unfitted preprocessor.

    """
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        remainder="drop",
    )


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into a training set and a test set kept for the final evaluation.

    Args:
        features: Features ``X``.
        target: Target ``y``.

    Returns:
        tuple: ``X_train, X_test, y_train, y_test``.

    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    return x_train, x_test, y_train, y_test
