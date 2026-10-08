"""Feature engineering: new columns computed from the clean data."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model features from the clean data.

    Examples of features to add: ratios (``df["price_per_m2"] = df["price"] / df["surface"]``),
    bins, dates split into year/month, aggregations by group…

    Args:
        df: Clean data.

    Returns:
        pd.DataFrame: Data ready for training.

    """
    df = df.copy()
    return df
