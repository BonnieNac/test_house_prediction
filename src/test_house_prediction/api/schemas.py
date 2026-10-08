"""Input and output schemas of the prediction API.

``records`` accepts any column so that the API works before the dataset is known. Once the
features are fixed, replace it with an explicit model, for instance::

    class Features(BaseModel):
        surface: float = Field(gt=0)
        city: str


    class PredictionRequest(BaseModel):
        records: list[Features]
"""

from pydantic import BaseModel, Field

FeatureValue = float | int | str | bool | None


class PredictionRequest(BaseModel):
    """Rows to predict, with the columns used for training."""

    records: list[dict[str, FeatureValue]] = Field(min_length=1, examples=[[{"feature_1": 1.5, "feature_2": "a"}]])


class PredictionResponse(BaseModel):
    """One prediction per row, in the same order."""

    predictions: list[float | int | str]
