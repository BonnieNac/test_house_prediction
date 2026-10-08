"""ML settings of the project: the single place to adapt to the real dataset.

Column names are those obtained AFTER ``clean.standardize_column_names``
(lower case, spaces and dashes replaced by ``_``).
"""

from typing import Literal

ProblemType = Literal["classification", "regression", "timeseries"]
PROBLEM_TYPE: ProblemType = "regression"

# Data files (data/raw/ and data/processed/): .csv, .xlsx, .xls, .json or .parquet.
# A pattern such as "api_*.parquet" reads the latest dated snapshot written by make fetch.
# profil : fichier fourni, copié dans data/raw/
RAW_DATA_FILE: str = "house_data.csv"
PROCESSED_DATA_FILE: str = "dataset_processed.csv"

# Columns
# description : "prédire le prix de vente d'une maison" ; profil : colonne continue `price`
# (le flag `identifier` du profil est un faux positif : 83 valeurs distinctes sur 84 lignes pour un prix)
TARGET: str = "price"
ID_COLUMNS: list[str] = []  # identifiers, never used as features
# None: inferred from the dtypes (numbers → numeric, everything else → categorical)
NUMERIC_FEATURES: list[str] | None = None
CATEGORICAL_FEATURES: list[str] | None = None

# Metric used to rank the models (make train, make tune)
# rmse | mae (less sensitive to outliers) | r2
# règle setup-ds : 8,3 % de valeurs extrêmes dans price, la MAE y est moins sensible
PRIMARY_METRIC: str = "mae"

# Training
TEST_SIZE: float = 0.2  # share of the rows kept for the final evaluation
CV_FOLDS: int = 5
RANDOM_STATE: int = 42
N_JOBS: int = -1  # parallel cross-validation: -1 = every CPU core
MODEL_FILE: str = "model.joblib"  # saved in models/

# Hyperparameter search (make tune)
TUNE_TOP_K: int = 2  # number of best models of the comparison to tune
N_ITER: int = 20  # random combinations tried per model (each one is cross-validated)
