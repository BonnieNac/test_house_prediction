# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Projet data science généré à partir du template cookiecutter *astrodata*. Il prédit le prix d'une maison (régression) à partir de `data/raw/house_data.csv`. Ce fichier est suivi par DVC (`house_data.csv.dvc`) et n'est pas versionné dans git. Python 3.12, dépendances gérées par **uv** (`uv sync`). N'utilise jamais `pip install`, c'est bloqué dans `.claude/settings.json`.

## Commandes

```bash
make dev-install              # uv sync + hooks pre-commit
make check                    # lint (ruff check + format --check) → mypy strict → pytest : ce que lance la CI
make format                   # ruff --fix + ruff format sur src/ tests/ app/
make test-fast                # pytest -x
uv run pytest tests/unit_test/test_api.py::test_xxx   # un seul test
```

pytest s'exécute avec `--cov=src --cov-fail-under=80` (cf. `pyproject.toml`). Lancer un seul test fait donc échouer le seuil de couverture : ajoute `--no-cov` pour l'ignorer. Le marqueur `ci_exclude` sert à sortir un test de la CI.

Pipeline ML (scripts minces dans `scripts/` qui appellent `core/`) :

```bash
make profile FILE=data/raw/house_data.csv TARGET=price   # profil statistique, n'affiche aucune ligne
make preprocess        # raw → clean → build_features → data/processed/dataset_processed.csv
make train             # compare les modèles en CV, réentraîne le meilleur → models/model.joblib (+ run MLflow)
make train-list        # liste les modèles ; uv run python scripts/run_training.py --models ridge lasso
make tune              # RandomizedSearchCV sur les TUNE_TOP_K meilleurs modèles
make mlflow-ui         # UI sur le store local sqlite:///mlflow.db (port 5000)
make run / make run_api    # Streamlit :8501 / FastAPI :8000 (/docs)
make up / make down        # docker compose (api + streamlit + mlflow), deployment/docker-compose.yml
```

`train` et `tune` acceptent `--register NAME --alias staging|production` pour enregistrer le modèle dans le Model Registry MLflow.

## Architecture

- **`src/test_house_prediction/core/config.py`** centralise tous les réglages liés au jeu de données : cible, colonnes ID, features numériques et catégorielles (`None` = déduites des dtypes), `PRIMARY_METRIC` (`mae`), taille du jeu de test, nombre de folds de CV, `MODEL_FILE`. Les noms de colonnes sont ceux obtenus **après** `clean.standardize_column_names` (minuscules, `_`). La skill `/adapt-config` met ce fichier à jour à partir du profil, sans jamais lire les données.
- **Flux** : `data_io/load.py` (csv/xlsx/json/parquet, motifs `api_*.parquet` acceptés) → `data_io/clean.py` → `features/build_features.py` (encore vide, c'est là qu'on ajoute les features dérivées) → `features/preprocess.py` (`ColumnTransformer` + split) → `models/train.py`. `core/pipelines.py` enchaîne ces étapes.
- **Un seul artefact** : `train.py` sauvegarde un `Pipeline` sklearn complet (préprocesseur + modèle). La prédiction n'a donc pas à refaire le préprocessing. En revanche, les lignes envoyées à `predict` doivent déjà contenir les colonnes produites par `build_features`.
- **Ajouter un modèle** dans `train.py` : écrire une fonction qui le construit, l'ajouter au dict `MODELS`, puis ajouter son espace de recherche dans `PARAM_SPACES` (clés `model__<param>`) pour que `tune.py` le prenne en compte. `dummy_baseline` sert de référence minimale.
- **MLflow** (`core/tracking.py`) : `MLFLOW_TRACKING_URI` et `MLFLOW_EXPERIMENT_NAME` sont lus depuis l'environnement ou `.env`. Si `MLFLOW_TRACKING_URI` n'est pas défini, MLflow utilise le store SQLite `mlflow.db` à la racine du projet.
- **Chargement du modèle par l'API** (`core/models/predict.py`, mis en cache par `lru_cache`) : si `MLFLOW_MODEL_URI` est défini (ex. `models:/test_house_prediction@production`), le modèle vient du registre ; sinon, de `models/model.joblib`. S'il n'y a pas de modèle, `/predict` renvoie 503, et 422 si l'entrée est invalide.
- **API** (`api/`) : `main.py` déclare les routers `base`, `system`, `greetings` et `predict`, plus `LimitUploadSizeMiddleware` (`API_MAX_UPLOAD_SIZE`). `PredictionRequest.records` accepte n'importe quel dictionnaire. Une fois les features fixées, le docstring de `schemas.py` recommande de remplacer ce dictionnaire par un modèle pydantic explicite.
- **Streamlit** (`app/streamlit_app.py`) n'est pour l'instant qu'une page de démonstration : elle n'appelle pas encore l'API.
- **Tests** (`tests/unit_test/`) : ils tournent sur des données synthétiques, indépendamment de `config.py`. Ne les adapte pas au vrai jeu de données.

## Déploiement

Voir `deployment/README.md`. L'image Docker multi-stage (cibles `api` et `streamlit`) ne contient **que du code** : le modèle est soit monté dans le conteneur, soit téléchargé depuis MLflow au démarrage. Un tag `vX.Y.Z` (créé par `make bump` puis `make release`) déclenche `.github/workflows/docker.yml`, qui publie l'image sur `ghcr.io/bonnienac/test_house_prediction-api`. Sur le serveur, on déploie avec `./deploy.sh <staging|prod> <version>` ; les réglages viennent de `config/<env>.env`, qui n'est jamais commité.

## Conventions

- ruff (lignes de 120 caractères, règles `D` pour les docstrings et `ANN` pour les annotations) et mypy `strict` avec le plugin pydantic. Toute fonction doit avoir des annotations et une docstring. Après chaque Write/Edit, un hook (`.claude/hooks/ruff-format.js`) formate automatiquement les fichiers `.py`.
- Commits au format Conventional Commits (`uv run cz c`). `make bump` (commitizen) met à jour `VERSION`, `pyproject.toml` et `CHANGELOG.md`.
- `git push`, `rm -rf` et la lecture ou l'écriture de `.env` sont interdits par `.claude/settings.json`.
