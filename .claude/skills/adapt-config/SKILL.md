---
name: adapt-config
description: Adapte la configuration ML du projet (src/test_house_prediction/core/config.py, métrique, liste des points de nettoyage) à un jeu de données réel, à partir de son profil statistique, sans lire les données. À utiliser quand un jeu de données arrive (fichier, instantané de make fetch) ou quand l'utilisatrice veut adapter le squelette ML à ses données.
argument-hint: "[chemin_du_fichier] [colonne_cible] [description du problème]"
---

Tu adaptes `src/test_house_prediction/core/config.py` à un jeu de données réel. C'est un réglage, pas une analyse : tu n'écris pas de code de nettoyage ni de features, tu ne lances pas d'entraînement.

## Règles

- **Ne lis jamais les données** : pas de `cat`, `head`, `pandas.read_*` ni d'aperçu de lignes. Tu ne travailles que sur le profil (statistiques, noms de colonnes), produit par `scripts/profile_dataset.py`.
- **N'invente rien.** Une valeur que ni le profil ni la description ne permettent de fixer reste à son défaut, avec un commentaire `# TODO(adapt-config): …`, et va dans le rapport.
- Les noms de colonnes de `config.py` sont ceux **après** `clean.standardize_column_names` : utilise `config_names` du profil.
- Ne modifie que `config.py` et le commentaire de `clean()` dans `src/test_house_prediction/core/data_io/clean.py`. Les tests tournent sur des données synthétiques, indépendamment de ces réglages : ne les modifie pas.

## 1. Entrées

Fichier de données, colonne cible et description du problème : prends-les dans `$ARGUMENTS` ou dans la demande.
- **Mode interactif** (`/adapt-config`) : s'il manque le fichier, demande-le ; s'il manque la cible et que la description et les noms de colonnes ne suffisent pas à la trouver sans ambiguïté, demande-la.
- **Mode non interactif** (agent setup-ds) : tu ne peux pas poser de question. Cible introuvable ou ambiguë → arrête-toi après le profil, laisse `config.py` intact et rapporte pourquoi.

Le fichier doit être dans `data/raw/`. S'il est ailleurs : jusqu'à 100 Mo, copie-le (`cp`) ; au-delà, crée un lien (`ln -s <chemin absolu> data/raw/<nom>`). N'écrase jamais un fichier existant. Il ne doit pas être commité : avec DVC (dossier `.dvc/` présent), `uv run dvc add data/raw/<nom>` ; sinon vérifie que `git check-ignore data/raw/<nom>` le confirme, et sinon ajoute `data/raw/*` et `!data/raw/.gitkeep` au `.gitignore`.

## 2. Profil

```bash
uv run python scripts/profile_dataset.py data/raw/<fichier> --target "<colonne brute>" --problem-type <type>
```
`<type>` : la valeur de `PROBLEM_TYPE` dans `config.py`. Sans cible connue, lance-le d'abord sans `--target`, choisis la cible à partir de la description et des noms de colonnes, puis relance avec `--target`.

Le profil donne, par colonne : type, % de manquants, nombre de valeurs distinctes, `flags` (`date`, `date_as_text`, `number_as_text`, `identifier`, `constant`, `high_cardinality`), `suspicious_values` (-999, "?"… avec leur part en %). La colonne de date principale peut rester en texte : `clean()` la convertit (formats mélangés, jour en premier). Et aussi : `time` (fréquence, historique, trous, `series_candidates`), `target` (répartition des classes sur tout le fichier, ou distribution et part de valeurs extrêmes), `metric` (règle fixe : `PRIMARY_METRIC`, `POSITIVE_CLASS`, raison).

## 3. `config.py`

| Réglage | Comment le fixer |
|---|---|
| `RAW_DATA_FILE` | Nom du fichier dans `data/raw/` ; garde un motif `api_*.parquet` / `sql_*.parquet` s'il est déjà là (instantanés de `make fetch`). |
| `TARGET` | Nom standardisé de la cible. |
| `ID_COLUMNS` | Colonnes `identifier` et **données personnelles** (nom, email, téléphone, adresse) d'après leur nom, même sans flag. Jamais de features. |
| `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` | Laisse `None` (déduit des types), sauf si une colonne serait mal typée : code numérique qui est une catégorie (code postal, code produit, identifiant de magasin…) ou colonne `number_as_text`. Liste alors explicitement les deux. Une colonne `number_as_text` n'entre dans **aucune** des deux listes tant que `clean()` ne la convertit pas (le préprocesseur l'ignore ; une imputation numérique sur du texte ferait échouer `make train`) : commentaire « à ajouter à NUMERIC_FEATURES une fois convertie (voir clean) ». |
| `DATE_COLUMN` (série temporelle) | `time.column`. |
| `GROUP_COLUMNS` (série temporelle) | `time.series_candidates[0]` si `several_rows_per_date`, sinon `[]`. Plusieurs candidats : celui que la description désigne (« par magasin ») ; sinon le premier, avec un TODO. |
| `LAGS`, `ROLLING_WINDOWS` (série temporelle) | Selon `time.frequency` : daily → `[1, 2, 7, 14]` et `[7, 28]` ; weekly → `[1, 2, 4, 52]` si `history_periods` ≥ 104, sinon `[1, 2, 4]`, et `[4]` ; monthly → `[1, 2, 3, 12]` si ≥ 24 périodes, sinon `[1, 2, 3]`, et `[3]` ; autre → laisse et TODO. Le plus grand lag doit rester sous 20 % de `history_periods`. |
| `CALENDAR_FEATURES` (série temporelle) | daily : `day_of_week`, `is_weekend`, plus `month` si `history_periods` ≥ 730 ; weekly / monthly : `month` si au moins deux ans d'historique, sinon `[]`. |
| `TEST_SIZE`, `CV_FOLDS` | Moins de 1 000 lignes : `0.2` et `5` ; plus de 1 000 000 : `0.1` et `3` ; sinon les défauts. |
| `PRIMARY_METRIC`, `POSITIVE_CLASS` | Ceux de `metric`, avec la raison en commentaire (`# règle setup-ds : classe minoritaire 'yes' = 3,2 % ; à revoir selon le coût des erreurs`). Si la description exprime un coût métier explicite (« ne rater aucune fraude »), garde la métrique de la règle et ajoute un TODO qui le rappelle. `POSITIVE_CLASS` garde le type de la valeur (1, pas "1"). Pas de `metric` (cible de régression stockée en texte) → défaut et TODO. |

Mets un commentaire au-dessus de chaque valeur que tu changes, avec sa source (« profil : … », « description : … »). Lignes de 120 caractères au plus (ruff) : coupe les commentaires longs sur plusieurs lignes.

## 4. Constats dans `clean()`

Dans `src/test_house_prediction/core/data_io/clean.py`, ajoute au début du corps de `clean()` un bloc de commentaires `# TODO(adapt-config): …`, un constat par ligne, **sans code** :
- `number_as_text` → « `<col>` : nombres stockés en texte (virgule décimale, espaces) à convertir » ;
- `suspicious_values` (`value`, `pct`) → « `<col>` : -999 = valeur manquante probable (3,1 % des valeurs) » ;
- `date_as_text` hors colonne de date principale → « `<col>` : date en texte, à convertir ou à découper en features » ;
- `constant` → à supprimer ; `high_cardinality` → à regrouper ou à encoder autrement ;
- colonnes que la description rend suspectes de **fuite de données** (connues seulement après l'événement à prédire : date de résiliation, montant remboursé…) → « à exclure des features » : ajoute-les aussi à `ID_COLUMNS` (le constat explique pourquoi) ;
- taux de manquants > 30 % → à traiter (imputation, indicateur, suppression).

Les identifiants et données personnelles déjà rangés dans `ID_COLUMNS` n'ont pas besoin de constat (seules les fuites de données en ont un, pour expliquer leur exclusion). Aucun constat → pas de bloc.

## 5. Vérifications

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy --explicit-package-bases src/ app/
uv run pytest -q
```
Lance pytest avec un délai long (plusieurs minutes avec MLOps). Tout doit passer. En mode non interactif, commite `src/test_house_prediction/core/config.py`, `src/test_house_prediction/core/data_io/clean.py` et, le cas échéant, `data/raw/<fichier>.dvc`, `data/raw/.gitignore` et `.gitignore` : `feat: adapt ML config to <fichier>`. Ne touche pas aux autres fichiers non suivis (`data/processed/`…). En mode interactif, propose ce commit.

## 6. Rapport

- Réglages changés, avec leur source ; métrique et raison.
- Liste des TODO (config et nettoyage), dont les fuites de données suspectées.
- Suite : `make preprocess`, `make train`, puis `make tune` ; ouvrir `notebooks/` pour l'EDA des points signalés.
