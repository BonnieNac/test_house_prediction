# Déploiement

Tout ce qui sert à construire et faire tourner l'application est ici.

```
deployment/
├── docker/Dockerfile          image : cibles api et streamlit (code seulement, jamais de données ni de modèle)
├── docker-compose.yml         développement sur ton poste : make up / make down / make docker-build
├── docker-compose.prod.yml    serveur : images du registre, rien n'est construit
├── deploy.sh                  à lancer sur le serveur : ./deploy.sh <staging|prod> [version]
└── config/
    ├── staging.env.example    modèles des réglages ; les vrais fichiers staging.env / prod.env
    └── prod.env.example       restent sur le serveur et ne sont jamais commités
```

## Le trajet d'une version

1. **Image** : `make bump` puis `make release` poussent un tag `vX.Y.Z`.
   Le workflow `.github/workflows/docker.yml` construit l'image de l'API et la pousse sur `ghcr.io/bonnienac/test_house_prediction-api:X.Y.Z` (et `:latest`). Aucun secret à configurer.
2. **Modèle** : il ne voyage pas dans l'image. Il est enregistré dans le MLflow du serveur, et l'API le télécharge au démarrage d'après `MLFLOW_MODEL_URI` :
   ```bash
   ssh -L 5000:localhost:5000 <serveur>        # tunnel vers le MLflow du serveur (non exposé sur internet)
   MLFLOW_TRACKING_URI=http://localhost:5000 uv run test_house_prediction-train --register test_house_prediction --alias staging
   ```
   Une fois la version validée en staging, déplace l'alias `production` sur cette version (interface MLflow → Models → test_house_prediction → Aliases, ou `--alias production`).
3. **Serveur** (une seule fois) : installe Docker, copie ce dossier `deployment/`, crée `config/staging.env` et `config/prod.env` à partir des modèles.
   Si le paquet `ghcr.io` est privé : `docker login ghcr.io -u <compte>` avec un jeton GitHub ayant le droit `read:packages`.
4. **Déploiement** : `./deploy.sh staging 1.2.0`, vérifie, puis `./deploy.sh prod 1.2.0`. Le script télécharge l'image, redémarre et attend que `/health` réponde.
5. **Retour arrière** : `./deploy.sh prod 1.1.0` relance la version précédente.

## Règles

- Une seule image par version, déployée telle quelle en staging puis en prod : on ne reconstruit jamais entre les deux.
- Les réglages et les secrets viennent de `config/<env>.env`, jamais de l'image ni du code.
- Le code est dans git, les données dans DVC, les modèles dans un registre (MLflow) ou un dossier monté.

## Passer à Kubernetes

Le projet remplit déjà ce que Kubernetes attend :

- une image publiée dans un registre, qui ne contient que du code ;
- une configuration par variables d'environnement ;
- une route `/health` ;
- un modèle chargé depuis le registre MLflow, jamais embarqué dans l'image ;

Seule la façon de lancer l'image change :

- remplacer `docker-compose.prod.yml` et `deploy.sh` par des manifests dans `deployment/k8s/` : un `Deployment` (image, nombre de copies), un `Service`, un `Ingress`, avec une base commune et un overlay Kustomize par environnement (`staging/`, `prod/`) ;
- passer les réglages de `config/<env>.env` dans une `ConfigMap`, et les identifiants dans un `Secret`, sous les mêmes noms de variables ;
- déclarer des `readinessProbe` et `livenessProbe` sur `/health` (Kubernetes ignore le `HEALTHCHECK` du Dockerfile) ;
- déployer avec `kubectl apply -k deployment/k8s/overlays/<env>`, à la main ou en dernière étape de la CI.

Le vrai chantier est MLflow : dans un cluster, la base SQLite et le disque local ne conviennent plus. Il faut une base PostgreSQL et un stockage d'objets (S3, GCS, Azure Blob) pour les modèles, ou un MLflow hébergé (Databricks, offre de ton cloud). L'API ne change pas : seule l'URL de `MLFLOW_TRACKING_URI` change.
