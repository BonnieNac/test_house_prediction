#!/usr/bin/env bash
# Deploy on this server: pull the image(s), restart and wait until the API answers its health check.
# Usage: ./deploy.sh <staging|prod> [image tag]     (e.g. ./deploy.sh prod 1.2.0; default: API_TAG of config/<env>.env)
# Needs only this deployment/ folder and Docker on the server. Each environment is a separate compose
# project (own containers and volumes): give them different API_PORT values in config/<env>.env.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME="${1:-}"
case "$ENV_NAME" in staging|prod) ;; *) echo "usage: ./deploy.sh <staging|prod> [image tag]"; exit 2 ;; esac
export ENV_FILE="config/$ENV_NAME.env"
[ -f "$ENV_FILE" ] || { echo "❌ $ENV_FILE absent: cp $ENV_FILE.example $ENV_FILE, then fill it in"; exit 1; }
[ -n "${2:-}" ] && export API_TAG="$2"

compose() { docker compose -p "test_house_prediction-$ENV_NAME" -f docker-compose.prod.yml --env-file "$ENV_FILE" "$@"; }

compose config -q
compose pull
compose up -d --remove-orphans

container=$(compose ps -q api)
for _ in $(seq 1 30); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo starting)
  if [ "$status" = healthy ]; then
    echo "✅ $ENV_NAME: API healthy ($(docker inspect -f '{{.Config.Image}}' "$container"))"
    exit 0
  fi
  sleep 2
done
echo "❌ $ENV_NAME: API not healthy after 60 s"
compose logs --tail 50 api
exit 1
