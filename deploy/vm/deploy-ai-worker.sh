#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <artifact-registry-image>" >&2
  exit 2
fi

IMAGE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.vm"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

cd "${SCRIPT_DIR}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing ${ENV_FILE}. Create it from .env.vm.example and fill secrets first." >&2
  exit 1
fi

if [ ! -f "${COMPOSE_FILE}" ]; then
  echo "Missing ${COMPOSE_FILE}" >&2
  exit 1
fi

REGISTRY="${IMAGE%%/*}"
if command -v gcloud >/dev/null 2>&1; then
  gcloud auth configure-docker "${REGISTRY}" --quiet
fi

TMP_ENV="$(mktemp)"
awk -v image="${IMAGE}" '
  BEGIN { replaced = 0 }
  /^AI_WORKER_IMAGE=/ {
    print "AI_WORKER_IMAGE=" image
    replaced = 1
    next
  }
  { print }
  END {
    if (replaced == 0) {
      print "AI_WORKER_IMAGE=" image
    }
  }
' "${ENV_FILE}" > "${TMP_ENV}"
cat "${TMP_ENV}" > "${ENV_FILE}"
rm -f "${TMP_ENV}"
chmod 600 "${ENV_FILE}"

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" pull ai-worker
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --no-deps ai-worker
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" ps ai-worker
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" logs --tail=80 ai-worker
