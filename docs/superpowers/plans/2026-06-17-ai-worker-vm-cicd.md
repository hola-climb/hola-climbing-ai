# AI Worker VM CI/CD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub Actions CI/CD and VM rollout files so the AI worker is built as an immutable Docker image and deployed to the existing `hola-climbing-server` VM without touching PostgreSQL or Redis.

**Architecture:** GitHub Actions runs `uv` quality gates, builds a `linux/amd64` worker image, pushes it to Artifact Registry, copies deployment files to the VM, and runs a VM-local rollout script. The rollout script updates only `AI_WORKER_IMAGE` in `.env.vm` and recreates only the `ai-worker` compose service.

**Tech Stack:** GitHub Actions, Workload Identity Federation, GCP Artifact Registry, `gcloud compute ssh/scp`, Docker Compose, Python 3.11, `uv`, ruff, mypy, pytest.

---

## File Structure

- Create `.github/workflows/ai-worker-ci.yml`: repository CI for pull requests, master pushes, and manual runs.
- Create `.github/workflows/deploy-ai-worker.yml`: production deployment workflow for master pushes and manual runs.
- Create `deploy/vm/docker-compose.yml`: VM-side production compose file for the AI worker service only.
- Create `deploy/vm/.env.vm.example`: non-secret example environment file with redacted sample values for sensitive fields.
- Create `deploy/vm/deploy-ai-worker.sh`: idempotent VM rollout script that updates only the worker image and restarts only `ai-worker`.
- Create `deploy/vm/README.md`: operator runbook including the exact terminal commands the user runs for sensitive values.

## Task 1: AI Worker CI Workflow

**Files:**
- Create: `.github/workflows/ai-worker-ci.yml`

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ai-worker-ci.yml` with:

```yaml
name: ai-worker-ci

on:
  pull_request:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ai-worker-ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"

      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --frozen

      - name: Ruff
        run: uv run ruff check app tests scripts

      - name: Mypy
        run: uv run mypy app

      - name: Pytest
        run: uv run pytest
```

- [ ] **Step 2: Validate YAML by inspection**

Run:

```bash
sed -n '1,220p' .github/workflows/ai-worker-ci.yml
```

Expected: workflow has `contents: read`, `uv sync --frozen`, ruff, mypy, and pytest steps.

## Task 2: VM Runtime Files

**Files:**
- Create: `deploy/vm/docker-compose.yml`
- Create: `deploy/vm/.env.vm.example`
- Create: `deploy/vm/deploy-ai-worker.sh`

- [ ] **Step 1: Create VM compose file**

Create `deploy/vm/docker-compose.yml` with:

```yaml
services:
  ai-worker:
    image: ${AI_WORKER_IMAGE}
    container_name: hola-ai-worker
    platform: linux/amd64
    restart: unless-stopped
    environment:
      REDIS_HOST: ${REDIS_HOST:-host.docker.internal}
      REDIS_PORT: ${REDIS_PORT:-6379}
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      REDIS_DB: ${REDIS_DB:-0}
      REDIS_STREAM_KEY: ${REDIS_STREAM_KEY:-analysis:requests}
      REDIS_CONSUMER_GROUP: ${REDIS_CONSUMER_GROUP:-hola-ai-worker}
      REDIS_CONSUMER_NAME: ${REDIS_CONSUMER_NAME:-worker-1}
      REDIS_PROGRESS_CHANNEL: ${REDIS_PROGRESS_CHANNEL:-analysis:progress}
      REDIS_BLOCK_MS: ${REDIS_BLOCK_MS:-5000}
      REDIS_DLQ_KEY: ${REDIS_DLQ_KEY:-analysis:requests:dlq}
      REDIS_PENDING_MIN_IDLE_MS: ${REDIS_PENDING_MIN_IDLE_MS:-60000}
      GCS_BUCKET: ${GCS_BUCKET:-hola-climbing-log-videos}
      GCS_DOWNLOAD_DIR: ${GCS_DOWNLOAD_DIR:-/tmp/hola-videos}
      AI_CALLBACK_SECRET: ${AI_CALLBACK_SECRET}
      CALLBACK_TIMEOUT_SECONDS: ${CALLBACK_TIMEOUT_SECONDS:-10}
      CALLBACK_MAX_RETRIES: ${CALLBACK_MAX_RETRIES:-3}
      CALLBACK_RETRY_INITIAL_SECONDS: ${CALLBACK_RETRY_INITIAL_SECONDS:-1}
      WORKER_HOST: ${WORKER_HOST:-0.0.0.0}
      WORKER_PORT: ${WORKER_PORT:-8000}
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      MODEL_VERSION: ${MODEL_VERSION:-rule_v3}
      MP_MODEL_COMPLEXITY: ${MP_MODEL_COMPLEXITY:-1}
      MP_MIN_DETECTION_CONFIDENCE: ${MP_MIN_DETECTION_CONFIDENCE:-0.5}
      MP_TASK_MODEL_PATH: ${MP_TASK_MODEL_PATH:-models/mediapipe/pose_landmarker_lite.task}
      FRAME_TARGET_FPS: ${FRAME_TARGET_FPS:-15}
      FLOW_GATE_MODEL_PATH: ${FLOW_GATE_MODEL_PATH:-models/flow_qa_rf_v2.joblib}
      FLOW_GATE_STATIC_THRESHOLD: ${FLOW_GATE_STATIC_THRESHOLD:-0.30}
      FLOW_GATE_DYNAMIC_THRESHOLD: ${FLOW_GATE_DYNAMIC_THRESHOLD:-0.70}
      FLOW_GATE_LABEL_THRESHOLD: ${FLOW_GATE_LABEL_THRESHOLD:-0.50}
      FLOW_GATE_DEMOTE_CONFIDENCE: ${FLOW_GATE_DEMOTE_CONFIDENCE:-0.55}
      FLOW_GATE_VERSION_SUFFIX: ${FLOW_GATE_VERSION_SUFFIX:-flow_rf_v2}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ports:
      - "127.0.0.1:${WORKER_PORT:-8000}:8000"
    volumes:
      - worker_tmp:/tmp/hola-videos
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)",
        ]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s

volumes:
  worker_tmp:
```

- [ ] **Step 2: Create VM env example**

Create `deploy/vm/.env.vm.example` with non-secret redacted sample values:

```dotenv
AI_WORKER_IMAGE=asia-northeast3-docker.pkg.dev/hola-climbing-log/hola-climb/hola-ai-worker:replace-with-git-sha

REDIS_HOST=host.docker.internal
REDIS_PORT=6379
REDIS_PASSWORD=replace-with-real-redis-password
REDIS_DB=0
REDIS_STREAM_KEY=analysis:requests
REDIS_CONSUMER_GROUP=hola-ai-worker
REDIS_CONSUMER_NAME=worker-1
REDIS_PROGRESS_CHANNEL=analysis:progress
REDIS_BLOCK_MS=5000
REDIS_DLQ_KEY=analysis:requests:dlq
REDIS_PENDING_MIN_IDLE_MS=60000

GCS_BUCKET=hola-climbing-log-videos
GCS_DOWNLOAD_DIR=/tmp/hola-videos

AI_CALLBACK_SECRET=replace-with-real-ai-callback-secret
CALLBACK_TIMEOUT_SECONDS=10
CALLBACK_MAX_RETRIES=3
CALLBACK_RETRY_INITIAL_SECONDS=1

WORKER_HOST=0.0.0.0
WORKER_PORT=8000
LOG_LEVEL=INFO
MODEL_VERSION=rule_v3

MP_MODEL_COMPLEXITY=1
MP_MIN_DETECTION_CONFIDENCE=0.5
MP_TASK_MODEL_PATH=models/mediapipe/pose_landmarker_lite.task
FRAME_TARGET_FPS=15

FLOW_GATE_MODEL_PATH=models/flow_qa_rf_v2.joblib
FLOW_GATE_STATIC_THRESHOLD=0.30
FLOW_GATE_DYNAMIC_THRESHOLD=0.70
FLOW_GATE_LABEL_THRESHOLD=0.50
FLOW_GATE_DEMOTE_CONFIDENCE=0.55
FLOW_GATE_VERSION_SUFFIX=flow_rf_v2
```

- [ ] **Step 3: Create deploy script**

Create `deploy/vm/deploy-ai-worker.sh` with:

```bash
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
```

- [ ] **Step 4: Make deploy script executable**

Run:

```bash
chmod +x deploy/vm/deploy-ai-worker.sh
```

Expected: `deploy/vm/deploy-ai-worker.sh` is executable.

## Task 3: AI Worker Deploy Workflow

**Files:**
- Create: `.github/workflows/deploy-ai-worker.yml`

- [ ] **Step 1: Create deployment workflow**

Create `.github/workflows/deploy-ai-worker.yml` with:

```yaml
name: Deploy AI Worker VM

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

concurrency:
  group: deploy-ai-worker-${{ github.ref }}
  cancel-in-progress: true

env:
  IMAGE: ${{ vars.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT_ID }}/${{ vars.ARTIFACT_REPOSITORY }}/hola-ai-worker:${{ github.sha }}

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"

      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --frozen

      - name: Ruff
        run: uv run ruff check app tests scripts

      - name: Mypy
        run: uv run mypy app

      - name: Pytest
        run: uv run pytest

  deploy:
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v6

      - id: auth
        uses: google-github-actions/auth@v3
        with:
          project_id: ${{ vars.GCP_PROJECT_ID }}
          workload_identity_provider: ${{ vars.WIF_PROVIDER }}
          service_account: ${{ vars.WIF_SERVICE_ACCOUNT }}

      - uses: google-github-actions/setup-gcloud@v3

      - name: Show Google Cloud identity
        run: gcloud auth list --filter=status:ACTIVE --format="value(account)"

      - name: Configure Docker for Artifact Registry
        run: |
          gcloud auth configure-docker "${{ vars.GCP_REGION }}-docker.pkg.dev" --quiet
          gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://${{ vars.GCP_REGION }}-docker.pkg.dev"

      - name: Build image
        run: docker build --platform linux/amd64 -t "$IMAGE" .

      - name: Push image
        run: docker push "$IMAGE"

      - name: Ensure VM deploy directory
        run: |
          gcloud compute ssh "${{ vars.AI_VM_NAME }}" \
            --zone "${{ vars.AI_VM_ZONE }}" \
            --tunnel-through-iap \
            --quiet \
            --command "mkdir -p '${{ vars.AI_VM_DEPLOY_DIR }}'"

      - name: Copy VM deploy files
        run: |
          gcloud compute scp \
            --zone "${{ vars.AI_VM_ZONE }}" \
            --tunnel-through-iap \
            --quiet \
            deploy/vm/docker-compose.yml \
            deploy/vm/.env.vm.example \
            deploy/vm/deploy-ai-worker.sh \
            "${{ vars.AI_VM_NAME }}:${{ vars.AI_VM_DEPLOY_DIR }}/"

      - name: Roll out AI Worker
        run: |
          gcloud compute ssh "${{ vars.AI_VM_NAME }}" \
            --zone "${{ vars.AI_VM_ZONE }}" \
            --tunnel-through-iap \
            --quiet \
            --command "chmod +x '${{ vars.AI_VM_DEPLOY_DIR }}/deploy-ai-worker.sh' && '${{ vars.AI_VM_DEPLOY_DIR }}/deploy-ai-worker.sh' '$IMAGE'"
```

- [ ] **Step 2: Validate YAML by inspection**

Run:

```bash
sed -n '1,260p' .github/workflows/deploy-ai-worker.yml
```

Expected: workflow has a `test` job, a `deploy` job depending on `test`, WIF auth, Artifact Registry push, VM scp, and VM ssh rollout.

## Task 4: VM Runbook And Sensitive Commands

**Files:**
- Create: `deploy/vm/README.md`

- [ ] **Step 1: Create runbook**

Create `deploy/vm/README.md` with sections for:

- Runtime architecture
- First VM setup
- User-only sensitive setup commands
- GitHub variables
- Deploy flow
- Health checks
- Rollback
- Troubleshooting

The "User-only sensitive setup commands" section must include:

```bash
gcloud compute ssh hola-climbing-server \
  --zone asia-northeast3-a \
  --tunnel-through-iap

mkdir -p /home/minjoun/hola-ai
cd /home/minjoun/hola-ai
cp .env.vm.example .env.vm
chmod 600 .env.vm

nano .env.vm
```

The section must explicitly tell the user to fill only:

```text
REDIS_PASSWORD
AI_CALLBACK_SECRET
```

It must also include `gh variable set` commands for non-secret repository variables and a note that no production secret should be committed.

## Task 5: Verification

**Files:**
- Verify: `.github/workflows/ai-worker-ci.yml`
- Verify: `.github/workflows/deploy-ai-worker.yml`
- Verify: `deploy/vm/docker-compose.yml`
- Verify: `deploy/vm/.env.vm.example`
- Verify: `deploy/vm/deploy-ai-worker.sh`
- Verify: `deploy/vm/README.md`

- [ ] **Step 1: Check forbidden markers**

Run:

```bash
rg -n 'T''BD|TO''DO|place''holder|\\.\\.\\.' .github deploy/vm docs/superpowers/plans/2026-06-17-ai-worker-vm-cicd.md
```

Expected: no matches except intentional `replace-with-*` example values in `.env.vm.example` and README.

- [ ] **Step 2: Check compose rendering**

Run:

```bash
docker compose --env-file deploy/vm/.env.vm.example -f deploy/vm/docker-compose.yml config
```

Expected: compose renders one `ai-worker` service and one `worker_tmp` volume.

- [ ] **Step 3: Run local quality gates**

Run:

```bash
uv run ruff check app tests scripts
uv run mypy app
uv run pytest
```

Expected: all commands pass.

- [ ] **Step 4: Review git diff**

Run:

```bash
git diff --check
git diff --stat
```

Expected: no whitespace errors and only planned files changed.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add .github/workflows/ai-worker-ci.yml \
  .github/workflows/deploy-ai-worker.yml \
  deploy/vm/docker-compose.yml \
  deploy/vm/.env.vm.example \
  deploy/vm/deploy-ai-worker.sh \
  deploy/vm/README.md \
  docs/superpowers/plans/2026-06-17-ai-worker-vm-cicd.md
git commit -m "ci: deploy AI worker to VM"
```

Expected: commit contains only the CI/CD implementation and plan.
