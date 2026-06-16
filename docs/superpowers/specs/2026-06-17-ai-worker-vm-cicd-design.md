# AI Worker VM CI/CD Design

Date: 2026-06-17
Status: Approved for implementation planning

## Context

`hola-climbing-ai` is a long-running Python worker that consumes Spring-dispatched Redis Stream jobs, downloads uploaded videos from GCS, analyzes them with MediaPipe/OpenCV, publishes progress through Redis Pub/Sub, and callbacks to Spring with `X-AI-Callback-Secret`.

The backend deployment is already moving toward a hybrid GCP shape:

- Spring backend runs on Cloud Run.
- PostgreSQL with pgvector and Redis run on the existing Compute Engine VM `hola-climbing-server`.
- The VM private IP is `10.178.0.2`.
- Spring Cloud Run connects to VM PostgreSQL and Redis through private VPC egress.
- AI worker should run on the same VM as Redis to keep Redis Streams local and low-cost.

The current AI repository already has the important runtime foundations: `Dockerfile`, `docker-compose.yml` for local development, `/health`, `/health/ready`, `uv` lockfile, `ruff`, `mypy`, `pytest`, GCS ADC support, Redis consumer group creation, `XAUTOCLAIM`, and DLQ behavior.

## Goals

- Add CI for the AI worker repository.
- Build and push a reproducible AI worker Docker image to GCP Artifact Registry.
- Deploy the worker to the existing VM with GitHub Actions.
- Keep PostgreSQL and Redis on the VM unchanged.
- Make worker rollout idempotent and rollback-friendly.
- Document the GitHub variables, VM env file, IAM requirements, and smoke checks needed to operate the worker.

## Non-Goals

- Do not move the AI worker to Cloud Run Jobs, Cloud Run Worker Pools, GKE, or Cloud SQL/Memorystore.
- Do not change the Redis Stream, Pub/Sub, callback body, or GCS object path contracts.
- Do not deploy the frontend.
- Do not rebuild PostgreSQL or Redis data volumes as part of normal AI worker deploys.
- Do not store production secrets in GitHub repository files.

## Recommended Approach

Use GitHub Actions as the build and deploy controller:

```text
GitHub Actions
  -> uv sync
  -> ruff / mypy / pytest
  -> docker build --platform linux/amd64
  -> docker push Artifact Registry
  -> gcloud compute scp deploy files to VM
  -> gcloud compute ssh VM
  -> deploy-ai-worker.sh IMAGE
  -> docker compose pull/up ai-worker
```

The VM keeps the durable services running:

```text
Existing VM hola-climbing-server
  -> PostgreSQL container
  -> Redis container
  -> AI worker container
```

Only the `ai-worker` service is replaced during worker deployments. Redis Streams keeps unacknowledged jobs in the PEL, and the worker already reclaims stale pending jobs with `XAUTOCLAIM`.

## Architecture

```text
Client
  -> Spring Cloud Run
      -> VM private IP 10.178.0.2:5432 PostgreSQL
      -> VM private IP 10.178.0.2:6379 Redis
      -> GCS Signed URL issuance

AI Worker on VM
  -> Redis Stream analysis:requests
  -> GCS bucket hola-climbing-log-videos
  -> Redis Pub/Sub analysis:progress
  -> Spring Cloud Run callback URL from stream message
```

The AI worker image is immutable and tagged by commit SHA:

```text
asia-northeast3-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPOSITORY/hola-ai-worker:$GITHUB_SHA
```

## Components

### GitHub CI Workflow

Add `.github/workflows/ai-worker-ci.yml`.

Triggers:

- `pull_request`
- `push` to `master`
- `workflow_dispatch`

Checks:

- `uv sync --frozen`
- `uv run ruff check app tests scripts`
- `uv run mypy app`
- `uv run pytest`

The existing integration tests skip Docker-only Redis tests when Docker is unavailable, but GitHub-hosted Ubuntu runners have Docker, so the Redis stream integration tests should run normally.

### GitHub Deployment Workflow

Add `.github/workflows/deploy-ai-worker.yml`.

Triggers:

- `push` to `master`
- `workflow_dispatch`

Deployment steps:

- Authenticate to GCP with Workload Identity Federation.
- Configure Docker for Artifact Registry.
- Build the worker image for `linux/amd64`.
- Push the image.
- Copy VM deployment files to `AI_VM_DEPLOY_DIR`.
- SSH to the VM through IAP.
- Run `deploy-ai-worker.sh "$IMAGE"`.
- Print `docker compose ps ai-worker` and recent worker logs.

Use `concurrency` so only one production worker deploy runs at a time.

### VM Runtime Files

Add `deploy/vm/docker-compose.yml`.

This compose file describes the VM production runtime. It should include `ai-worker` and can document the existing PostgreSQL/Redis services, but normal deploy commands should target only `ai-worker`.

The worker service should:

- Use `image: ${AI_WORKER_IMAGE}`.
- Use `restart: unless-stopped`.
- Read `.env.vm`.
- Set `REDIS_HOST` to the VM Redis service or host address.
- Mount a temp volume for `/tmp/hola-videos`.
- Keep `FLOW_GATE_MODEL_PATH=models/flow_qa_rf_v2.joblib`.
- Expose worker HTTP only on localhost or not publish it externally unless needed for VM-local health checks.
- Use `/health` as liveness and `/health/ready` for manual dependency checks.

Add `deploy/vm/.env.vm.example`.

Required values:

- `AI_WORKER_IMAGE`
- `REDIS_HOST`
- `REDIS_PORT`
- `REDIS_PASSWORD`
- `GCS_BUCKET`
- `AI_CALLBACK_SECRET`
- `GCS_DOWNLOAD_DIR`
- `MODEL_VERSION`
- Redis stream keys and consumer settings
- MediaPipe and flow gate settings
- Callback timeout/retry settings

`AI_CALLBACK_SECRET` and `REDIS_PASSWORD` must match the values used by Spring Cloud Run through Secret Manager.

Add `deploy/vm/deploy-ai-worker.sh`.

Responsibilities:

- Require an image argument.
- Validate `.env.vm` exists.
- Configure Docker auth for Artifact Registry when `gcloud` is available.
- Update `AI_WORKER_IMAGE` inside `.env.vm`.
- Pull the new image.
- Recreate only `ai-worker`.
- Show container status and health.
- Avoid touching PostgreSQL and Redis unless explicitly requested.

Add `deploy/vm/README.md`.

It should cover first-time VM setup, GitHub variables, GCP IAM, secrets, deploy flow, smoke checks, rollback, and troubleshooting.

## GitHub Variables

The AI repository needs these variables:

```text
GCP_PROJECT_ID=hola-climbing-log
GCP_REGION=asia-northeast3
GCP_ZONE=asia-northeast3-a
ARTIFACT_REPOSITORY=hola-climb
WIF_PROVIDER=same value as the existing backend Cloud Run deploy workflow
WIF_SERVICE_ACCOUNT=same value as the existing backend Cloud Run deploy workflow
AI_VM_NAME=hola-climbing-server
AI_VM_ZONE=asia-northeast3-a
AI_VM_DEPLOY_DIR=/home/deploy/hola-ai
```

If the VM deploy user or path differs, only `AI_VM_DEPLOY_DIR` and the compute SSH target need to change.

## GCP IAM Requirements

The GitHub WIF service account needs:

- Push access to the Artifact Registry repository.
- `compute.instances.get`
- `compute.instances.list`
- `compute.instances.osLogin` or equivalent SSH access path.
- IAP tunnel permission if `--tunnel-through-iap` is used.

The VM runtime service account needs:

- Pull access to Artifact Registry.
- GCS read access for `hola-climbing-log-videos`.

The Spring Cloud Run runtime service account keeps its existing Secret Manager access for backend secrets. The AI worker reads runtime secrets from `.env.vm`, not from GitHub files.

## Data Flow

1. Spring writes a message to `analysis:requests`.
2. AI worker reads it with `XREADGROUP`.
3. AI worker downloads the object from GCS using ADC.
4. AI worker publishes `PROCESSING` events to `analysis:progress`.
5. AI worker POSTs the callback URL from the stream message.
6. Spring stores results, publishes terminal progress, and sends notifications.

Deploying a new worker image does not change the stream contract. If deployment interrupts a job before ACK, the job remains pending and is eligible for `XAUTOCLAIM`.

## Error Handling And Rollback

### Failed CI

No image is pushed and no VM state changes.

### Failed Image Build Or Push

Deployment stops before touching the VM.

### Failed VM Rollout

The deployment script should show `docker compose ps ai-worker` and recent logs. Redis/PostgreSQL continue running. If the new worker fails before ACKing a claimed job, Redis PEL recovery handles retry after `REDIS_PENDING_MIN_IDLE_MS`.

### Rollback

Rollback is image based:

```bash
cd $AI_VM_DEPLOY_DIR
./deploy-ai-worker.sh asia-northeast3-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPOSITORY/hola-ai-worker:<previous-sha>
```

The deploy README should include a command to list recent image tags from Artifact Registry.

## Observability And Smoke Checks

After deploy:

- `docker compose ps ai-worker`
- `docker compose logs --tail=100 ai-worker`
- VM-local `/health`
- VM-local `/health/ready`
- Redis `XINFO GROUPS analysis:requests`
- Redis `XPENDING analysis:requests hola-ai-worker`
- Redis `XLEN analysis:requests:dlq`
- One GCS-backed Spring upload/analysis E2E smoke when credentials and a sample video are available

Expected healthy signs:

- Worker logs show startup and consumer group information.
- `/health` returns 200.
- `/health/ready` returns 200 when Redis and GCS ADC are configured.
- New jobs move from `analysis:requests` to Spring callback results.
- DLQ does not grow unexpectedly.

## Testing Strategy

Before implementation is considered complete:

- Run `uv run ruff check app tests scripts`.
- Run `uv run mypy app`.
- Run `uv run pytest`.
- Run `docker compose -f deploy/vm/docker-compose.yml --env-file deploy/vm/.env.vm.example config`.
- Build the AI image locally if Docker and network are available.
- Verify workflow YAML syntax by inspection or a dry run where available.

## Risks

- GitHub Actions build may need network access for the Dockerfile `ADD` of the MediaPipe model. This is acceptable because CI runners have network access, but failed external download should stop deployment before VM changes.
- VM ADC must be configured correctly. Without GCS access, `/health/ready` fails and jobs fail at download time.
- `gcloud compute ssh --tunnel-through-iap` requires IAM and firewall setup. If IAP is unavailable, the deploy workflow must switch to the approved SSH route for the VM.
- The VM may already have a hand-written compose file. Implementation must inspect the VM setup before replacing anything production-critical.

## Acceptance Criteria

- Pull requests and master pushes run AI CI.
- Master pushes build and push an AI worker image.
- Deployment replaces only the `ai-worker` container on the existing VM.
- PostgreSQL and Redis containers remain untouched by normal AI worker deploys.
- Worker readiness succeeds with Redis and GCS access configured.
- A Redis Stream job can be consumed and callback results reach Spring.
- Rollback to a previous image tag is documented and can be run with one script command.
