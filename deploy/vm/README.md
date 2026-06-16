# Hola AI Worker VM Deployment

This directory contains the production-side VM runtime files for `hola-climbing-ai`.

The normal deployment path is:

```text
GitHub Actions
  -> test with uv, ruff, mypy, pytest
  -> build linux/amd64 Docker image
  -> push image to Artifact Registry
  -> copy deploy/vm files to the VM
  -> run deploy-ai-worker.sh IMAGE on the VM
  -> recreate only the ai-worker container
```

## Runtime Shape

```text
Spring backend
  -> Cloud Run

Existing VM hola-climbing-server
  -> PostgreSQL container
  -> Redis container
  -> hola-ai-worker container
```

The AI worker connects to the existing VM Redis through `host.docker.internal:6379`.
The compose file adds Docker's Linux host gateway mapping so the worker container can reach
the host-published Redis port without joining or changing the existing Redis container.

## User-Only Sensitive Setup

Run these commands from your local machine. You only type production-sensitive values inside
the VM-local `.env.vm` file.

```bash
cd /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-ai

gcloud compute ssh hola-climbing-server \
  --zone asia-northeast3-a \
  --tunnel-through-iap \
  --command "mkdir -p /home/minjoun/hola-ai"

gcloud compute scp \
  --zone asia-northeast3-a \
  --tunnel-through-iap \
  deploy/vm/.env.vm.example \
  hola-climbing-server:/home/minjoun/hola-ai/.env.vm.example

gcloud compute ssh hola-climbing-server \
  --zone asia-northeast3-a \
  --tunnel-through-iap
```

Inside the VM SSH session:

```bash
cd /home/minjoun/hola-ai
cp -n .env.vm.example .env.vm
chmod 600 .env.vm
nano .env.vm
```

Fill only these sensitive values:

```text
REDIS_PASSWORD
AI_CALLBACK_SECRET
```

`AI_WORKER_IMAGE` can stay as the sample value before the first workflow deploy. The rollout
script rewrites it to the pushed image tag every time.

Do not commit `.env.vm`. It must live only on the VM.

## GitHub Variables

These are not production secrets. Set them on the GitHub repository that runs the workflow.
The commands below target the team repository. If you run Actions from a personal mirror,
replace `hola-climb/hola-climbing-ai` with that repository.

```bash
gh variable set GCP_PROJECT_ID \
  --repo hola-climb/hola-climbing-ai \
  --body hola-climbing-log

gh variable set GCP_REGION \
  --repo hola-climb/hola-climbing-ai \
  --body asia-northeast3

gh variable set GCP_ZONE \
  --repo hola-climb/hola-climbing-ai \
  --body asia-northeast3-a

gh variable set ARTIFACT_REPOSITORY \
  --repo hola-climb/hola-climbing-ai \
  --body hola-climb

gh variable set AI_VM_NAME \
  --repo hola-climb/hola-climbing-ai \
  --body hola-climbing-server

gh variable set AI_VM_ZONE \
  --repo hola-climb/hola-climbing-ai \
  --body asia-northeast3-a

gh variable set AI_VM_DEPLOY_DIR \
  --repo hola-climb/hola-climbing-ai \
  --body /home/minjoun/hola-ai
```

Copy Workload Identity Federation values from the backend repository:

```bash
gh variable set WIF_PROVIDER \
  --repo hola-climb/hola-climbing-ai \
  --body "$(gh variable list --repo hola-climb/hola-climbing-server --json name,value -q '.[] | select(.name == "WIF_PROVIDER") | .value')"

gh variable set WIF_SERVICE_ACCOUNT \
  --repo hola-climb/hola-climbing-ai \
  --body "$(gh variable list --repo hola-climb/hola-climbing-server --json name,value -q '.[] | select(.name == "WIF_SERVICE_ACCOUNT") | .value')"
```

## GCP Permissions

The WIF service account used by GitHub Actions needs:

```text
roles/artifactregistry.writer
roles/compute.instanceAdmin.v1 or narrower compute SSH/scp permissions
roles/iap.tunnelResourceAccessor when --tunnel-through-iap is used
```

The VM runtime service account needs:

```text
Artifact Registry read access
GCS read access for hola-climbing-log-videos
```

If the VM can already pull images and the worker readiness can access GCS, no extra runtime
credential file is needed. The Python GCS client inside the container can use Compute Engine
metadata credentials through ADC.

## Manual Rollout

After `.env.vm` exists on the VM, a manual rollout is:

```bash
cd /home/minjoun/hola-ai
./deploy-ai-worker.sh asia-northeast3-docker.pkg.dev/hola-climbing-log/hola-climb/hola-ai-worker:GIT_SHA
```

The script:

```text
1. Updates AI_WORKER_IMAGE in .env.vm.
2. Pulls the new image.
3. Runs docker compose up -d --no-deps ai-worker.
4. Prints ai-worker status and recent logs.
```

It does not recreate PostgreSQL or Redis.

## Health Checks

Run on the VM:

```bash
cd /home/minjoun/hola-ai
docker compose --env-file .env.vm -f docker-compose.yml ps ai-worker
docker compose --env-file .env.vm -f docker-compose.yml logs --tail=100 ai-worker
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/health/ready
```

Healthy signs:

```text
/health returns 200
/health/ready returns 200 after Redis and GCS ADC are reachable
logs show consumer starting
logs show stream=analysis:requests and group=hola-ai-worker
```

Redis stream checks from the VM:

```bash
docker exec -it hola-redis redis-cli
```

Inside the Redis prompt, type:

```text
AUTH <password>
XINFO GROUPS analysis:requests
XPENDING analysis:requests hola-ai-worker
XLEN analysis:requests:dlq
```

## Rollback

Rollback is image-based. Pick a previous image tag and run:

```bash
cd /home/minjoun/hola-ai
./deploy-ai-worker.sh asia-northeast3-docker.pkg.dev/hola-climbing-log/hola-climb/hola-ai-worker:PREVIOUS_GIT_SHA
```

List recent tags:

```bash
gcloud artifacts docker tags list \
  asia-northeast3-docker.pkg.dev/hola-climbing-log/hola-climb/hola-ai-worker \
  --format="table(tag,version,createTime)" \
  --limit=20
```

## Troubleshooting

### `.env.vm` missing

```bash
cd /home/minjoun/hola-ai
cp -n .env.vm.example .env.vm
chmod 600 .env.vm
nano .env.vm
```

Fill `REDIS_PASSWORD` and `AI_CALLBACK_SECRET`.

### Worker cannot connect to Redis

Check that the existing Redis container publishes port `6379` on the VM host:

```bash
docker ps --filter name=hola-redis
docker port hola-redis
```

If Redis is only available on a Docker network, set `REDIS_HOST` in `.env.vm` to the reachable
host or attach the worker to the existing external network after confirming the network name.

### `/health/ready` reports GCS unavailable

Confirm the VM service account can read the bucket:

```bash
gcloud storage ls gs://hola-climbing-log-videos
```

Then check worker logs:

```bash
docker compose --env-file .env.vm -f docker-compose.yml logs --tail=100 ai-worker
```

### Callback returns 401

`AI_CALLBACK_SECRET` in `.env.vm` must exactly match the Spring Cloud Run
`AI_CALLBACK_SECRET` Secret Manager value.
