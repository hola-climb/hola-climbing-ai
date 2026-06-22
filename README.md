# Hola (올라) — AI 클라이밍 영상 분석 워커

> Spring 서버(`hola-climbing-server`)와 한 쌍으로 동작하는 Python AI 워커
> SSAFY 자율 프로젝트 · 2026.05.15 ~ 2026.06.25

---

## 프로젝트 소개

**Hola(올라)** 는 클라이밍을 즐기는 사람들을 위한 **AI 동작 분석 기반 영상 SNS** 입니다.
이 저장소(`hola-climbing-ai`)는 그 중 **AI 분석 워커**를 담당합니다.

클라이밍 영상이 업로드되면 워커는 GCS에서 영상을 가져와 MediaPipe Pose로 33개 키포인트를
프레임별로 추출하고, 규칙 기반 분류기로 클라이밍 기술(하이스텝/플래깅/훅/락오프/다이노/
코디네이션)을 구간별로 라벨링합니다. 결과는 Spring 서버의 `/api/analysis/videos/{id}`
엔드포인트로 콜백하며, 진행 상태는 Redis Pub/Sub으로 발행되어 사용자가 SSE로 실시간
구독할 수 있습니다.

영상 바이너리는 서버를 거치지 않습니다 — 클라이언트가 GCS Signed URL로 직접 업로드한
객체를 워커가 ADC(Application Default Credentials)로 직접 받아 처리합니다.

### 단일 진실 원천 (SSOT)

워커의 모든 입출력 계약(Redis 스트림 키, 콜백 body shape, ErrorCode 등)은
**Spring 서버(`hola-climbing-server`)가 SSOT**입니다. 본 워커 코드와 Spring 측 코드가
충돌하면 **워커를 수정**합니다. 매핑 표는 `_workspace/04_integration_contract_match.md`
에 라인 단위로 정리되어 있습니다.

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| Language / Runtime | Python 3.11 |
| Web | FastAPI 0.115+, uvicorn |
| Validation | Pydantic v2, pydantic-settings |
| AI / CV | MediaPipe Pose 0.10+, OpenCV (headless) 4.10+, NumPy |
| Messaging | Redis 7 (Streams + Pub/Sub), redis-py async |
| Storage | Google Cloud Storage (ADC) |
| HTTP Client | httpx, tenacity (지수 백오프) |
| Packaging | uv (lock-based), hatchling |
| Test | pytest, pytest-asyncio, testcontainers, respx |
| Lint / Type | ruff, mypy strict |

### 설계 원칙

- **Spring 계약이 SSOT** — 워커 모델은 항상 Spring DTO에 맞춘다
- **워커는 분석만 한다** — DB/인증/SSE는 Spring 책임, 워커는 Redis로만 통신
- **장기 실행 컨슈머는 FastAPI lifespan task** — 별도 프로세스 분리 없음 (단일 컨테이너)
- **vision 모듈은 동기 / CPU-bound** — `asyncio.to_thread`로 이벤트 루프와 분리
- **모든 실패는 `status="failed"` 콜백 한 가지로 통신** — ErrorCode를 워커가 만들지 않음
- **재시도는 멱등** — Spring `ingestResult`가 `deleteByVideoId` 후 insert이므로 동일 videoId 재호출 안전

---

## 아키텍처

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Client (Vue3/Capacitor)                                                     │
│    │ ① Signed URL 받아 영상 업로드 (Spring 미경유)                            │
│    ▼                                                                         │
│  GCS  hola-climbing-log-videos/videos/uploads/2026/.../abc.mp4               │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ▲                          │
                                  │ ⑥ download_to_filename   │ ② client POST objectPath
                                  │   (ADC)                  ▼
┌──────────────────────────────────┴────────────────────┐  ┌──────────────────┐
│  Spring (hola-climbing-server)                        │  │  hola-climbing-ai│
│    AnalysisDispatcher                                 │  │  (this repo)     │
│    │                                                  │  │                  │
│    │ ③ XADD analysis:requests                         │  │                  │
│    ▼                                                  │  │                  │
│  ┌────────────────────────────────────────────────────┴──┴───────────┐      │
│  │ Redis 7                                                            │      │
│  │   Stream: analysis:requests   (fields: videoId, gcsPath, callback) │      │
│  │   PubSub: analysis:progress                                        │      │
│  │   Stream: analysis:requests:dlq                                    │      │
│  └────────┬──────────────────────────────────────────────────┬───────┘      │
│           │ ④ XREADGROUP                            ⑦ PUBLISH │              │
│           │    group=hola-ai-worker                analysis:  │              │
│           ▼                                       progress   ▼              │
│                                                  ┌──────────────────────┐    │
│  AnalysisProgressListener (subscriber)           │ Worker pipeline      │    │
│    ├─ statusStore.save                           │  ├─ ⑥ GCS download   │    │
│    └─ SSE fan-out                                │  ├─ iter_frames      │    │
│           ▲                                       │  ├─ extract_pose    │    │
│           │ ⑨ COMPLETED/FAILED publish            │  ├─ split_segments  │    │
│           │                                       │  └─ classify_segs   │    │
│  AnalysisController                              └──────────┬───────────┘    │
│    POST /api/analysis/videos/{id}                            │                │
│    └─ ingestResult (멱등)  ◄────────── ⑧ POST callback ──────┘                │
└───────────────────────────────────────────────────────────────────────────────┘
```

1. 클라이언트가 Signed URL로 GCS에 직접 업로드
2. 클라이언트가 Spring에 objectPath 등록
3. Spring이 `analysis:requests` 스트림에 (videoId, gcsPath, callbackUrl) 적재
4. 워커가 `XREADGROUP`으로 메시지 소비
5. 워커가 `PROCESSING` 진행률을 `analysis:progress` 채널에 publish
6. 워커가 GCS에서 영상 다운로드 (ADC)
7. 단계별로 진행률 publish
8. 분석 완료/실패 후 워커가 Spring 콜백 POST (멱등)
9. Spring이 `COMPLETED`/`FAILED`를 다시 publish → SSE fan-out → FCM push

---

## 빠른 시작

### 사전 요구사항

- Python 3.11 (3.13 미지원 — MediaPipe wheel 호환성)
- [uv](https://docs.astral.sh/uv/) 0.5+ — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker / Docker Compose (로컬 통합 테스트 시)
- GCS Service Account 키 (또는 `gcloud auth application-default login`)
- Spring 서버(`hola-climbing-server`)가 동일 Redis 인스턴스를 사용 중이어야 함

### 1. 의존성 설치

```bash
uv sync                 # 런타임 + dev deps 모두
uv sync --no-dev        # 운영 빌드 시 dev 제외
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 편집 — 최소 REDIS_PASSWORD, GCS_BUCKET, GOOGLE_APPLICATION_CREDENTIALS 확인
```

### 3. 로컬 실행 (uv)

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
# 또는 reload 모드
uv run uvicorn app.main:app --reload
```

기동 시 `XGROUP CREATE analysis:requests hola-ai-worker $ MKSTREAM`을 자동 호출합니다.
이미 그룹이 있으면 `BUSYGROUP`을 무시하고 계속 진행합니다.

### 4. Docker Compose (Redis 포함 통합 테스트)

```bash
docker compose up --build       # redis + worker 동시 기동
docker compose logs -f worker   # 워커 로그
```

> Apple Silicon에서는 MediaPipe ARM64 wheel 부재로 인해 `platform: linux/amd64`를
> 명시했습니다. Rosetta 2가 설치되어 있어야 합니다 (`softwareupdate --install-rosetta`).

### 5. 헬스체크

```bash
curl http://localhost:8000/health        # liveness
curl http://localhost:8000/health/ready  # readiness (Redis/GCS)
```

---

## 환경 변수

`.env.example`이 SSOT. 본 표는 가독성을 위해 복제. 변경 시 양쪽 동시 수정.

### Worker

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|------|------|
| `WORKER_HOST` | `0.0.0.0` | no | FastAPI bind host |
| `WORKER_PORT` | `8000` | no | FastAPI bind port |
| `WORKER_CONCURRENCY` | `1` | no | 동시에 실행할 Redis Streams consumer 수. `e2-medium` 운영 VM은 `2`부터 측정 권장 |
| `LOG_LEVEL` | `INFO` | no | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `MODEL_VERSION` | `rule_v3` | no | 콜백 body `model_version` 값. MVP는 규칙 기반 |

### Redis (Spring과 동일 인스턴스 공유)

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|------|------|
| `REDIS_HOST` | `localhost` | yes | Spring `spring.data.redis.host`와 동일 |
| `REDIS_PORT` | `6379` | no | Spring 측과 동일 |
| `REDIS_PASSWORD` | (빈 값) | env 따라 | Spring `REDIS_PASSWORD`와 동일 값 |
| `REDIS_DB` | `0` | no | Redis DB 인덱스 |
| `REDIS_STREAM_KEY` | `analysis:requests` | no | **Spring 확정값 — 변경 금지** |
| `REDIS_CONSUMER_GROUP` | `hola-ai-worker` | no | 워커 측 정의. Spring 무관 |
| `REDIS_CONSUMER_NAME` | `worker-1` | no | 기본값일 때 hostname+pid로 자동 고유화. 병렬 실행 시 slot suffix 추가 |
| `REDIS_PROGRESS_CHANNEL` | `analysis:progress` | no | **Spring 확정값 — 변경 금지** |
| `REDIS_BLOCK_MS` | `5000` | no | `XREADGROUP BLOCK` ms |
| `REDIS_DLQ_KEY` | `analysis:requests:dlq` | no | Dead-letter 스트림 키 |
| `REDIS_PENDING_MIN_IDLE_MS` | `60000` | no | 이 시간 이상 idle인 PEL 메시지를 `XAUTOCLAIM`으로 회수 |

### GCS

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|------|------|
| `GCS_BUCKET` | `hola-climbing-log-videos` | yes | Spring `gcs.bucket`과 동일 |
| `GOOGLE_APPLICATION_CREDENTIALS` | `./keys/gcs-sa.json` | local | SA 키 파일 경로. 운영 VM은 ADC 자동 → 미설정 OK |
| `GCS_DOWNLOAD_DIR` | `/tmp/hola-videos` | no | 다운로드 작업 디렉토리 |

### 콜백 (Worker → Spring)

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|------|------|
| `AI_CALLBACK_SECRET` | (빈 값) | Spring 설정 시 yes | Spring `AI_CALLBACK_SECRET`과 동일. 값이 있으면 `X-AI-Callback-Secret` 헤더로 전송 |
| `CALLBACK_TIMEOUT_SECONDS` | `10` | no | httpx 요청 타임아웃 |
| `CALLBACK_MAX_RETRIES` | `3` | no | tenacity `stop_after_attempt` |
| `CALLBACK_RETRY_INITIAL_SECONDS` | `1` | no | 지수 백오프 초기값 |

### MediaPipe / OpenCV

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|------|------|
| `MP_MODEL_COMPLEXITY` | `1` | no | 0=light, 1=full, 2=heavy. 1이 CPU/정확도 균형 |
| `MP_MIN_DETECTION_CONFIDENCE` | `0.5` | no | landmark detection threshold |
| `FRAME_TARGET_FPS` | `15` | no | OpenCV 다운샘플링 (원본 30fps → 15fps) |

### Flow gate (optional ML 보정)

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|------|------|
| `FLOW_GATE_MODEL_PATH` | `models/flow_qa_rf_v2.joblib` | no | flow RF artifact 경로. 빈 값으로 명시하면 off |
| `FLOW_GATE_STATIC_THRESHOLD` | `0.30` | no | prob_dynamic이 이 값 미만이면 static 확신 → 보정 발동 |
| `FLOW_GATE_DYNAMIC_THRESHOLD` | `0.70` | no | 이 값 초과면 dynamic 확신 (현재 개입 없음, 예약) |
| `FLOW_GATE_LABEL_THRESHOLD` | `0.50` | no | callback top-level `is_dynamic` 판정 기준 (`prob_dynamic >= threshold`) |
| `FLOW_GATE_DEMOTE_CONFIDENCE` | `0.55` | no | static 확신 시 이 confidence 미만의 dynamic segment를 drop |
| `FLOW_GATE_VERSION_SUFFIX` | `flow_rf_v2` | no | 게이트 적용 시 `model_version`에 붙는 suffix |

영상 단위 dynamic/static RF (group-kfold balanced accuracy **0.8381**, 2026-06-10)가
rule 출력의 사후 보정 prior로 동작합니다. flow RF가 "static 위주 영상"이라
확신하고 rule confidence도 약한 dynamic segment만 drop — 두 신호가 동시에
약할 때만 개입하므로 백다이노류 미세 다이나믹 무브는 보존됩니다.

- 게이트 on + 추론 성공 시 콜백 `model_version`이 `rule_v3+flow_rf_v2`로 바뀌고,
  top-level `dynamic_probability`와 `is_dynamic`이 함께 전송됩니다.
- 게이트 off 또는 추론 실패 시 top-level `dynamic_probability`/`is_dynamic`은 `null`입니다.
  이 값은 segment별 dynamic 비율로 fallback 계산하지 않습니다.
- 추론 비용: 영상당 약 +7~11초 (Farneback optical flow 전체 패스)
- 모델 로딩/추론 실패 시 rule 출력으로 자동 fallback (분석 실패 아님)
- artifact (`models/flow_qa_rf_v2.joblib`, 816KB)는 git 추적 + Docker 이미지 포함.
  기본 설정과 docker-compose 모두 게이트 on (`FLOW_GATE_MODEL_PATH` 빈 값 override로 off)
- 추론 의존성 (joblib/scikit-learn/scipy)은 main 의존성. `ml` 그룹(torch)은 학습 전용
- v2 artifact는 legacy 42-dim feature로, v3 artifact는 burst-aware 46-dim feature로 추론합니다.
  v3 실험(2026-06-10)은 group-kfold balanced accuracy `0.8384`로 v2 대비 보합이라
  운영 기본 artifact는 v2를 유지합니다.

---

## Redis Streams 계약

워커가 소비하는 메시지의 raw shape입니다. **Spring `RedisStreamAnalysisJobQueue.java`가
적재하는 그대로**이므로 변경 시 양쪽 동기화가 필수입니다.

### Stream `analysis:requests`

| 필드 | 타입 | 비고 |
|------|------|------|
| `videoId` | string (Long 직렬화) | camelCase. 워커가 int로 파싱 |
| `gcsPath` | string | 객체 경로 (`gs://` 또는 https prefix 없음) |
| `callbackUrl` | string | Spring이 미리 조립한 절대 URL. 워커는 그대로 사용 |

> Stream payload는 Spring Jackson SNAKE_CASE 정책의 영향을 받지 **않습니다** (HTTP JSON에만
> 적용). 따라서 키는 camelCase 원본 그대로입니다.

### Pub/Sub `analysis:progress`

워커는 `PROCESSING` 단계만 publish합니다. `COMPLETED`/`FAILED`는 Spring이 콜백 처리 후
자동 발행합니다.

```json
{
  "video_id": 42,
  "stage": "PROCESSING",
  "message": "프레임 추출 중",
  "updated_at": "2026-05-28T10:32:45.123Z"
}
```

| 필드 | 타입 | 비고 |
|------|------|------|
| `video_id` | number | snake_case (Jackson SNAKE_CASE 적용) |
| `stage` | enum string | `QUEUED` / `PROCESSING` / `COMPLETED` / `FAILED` (대문자) |
| `message` | string | 한국어 진행 메시지 (예: `"분석 시작"`, `"포즈 추정 완료"`) |
| `updated_at` | ISO-8601 string | UTC, `Z` suffix (Spring `Instant` 호환) |

### Consumer group

워커가 직접 생성합니다 — Spring은 group을 만들지 않습니다.

```
XGROUP CREATE analysis:requests hola-ai-worker $ MKSTREAM
```

`BUSYGROUP` 에러는 무시하고 진행합니다 (이미 존재).

### Dead-letter

콜백 4xx, max retry 초과, 파싱 실패 등은 `analysis:requests:dlq` 스트림으로 이동
후 `XACK`합니다 (PEL 누적 방지). DLQ 컨슈머는 별도 운영 도구가 처리합니다.

워커는 새 메시지를 읽기 전에 `XAUTOCLAIM`으로 `REDIS_PENDING_MIN_IDLE_MS` 이상 idle인
pending 메시지를 현재 consumer로 회수합니다. 처리 중 워커가 종료되어 ACK가 보류된 메시지는
다음 루프/재시작 후 재처리 대상이 됩니다.

---

## 콜백 계약 (Worker → Spring)

워커가 분석 완료/실패 시 호출합니다.

- **URL**: 메시지의 `callbackUrl` 그대로 사용 (워커가 path 조립 금지)
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
  - `AI_CALLBACK_SECRET`이 설정되어 있으면 `X-AI-Callback-Secret`를 함께 전송합니다.
  - Spring도 `AI_CALLBACK_SECRET`이 설정된 환경에서는 `POST /api/analysis/**`에 같은 헤더를 요구합니다.

### Body (성공)

```json
{
  "status": "done",
  "model_version": "rule_v3",
  "techniques": ["high_step", "dyno"],
  "is_dynamic": true,
  "dynamic_probability": 0.82,
  "segments": [
    {
      "sequence_index": 0,
      "start_time_ms": 0,
      "end_time_ms": 1240,
      "technique": "high_step",
      "is_dynamic": false,
      "confidence": 0.87
    }
  ]
}
```

- `techniques`: `segments[].technique`를 영상 단위로 중복 제거한 목록입니다. 순서는
  `high_step`, `flagging`, `toe_hook`, `heel_hook`, `lock_off`, `dyno`, `coordination`입니다.
- `is_dynamic`: flow gate가 성공했을 때만 `dynamic_probability >= FLOW_GATE_LABEL_THRESHOLD`
  기준으로 산출합니다. segment별 `is_dynamic` 비율로 계산하지 않습니다.
- `dynamic_probability`: flow gate의 영상 단위 `prob_dynamic`입니다. 게이트 off/실패 시 `null`.
- `segments[].is_dynamic`: 기존과 동일하게 raw segment metadata입니다.

### Body (실패)

```json
{
  "status": "failed",
  "model_version": "rule_v3",
  "techniques": [],
  "is_dynamic": null,
  "dynamic_probability": null,
  "segments": []
}
```

### 재시도 정책

- **2xx**: 성공. body는 무시 (`is_success=false`만 경고 로깅).
- **4xx**: 즉시 dead-letter (계약 위반 — videoId 없음/INVALID_INPUT 등).
- **5xx / 429 / 네트워크 오류**: 지수 백오프 재시도, `CALLBACK_MAX_RETRIES` 회 소진 후 dead-letter.
- **재시도는 멱등 안전**: Spring `AnalysisServiceImpl.ingestResult`가 `deleteByVideoId` 후 insert.

---

## 인식하는 클라이밍 기술

워커는 다음 7개 라벨 중 하나를 각 구간에 부여합니다 (모두 snake_case).

| 라벨 | 설명 | 동적/정적 |
|------|------|-----------|
| `high_step` | 발을 골반 이상 높이로 올리는 기술 | static |
| `flagging` | 한쪽 다리를 반대 방향으로 뻗어 무게중심 보정 | static |
| `toe_hook` | 발끝(toe)을 홀드 위에 걸어 당김 | static |
| `heel_hook` | 발뒤꿈치(heel)를 홀드 위에 걸어 당김 | static |
| `lock_off` | 한 팔로 몸을 고정한 정적 자세 | static |
| `dyno` | 양손 도약 점프 무브 | **dynamic** |
| `coordination` | 다수 limb이 동시에 이동하는 복합 무브 | **dynamic** |

각 기술의 정확한 임계값과 우선순위는 `_workspace/02_vision_technique_rules.md`와
`app/services/vision/_thresholds.py`를 참조하세요.

알고리즘 흐름:
1. `iter_frames(target_fps=15)` — OpenCV로 다운샘플링하며 프레임 yield
2. `extract_pose_landmarks(...)` — MediaPipe Pose 33 landmark 추출
3. `split_segments(...)` — 정지 구간(quiet) + 골반 정점으로 동작 단위 분할
4. `classify_segments(...)` — 우선순위 + score로 단일 라벨 부여 (임계 미달 시 drop)

### 6기술 MVP 출력/DB 검증 절차

실제 MediaPipe 영상 추론은 macOS sandbox 안에서 Metal/GL service 제약이 있을 수 있으므로,
아래 smoke 명령은 로컬 터미널의 네이티브 `uv` 환경에서 실행하는 것을 권장합니다.

#### 1. 워커 단독 segments 덤프

캐시된 GCS 영상에서 dynamic 5개 + static 5개를 라벨 균형으로 골라 rule 출력과 flow gate
출력을 각각 JSON으로 저장합니다.

```bash
uv run python scripts/dump_technique_segments.py \
  data/gcs_cache/videos/original \
  --labels-csv data/gcs_cache/labels_gcs_matched.csv \
  --sample-per-label 5 \
  --output-dir data/technique_segments/rule_v3

uv run python scripts/dump_technique_segments.py \
  data/gcs_cache/videos/original \
  --labels-csv data/gcs_cache/labels_gcs_matched.csv \
  --sample-per-label 5 \
  --flow-gate-model models/flow_qa_rf_v2.joblib \
  --output-dir data/technique_segments/rule_v3_flow_rf_v2
```

각 JSON에는 `segments[]`, `model_version`, 기술 분포, rule drop 수, flow gate demote 수,
`prob_dynamic`, sanity 결과가 포함됩니다. sanity는 다음 조건을 검사합니다.

- `technique`이 7개 라벨 중 하나인지
- `sequence_index`가 0부터 연속인지
- `start_time_ms < end_time_ms`이고 영상 길이를 넘지 않는지
- `confidence`가 `[0, 1]` 범위인지
- `dyno`/`coordination`만 `is_dynamic=true`인지

#### 2. Redis → Spring callback → DB 저장 확인

로컬 Spring/Redis/DB를 같은 환경변수로 맞춘 뒤, Spring이 만든 job을 그대로 쓰거나
직접 `XADD`로 주입합니다. 직접 주입 시 `videoId`는 Spring DB의 `videos`에 이미 존재해야 합니다.

```bash
redis-cli XADD analysis:requests '*' \
  videoId 123 \
  gcsPath videos/uploads/1/sample.mp4 \
  callbackUrl http://localhost:8080/api/analysis/videos/123
```

저장 확인은 DB CSV export 또는 Spring 조회 API 응답을 Phase 1 JSON과 비교합니다.
Spring 조회 응답은 camelCase이고, 워커/DB dump는 snake_case이므로 비교 스크립트가 두 형태를
정규화합니다.

```bash
# Spring GET 응답(JSON)과 비교
curl http://localhost:8080/api/videos/123/analysis \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -o data/technique_segments/spring-video-123.json

uv run python scripts/compare_analysis_segments.py \
  --expected-dump data/technique_segments/rule_v3_flow_rf_v2/sample.segments.json \
  --actual-json data/technique_segments/spring-video-123.json

# DB CSV export와 비교 (psql 예시)
psql "$DATABASE_URL" --csv \
  -c "select sequence_index,start_time_ms,end_time_ms,technique,is_dynamic,confidence,model_version from analysis_results where video_id = 123 order by sequence_index" \
  > data/technique_segments/db-video-123.csv

uv run python scripts/compare_analysis_segments.py \
  --expected-dump data/technique_segments/rule_v3_flow_rf_v2/sample.segments.json \
  --actual-csv data/technique_segments/db-video-123.csv
```

Spring의 재처리 정책은 append/upsert가 아니라 `deleteByVideoId(videoId)` 후 새 insert입니다.
따라서 같은 `videoId`로 재처리하면 이전 `analysis_results` 행은 지워지고 최신 결과만 남아야 합니다.

#### 3. 스팟체크 오버레이 생성

덤프 JSON과 원본 영상을 입력으로 라벨/confidence가 burn-in된 mp4를 생성합니다.

```bash
uv run python scripts/render_technique_overlay.py \
  --video data/gcs_cache/videos/original/sample.mp4 \
  --segments-json data/technique_segments/rule_v3_flow_rf_v2/sample.segments.json \
  --output data/technique_segments/overlays/sample.overlay.mp4
```

명백한 오분류가 반복되면 `app/services/vision/_thresholds.py`를 조정하고 같은 덤프를 재실행한 뒤
before/after JSON diff로 영향을 확인합니다. 임계값 변경을 MVP에 채택하면 `MODEL_VERSION`을
새 rule 버전으로 올리고 `_workspace/02_vision_technique_rules.md`와 함께 갱신합니다.

---

## 개발

### 디렉토리 구조

```
app/
├── api/                 # FastAPI 라우터 (현재 /health, /health/ready)
├── core/                # config(env), errors, logging
├── infra/               # redis_bus, gcs (외부 인프라 어댑터)
├── models/              # Pydantic 모델: stream, progress, callback, response
├── services/
│   ├── callback/        # Spring 콜백 HTTP 클라이언트 (tenacity)
│   ├── pipeline/        # frames(OpenCV), orchestrator (1 job 처리)
│   └── vision/          # MediaPipe pose + segmenter + classifier
├── workers/             # stream_consumer (장기 실행 XREADGROUP 루프)
└── main.py              # FastAPI + lifespan (consumer task spawn)

_workspace/              # 에이전트 산출물 (git ignored)
scripts/                 # 개발 보조 스크립트
```

### 자주 쓰는 명령

```bash
uv sync                              # 의존성 설치
uv run uvicorn app.main:app --reload # 로컬 실행
uv run ruff check app                # lint
uv run ruff format app               # format
uv run mypy app                      # type check
uv run pytest                        # 테스트
uv run pytest --cov=app              # 커버리지
```

### 학습 기반 dynamic/static 분류기

휴리스틱 기술 분류와 별도로, MediaPipe Pose 시퀀스를 학습하는 GRU 기반 영상 단위
`dynamic`/`static` 2진 분류기를 만들 수 있습니다. 학습용 의존성은 `ml` 그룹에 분리되어
있습니다.

```bash
uv sync --group ml

mkdir -p models/mediapipe
curl -L -o models/mediapipe/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

uv run python scripts/build_pose_dataset.py \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
  --videos /Users/minjoun/Movies/Original \
  --out data/pose_dataset \
  --target-frames 128

uv run python scripts/train_pose_sequence.py \
  --data data/pose_dataset \
  --out models/pose_dynamic_static.pt \
  --epochs 20
```

`data/pose_dataset/`와 `models/`는 학습 산출물이므로 git에 포함하지 않습니다. 현재 워커의
기본 분석 결과는 기존 휴리스틱 파이프라인을 유지하며, 학습 모델은 평가 후 optional로 연결합니다.
운영 Docker 이미지는 `mediapipe` Tasks 런타임에 필요한
`models/mediapipe/pose_landmarker_lite.task`를 build 단계에서 checksum으로 고정 다운로드합니다.

성능 진단과 비교 실험은 같은 cache에서 바로 실행할 수 있습니다.

```bash
uv run python scripts/train_pose_sequence.py \
  --data data/pose_dataset \
  --out models/pose_dynamic_static_raw_kfold.pt \
  --epochs 20 \
  --feature-set raw \
  --folds 5 \
  --min-raw-pose-frames 30 \
  --run-name pose_dynamic_static_raw_kfold

uv run python scripts/train_pose_sequence.py \
  --data data/pose_dataset \
  --out models/pose_dynamic_static_motion_kfold.pt \
  --epochs 20 \
  --feature-set motion \
  --folds 5 \
  --min-raw-pose-frames 30 \
  --run-name pose_dynamic_static_motion_kfold
```

각 실행은 `models/reports/*_predictions.csv`와 `models/reports/*_metrics.json`을 생성합니다.

모델 개선용 QA 리뷰 큐는 raw/motion k-fold 예측 리포트를 비교해서 생성합니다.

```bash
uv run python scripts/build_dynamic_static_review_queue.py \
  --raw-predictions models/reports/pose_dynamic_static_raw_kfold_predictions.csv \
  --motion-predictions models/reports/pose_dynamic_static_motion_kfold_predictions.csv \
  --data-dir data/pose_dataset \
  --videos-dir /Users/minjoun/Movies/Original \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
  --out data/review/dynamic_static_review_queue.csv \
  --known-failure IMG_8942:no_pose_detected \
  --contact-sheets-dir data/review/contact_sheets
```

CSV의 우선순위는 `P0` 포즈 추출 실패/저프레임, `P1` raw/motion 공통 오분류,
`P2` raw 고확신 오분류, `P4` 정상 샘플입니다. 사람이 확인한 뒤에는
`suggested_status`, `new_label`, `reason`, `notes`를 채우고 라벨을 정리한 다음 dataset을
다시 빌드해 재학습합니다.

완료된 QA CSV는 `_complete` 파일로 저장한 뒤 라벨 CSV와 pose cache에 반영합니다.

```bash
uv run python scripts/apply_dynamic_static_review.py \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
  --review data/review/dynamic_static_review_queue_complete.csv \
  --labels-out data/review/labels_완료_qa.csv \
  --cache-in data/pose_dataset \
  --cache-out data/pose_dataset_reviewed

uv run python scripts/train_pose_sequence.py \
  --data data/pose_dataset_reviewed \
  --out models/pose_dynamic_static_raw_qa_kfold.pt \
  --epochs 20 \
  --feature-set raw \
  --folds 5 \
  --min-raw-pose-frames 30 \
  --run-name pose_dynamic_static_raw_qa_kfold
```

Pose GRU와 별도로 tabular pose, optical flow, fusion baseline도 비교할 수 있습니다.

GCS에 있는 원본 영상으로 flow dataset을 만들 때는 먼저 라벨 CSV와 GCS object를 stem 기준으로
매칭합니다. 2026-06-10 확인 기준 버킷에는 `videos/Original/` prefix 아래 영상 452개가 있고,
기존 라벨 파일 425행은 모두 매칭됩니다 (`static=218`, `dynamic=207`). 로컬 캐시는
`data/gcs_cache/` 아래에 생성되며 git에 포함하지 않습니다.

```bash
uv run python scripts/cache_gcs_labeled_videos.py \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
  --bucket hola-climbing-log-videos \
  --prefix videos/Original/ \
  --cache-dir data/gcs_cache/videos/original \
  --matched-labels-out data/gcs_cache/labels_gcs_matched.csv \
  --manifest-out data/gcs_cache/gcs_original_manifest.csv \
  --backend gsutil

# 전체 영상 캐시가 필요할 때만 실행. 다운로드 용량이 크므로 먼저 --limit 2 등으로 smoke 권장.
uv run python scripts/cache_gcs_labeled_videos.py \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
  --bucket hola-climbing-log-videos \
  --prefix videos/Original/ \
  --cache-dir data/gcs_cache/videos/original \
  --matched-labels-out data/gcs_cache/labels_gcs_matched.csv \
  --manifest-out data/gcs_cache/gcs_original_manifest.csv \
  --backend gsutil \
  --download
```

```bash
uv run python scripts/build_pose_tabular_dataset.py \
  --labels data/review/labels_완료_qa.csv \
  --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
  --out data/tabular_dataset/qa_normalized \
  --variant normalized

uv run python scripts/build_flow_dataset.py \
  --labels data/gcs_cache/labels_gcs_matched.csv \
  --videos-dir data/gcs_cache/videos/original \
  --out data/flow_dataset/gcs_flow_v1

uv run python scripts/build_fusion_dataset.py \
  --left data/tabular_dataset/qa_normalized \
  --right data/flow_dataset/gcs_flow_v1 \
  --out data/fusion_dataset/qa_normalized_flow

uv run python scripts/train_tabular_dynamic_static.py \
  --data data/flow_dataset/gcs_flow_v1 \
  --out models/flow_gcs_rf_v1.joblib \
  --run-name flow_gcs_v1 \
  --splits holdout,kfold,group-kfold

uv run python scripts/build_flow_miss_review_queue.py \
  --predictions models/reports/flow_gcs_v1_predictions.csv \
  --out data/review/flow_gcs_v1_miss_review_queue.csv \
  --model rf \
  --split group-kfold

uv run python scripts/propagate_review_decisions.py \
  --target data/review/flow_gcs_v1_miss_review_queue.csv \
  --source data/review/flow_miss_review_queue_normalized.csv \
  --source data/review/dynamic_static_review_queue_complete.csv \
  --out data/review/flow_gcs_v1_miss_review_queue.csv \
  --summary-out data/review/flow_gcs_v1_previous_review_applied.csv
```

2026-06-10 기준 운영 artifact는 `flow_qa_rf_v2.joblib`입니다. QA 라벨 205개에서 v2 RF는
group-kfold balanced accuracy `0.8381`, dynamic recall `0.8147`이고, v3(동적 낙하 trimming +
burst-aware 46-dim feature)는 `0.8384`, dynamic recall `0.8247`로 보합입니다. v3는 회복 6건 /
악화 6건이라 운영 기본값은 v2로 유지하고, v3 산출물은 다음 ROI flow/라벨 확장 실험의 입력으로
둡니다.

GCS 원본 425개 전체(`static=218`, `dynamic=207`)로 만든 `flow_gcs_v1`은 더 다양한/덜 정제된
분포를 반영합니다. group-kfold 기준 RF는 balanced accuracy `0.8114`, dynamic recall `0.8159`,
logreg는 balanced accuracy `0.8176`, dynamic recall `0.7729`입니다. 샘플 수는 늘었지만 운영 v2보다
낮으므로 artifact 승격은 보류하고, `data/review/flow_gcs_*_miss_review_queue.csv`에서 고신뢰
오분류를 먼저 라벨 리뷰합니다.

RF 오분류 리뷰 80건을 반영한 `flow_gcs_reviewed_v1`은 419개 샘플(`static=212`, `dynamic=207`)이며
group-kfold 기준 RF balanced accuracy `0.8378`, dynamic recall `0.8359`, static specificity
`0.8398`입니다. 운영 v2와 balanced accuracy는 거의 같지만 static specificity가 낮아 기본 artifact
승격은 보류합니다. reviewed 모델의 남은 새 오분류는
`data/review/flow_gcs_reviewed_v1_miss_review_queue.csv`의 `review` 8건입니다.

추가 리뷰 8건 중 2건 라벨 수정(`IMG_6755`, `IMG_6726`)을 반영한 `flow_gcs_reviewed_round2_v1`은
419개 샘플(`static=210`, `dynamic=209`)이며 group-kfold 기준 RF balanced accuracy `0.8449`,
dynamic recall `0.8469`, static specificity `0.8429`입니다. 운영 v2(`0.8381`)보다 balanced
accuracy와 dynamic recall은 높지만 평가셋이 다르고 static specificity는 운영 v2(`0.8615`)가 더 높으므로
기본 artifact 교체 전에는 static FP 비용을 함께 봐야 합니다. round2 오분류 큐의 남은 `review` 1건
(`IMG_6881`)은 기존 dynamic 라벨 유지로 확인됐고, `flow_gcs_reviewed_round3_v1`은 round2와 동일한
성능입니다. round3 오분류 큐는 65건 모두 기존 리뷰 결정으로 처리되어 남은 수동 리뷰가 없습니다.

방향 분해 v4(`flow_gcs_v4_direction`, feature_dim 58)는 Farneback magnitude에 `vy` 시계열 통계를
추가한 실험입니다. 동일 419개/group-kfold 기준 RF balanced accuracy `0.8377`, dynamic recall
`0.8517`, static specificity `0.8238`로 round3 RF `0.8449`보다 낮았습니다. round3 miss 65건 중
5건은 회복했지만 새 오답 8건이 생겼고, 고확신 miss 20건은 0건 회복했습니다. 따라서 ROI flow Task로
바로 확장하지 않고 고확신 FN 12건 + 정분류 dynamic 30건만 ROI 진단 probe를 먼저 실행했습니다.
probe 결과는 `models/reports/roi_flow_probe_round3_summary.json`에 있으며, pose 미검출 1건을 제외한
41건에서 최대 효과크기 `1.4914`(`roi_mag_p95`)가 나왔습니다. 기준 `1.0`을 넘었으므로 full ROI
flow v5 후보로 보였지만, 후속 static gate에서 결론이 뒤집혔습니다. static gate는 기존 41건을
재사용하고 정분류 static 30건 + 고확신 FP static 8건을 추가 처리해
`models/reports/roi_flow_probe_round3_static_gate_summary.json`에 저장했습니다. FN vs correct_static
최대 효과크기는 `0.9725`, FN vs static_pool은 `0.8261`로 기준 `1.0` 미만이므로 Task 3 full ROI는
보류하고 pretrained video encoder probe를 다음 후보로 둡니다.

pretrained video encoder probe도 2026-06-10에 실행했습니다. `torchvision r3d_18` Kinetics-400
frozen embedding은 single clip best `0.5777`, 4-clip mean+std best `0.6301`로 round3 flow RF
`0.8449`보다 크게 낮았습니다. 4-clip logreg가 고확신 miss 20건 중 10건을 회복했지만 새 오답
137건을 만들었고, flow+encoder fusion RF도 balanced accuracy `0.8065`, static specificity
`0.7667`로 하락했습니다. 따라서 encoder artifact도 승격하지 않고 운영 기본값은 계속
`models/flow_qa_rf_v2.joblib`입니다.

2026-06-11에는 1차 encoder probe의 약점(약한 encoder, uniform sampling, 와이드샷 미크롭)을
줄인 v2 probe를 실행했습니다. `r3d_18` + burst + person crop은 best `0.6940`, VideoMAE
(`MCG-NJU/videomae-base-finetuned-kinetics`) + burst + crop은 best `0.6849`, DINOv2
(`facebook/dinov2-base`, 8프레임) + burst + crop은 best `0.6135`였습니다. 세 구성 모두
사전 기각 기준인 `0.75` 미만이고, 고확신 miss는 9~11건 회복했지만 새 오답이 103~130건으로
훨씬 많았습니다. 따라서 frozen pretrained embedding 트랙은 MVP 경로에서 종료하고,
정확도 보완은 fine-tuning 또는 제품 측 유보/재촬영 UX로 분리합니다.

### 추후 작업

- [ ] `.github/workflows/ci.yml` — uv sync + ruff + mypy + pytest
- [ ] Prometheus `/metrics` 엔드포인트 (deps에 `prometheus-client` 있음)
- [ ] DLQ 재처리 도구 (`scripts/replay_dlq.py`)

---

## 트러블슈팅

### MediaPipe가 Apple Silicon에서 import 실패

`mediapipe` 0.10.x는 `manylinux2014_x86_64` wheel만 제공합니다. 다음 중 하나로 해결:

1. **Docker로 실행** (권장) — `docker-compose.yml`이 `platform: linux/amd64`를 명시하여
   Rosetta로 동작. 사전에 `softwareupdate --install-rosetta` 실행 필요.
2. **conda-forge mediapipe** — `conda install -c conda-forge mediapipe` (네이티브 ARM64 wheel 제공).
3. **AMD64 가상머신 / 빌드 머신** — CI나 운영은 Linux x86_64에서 직접 빌드.

### GCS 다운로드가 `403 Permission denied`

ADC가 잡히지 않은 경우입니다.

```bash
gcloud auth application-default login
# 또는 SA 키 사용
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/keys/gcs-sa.json
```

compose는 `~/.config/gcloud`를 자동 마운트하므로 호스트에서 로그인 후 컨테이너 재기동.

### Redis 연결이 안 됨

1. Spring 서버와 **같은 Redis 인스턴스**를 가리키는지 확인.
2. `REDIS_PASSWORD`가 Spring 측과 정확히 일치하는지 확인 (compose는 `${REDIS_PASSWORD:-changeme}` 기본값).
3. compose 환경에서는 `REDIS_HOST=redis` (서비스명), 로컬 실행에서는 `localhost`.

### `XGROUP CREATE` 권한 오류

Redis 5.0 이상이 필요합니다 (`XGROUP MKSTREAM` 지원). 7-alpine 사용 권장.

### 워커가 메시지를 받지 못함

```bash
# 스트림이 비어있는지 확인
redis-cli -a "$REDIS_PASSWORD" XLEN analysis:requests
# 그룹이 만들어졌는지 확인
redis-cli -a "$REDIS_PASSWORD" XINFO GROUPS analysis:requests
# Pending Entries List 확인 (재처리 필요한 메시지)
redis-cli -a "$REDIS_PASSWORD" XPENDING analysis:requests hola-ai-worker
```

Spring이 `XADD`를 호출하지 않고 있을 수도 있습니다 — Spring 측 `AnalysisDispatcher` 로그
확인.

### 콜백이 5xx로 계속 실패

- Spring `AnalysisServiceImpl.ingestResult` 로그를 확인 (`videos.id`가 DB에 있는지).
- `videoId`가 DB에 없으면 `V001` (404) → 재시도 무의미, dead-letter로 즉시 이동.

---

## 변경 이력

| 날짜 | 변경 | 작성자 |
|------|------|--------|
| 2026-05-28 | 부트스트랩 — FastAPI/MediaPipe/Redis Streams 골격, Spring 계약 정합 검증, Docker 패키징 | architect / vision / pipeline / integration 에이전트 |

---

## 라이선스

Proprietary — Hola Climbing 팀 내부 프로젝트.
