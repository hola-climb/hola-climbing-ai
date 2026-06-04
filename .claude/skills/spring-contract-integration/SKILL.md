---
name: spring-contract-integration
description: "Spring 서버(hola-climbing-server)와 AI 워커 간의 계약 정합성을 검증한다. ApiResponse 래퍼, ErrorCode enum, snake_case JSON, Redis Streams key, 환경변수가 양쪽에서 일치하는지 확인하고 Dockerfile·docker-compose·README를 작성한다. 'Spring 계약 검증', '경계면 확인', 'Dockerfile 작성', 'README 작성' 요청 시 반드시 사용."
---

# Spring Contract Integration

워커와 Spring 서버의 **경계면 정합성 검증 + 배포 구성** 스킬. integration-engineer 전용.

## 언제 사용하는가

- Spring `domain/analysis/` 변경 후 워커가 따라가야 할 때
- 워커 응답이 Spring DTO와 일치하는지 검증
- Dockerfile, docker-compose, .env.example 작성
- README 초안 작성 (server README와 동일 톤)

## 핵심 원칙

1. **두 저장소를 항상 같이 본다.** `hola-climbing-server`와 `hola-climbing-ai`를 동시에 grep.
2. **추정 금지.** 의심되면 실제 코드를 확인. Spring DTO 필드명, Jackson 설정, ErrorCode enum 값.
3. **Spring이 SSOT.** 불일치 발견 시 기본은 **워커를 수정**한다. Spring 버그로 판단되면 사용자에게 보고.
4. **README는 server와 동일 톤.** 구조·섹션·말투가 일관되어야 한다.

## Spring contract 확정값 (2026-05-27 스냅샷)

> 아래는 실제 hola-climbing-server 코드에서 확인된 값이다. 변경 의심 시 재추출.

| 영역 | 값 | 출처 |
|------|-----|------|
| 디스패치 방식 | Redis Streams `XADD` | `infrastructure/ai/RedisStreamAnalysisJobQueue.java` |
| Stream key | `analysis:requests` | 위 파일 line 21 |
| Pub/Sub channel | `analysis:progress` | `RedisAnalysisProgressBus.java` line 18 |
| 콜백 path | `POST /api/analysis/videos/{videoId}` | `AnalysisController.java` line 52~ |
| 콜백 URL 전달 | 메시지 페이로드 `callbackUrl` 필드에 절대 URL | `AnalysisDispatcher.java` |
| 메시지 페이로드 키 | `videoId` (String), `gcsPath`, `callbackUrl` | `AnalysisJob` record |
| 콜백 body | `AnalysisIngestRequest { status, model_version, segments[] }` | record class |
| segment 필드 | `sequence_index, start_time_ms, end_time_ms, technique, is_dynamic, confidence` | `AnalysisSegmentPayload` |
| ApiResponse | `{is_success, code, message, data, timestamp}` (`is_success` bool, **`status` 아님**) | `common/response/ApiResponse.java` |
| 성공 code | `"OK"` | 위 파일 line 15 |
| 분석 ErrorCode | `V005 ANALYSIS_FAILED` (500), `S002 AI_SERVER_UNAVAILABLE` (503) | `common/exception/error/ErrorCode.java` |
| Stage enum | `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` | `AnalysisStage.java` |
| SSE endpoint | `GET /api/videos/{videoId}/analysis/stream`, 이벤트명 `progress`, timeout 10분 | `VideoAnalysisSseService.java` |
| 환경변수 | `AI_ANALYSIS_URL` (URL, 비우면 비활성), `APP_BASE_URL` (콜백 URL 생성 기준) | `application.yaml` |

워커 시작 시 본 표가 여전히 유효한지 재검증:

```bash
# 핵심 stream key 변경 감지
grep -r "analysis:requests\|analysis:progress" \
  /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/
```

스냅샷을 `_workspace/00_input/spring-server-snapshot.md` 에 저장. 추가 추출 대상:

### 1. 컨트롤러 시그니처

```bash
# 예: AnalysisController 메서드 시그니처 추출
grep -A 5 "@PostMapping\|@GetMapping\|@PutMapping" \
  /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/com/holaclimbing/server/domain/analysis/*.java
```

확인할 정보:
- 워커가 호출받는 엔드포인트 (있다면)
- 워커가 콜백할 엔드포인트
- 요청/응답 DTO 클래스명

### 2. DTO 필드명 (snake_case 검증)

Spring Jackson은 `application.yml`에서 `property-naming-strategy: SNAKE_CASE` 설정 시 자동 변환.
워커 Pydantic도 동일하게 snake_case alias_generator 사용.

각 DTO에 대해 필드명 표를 만든다:

| Spring DTO | Java 필드 | JSON 키 | 워커 Pydantic | 일치? |
|------------|----------|---------|----------------|-------|
| `AnalysisJob` (stream msg) | `videoId` (String) | `videoId` (raw) | `int(fields[b"videoId"])` | ✅ |
| `AnalysisJob` | `gcsPath` | `gcsPath` | `gcs_path: str` | ✅ |
| `AnalysisJob` | `callbackUrl` | `callbackUrl` | `callback_url: str` | ✅ |
| `AnalysisIngestRequest` | `status` | `status` | `status: Literal["done","failed"]` | ✅ |
| `AnalysisIngestRequest` | `modelVersion` | `model_version` | `model_version: str` | ✅ |
| `AnalysisIngestRequest` | `segments` | `segments` | `segments: list[Segment]` | ✅ |
| `AnalysisSegmentPayload` | `sequenceIndex` | `sequence_index` | `sequence_index: int` | ✅ |
| `AnalysisSegmentPayload` | `startTimeMs` | `start_time_ms` | `start_time_ms: int \| None` | ✅ |
| `AnalysisSegmentPayload` | `endTimeMs` | `end_time_ms` | `end_time_ms: int \| None` | ✅ |
| `AnalysisSegmentPayload` | `technique` | `technique` | `technique: str` | ✅ |
| `AnalysisSegmentPayload` | `isDynamic` | `is_dynamic` | `is_dynamic: bool \| None` | ✅ |
| `AnalysisSegmentPayload` | `confidence` | `confidence` | `confidence: float \| None` | ✅ |
| `ApiResponse` | `isSuccess` (bool) | `is_success` | `is_success: bool` | ✅ |

**Stream payload 주의:** Spring이 `videoId`를 **String**으로 XADD함 (`String.valueOf(...)`). 워커는 `int(...)`로 파싱. XADD 필드명은 snake_case 변환 없는 **raw camelCase**.

### 3. ErrorCode enum (확정)

Spring의 분석 관련 ErrorCode는 **단 2개**. 접두사는 `ANALYSIS_`가 아니라 도메인 글자 + 번호:
- `V005` (`ANALYSIS_FAILED`) — HTTP 500
- `S002` (`AI_SERVER_UNAVAILABLE`) — HTTP 503

워커 정책: 콜백 body에는 워커 자체 코드 안 넣음. `status: "done"|"failed"`로만 통신. 내부 로깅용 분류 (`AnalysisFailureReason`)는 워커 전용.

워커가 Spring 응답에서 `S002`를 받으면 재시도 무의미 → 즉시 dead-letter.

### 4. Redis 설정 (확정)

| 키 | 값 | 비고 |
|----|-----|------|
| Stream key | `analysis:requests` | Spring `XADD` → 워커 `XREADGROUP` |
| Stream 메시지 필드 | `videoId`(String), `gcsPath`, `callbackUrl` | snake_case 변환 없음 (raw XADD field) |
| Consumer group | (Spring은 강제 안 함) | 워커가 `hola-ai-worker`로 정의 |
| Progress channel | `analysis:progress` | Pub/Sub, 단일 채널 |
| Progress 페이로드 | `AnalysisProgressEvent` 형식 | Java 클래스 확인 후 일치 |

`application.yaml` Redis 부분 (Spring 기본값):
- host: `localhost`, port: `6379`, password: `${REDIS_PASSWORD}`
- timeout: 3s, lettuce pool max-active: 8

### 5. 환경변수

```bash
grep -A 2 "@Value\|@ConfigurationProperties" \
  /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/resources/application*.yml
```

워커 `.env.example`에 동일한 값으로 매핑 (특히 GCS bucket, Redis host/port).

## 정합성 검증 표

`_workspace/04_integration_contract_match.md` 에 다음 표를 채운다:

```markdown
| 영역 | Spring 값 | 워커 값 | 일치 여부 | 액션 |
|------|----------|--------|----------|------|
| Stream key | `hola:analysis:jobs` | `hola:analysis:jobs` | ✅ | - |
| Consumer group | `analysis-workers` | `hola-ai-worker` | ❌ | 워커 수정 |
| Progress channel | `hola:analysis:progress:{jobId}` | `hola:analysis:progress` | ❌ | 워커 수정 |
| AnalysisRequest.videoUrl | `video_url` (snake) | `video_url` | ✅ | - |
| ErrorCode.ANALYSIS_TIMEOUT | 존재 | 누락 | ❌ | 워커 추가 |
```

## Docker 구성

### Dockerfile

```dockerfile
FROM python:3.11-slim AS base

# 시스템 의존성: OpenCV, mediapipe에 필요
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 레이어 분리 (캐시 효율)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> Apple Silicon에서 빌드 시 `--platform linux/amd64` 사용. arm64 네이티브 이미지가 가능하면 그쪽이 빠름.

### docker-compose.yml (개발용)

Spring server의 Redis와 같은 인스턴스 사용. server `docker-compose` 또는 `docker run` 명령과 충돌 없도록 네트워크 공유.

```yaml
services:
  worker:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./keys:/app/keys:ro      # GCS service account
    depends_on:
      - redis
  redis:
    image: redis:7
    ports: ["6379:6379"]
```

> server가 이미 redis 컨테이너를 운영 중이면 worker만 띄우고 redis 서비스 제거.

## README 작성

`hola-climbing-server/README.md`의 구조를 그대로 따른다:

```markdown
# Hola AI Worker

> AI 동작 분석 파이프라인 (Python)
> SSAFY 자율 프로젝트 · 2026.05.15 ~ 2026.06.25

## 프로젝트 소개
... (server README와 일관된 톤)

## 기술 스택
| 구분 | 기술 |
|------|------|
| Language / Runtime | Python 3.11+ |
| Framework | FastAPI |
| ML | MediaPipe Pose, OpenCV |
| Queue | Redis Streams (Spring과 공유) |
| Storage | Google Cloud Storage |
| Test | pytest, testcontainers-python |

## 실행 방법
### 사전 요구사항
### 1. 인프라
### 2. 환경 변수
### 3. 워커 실행
### 4. 테스트

## 전체 기능 설명
- 영상 분석 dispatch (HTTP 또는 Streams)
- 진행률 발행 (Pub/Sub)
- 결과 콜백
- 정확도 측정 (라벨링 데이터)

## 프로젝트 구조
```
```

## 산출물 체크리스트

- [ ] `_workspace/00_input/spring-server-snapshot.md` — 가장 먼저 작성
- [ ] `_workspace/04_integration_contract_match.md` — 정합성 매핑 표
- [ ] `_workspace/04_integration_dockerfile.md` — Docker 설계 근거
- [ ] `_workspace/04_integration_readme.md` — README 초안
- [ ] 실제 파일: `Dockerfile`, `docker-compose.yml`, `.env.example`, `README.md`
- [ ] 불일치 발견 시 책임 에이전트에게 SendMessage
