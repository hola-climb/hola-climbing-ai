---
name: fastapi-worker-design
description: "Hola AI 워커의 FastAPI 구조·인터페이스 계약·의존성·환경변수를 설계한다. Spring 서버(hola-climbing-server)의 분석 도메인이 기대하는 요청/응답/콜백 shape을 단일 진실 원천(SSOT)으로 삼는다. '워커 구조 설계', 'API 계약 정리', '의존성 정리', '디렉토리 구조', '환경변수 설계' 요청 시 반드시 사용."
---

# FastAPI Worker Design

Hola AI 워커의 **외부 인터페이스 + 내부 모듈 구조**를 결정하는 스킬. architect 에이전트의 주력 도구.

## 언제 사용하는가

- 워커 프로젝트 초기 부트스트랩 (디렉토리/의존성/환경변수 일괄 결정)
- Spring 서버 분석 도메인 변경에 따른 워커 인터페이스 재설계
- 의존성 버전 핀 (Python 3.11, FastAPI, MediaPipe, OpenCV 등)
- 신규 엔드포인트 추가 시 계약 매핑

## 핵심 원칙

1. **Spring contract가 SSOT.** 워커의 어떤 인터페이스도 Spring 측 계약과 충돌하면 안 된다. 충돌 발견 시 워커를 수정한다.
2. **snake_case JSON 강제.** Pydantic v2 모델은 alias_generator로 snake_case 직렬화.
3. **식별자는 `int`.** Spring의 BIGSERIAL = Python `int`. `str(id)` 변환 금지.
4. **결정론적.** 같은 입력 → 같은 출력. AI 결과도 시드 고정.
5. **README는 단일 진실 원천 위치.** 워커도 server와 동일하게 README가 모든 운영 진실을 담는다.

## 워크플로우

### Step 1: Spring contract (확정값, 2026-05-27 추출)

> ⚠️ 아래는 실제 hola-climbing-server 코드에서 추출한 확정값이다. 코드 변경이 의심되면 재추출:
> `/Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/com/holaclimbing/server/`

**디스패치:** Redis Streams `XADD` (HTTP 아님). 워커는 `XREADGROUP`으로 소비.

**작업 메시지 (Stream `analysis:requests`, payload):**
```python
{
    "videoId": str,          # Long → String 변환됨, 워커는 int(value)로 파싱
    "gcsPath": str,          # GCS 객체 경로
    "callbackUrl": str,      # 절대 URL. 예: http://localhost:8080/api/analysis/videos/123
}
```

**콜백 (Spring 수신):** `POST {callbackUrl}` — 메시지의 `callbackUrl`을 그대로 사용. 워커가 직접 path를 구성하지 않는다.

**콜백 body (`AnalysisIngestRequest`):**
```python
{
    "status": "done" | "failed",
    "model_version": str,    # 예: "rule_v1", "lstm_v1"
    "segments": [
        {
            "sequence_index": int,
            "start_time_ms": int | None,
            "end_time_ms": int | None,
            "technique": str,           # vision-engineer의 6개 라벨 중 하나
            "is_dynamic": bool | None,
            "confidence": float | None,
        },
        ...
    ]
}
```

**진행률 발행:** Redis **Pub/Sub** 채널 `analysis:progress`. 페이로드는 Spring `AnalysisProgressEvent` shape (실제 클래스 확인 필요). Stage enum: `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`.

**ErrorCode (워커가 알아야 할 두 가지):**
- `V005` (`ANALYSIS_FAILED`) — HTTP 500, "영상 분석에 실패했습니다."
- `S002` (`AI_SERVER_UNAVAILABLE`) — HTTP 503, "AI 분석 서버에 연결할 수 없습니다."

워커는 콜백 실패 시 Spring이 S002로 응답할 수 있음을 처리. 콜백 body에는 워커 자체 에러 코드를 넣지 않는다 (status="failed" + segments=[]로 보고).

### Step 2: 디렉토리 구조 결정

권장 구조 (FastAPI 관용 + Hola 도메인 반영):

```
hola-climbing-ai/
├── app/
│   ├── main.py                 # FastAPI 진입점
│   ├── api/
│   │   ├── __init__.py
│   │   ├── analysis.py         # /analyze 엔드포인트 (HTTP 모드일 때)
│   │   └── health.py           # /health
│   ├── core/
│   │   ├── config.py           # Pydantic Settings
│   │   ├── logging.py
│   │   └── errors.py           # ErrorCode + AnalysisException
│   ├── models/                 # Pydantic v2 schemas
│   │   ├── request.py
│   │   ├── response.py         # ApiResponse, PageResponse 호환
│   │   └── analysis.py         # 분석 결과 스키마
│   ├── services/
│   │   ├── vision/             # vision-engineer 영역
│   │   ├── pipeline/           # pipeline-engineer 영역
│   │   └── callback/           # 결과 콜백 발행
│   ├── workers/
│   │   └── stream_consumer.py  # Redis Streams 컨슈머 (장기 실행)
│   └── infra/
│       ├── gcs.py
│       └── redis_bus.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── pyproject.toml
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

근거:
- Spring과 동일한 `domain/` 패턴은 FastAPI에서 부자연스럽다. 대신 layer(`api/services/models/infra`) + 도메인은 layer 안에 폴더로.
- `workers/`는 장기 실행 프로세스 (Redis Streams 컨슈머). `api/`는 HTTP 단발 요청.
- 동일 워커가 HTTP 모드와 Streams 모드를 모두 지원해야 함 (Spring `AI_ANALYSIS_URL` 비어있으면 비활성, 채워져 있으면 HTTP, 추후 Streams 전환 가능).

### Step 3: 의존성 핀

`pyproject.toml` 기준 권장 핀:

```toml
[project]
name = "hola-climbing-ai"
requires-python = ">=3.11,<3.13"
dependencies = [
    "fastapi>=0.115,<0.120",
    "uvicorn[standard]>=0.32,<0.40",
    "pydantic>=2.9,<3",
    "pydantic-settings>=2.6",
    "mediapipe>=0.10.18",       # Apple Silicon 호환은 0.10.14+
    "opencv-python-headless>=4.10",
    "numpy>=1.26,<2.2",         # mediapipe 호환 제약
    "redis[hiredis]>=5.2",
    "google-cloud-storage>=2.18",
    "httpx>=0.28",              # 콜백 클라이언트
    "structlog>=24.4",
    "tenacity>=9.0",            # 재시도 데코레이터
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "testcontainers[redis]>=4.8",
    "ruff>=0.7",
    "mypy>=1.13",
]
```

핀 사유 (각 줄에 주석으로 남기지 말고 `_workspace/01_architect_deps.md`에 따로 정리):
- `numpy<2.2`: mediapipe 0.10.x는 numpy 2.x와 부분적 비호환 → 보수적 핀
- `opencv-python-headless`: 워커는 GUI 불필요, headless로 이미지 50% 절약
- `mediapipe>=0.10.18`: Apple Silicon 네이티브 휠 제공

### Step 4: 환경변수 계약

`.env.example`:

```dotenv
# Server
WORKER_HOST=0.0.0.0
WORKER_PORT=8000
LOG_LEVEL=INFO

# Spring 콜백
SPRING_CALLBACK_BASE_URL=http://localhost:8080
CALLBACK_TIMEOUT_SECONDS=10
CALLBACK_MAX_RETRIES=3

# GCS
GCS_BUCKET=hola-climbing-log-videos
GOOGLE_APPLICATION_CREDENTIALS=./keys/gcs-sa.json

# Redis (Spring과 동일 인스턴스)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=                              # Spring과 동일. 비어있으면 인증 생략
REDIS_STREAM_KEY=analysis:requests           # ✅ Spring 확정값
REDIS_CONSUMER_GROUP=hola-ai-worker          # 워커 측에서 정의 (Spring은 group 강제 안 함)
REDIS_CONSUMER_NAME=worker-1
REDIS_PROGRESS_CHANNEL=analysis:progress     # ✅ Spring Pub/Sub 채널 (단일)

# MediaPipe
MP_MODEL_COMPLEXITY=1            # 0: light, 1: full, 2: heavy
MP_MIN_DETECTION_CONFIDENCE=0.5
```

규칙:
- Spring과 공유하는 Redis 호스트/포트는 server `.env`와 동일.
- `REDIS_STREAM_KEY`, `REDIS_CONSUMER_GROUP`은 server `RedisStreamConfig` 코드와 정확히 일치 (integration-engineer 검증).
- `GOOGLE_APPLICATION_CREDENTIALS` 파일은 git ignore.

### Step 5: 응답 모델 (Spring ApiResponse 확정값)

Spring `ApiResponse` 실제 필드 (확정):
- `isSuccess: boolean` → JSON `is_success` (boolean, **`status` 문자열 아님**)
- `code: String` (성공: `"OK"`, 오류: ErrorCode.code 값)
- `message: String` (null 제외)
- `data: T` (null 제외)
- `timestamp: Instant` (ISO-8601)

```python
# app/models/response.py
from typing import Generic, TypeVar
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    model_config = ConfigDict(populate_by_name=True)
    is_success: bool
    code: str = "OK"
    message: str | None = None
    data: T | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def ok(cls, data: T | None = None) -> "ApiResponse[T]":
        return cls(is_success=True, code="OK", data=data)

    @classmethod
    def error(cls, code: str, message: str) -> "ApiResponse[None]":
        return cls(is_success=False, code=code, message=message, data=None)
```

> 워커가 직접 ApiResponse를 응답할 일은 거의 없음 (콜백 body는 `AnalysisIngestRequest` 형식). 본 모델은 워커 자체 endpoint(`/health` 등)에서 사용.

### Step 6: ErrorCode (Spring과 호환되는 두 값 + 워커 내부 분류)

Spring `ErrorCode`에는 분석 관련 enum이 두 개뿐이다 (`ANALYSIS_` 접두사 없음):
- `V005` (`ANALYSIS_FAILED`) — 500, "영상 분석에 실패했습니다."
- `S002` (`AI_SERVER_UNAVAILABLE`) — 503, "AI 분석 서버에 연결할 수 없습니다."

워커는 콜백 body에 ErrorCode를 직접 넣지 않는다 (`status="failed"` + `message`로만 보고). 내부 로깅용 분류만 유지:

```python
# app/core/errors.py
from enum import Enum

class AnalysisFailureReason(str, Enum):
    """워커 내부 로깅/메트릭용. Spring 콜백 body에는 들어가지 않음."""
    VIDEO_DOWNLOAD = "video_download"
    VIDEO_DECODE = "video_decode"
    POSE_NOT_DETECTED = "pose_not_detected"
    CALLBACK_FAILED = "callback_failed"
    INTERNAL = "internal"

class AnalysisException(Exception):
    def __init__(self, reason: AnalysisFailureReason, message: str):
        self.reason = reason
        self.message = message

# Spring 측 코드 (워커가 Spring 응답에서 받을 수 있는 값)
SPRING_ANALYSIS_FAILED = "V005"
SPRING_AI_SERVER_UNAVAILABLE = "S002"
```

콜백 실패 시 워커 동작:
- HTTP 5xx (Spring 일시 다운) → 지수 백오프 재시도 (pipeline-engineer 영역)
- HTTP 4xx (계약 불일치) → 즉시 dead-letter, 알림
- `S002` 수신 → 재시도 무의미. dead-letter

## 산출물 체크리스트

- [ ] `_workspace/01_architect_contract.md` — Spring 계약 + 워커 인터페이스 매핑
- [ ] `_workspace/01_architect_directory.md` — 디렉토리 구조 + 모듈 책임
- [ ] `_workspace/01_architect_deps.md` — `pyproject.toml` + 핀 사유
- [ ] `_workspace/01_architect_env.md` — `.env.example` + 변수 설명
- [ ] 다른 에이전트에게 계약 확정 SendMessage 발송

## 참고

- 글로벌 스킬: `~/.Codex/skills/harness/references/skill-writing-guide.md`
- vault MOC: `/Users/minjoun/Documents/DevKnowledge/10_Projects/hola-climbing-server/MOC.md`
- 결정 노트: `30_Decisions/2026-05-25-hola-*` 시리즈
