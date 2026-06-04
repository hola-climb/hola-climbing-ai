---
name: ai-worker-qa
description: "Hola AI 워커의 통합 테스트와 경계면 정합성을 점진적으로 검증한다. pytest + testcontainers로 Redis/Mock GCS 컨테이너를 띄우고, Spring contract와 워커 응답 shape을 교차 비교하며, vision 모듈의 정확도를 라벨링 데이터로 측정한다. '워커 테스트', '통합 테스트 작성', '정확도 측정', '경계면 검증', 'pytest fixture' 요청 시 반드시 사용."
---

# AI Worker QA

워커 전체의 **점진적 품질 게이트**. qa-engineer 전용.

## 언제 사용하는가

- 각 모듈 완성 직후 (architect → vision → pipeline → integration 순차)
- Spring contract와 워커 응답 shape diff
- vision 모듈의 정확도/recall 측정
- 통합 테스트 (`pytest` + `testcontainers`)
- 회귀 케이스 보존

## 핵심 원칙

1. **존재 확인 ≠ 검증.** 파일이 있다는 것과 두 코드의 shape이 일치한다는 것은 다르다.
2. **양쪽 코드를 동시에 읽는다.** Spring DTO와 워커 Pydantic 모델을 같은 컨텍스트에 두고 필드별 대조.
3. **정량 지표.** "잘 됨" 금지. "20개 중 17개 일치, precision 0.85"처럼 숫자로.
4. **점진적 QA.** 전체 완성 후 1회가 아니다. 각 모듈 완성 즉시 검증.

## 경계면 교차 비교

### 1. Spring DTO vs 워커 Pydantic

```bash
# Spring 측 DTO 필드 추출
grep -A 20 "class AnalysisResult" \
  /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/com/holaclimbing/server/domain/analysis/dto/*.java

# 워커 측 Pydantic 모델 추출
grep -A 20 "class AnalysisResult" app/models/analysis.py
```

필드별 비교표 (Spring 확정값):

```markdown
| Spring 필드 | Java 타입 | JSON 키 | 워커 Pydantic | 일치? |
|------------|----------|--------|----------------|------|
| `AnalysisIngestRequest.status` | `String` | `status` | `Literal["done","failed"]` | ✅ |
| `AnalysisIngestRequest.modelVersion` | `String` | `model_version` | `model_version: str` | ✅ |
| `AnalysisIngestRequest.segments` | `List<AnalysisSegmentPayload>` | `segments` | `list[Segment]` | ✅ |
| `AnalysisSegmentPayload.sequenceIndex` | `Integer` | `sequence_index` | `int` | ✅ |
| `AnalysisSegmentPayload.startTimeMs` | `Integer` (nullable) | `start_time_ms` | `int \| None` | ✅ |
| `AnalysisSegmentPayload.endTimeMs` | `Integer` (nullable) | `end_time_ms` | `int \| None` | ✅ |
| `AnalysisSegmentPayload.technique` | `String` | `technique` | `str` (6 라벨) | ✅ |
| `AnalysisSegmentPayload.isDynamic` | `Boolean` (nullable) | `is_dynamic` | `bool \| None` | ✅ |
| `AnalysisSegmentPayload.confidence` | `Float` (nullable) | `confidence` | `float \| None` | ✅ |
| `ApiResponse.isSuccess` | `boolean` | `is_success` | `is_success: bool` | ✅ |
```

테스트는 `pytest` + 실제 Spring 클래스 파일 grep으로 매번 재검증:

```python
def test_segment_field_names_match_spring():
    import re
    java = Path("/Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/com/holaclimbing/server/domain/analysis/dto/AnalysisSegmentPayload.java").read_text()
    java_fields = set(re.findall(r"(?:Integer|String|Boolean|Float)\s+(\w+)", java))
    # camelCase → snake_case 변환
    expected_keys = {re.sub(r"(?<!^)(?=[A-Z])", "_", f).lower() for f in java_fields}
    from app.models.analysis import AnalysisSegment
    worker_keys = set(AnalysisSegment.model_json_schema()["properties"].keys())
    assert expected_keys <= worker_keys, f"missing in worker: {expected_keys - worker_keys}"
```

### 2. ErrorCode 매핑

```bash
grep "ANALYSIS_" /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/com/holaclimbing/server/common/exception/ErrorCode.java
grep "ANALYSIS_" app/core/errors.py
```

누락된 enum 값 발견 시 architect/integration에게 즉시 알림.

### 3. Redis 명명 (확정값 대조)

| 키 | Spring 값 | 워커 환경변수 | 일치 검증 |
|----|----------|--------------|----------|
| Stream key | `analysis:requests` (RedisStreamAnalysisJobQueue.java:21) | `REDIS_STREAM_KEY` | 같아야 함 |
| Progress channel | `analysis:progress` (RedisAnalysisProgressBus.java:18) | `REDIS_PROGRESS_CHANNEL` | 같아야 함 |
| Consumer group | (Spring 강제 안 함) | `REDIS_CONSUMER_GROUP=hola-ai-worker` | 워커 자율 |
| Stream payload key | `videoId`(String), `gcsPath`, `callbackUrl` | 워커 parse_job() | camelCase raw |

테스트:

```python
def test_stream_key_matches_spring():
    spring_src = Path("/Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/src/main/java/com/holaclimbing/server/infrastructure/ai/RedisStreamAnalysisJobQueue.java").read_text()
    assert 'analysis:requests' in spring_src, "Spring stream key changed"
    from app.core.config import settings
    assert settings.REDIS_STREAM_KEY == "analysis:requests"
```

## 통합 테스트 (pytest + testcontainers)

### 픽스처

```python
# tests/conftest.py
import pytest
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7") as r:
        yield r

@pytest.fixture
def redis_url(redis_container):
    return f"redis://localhost:{redis_container.get_exposed_port(6379)}"

@pytest.fixture
def fake_gcs(tmp_path):
    """샘플 영상 파일을 로컬 경로로 마운트. GCS 인터페이스 mock."""
    sample = tmp_path / "sample.mp4"
    # 작은 테스트 영상 복사
    import shutil
    shutil.copy("tests/fixtures/sample_short.mp4", sample)
    return f"file://{sample}"
```

### 경계면 테스트 예시

```python
# tests/integration/test_callback_shape.py
import json
from app.models.analysis import AnalysisIngestRequest, AnalysisSegment

def test_callback_matches_spring_ingest_request():
    """워커가 콜백할 JSON이 Spring AnalysisIngestRequest와 일치하는지."""
    req = AnalysisIngestRequest(
        status="done",
        model_version="rule_v1",
        segments=[AnalysisSegment(
            sequence_index=0,
            start_time_ms=1500,
            end_time_ms=2800,
            technique="high_step",
            is_dynamic=False,
            confidence=0.7,
        )],
    )
    payload = req.model_dump()
    assert payload["status"] == "done"
    assert payload["model_version"] == "rule_v1"          # snake, not modelVersion
    assert payload["segments"][0]["sequence_index"] == 0
    assert payload["segments"][0]["is_dynamic"] is False  # not isDynamic
    assert "modelVersion" not in payload                  # camelCase 누출 검출
```

```python
# tests/integration/test_api_response_shape.py
def test_api_response_uses_is_success_not_status():
    from app.models.response import ApiResponse
    r = ApiResponse.ok(data={"x": 1})
    payload = r.model_dump()
    assert "is_success" in payload  # NOT "status"
    assert payload["is_success"] is True
    assert payload["code"] == "OK"
    assert "timestamp" in payload
```

### Streams 컨슈머 테스트

```python
# tests/integration/test_stream_consumer.py
import pytest, json
import redis.asyncio as aioredis

@pytest.mark.asyncio
async def test_consumer_acks_message(redis_url):
    r = aioredis.from_url(redis_url)
    stream, group = "hola:analysis:jobs", "test-group"
    await r.xgroup_create(stream, group, id="0", mkstream=True)
    msg_id = await r.xadd(stream, {"job_id": "1", "video_url": "file:///tmp/x.mp4"})

    from app.workers.stream_consumer import handle_one
    await handle_one(r, stream, group, "consumer-1", expect=msg_id)

    pending = await r.xpending(stream, group)
    assert pending["pending"] == 0, "메시지가 ACK되지 않음"
```

## 정확도 측정

```python
# tests/accuracy/test_vision_baseline.py
import pytest, csv, json
from pathlib import Path
from app.services.vision import analyze_video

LABEL_CSV = "/Users/minjoun/Workspace/projects/Hola-Climbing/labels.csv"

def load_labels():
    with open(LABEL_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["label"].strip()]
    return rows

@pytest.mark.parametrize("row", load_labels())
def test_video_accuracy(row, request):
    """라벨링된 영상에 대해 모델 출력과 ground truth 비교."""
    # 실제 영상이 로컬에 없으므로 skip 처리. 사용자가 직접 실행 시에만 활성.
    video_path = Path("tests/fixtures/videos") / row["filename"].replace(".json", ".mp4")
    if not video_path.exists():
        pytest.skip(f"video not local: {video_path}")
    result = analyze_video(video_path)
    truth = json.loads(row["label"]) if row["label"] else {}
    # precision/recall 계산
    # ... (기술별 비교)
```

baseline 결과는 `_workspace/05_qa_accuracy.md`에:

```markdown
| 기술 | 영상 수 | True Positive | False Positive | False Negative | Precision | Recall |
|------|---------|---------------|----------------|----------------|-----------|--------|
| high_step | 15 | 12 | 2 | 3 | 0.857 | 0.800 |
| flagging  | 10 | 7  | 1 | 3 | 0.875 | 0.700 |
| deadpoint | 12 | 8  | 2 | 4 | 0.800 | 0.667 |
```

## 회귀 케이스 보존

발견된 버그/엣지 케이스를 테스트로 박제:

```python
# tests/regression/test_no_person_frames.py
def test_handles_video_with_no_person_in_first_10s():
    """벽만 보이고 사람이 늦게 들어오는 영상에서 크래시 안 함."""
    result = analyze_video("tests/fixtures/wall_only_start.mp4")
    assert result.pose_detection_rate < 0.5
    assert "high_step" in result.techniques  # 빈 dict라도 키는 존재

# tests/regression/test_hevc_codec.py
def test_handles_hevc_encoded_video():
    """iPhone HEVC 영상 처리."""
    result = analyze_video("tests/fixtures/iphone_hevc.mov")
    assert result is not None
```

## 산출물 체크리스트

- [ ] `_workspace/05_qa_boundary_diff.md` — Spring vs 워커 필드 diff
- [ ] `_workspace/05_qa_test_plan.md` — 단위/통합/정확도 전략
- [ ] `_workspace/05_qa_accuracy.md` — vision baseline 수치
- [ ] `_workspace/05_qa_findings.md` — 버그 목록 + 책임 에이전트 + 심각도
- [ ] 실제 파일: `tests/conftest.py`, `tests/unit/`, `tests/integration/`, `tests/accuracy/`, `tests/regression/`

## 참고

- 글로벌 QA 가이드: `~/.claude/skills/harness/references/qa-agent-guide.md`
- testcontainers-python: https://testcontainers-python.readthedocs.io/
