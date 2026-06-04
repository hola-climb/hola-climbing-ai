---
name: hola-ai-orchestrator
description: "Hola 클라이밍 AI 워커(Python/FastAPI + MediaPipe)의 부트스트랩·개발·기능 추가·기능 수정·통합 검증을 조율하는 5명 에이전트 팀 오케스트레이터. '워커 부트스트랩', 'AI 워커 시작', '분석 엔드포인트 추가', 'pose 기술 추가', 'Spring 계약 동기화', '워커 통합 테스트' 요청 시 반드시 사용. 후속 작업: '다시 실행', '재실행', '업데이트', '수정', '보완', '결과 개선', '이전 산출물 기반으로 작업' 요청 시에도 반드시 이 스킬을 사용."
---

# Hola AI Worker Orchestrator

Hola 클라이밍 AI 워커의 **5명 에이전트 팀**을 조율하여 워커 부트스트랩부터 통합 검증까지 완수하는 메인 스킬.

## 실행 모드: 에이전트 팀

`TeamCreate` + `SendMessage` + `TaskCreate`를 사용한다. 5명 이상 협업이 필요하며 경계면(Spring contract)에서 빈번한 조율이 필요하기 때문.

## 에이전트 구성

| 팀원 | 타입 | 모델 | 역할 | 주력 스킬 | 출력 위치 |
|------|------|------|------|----------|-----------|
| `architect` | general-purpose | opus | FastAPI 구조·계약·의존성·환경변수 | `fastapi-worker-design` | `_workspace/01_architect_*` |
| `vision-engineer` | general-purpose | opus | MediaPipe Pose + 기술 분류 휴리스틱 | `pose-technique-analysis` | `_workspace/02_vision_*` |
| `pipeline-engineer` | general-purpose | opus | GCS 다운로드, OpenCV, Redis Streams | `video-pipeline-redis` | `_workspace/03_pipeline_*` |
| `integration-engineer` | general-purpose | opus | Spring 계약 검증·Docker·README | `spring-contract-integration` | `_workspace/04_integration_*` |
| `qa-engineer` | general-purpose | opus | 통합 테스트·정확도·경계면 검증 | `ai-worker-qa` | `_workspace/05_qa_*` |

## 워크플로우

### Phase 0: 컨텍스트 확인 (후속 작업 지원)

작업 시작 전 기존 산출물을 확인하여 실행 모드를 결정한다:

1. `_workspace/` 디렉토리 존재 여부 확인
2. 실행 모드 결정:
   - **`_workspace/` 미존재** → 초기 부트스트랩. Phase 1로 진행
   - **`_workspace/` 존재 + 사용자가 부분 수정 요청** (예: "vision 모듈만 다시", "콜백 로직 보완") → **부분 재실행**. 해당 에이전트만 재호출
   - **`_workspace/` 존재 + 사용자가 새 입력 (예: 새 기술 추가, Spring 계약 변경)** → **확장 실행**. 기존 산출물을 입력으로 사용하면서 변경분만 처리
   - **사용자가 명시적 재시작 요청** → 기존 `_workspace/`를 `_workspace_{YYYYMMDD_HHMMSS}/`로 이동 후 새로 시작
3. 부분/확장 실행 시 이전 산출물 경로를 에이전트 프롬프트에 포함

### Phase 1: 준비

1. 사용자 입력 분석:
   - 부트스트랩 요청인가? 기능 추가/수정 요청인가?
   - 명시된 제약 (성능, 데드라인, 새 기술명 등) 추출
2. 작업 디렉토리 준비:
   - `_workspace/00_input/` — 사용자 입력 + Spring 코드 위치 + vault 컨텍스트 링크
   - `_workspace/_log/` — 팀 통신/진행 로그
3. **Spring 코드 위치 확인** (필수):
   - `/Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server/` 가 접근 가능한지
   - 분석 도메인 (`domain/analysis/`) 의 최신 commit 확인
4. **vault 컨텍스트 로딩** (CLAUDE.md 규칙 0~1):
   - `01_Profile/About-Me.md`
   - `10_Projects/hola-climbing-server/MOC.md` (자매 프로젝트)
   - 최근 `50_SessionLogs/` 중 hola 관련
   - `30_Decisions/2026-05-25-hola-*` 시리즈

### Phase 2: 팀 구성

```
TeamCreate(
  team_name: "hola-ai-team",
  members: [
    {
      name: "architect", agent_type: "general-purpose", model: "opus",
      prompt: """너는 hola-climbing-ai 워커의 architect다.
        Spring 서버(/Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-server)의
        domain/analysis/ 를 단일 진실 원천으로 삼아 워커의 FastAPI 구조·계약·의존성을 설계한다.
        스킬: fastapi-worker-design
        에이전트 정의: .claude/agents/architect.md
        출력: _workspace/01_architect_*.md + app/ 구조 + pyproject.toml + .env.example"""
    },
    {
      name: "vision-engineer", agent_type: "general-purpose", model: "opus",
      prompt: """너는 vision-engineer다. MediaPipe Pose + 휴리스틱 규칙으로 클라이밍 기술
        (하이스텝/플래깅/데드포인트)을 인식한다. 4주 데드라인이므로 ML 학습 금지, 규칙 기반 우선.
        스킬: pose-technique-analysis
        에이전트 정의: .claude/agents/vision-engineer.md
        라벨 데이터: /Users/minjoun/Workspace/projects/Hola-Climbing/labels.csv
        출력: _workspace/02_vision_*.md + app/services/vision/"""
    },
    {
      name: "pipeline-engineer", agent_type: "general-purpose", model: "opus",
      prompt: """너는 pipeline-engineer다. GCS Signed URL 다운로드, OpenCV 프레임 추출,
        Redis Streams 컨슈머, Spring 콜백을 담당한다.
        스킬: video-pipeline-redis
        에이전트 정의: .claude/agents/pipeline-engineer.md
        출력: _workspace/03_pipeline_*.md + app/infra/, app/workers/"""
    },
    {
      name: "integration-engineer", agent_type: "general-purpose", model: "opus",
      prompt: """너는 integration-engineer다. Spring 계약(ApiResponse/ErrorCode/snake_case JSON/
        Redis key)이 워커와 일치하는지 검증하고 Docker/README를 작성한다.
        스킬: spring-contract-integration
        에이전트 정의: .claude/agents/integration-engineer.md
        출력: _workspace/00_input/spring-server-snapshot.md (최우선),
              _workspace/04_integration_*.md, Dockerfile, docker-compose.yml, README.md"""
    },
    {
      name: "qa-engineer", agent_type: "general-purpose", model: "opus",
      prompt: """너는 qa-engineer다. 각 모듈 완성 직후 점진적으로 검증한다.
        pytest + testcontainers로 Redis mock, vision은 라벨 데이터로 정확도 측정.
        스킬: ai-worker-qa
        에이전트 정의: .claude/agents/qa-engineer.md
        출력: _workspace/05_qa_*.md + tests/ 디렉토리"""
    },
  ]
)
```

### Phase 3: 작업 등록 + 자체 조율

```
TaskCreate(tasks: [
  // 우선순위 0: Spring 계약 스냅샷 (모든 다른 작업의 입력)
  { title: "Spring contract 스냅샷 추출", assignee: "integration-engineer",
    description: "_workspace/00_input/spring-server-snapshot.md 작성. domain/analysis/, ErrorCode enum, Redis 설정, application.yml 환경변수 모두 포함" },

  // 우선순위 1: architect — 다른 모든 에이전트의 입력
  { title: "워커 외부 인터페이스 계약 확정", assignee: "architect",
    depends_on: ["Spring contract 스냅샷 추출"],
    description: "Spring 스냅샷 기반으로 워커 API/Redis Streams 인터페이스 확정. _workspace/01_architect_contract.md" },
  { title: "디렉토리 구조 + pyproject.toml 작성", assignee: "architect",
    depends_on: ["워커 외부 인터페이스 계약 확정"],
    description: "_workspace/01_architect_directory.md + 실제 pyproject.toml + .env.example" },

  // 우선순위 2: vision/pipeline 병렬 (architect 완료 후)
  { title: "MediaPipe Pose 통합 모듈", assignee: "vision-engineer",
    depends_on: ["디렉토리 구조 + pyproject.toml 작성"],
    description: "app/services/vision/pose.py + 키포인트 추출" },
  { title: "기술 휴리스틱 분류기", assignee: "vision-engineer",
    depends_on: ["MediaPipe Pose 통합 모듈"],
    description: "하이스텝/플래깅/데드포인트 + _workspace/02_vision_technique_rules.md (사용자 검토용)" },
  { title: "GCS 다운로드 + OpenCV 프레임 iterator", assignee: "pipeline-engineer",
    depends_on: ["디렉토리 구조 + pyproject.toml 작성"],
    description: "app/infra/gcs.py + app/services/pipeline/frames.py" },
  { title: "Redis Streams 컨슈머 + 진행률 발행", assignee: "pipeline-engineer",
    depends_on: ["GCS 다운로드 + OpenCV 프레임 iterator"],
    description: "app/workers/stream_consumer.py + app/infra/redis_bus.py" },
  { title: "Spring 콜백 클라이언트 + 재시도", assignee: "pipeline-engineer",
    depends_on: ["Redis Streams 컨슈머 + 진행률 발행"],
    description: "app/services/callback/ + tenacity retry" },

  // 우선순위 3: integration — vision/pipeline 산출물 검증
  { title: "정합성 검증 표 작성", assignee: "integration-engineer",
    depends_on: ["기술 휴리스틱 분류기", "Spring 콜백 클라이언트 + 재시도"],
    description: "_workspace/04_integration_contract_match.md — 필드/Redis key/ErrorCode 매핑" },
  { title: "Dockerfile + docker-compose + README", assignee: "integration-engineer",
    depends_on: ["정합성 검증 표 작성"],
    description: "실제 파일 생성. README는 server README와 동일 톤" },

  // 우선순위 4: QA — 점진적 검증 (각 모듈 완성 시점에 즉시 시작)
  { title: "Boundary diff (Spring DTO vs 워커 Pydantic)", assignee: "qa-engineer",
    depends_on: ["워커 외부 인터페이스 계약 확정"],
    description: "_workspace/05_qa_boundary_diff.md. architect 산출물 완성 즉시 실행" },
  { title: "Vision 정확도 baseline", assignee: "qa-engineer",
    depends_on: ["기술 휴리스틱 분류기"],
    description: "labels.csv로 precision/recall 측정. _workspace/05_qa_accuracy.md" },
  { title: "통합 테스트 (testcontainers + pytest)", assignee: "qa-engineer",
    depends_on: ["Dockerfile + docker-compose + README"],
    description: "tests/conftest.py + tests/integration/. Redis 컨테이너 + Mock GCS" },
  { title: "최종 검증 리포트", assignee: "qa-engineer",
    depends_on: ["통합 테스트 (testcontainers + pytest)"],
    description: "_workspace/05_qa_findings.md — 잔여 이슈 + 권장 액션" },
])
```

### Phase 4: 실행 + 모니터링

팀원들이 자체 조율한다 (SendMessage + TaskCreate).

**오케스트레이터 모니터링 책임:**
- 각 작업이 30분 이상 진척 없으면 해당 에이전트에게 상태 요청
- 경계면 불일치 (integration이 architect/pipeline에게 SendMessage)는 즉시 추적
- QA가 발견한 버그는 책임 에이전트에게 즉시 라우팅

**팀원 간 통신 규칙:**
- `integration-engineer` ↔ 모든 에이전트: Spring 계약 변경 알림
- `architect` ↔ `vision`/`pipeline`: 인터페이스 변경
- `vision` ↔ `pipeline`: 프레임 iterator API, 샘플링 주기
- `qa-engineer` → 모든 에이전트: 버그 보고
- 모두 → `qa-engineer`: 모듈 완성 시 "QA 가능" 알림

### Phase 5: 산출물 정리 + 후속

1. 작업 디렉토리에 최종 산출물 확인:
   ```
   hola-climbing-ai/
   ├── app/                    # FastAPI 워커 코드 (architect/vision/pipeline)
   ├── tests/                  # pytest 스위트 (qa)
   ├── pyproject.toml          # architect
   ├── .env.example            # architect
   ├── Dockerfile              # integration
   ├── docker-compose.yml      # integration
   ├── README.md               # integration
   └── _workspace/             # 모든 중간 산출물 보존
   ```

2. 사용자 피드백 요청:
   - "vision 휴리스틱 임계값 검토 가능하신가요?" (도메인 전문가 검토 필요)
   - "Spring contract 추정 부분 확인 가능하신가요?" (스냅샷의 known_unknowns)
   - "다음 우선순위는 어느 기능인가요?" (예: 추가 기술 인식, ML 분류기 도입)

3. CLAUDE.md (또는 `.claude/HARNESS.md`) 변경 이력 업데이트

4. vault에 세션 로그 작성 제안: `50_SessionLogs/YYYY-MM-DD-hola-ai-bootstrap.md`

## 에러 핸들링

| 에러 | 처리 |
|------|------|
| Spring 코드 접근 불가 | vault MOC를 임시 SSOT로 사용. `_workspace/known_unknowns.md` 명시 |
| MediaPipe 환경 의존성 (Apple Silicon) | Docker linux/amd64 컨테이너 내부에서만 통합 테스트 실행 |
| 라벨링 데이터 빈약 | 정확도 검증 생략, 사용자 수동 검증 요청 명시 |
| testcontainers 실패 | docker-compose 폴백, 또는 mock-only 테스트 (정보 보존) |
| 에이전트 1회 재시도 후 재실패 | 해당 산출물 없이 진행, `_workspace/_log/missing.md`에 기록, 최종 리포트에 누락 명시 |
| 경계면 불일치 발견 | 기본은 워커 수정. Spring 측 버그 의심 시 사용자에게 보고 |

## 테스트 시나리오

### 정상 흐름 (초기 부트스트랩)

1. 사용자: "하네스 구성해줘" → 본 오케스트레이터 트리거
2. `_workspace/` 미존재 → 초기 부트스트랩 모드
3. Phase 1~5 순차 진행, 약 5명 × 5~6 task = 25~30 작업
4. 최종 산출물: 실행 가능한 워커 + 통합 테스트 + README

### 후속 1: 기술 추가 요청

1. 사용자: "풋체인지 기술도 인식하도록 확장해줘"
2. Phase 0에서 `_workspace/` 존재 확인 → 부분 재실행 모드
3. 활성화 에이전트: `vision-engineer` (메인), `qa-engineer` (검증), `architect` (출력 schema 변경 시)
4. 다른 에이전트는 idle

### 후속 2: Spring 계약 변경

1. 사용자: "Spring 측에서 콜백 URL이 바뀌었어"
2. 활성화 에이전트: `integration-engineer` (감지), `pipeline-engineer` (수정), `qa-engineer` (검증)
3. `_workspace/04_integration_contract_match.md` 갱신
4. CLAUDE.md 변경 이력에 기록

### 에러 흐름

1. Spring 코드 디렉토리가 비어있음 (정리됨/이동됨)
2. integration-engineer가 스냅샷 작성 실패 → `_workspace/known_unknowns.md`에 명시
3. architect는 vault MOC 기반으로 추정 계약 작성
4. 최종 리포트에 "Spring 코드 재확인 필요" 경고

## 산출물 체크리스트

- [ ] `.claude/agents/` 5개 (architect, vision-engineer, pipeline-engineer, integration-engineer, qa-engineer)
- [ ] `.claude/skills/` 6개 (본 오케스트레이터 + 5개 전문 스킬)
- [ ] `_workspace/` 산출물 보존 (00_input ~ 05_qa)
- [ ] 실제 코드: `app/`, `tests/`, `pyproject.toml`, `.env.example`, `Dockerfile`, `docker-compose.yml`, `README.md`
- [ ] 변경 이력 기록 (`.claude/HARNESS.md`)

## 참고

- 하네스 메타 스킬: `~/.claude/skills/harness/SKILL.md`
- 에이전트 패턴: `~/.claude/skills/harness/references/agent-design-patterns.md`
- 오케스트레이터 템플릿: `~/.claude/skills/harness/references/orchestrator-template.md`
- 자매 프로젝트 MOC: `/Users/minjoun/Documents/DevKnowledge/10_Projects/hola-climbing-server/MOC.md`
