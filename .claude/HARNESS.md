# 하네스: Hola Climbing AI Worker

> 프로젝트 루트의 `CLAUDE.md`는 vault `GLOBAL_AI_RULES.md`로의 심볼릭 링크이므로 직접 수정하지 않는다. 본 파일이 프로젝트 전용 하네스 포인터를 담는다.

**목표:** Python(FastAPI + MediaPipe)으로 클라이밍 영상 동작 분석 AI 워커를 부트스트랩·확장·유지보수한다. Spring 서버(`hola-climbing-server`)의 분석 도메인을 단일 진실 원천으로 삼는다.

**트리거 규칙:**
- 워커 관련 작업 요청 시 `hola-ai-orchestrator` 스킬을 사용한다
- 단순 코드 질문, 단일 파일 수정은 직접 응답 가능
- 부트스트랩, 기능 추가/수정, Spring 계약 동기화, 통합 검증, 정확도 측정은 반드시 오케스트레이터를 통한다

**에이전트 (`.claude/agents/`):**
| 이름 | 역할 |
|------|------|
| architect | FastAPI 구조·계약·의존성·환경변수 |
| vision-engineer | MediaPipe Pose + 클라이밍 기술 분류 |
| pipeline-engineer | GCS 다운로드·OpenCV·Redis Streams·콜백 |
| integration-engineer | Spring 계약 검증·Docker·README |
| qa-engineer | 통합 테스트·정확도·경계면 검증 |

**스킬 (`.claude/skills/`):**
- `hola-ai-orchestrator` — 메인 오케스트레이터
- `fastapi-worker-design` — architect 전용
- `pose-technique-analysis` — vision-engineer 전용
- `video-pipeline-redis` — pipeline-engineer 전용
- `spring-contract-integration` — integration-engineer 전용
- `ai-worker-qa` — qa-engineer 전용

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-05-27 | 초기 구성 (5 에이전트 + 6 스킬, 감독자 + 점진적 QA 패턴) | 전체 | 사용자 요청 "하네스 구성해줘". 스택 FastAPI + MediaPipe/OpenCV로 확정 |
| 2026-05-27 | Spring contract 추정 제거 → 확정값 박음 (stream key `analysis:requests`, channel `analysis:progress`, callback path `/api/analysis/videos/{videoId}`, ApiResponse `is_success` boolean, ErrorCode V005/S002, AnalysisIngestRequest schema) | fastapi-worker-design, video-pipeline-redis, spring-contract-integration, ai-worker-qa | hola-climbing-server 실제 코드 분석 (Explore agent) |
| 2026-05-27 | 기술 목록 6개로 확정 (하이스텝/플래깅/훅(토,힐)/락오프/다이노/코디네이션), 데드포인트는 다이노에 흡수 | pose-technique-analysis, vision-engineer | 사용자 도메인 결정. 임계값 정밀화 + 표 형식 (사용자 검토 대상 명시) |

**CLAUDE.md 통합 옵션 (선택):**

기본적으로 본 `.claude/HARNESS.md`만으로도 오케스트레이터 스킬 트리거는 작동한다 (Claude Code는 `.claude/skills/`를 자동 인식). 단, 새 세션의 컨텍스트에 본 하네스 정보를 명시적으로 노출하려면 다음 중 하나를 선택:

1. **권장 (현재 상태):** CLAUDE.md는 vault 글로벌 규칙 심볼릭 링크 유지. 본 `.claude/HARNESS.md`만 따로 둠. 트리거는 스킬 description으로 작동.
2. **CLAUDE.md 분리 (수동 작업 필요):**
   ```bash
   # 심볼릭 링크를 제거하고 글로벌 규칙을 inline + 하네스 섹션 추가
   rm CLAUDE.md
   cat /Users/minjoun/Documents/DevKnowledge/00_System/GLOBAL_AI_RULES.md \
       .claude/HARNESS.md > CLAUDE.md
   ```
   단점: 글로벌 규칙 갱신 시 수동 동기화 필요. scripts/link-ai-rules.sh 재실행 시 덮어쓰임 주의.
3. **참조 inject:** 글로벌 규칙 본체에 `## 프로젝트별 하네스` 섹션을 두고 각 프로젝트의 `.claude/HARNESS.md`를 참조하도록 한 줄 추가 (vault 본체 수정 필요 — 신중).

## Vault 연결

- 자매 프로젝트 MOC: `[[10_Projects/hola-climbing-server/MOC]]`
- 관련 결정: `30_Decisions/2026-05-25-hola-redis-streams-ai-dispatch.md`, `2026-05-25-hola-gcs-signed-url-direct-upload.md`
- 워커 부트스트랩 세션 로그 (작성 예정): `50_SessionLogs/2026-05-27-hola-ai-bootstrap.md`
