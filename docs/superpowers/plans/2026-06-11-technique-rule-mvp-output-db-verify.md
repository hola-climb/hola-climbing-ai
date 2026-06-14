# 2026-06-11 6기술 Rule-base MVP — 출력 확인 → DB 저장 확인 → 스팟체크 검증

## Goal

rule-base 6기술 분류기(이미 구현·배선 완료)를 MVP로 확정하기 위해
① 워커 단독으로 segments가 잘 출력되는지, ② Spring 콜백을 거쳐 DB(`analysis_results`)에
잘 저장되는지, ③ 스팟체크로 명백한 오분류가 없는지 순서로 확인한다.
라벨링·정량 평가는 MVP 범위에서 제외한다 (post-MVP).

## 현황 (전제)

- `app/services/vision/classifier.py` — 7라벨(high_step, flagging, toe_hook, heel_hook,
  lock_off, dyno, coordination) 스코어러 구현 완료, `orchestrator.py`에 배선됨.
- `app/services/vision/_thresholds.py` — 임계값 중앙화. `tests/unit/test_vision_rules.py` 존재.
- `model_version` 기본값 `rule_v1`, flow gate 적용 시 `rule_v1+flow_rf_v2`.
- 운영 게이트 artifact: `models/flow_qa_rf_v2.joblib` (dynamic/static 보정, MVP 확정).
- 캐시 영상: `data/gcs_cache/videos` (labels_완료.csv 425행 매칭).
- Spring 수신부: `POST /api/analysis/videos/{videoId}` (`AnalysisController.ingestResult`)
  → `analysis_results` 테이블. `technique`은 자유 문자열 컬럼.
  `AiCallbackSecretFilter`가 `X-AI-Callback-Secret` 헤더 검증.

---

## Phase 1 — 워커 단독 출력 스모크 (Spring/Redis 불필요)

네이티브 `uv` 실행 (arm64에서 Docker는 amd64 에뮬레이션이라 느림).

1. **출력 덤프 스크립트** `scripts/dump_technique_segments.py` 작성
   - 입력: 영상 디렉토리(또는 파일 목록), `--flow-gate-model` 옵션
   - 처리: `iter_frames → extract_pose_landmarks → split_segments → classify_segments`
     (+ 옵션으로 `apply_flow_gate`)
   - 출력: 영상별 segments JSON + 콘솔 요약 (영상별 segment 수 / 기술 분포 / drop된 구간 수)
2. **sanity 자동 체크** (스크립트에 내장):
   - `technique` ∈ 7종 라벨
   - `sequence_index` 0부터 연속
   - `start_time_ms < end_time_ms`, 영상 길이 이내
   - `confidence` ∈ [0,1]
   - dyno/coordination → `is_dynamic=true`, 나머지 → false (게이트 demote 케이스 제외)
3. **실행**: 캐시 영상 dynamic 5 + static 5 (라벨 균형) 선택, 게이트 off/on 각 1회
   → demote 동작과 출력 차이 기록
4. 기존 테스트 회귀 확인: `uv run pytest tests/unit/test_vision_rules.py -q`

**완료 기준**: 10개 영상 모두 예외 없이 산출 + sanity 체크 전부 통과.
**관찰 기록**: 빈 출력 영상 수(MIN_CONFIDENCE_TO_EMIT=0.35로 전 segment drop 가능),
특정 기술 쏠림 여부 (예: 전부 coordination이면 Phase 3 전에 임계값 1차 조정).

## Phase 2 — E2E: Redis Stream → 워커 → Spring 콜백 → DB 저장 확인

구성 (권장): Spring 로컬 기동(+DB) / Redis는 docker / 워커는 네이티브 `uv` 실행.
(전 구간 docker compose도 가능하지만 amd64 에뮬레이션으로 vision이 느려 디버깅에 불리)

1. **환경 정렬 체크리스트**
   - `REDIS_HOST/PORT/PASSWORD` 양쪽 일치
   - `AI_CALLBACK_SECRET` ↔ Spring 측 secret 일치 (`X-AI-Callback-Secret`)
   - callback URL 도달성: 워커가 네이티브면 `http://localhost:8080/...`
   - `FLOW_GATE_MODEL_PATH=models/flow_qa_rf_v2.joblib` (운영과 동일 조건)
2. **잡 주입** — 방법 B 권장 (빠름):
   - 방법 A: Spring 업로드 API로 실제 영상 등록 → 정공법, 전 구간 검증되지만 준비 김
   - 방법 B: `redis-cli XADD`로 StreamRequest 직접 주입.
     단, `video_id`는 Spring DB `videos`에 실재하는 행이어야 ingest가 성공하므로
     테스트용 video 행을 먼저 만들어 둔다. `gcs_path`는 캐시 매니페스트의 실제 경로 사용.
3. **관찰 포인트**
   - 워커 로그: `process_job done` + `segments N`, `flow gate applied` + prob_dynamic
   - Spring 로그: ingest 200 수신
   - 진행률 Pub/Sub 이벤트 (옵션): `SUBSCRIBE` 로 PROCESSING 메시지 4종 확인
4. **DB 검증**
   - `SELECT * FROM analysis_results WHERE video_id = ? ORDER BY sequence_index`
   - 행 수 = 워커가 보낸 N, 각 컬럼(technique/start·end/is_dynamic/confidence/model_version)이
     Phase 1 JSON과 **정확히 일치** (결정론 — 같은 영상 같은 결과)
   - 조회 API `GET`으로 `VideoAnalysisResponse` shape 확인 (프론트 소비 경로)
5. **엣지 케이스**
   - segments 0개 done 콜백 → Spring 저장/응답 동작 확인
   - 같은 video_id 재처리 시 중복 행 정책 (upsert vs append) 확인 — Spring 코드로 검증
   - failed 콜백 1회 (옵션): 존재하지 않는 gcs_path로 주입

**완료 기준**: 영상 2~3개에서 DB 행 ↔ 워커 출력 일치, 조회 API 정상.

## Phase 3 — 검증 (스팟체크)

1. 캐시 영상 15~20개 일괄 실행 (dynamic/static 균형, 파일명/기존 리뷰 큐 참고해 기술 다양성 확보)
2. **(권장) 오버레이 렌더 스크립트** `scripts/render_technique_overlay.py`:
   구간·기술 라벨·confidence를 영상에 burn-in한 mp4 생성 → 스팟체크 효율 대폭 상승
3. 사용자 스팟체크: 영상별 `맞음/틀림/애매` + 메모를 간단 CSV로 기록
4. 오분류 패턴 → `_thresholds.py` 조정 (flagging·coordination이 false positive 1순위 예상)
   → 재실행 → before/after segment diff 비교 (결정론이라 diff 가능)
5. 임계값 변경이 발생하면:
   - `model_version` bump: `rule_v1` → `rule_v2`
   - `_thresholds.py` ↔ `_workspace/02_vision_technique_rules.md` 1:1 동기화

**완료 기준**: 사용자 승인 ("MVP 수준"). 명백한 오작동(전 영상 동일 기술, 빈 출력 다수) 없음.

## Phase 4 — 마무리

- `uv run pytest -q`, `ruff check`, `mypy` 통과
- README: 6기술 출력 상태/버전/스팟체크 결과 요약 반영
- vault: 세션 로그 + 결정 기록 (스팟체크 채택 사유, 게이트 충돌 정책)

---

## 결정 필요 (사용자)

1. **Phase 2 잡 주입**: 방법 B(XADD 직접 주입) 권장 — 빠르고 반복 용이. 방법 A는 최종 1회만.
2. **오버레이 렌더 스크립트**: 제작 권장 (1~2시간 비용으로 스팟체크 시간 크게 절약).
3. **게이트 충돌 정책**: flow gate가 static 판정 시 dyno/coordination demote — 현행
   `demote_confidence=0.55` 유지 권장, 스팟체크에서 문제 보이면 재논의.

## 리스크

| 리스크 | 대응 |
|---|---|
| 전 segment drop으로 빈 결과 다수 | Phase 1에서 비율 측정, MIN_CONFIDENCE_TO_EMIT 하향 검토 |
| 특정 기술 쏠림 (priority 순서 영향) | Phase 1 분포 확인 후 TECHNIQUE_PRIORITY/임계 조정 |
| Spring ingest가 video 부재로 4xx | Phase 2에서 테스트 video 행 선생성 |
| 재처리 중복 행 | Spring upsert 정책 확인 — append라면 서버 측 이슈로 공유 |
