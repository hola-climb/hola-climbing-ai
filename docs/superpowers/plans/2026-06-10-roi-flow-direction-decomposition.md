# ROI Flow + Direction Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** flow dynamic/static 분류의 feature 병목을 해소한다. (1) Farneback flow의 **수직 방향 성분(vy)** 을 별도 시계열로 분해하고, (2) 효과 확인 후 **pose bbox ROI + 배경 카메라 모션 차감 + 몸통 스케일 정규화**를 추가한다. 분류기 변경은 하지 않는다 (capacity probe 음성 — 아래 Facts 참조).

**Architecture:** `flow_features.py`의 추출 단계만 확장하고 학습/평가 프로토콜(`train_tabular_dynamic_static.py`, group-kfold)은 동결한다. 운영 게이트(`flow_gate.py`)는 feature_dim 스위치 패턴(42 legacy / 46 v3)을 그대로 확장한다. 운영 artifact는 승격 기준을 통과하기 전까지 `models/flow_qa_rf_v2.joblib` 유지.

**Tech Stack:** Python, NumPy, OpenCV (Farneback), SciPy, scikit-learn, MediaPipe Pose (Task 3만), pytest.

---

## Execution Result 2026-06-10

- Task 1 완료: `extract_flow_series` 추가. 반환값은 `(T, 2)`이고 ch0=magnitude, ch1=`vy`.
  OpenCV 좌표계 기준 `vy > 0`은 아래 방향(낙하), `vy < 0`은 위 방향(다이노/상승)으로 테스트 고정.
- Task 2 완료: `data/flow_dataset/gcs_flow_v4_direction` 419개 생성 (`flow_v4`, feature_dim 58).
- 학습 결과: `flow_gcs_v4_direction` RF group-kfold balanced accuracy `0.8377`, dynamic recall `0.8517`,
  static specificity `0.8238`. round3 RF `0.8449`보다 낮음.
- 회귀 결과: round3 miss 65건 중 5건 회복, 8건 신규 오답, 고확신 miss 20건 회복 0건.
- Gate 판단: `0.85 미만 + 고확신 회복 < 5건` 조건에 해당하므로 Task 3(ROI flow)을 바로 진행하지 않고,
  아래 ROI 진단 probe로 축소 검증.

## Review 2026-06-10 (오케스트레이터 세션 사후 진단)

v4 실패 원인을 데이터로 분해한 결과:

- **0.8449 → 0.8377은 노이즈 범위** (fold std ±0.025~0.027). v4는 "해를 끼친" 게 아니라 "도움이 안 된" 것.
- **RF importance: vy 블록(12-dim) 합계 14.4%**, 차원당 ~1%로 균등 분산 — 강한 차원 없음. 약한/중복 신호.
- **결정적: 고확신 FN 12건 vs 정분류 dynamic 30건의 vy feature effect size 0.15~0.56 (전부 <0.6)** — 전역 vy 공간에서 hard case는 정분류 케이스와 구분 불가. 백다이노류는 화면 변위 자체가 작아 **전역 평균에서 희석**된다.

**해석: 이 음성 결과는 "전역 vy"를 기각하지만 "ROI vy"까지 기각하진 않는다** (희석 가설이 맞다면 ROI에서 살아날 수 있음). 단, 사전 확신은 낮아짐.

**갱신된 다음 단계 (비용 순):**
1. **(반나절) ROI 진단 probe**: full Task 3 착수 전에, 고확신 FN 12건 + 대조군 30건만 pose bbox + ROI vy/mag을 계산해 effect size 재측정. ≥1.0 분리가 나오면 Task 3 full 진행, 안 나오면 ROI 폐기.
2. **(1~2일) Escalation encoder probe** — ROI 진단과 독립적으로 가치 있음. 고확신 20건을 encoder가 가르면 hand-crafted feature 트랙 전체를 닫는다.
3. v4 artifact는 승격 비대상 (bal_acc 0.8377, specificity 0.8238 — 기준 둘 다 미달). 운영은 v2 유지.

---

## ROI Probe Result 2026-06-10

ROI 진단 probe 실행 완료:

- 산출물:
  - `scripts/probe_roi_flow_diagnostics.py`
  - `models/reports/roi_flow_probe_round3.csv`
  - `models/reports/roi_flow_probe_round3_summary.json`
- 표본: round3 RF 고확신 dynamic FN 12건 + 정분류 dynamic 30건. `IMG_8449` 1건은 pose 미검출로 실패,
  완료 표본은 41건 (`high_conf_fn=11`, `correct_dynamic=30`).
- 결과: 최대 효과크기 `1.4914` (`roi_mag_p95`), `adj_mag_p95=1.4435`, `adj_mag_mean=1.3132`,
  `adj_vy_p10=1.2317`, `roi_vy_p10=1.1478`.
- 1차 해석: 기준 `≥1.0`을 넘었으므로 ROI 후보를 유지. 단, 아래 사후 검토에서 대조군 설계 문제가
  확인되어 이 판정은 **최종 gate가 아님**.
- 실행 환경 메모: MediaPipe Tasks API는 sandbox 안에서 `DrishtiMetalHelper` service unavailable로 abort.
  ROI probe/full v5 dataset 생성은 샌드박스 밖에서 실행해야 한다.

### Review 2026-06-10 (오케스트레이터 사후 검토) — **"통과" 판정 불성립, Task 3 보류**

probe 수치 자체는 재현 확인. 그러나 **대조군 설계가 결론을 뒷받침하지 못한다:**

1. **잘못된 대조 방향.** probe는 FN vs **정분류 dynamic**을 비교했는데, top effect의 부호를 보면
   전부 **FN이 조용한(quiet) 쪽**이다: `roi_mag_p95` FN 중앙값 1.18 vs OK-dyn 3.68,
   `roi_vy_p10` FN −0.28 vs OK-dyn −0.60. 즉 "ROI에서도 hard FN은 여전히 일반 다이나믹보다
   덜 다이나믹해 보인다" — 이 분리가 큰 것은 **좋은 신호가 아니라 경고 신호**다.
2. **회복 가능성의 판정 기준은 FN vs static인데 static 대조군이 없다.** FN이 회복되려면
   ROI 공간에서 static과 달라 보여야 한다. FN의 `roi_mag_p95` 범위 [0.11, 2.92]가 static 분포와
   겹치는지 아닌지가 진짜 gate인데, 측정되지 않았다.
3. **pose 커버리지 문제: 41건 중 26건이 pose 검출 70% 미만** (최저 34%, fallback 다수).
   ROI feature의 절반 이상이 직전 bbox 유지/전역 fallback으로 만들어져 품질이 불균일하다.
   Task 3 full을 가도 이 문제가 그대로 재현된다.

**갱신된 gate (Task 3 착수 전 필수):**
- [x] probe에 **정분류 static 30건 + 고확신 FP static** cohort 추가 재실행 (샌드박스 밖, ~1시간)
  - `scripts/probe_roi_flow_diagnostics.py`에 static cohort 선택 로직 추가 완료 (TDD: `tests/unit/test_probe_roi_flow_diagnostics.py` 확장). 기존 41건 ROI feature는 `models/reports/roi_flow_probe_round3.csv`에서 재사용했고 static 38건만 새로 처리함.
  - summary에 **FN vs static** effect size 블록 추가 완료.
- [x] 판정: FN vs static effect size ≥ 1.0 **그리고 FN이 dynamic 쪽 방향** → Task 3 GO.
  FN이 static 분포 안에 묻히면 → **ROI 폐기, encoder probe로 직행**
- [x] pose 커버리지 ≥70% 부분집합으로 effect 재계산 (ROI 품질이 결과를 좌우하는지 분리)

### Static Gate Result 2026-06-10

- 산출물:
  - `models/reports/roi_flow_probe_round3_static_gate.csv`
  - `models/reports/roi_flow_probe_round3_static_gate_summary.json`
- 표본: 기존 41건 재사용 + 정분류 static 30건 + 고확신 FP static 8건 추가. 완료 78건, 실패 2건
  (`IMG_8449` high_conf_fn pose 미검출, `IMG_8731` correct_static pose 미검출).
- 전체 비교:
  - `high_conf_fn_vs_correct_static`: 최대 `adj_downward_ratio=0.9725`, dynamic 방향이지만 기준 `1.0` 미만.
  - `high_conf_fn_vs_static_pool`: 최대 `adj_downward_ratio=0.8261`, 기준 미달.
- pose coverage ≥70% subset:
  - `high_conf_fn_vs_correct_static`: `adj_mag_max=1.3943`, dynamic 방향.
  - 단, high_conf_fn subset이 2건뿐이라 최종 GO 근거로 쓰기에는 불안정.
- 최종 판정: **static gate 불통과. Task 3 full ROI v5 보류/폐기. 다음은 encoder probe.**

### Review 2026-06-10 (오케스트레이터 사후 검토) — **판정 승인 + 보존할 인사이트 2개**

수치 재검증 일치. 사전 등록 기준(FN vs static ≥1.0 + dynamic 방향) 대비 0.9725로 미달 — 판정 타당.
중앙값으로 보면 판정이 더 명확하다:

| feature | hard FN | correct static | correct dynamic |
|---|---|---|---|
| `roi_mag_p95` 중앙값 | 1.177 | **1.280** | 3.675 |
| `roi_max_upward_window_mean` | 0.392 | **0.394** | 0.566 |

**hard FN은 ROI magnitude에서 static보다도 조용하다.** ROI feature 자체는 easy dynamic vs static을
잘 가르므로 (1.28 vs 3.68), 결론은 "ROI 위치 보정의 실패"가 아니라 **"hard FN 영상들은 픽셀 변위
수준에서 static과 동일 — optical flow 표현의 본질적 한계"**다. encoder 전환이 맞다.

보존할 인사이트:
1. **잔여 신호: `adj_downward_ratio`** (FN 0.100 ≈ dynamic 0.103 vs static 0.062, effect 0.97).
   hard FN이 다이나믹처럼 보이는 유일한 축 — 백다이노/수직다이노의 하강(착지) 페이즈로 추정.
   encoder 실패 시 이 축 + dedicated person detector(YOLO류, pose 커버리지 문제 해소) 조합이
   마지막 hand-crafted 카드. 지금은 진행 안 함.
2. **이번 기각은 "MediaPipe-pose-bbox 기반 ROI"의 기각이다.** 표본 절반이 pose 커버리지 70% 미만인
   상태로 측정됐으므로 ROI 개념 전체의 사형선고는 아님. 단 encoder가 attention으로 localization을
   암묵 처리하므로 별도 ROI 트랙을 더 팔 이유는 현재 없음.

## Encoder Probe Result 2026-06-10

Escalation encoder probe 실행 완료. `torchvision` pretrained `r3d_18` Kinetics-400 frozen embedding을
419개 round3 라벨에 추출하고, 동일 `group-kfold` 프로토콜에서 RF/logreg/SVM을 비교했다.

- 산출물:
  - `scripts/build_video_encoder_dataset.py`
  - `tests/unit/test_build_video_encoder_dataset.py`
  - `data/video_encoder_dataset/gcs_r3d18_k400_v1` (single clip, feature_dim 512)
  - `data/video_encoder_dataset/gcs_r3d18_k400_4clip_ms` (4 clips mean+std, feature_dim 1024)
  - `data/fusion_dataset/gcs_flow_round3_r3d18_4clip_ms` (round3 flow + encoder, feature_dim 1070)
  - `models/reports/video_encoder_gcs_r3d18_k400_probe_summary.json`
  - `models/reports/fusion_gcs_flow_round3_r3d18_4clip_ms_probe_summary.json`
- Encoder-only 결과:
  - single clip best: logreg balanced accuracy `0.5777`
  - 4-clip mean+std best: SVM balanced accuracy `0.6301`
  - baseline round3 RF `0.8449` 대비 큰 폭 하락.
- 회귀 비교:
  - 4-clip SVM은 round3 miss 65건 중 36건을 회복했지만 새 오답 126건을 만들었다.
  - 4-clip logreg는 고확신 miss 20건 중 10건을 회복했지만 새 오답 137건을 만들었다.
- Fusion 결과:
  - flow round3 + encoder 4clip RF balanced accuracy `0.8065`, dynamic recall `0.8465`,
    static specificity `0.7667`.
  - round3 flow RF보다 낮고, 특히 static specificity가 크게 악화됐다.
- 최종 판정: **현재 r3d_18 K400 frozen embedding은 승격하지 않는다.** hard case를 일부 맞추는 보조
  신호는 있지만 새 오답 비용이 너무 커서 MVP 경로가 아니다. 더 강한 VideoMAE/DINOv2/X3D 또는
  fine-tuning은 별도 post-MVP 연구 트랙으로만 둔다.

### Review 2026-06-10 (오케스트레이터 사후 검토) — **MVP 판정 승인. 단, "encoder 가설 기각"으로 읽지 말 것**

수치 재검증 일치. MVP 승격 불가 판정은 명백히 타당 (encoder-only 0.63, fusion 0.81 + specificity 악화).

**그러나 이 probe는 encoder 가설의 가장 약한 형태를 테스트했다.** 결과를 깎은 설계 요인 3가지:

1. **r3d_18은 최약체 후보** — 2018년 18-layer, K400. VideoMAE/DINOv2 대비 수 세대 전 표현.
2. **uniform clip sampling이 이전 진단과 같은 함정에 빠짐** — 4클립×16프레임은 영상의 수 초만 보는데
   균등 샘플링이라 **다이나믹 burst가 클립에 안 들어갈 확률이 높다**. "짧은 다이나믹 구간이 희석된다"는
   flow 시절 진단이 클립 선택에서 그대로 재현된 것. 재시도 시 **burst-guided sampling**
   (기존 flow burst 피크 시점에서 클립 추출 — 인프라 재사용) 필수.
3. **128×171 입력에서 와이드샷 클라이머는 수 픽셀** — K400은 피사체가 화면을 채우는 영상으로 학습됨.
   person-crop 전처리 (YOLO류) 결합 필요 — ROI 트랙의 인사이트와 합류 지점.

**보존할 양성 신호 (처음 나온 것):** 고확신 miss 20건 중 **8~9건 회복** — flow가 원리적으로 못 보는
케이스를 표현 교체로 처음 맞췄다. 특히 `IMG_8449`(pose 완전 실패 영상)와 `IMG_0035`/`IMG_7864`
(백다이노/수직다이노) 회복. 모델별 회복 stem 합집합이 넓다 = 임베딩에 상보적 신호 존재.
**"방향은 맞고 도구가 약했다"가 정확한 결론.**

**post-MVP 연구 트랙 사양 (재시도 시):** VideoMAE-base 또는 DINOv2 + burst-guided clip sampling +
person-crop 전처리 + (선택) 마지막 블록 fine-tune. 성공 기준 동일: gk 0.88+ & 고확신 5건+ 회복,
신규 오답 ≤ 회복 수.

**MVP 잔여 기간 권장 (모델 트랙 종료):** 운영 v2 게이트 유지 확정. 남은 시간은
(a) 게이트 on E2E 재검증, (b) uncertain 구간(prob 0.3~0.7) 유보 UX — 제품 측 95% 경로,
(c) segment 기술 분석 품질 (사용자 체감 본체)로 전환.

---

## Current Facts To Preserve

- **현재 최고 baseline (round3):** `models/reports/flow_gcs_reviewed_round3_v1_metrics.json`
  - samples `419` (static 210 / dynamic 209), feature_dim `46`
  - RF group-kfold balanced accuracy `0.8449`, dynamic recall `0.8469`, static specificity `0.8429`
  - dataset: `data/flow_dataset/gcs_flow_v1_reviewed_round3/`
  - 라벨 SSOT: `data/review/labels_gcs_flow_reviewed_round3.csv`
  - 원본 영상 캐시: `data/gcs_cache/videos/original/` (425개, ~24GiB)
- **라벨 리뷰 소진:** round1~3 완료, 남은 review 0건. 라벨 정제로 더 얻을 게 없다.
- **분류기 capacity probe 음성 (2026-06-10, 재실험 금지):** 같은 419/group-kfold에서
  HGB `0.8348`, HGB 튜닝 `0.8271`, RF+HGB+LR soft voting `0.8493(±0.0485 노이즈)`.
  miss 65건 중 고확신 오답(≥0.85) 31% — feature가 신호를 못 보는 케이스.
  → **병목은 feature. 분류기/하이퍼파라미터/앙상블 변경 금지.**
- **운영 게이트:** `app/services/vision/flow_gate.py`. artifact `models/flow_qa_rf_v2.joblib`(42-dim legacy, bal_acc `0.8381`, static specificity `0.8615`).
  v3(46-dim) / GCS round3 artifact는 **승격 보류** — 사유: static specificity가 v2보다 낮음
  ([[30_Decisions/2026-06-10-hola-gcs-flow-not-promoted]]).
- **Encoder probe 음성 (2026-06-10):** `torchvision r3d_18` K400 frozen embedding은 encoder-only best
  `0.6301`, flow+encoder fusion RF `0.8065`로 round3 flow `0.8449`보다 낮다. 운영/승격 비대상.
- **miss 패턴 (리뷰에서 검증됨):**
  - FN 주범: 수직다이노·백다이노 (화면 변위 작거나 수직 이동)
  - FP 주범: 낙하·발 슬립 (아래 방향 단발 burst)
  - → magnitude는 둘을 구분 못 함. **vy 부호가 가른다.**
- **낙하 트리밍:** `trim_fall_segment` + `remove_fall_end(tail_ratio=0.25)` 이미 구현됨 (magnitude 기반).
- **vault 계획 문서:** `/Users/minjoun/Documents/DevKnowledge/10_Projects/hola-climbing-ai/2026-06-10-flow-accuracy-improvement-plan.md`

## 회귀 셋 (모든 Task 공통 검증)

`models/reports/flow_gcs_reviewed_round3_v1_predictions.csv`의 group-kfold rf miss **65건**이 고정 회귀 셋.
그중 **고확신 오답 20건** (라벨 방향 반대로 prob ≥ 0.85)이 핵심 추적 대상 — Task별로 몇 건 회복되는지 센다.

```python
# 고확신 miss 추출 스니펫
import csv
rows = [r for r in csv.DictReader(open('models/reports/flow_gcs_reviewed_round3_v1_predictions.csv'))
        if r['split']=='group-kfold' and r['model']=='rf' and r['label']!=r['pred']]
hi = [r for r in rows if (float(r['prob_dynamic']) if r['label']=='0' else 1-float(r['prob_dynamic'])) >= 0.85]
```

## Next Session Kickoff

Minjoun이 "진행해" / "이 plan 실행해줘"라고 하면:

1. 본 plan 읽기.
2. `hola-ai-orchestrator` + `superpowers:executing-plans` (또는 subagent-driven-development) 사용.
3. 아래 Preflight 실행.
4. `Encoder Probe Result 2026-06-10`까지 반영된 상태라면 Task 3 full ROI v5와 `r3d_18` encoder 승격은 모두 보류하고, 운영 artifact는 `flow_qa_rf_v2`를 유지.

Preflight:

```bash
pwd   # /Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-ai
git status --short
ls data/flow_dataset/gcs_flow_v1_reviewed_round3 | wc -l   # 419
ls data/gcs_cache/videos/original | wc -l                  # 425
uv run pytest tests/unit/test_flow_features.py tests/unit/test_flow_gate.py -q
python3 - <<'PY'
import json
m = json.load(open("models/reports/flow_gcs_reviewed_round3_v1_metrics.json"))
print(m["models"]["rf"]["group-kfold"]["aggregate_valid"]["balanced_accuracy_mean"])  # 0.8449
PY
```

---

## Task 1 — 방향 분해: vy 시계열 추출 (반나절, ROI 없이 단독)

`extract_flow_magnitude`가 magnitude만 반환하는 것을 (magnitude, vy) 2채널로 확장.

**⚠ 부호 규약 (반드시 문서화 + 테스트):** OpenCV 이미지 좌표는 **+y가 아래**.
`flow[..., 1].mean() > 0` = 아래 방향 이동 = **낙하**, `< 0` = 위 방향 = **다이노/상승**.

- [x] 1.1 `tests/unit/test_flow_features.py`에 vy 추출 테스트 추가 (합성 프레임: 아래로 이동하는 패턴 → vy > 0, 위로 → vy < 0)
- [x] 1.2 `extract_flow_magnitude` → `extract_flow_series`로 확장: 반환 `(NDArray[(T,2)], src_fps, duration)` (ch0=magnitude, ch1=vy). 기존 함수는 ch0만 반환하는 thin wrapper로 호환 유지
- [x] 1.3 vy 통계 추가: `_extract_flow_stats`에 vy 채널 feature 블록
  - 권장: `vy_mean, vy_std, vy_min, vy_max, vy_p10, vy_p90`, 상승 burst (`vy < -k·|vy|중앙값`) 횟수/지속비율, 하강 burst 횟수/지속비율, `max_upward_window_mean` (2초 윈도우)
  - `FLOW_FEATURE_VERSION = "flow_v4"`, `FLOW_FEATURE_DIM` 갱신 (46 + 추가분). legacy 42/46 경로는 보존
- [ ] 1.4 (선택) `trim_fall_segment`를 vy 기반으로 강화: 끝 구간에서 `vy > 0` 큰 burst = 낙하 확정 신호. magnitude-only 버전과 분리해 두고 ablation 가능하게
- [x] 1.5 `ruff` + `mypy` + 전체 단위 테스트 통과

## Task 2 — v4 dataset 재생성 + 재학습 + 회귀 평가

- [x] 2.1 dataset 생성 (영상 재처리 ~35분):
  ```bash
  uv run python scripts/build_flow_dataset.py \
    --labels data/review/labels_gcs_flow_reviewed_round3.csv \
    --videos-dir data/gcs_cache/videos/original \
    --out data/flow_dataset/gcs_flow_v4_direction
  ```
  (빌더가 새 extractor를 쓰도록 수정 포함. exclude 6건이 라벨 CSV에서 이미 빠져 있는지 확인 — 419개 나와야 함)
- [x] 2.2 학습/평가 (프로토콜 동결):
  ```bash
  uv run python scripts/train_tabular_dynamic_static.py \
    --data data/flow_dataset/gcs_flow_v4_direction \
    --out models/flow_gcs_v4_direction_rf.joblib \
    --run-name flow_gcs_v4_direction \
    --splits holdout,kfold,group-kfold
  ```
- [x] 2.3 회귀 셋 평가: round3 miss 65건 (고확신 20건) 중 회복 수 집계. 새 predictions와 round3 predictions를 stem 기준 join
- [x] 2.4 **Decision gate:**
  - group-kfold bal_acc **≥ 0.86** → Task 3 진행 (ROI로 추가 상승 노림)
  - **0.85 미만 + 고확신 miss 회복 < 5건** → Task 3을 건너뛰고 Escalation (pretrained encoder probe) 검토를 Minjoun에게 보고
  - 결과와 무관하게 수치를 vault 세션 로그에 기록

## Task 3 — pose bbox ROI + 배경 차감 + 스케일 정규화 (보류: static gate 불통과)

- [ ] 3.1 bbox 추출: `extract_pose_landmarks` 결과에서 프레임별 visible landmark min/max + 마진 25%. **pose 캐시(`data/pose_dataset/`)는 128프레임 리샘플이라 flow 프레임과 정렬 불가 — 빌드 시 새로 추출** (425개 1회성)
  - pose 소실 프레임: 직전 bbox 유지. 연속 30프레임(1초@30fps) 소실 시 해당 구간 전역 flow fallback
- [ ] 3.2 ROI flow: Farneback 결과에서 bbox 내부 픽셀만 평균 (magnitude + vy)
- [ ] 3.3 카메라 모션 차감: bbox **외부** 영역 median flow(dx, dy)를 카메라 모션 근사로 ROI flow에서 벡터 차감
- [ ] 3.4 스케일 정규화: flow 값을 몸통 픽셀 길이(어깨 중점~골반 중점)로 나눔. 몸통 길이도 pose 소실 시 직전 값 유지
- [ ] 3.5 dataset `gcs_flow_v5_roi` 생성 → Task 2와 동일 프로토콜 학습/평가/회귀 추적
- [ ] 3.6 ablation 표: v4(방향만) vs v5(방향+ROI) vs v5 variants(차감/정규화 on/off) — 어느 요소가 기여했는지 분리

## Task 4 — 운영 승격 검토 (성능 달성 시만)

**승격 기준 (둘 다 충족):** group-kfold bal_acc > 0.86 **그리고** static specificity ≥ 0.86 (v2의 0.8615 동등 이상 — 이전 보류 사유가 specificity였음)

- [ ] 4.1 `flow_gate.predict_prob_dynamic`의 feature_dim 스위치에 v4/v5 차원 분기 추가 (legacy 42 / v3 46 패턴 그대로)
- [ ] 4.2 Task 3 채택 시: orchestrator가 pose 결과(bbox)를 게이트에 전달하는 인터페이스 변경 + 추론 비용 재측정
- [ ] 4.3 Dockerfile COPY artifact 교체 + `.gitignore` 예외 갱신 + docker-compose `FLOW_GATE_MODEL_PATH` 갱신
- [ ] 4.4 vault: 승격 결정을 `30_Decisions/`에 기록 (기존 `2026-06-10-hola-gcs-flow-not-promoted` 갱신 또는 신규)

## Escalation (Task 2/3 모두 0.86 미달 시)

**2026-06-10 static gate 결과로 활성화했고, encoder probe까지 실행 완료.** ROI full v5와 `r3d_18`
frozen embedding 모두 승격하지 않는다.

진행한 probe:
- [x] 419 round3 라벨에 `torchvision r3d_18` K400 single-clip / 4-clip mean+std embedding 추출
- [x] 동일 419 라벨 + group-kfold + RF/logreg/SVM 평가
- [x] flow round3 + encoder fusion 평가

남은 선택지는 현재 MVP 범위 밖이다:
- VideoMAE/X3D/DINOv2처럼 더 강한 표현을 쓰거나 fine-tuning한다.
- 단, 지금 데이터 419개에서는 과적합/운영 비용이 커서 별도 post-MVP 연구 트랙으로 분리한다.

## 금지 사항

- 분류기 교체/튜닝/앙상블 실험 (capacity probe 음성 — Facts 참조)
- 추가 라벨 리뷰 요청 (round3에서 소진)
- 평가 프로토콜 변경 (group-kfold seed/folds 동결 — 비교 가능성 유지)
- 운영 artifact(`flow_qa_rf_v2.joblib`) 무단 교체 — Task 4 기준 통과 시만
