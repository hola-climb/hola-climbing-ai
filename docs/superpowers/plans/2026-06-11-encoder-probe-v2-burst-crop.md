# Encoder Probe v2 — Burst-Guided Sampling + Person-Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** r3d_18 1차 probe의 3중 핸디캡(약한 encoder / uniform sampling / 와이드샷 미크롭)을 각각 해소한 encoder probe 재실행. **post-MVP 연구 트랙** — MVP(2026-06-25) 전 운영 변경 없음, 운영은 `flow_qa_rf_v2` 유지.

**Architecture:** `scripts/build_video_encoder_dataset.py`를 확장 (신규 파일 아님). 학습/평가 프로토콜(`train_tabular_dynamic_static.py`, 419 라벨, group-kfold seed)은 동결. ablation A0→A2 순서로 진행해 **어느 수정이 효과를 내는지 분리**한다.

**Tech Stack:** Python, PyTorch (MPS), torchvision (r3d_18 + Faster R-CNN person detection), HuggingFace transformers (VideoMAE, DINOv2), OpenCV, scikit-learn.

---

## 1차 probe 결과 (비교 기준)

| 구성 | gk bal_acc | 고확신 회복 /20 | 신규 오답 |
|---|---|---|---|
| r3d_18 uniform 4-clip (1차) | 0.6301 (SVM) | 8~9 | 132~147 |
| flow round3 RF (baseline) | **0.8449** | — | — |

1차 probe의 양성 신호: 고확신 miss 8~9건 회복 (`IMG_8449`, `IMG_0035`, `IMG_7864` 포함) — 표현 교체가 hard case를 본다는 첫 증거. 상세: `docs/superpowers/plans/2026-06-10-roi-flow-direction-decomposition.md`의 Encoder Probe Review.

## 성공 기준 (사전 등록 — 변경 금지)

- **승격 검토 기준**: group-kfold bal_acc **≥ 0.88** AND 고확신 20건 중 **5건+ 회복** AND 신규 오답 ≤ 회복 수
- **중간 기준**: encoder-only ≥ 0.85면 flow와의 stacking 실험 추가 가치 있음
- **기각 기준**: A0~A2 전부 encoder-only < 0.75면 frozen embedding 트랙 종료, fine-tuning 또는 제품 측 보완(유보 UX)으로 전환

---

## 정확한 모델 명세

### M1. VideoMAE (1순위)

```python
# HuggingFace transformers
from transformers import VideoMAEImageProcessor, VideoMAEModel
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"   # ViT-B, K400 fine-tuned
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEModel.from_pretrained(MODEL_ID).eval()
# 입력: 16 frames × 224×224 RGB (processor가 정규화 처리)
# 출력: last_hidden_state (B, 1568, 768) → dim=1 mean-pool → 768-dim clip embedding
```
- ⚠ pre-train 전용 `MCG-NJU/videomae-base`가 아니라 **kinetics fine-tuned** 체크포인트를 쓸 것 (frozen feature의 의미성이 다름)
- 클립당 16프레임, 클립 4개 → 영상 embedding = 클립 4개의 mean+std concat = **1536-dim**

### M2. DINOv2 (2순위, CPU 친화)

```python
from transformers import AutoImageProcessor, AutoModel
MODEL_ID = "facebook/dinov2-base"    # ViT-B/14
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).eval()
# 입력: 프레임별 224×224 (14의 배수). 출력: pooler_output 또는 CLS 768-dim
# 클립(=윈도우)당 8프레임 CLS mean → 768 클립 embedding
# 영상 = 클립 4개 mean+std concat = 1536-dim
```

### M3. r3d_18 (A0 ablation 전용 — 기존 코드 재사용)

`make_torchvision_embedding_fn(encoder_model="r3d_18")` 그대로. 바뀌는 건 sampling/crop만.

### 디바이스

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```
425영상 × 4클립, VideoMAE-base MPS 기준 수 시간. 임베딩은 npz 캐시 (1회성).

### 의존성 추가

```toml
# pyproject.toml [dependency-groups] ml 에 추가
"transformers>=4.41",
```
`uv sync --group ml` 후 진행. **main 의존성에 넣지 말 것** (운영 이미지 불변).

---

## Burst-Guided Clip Sampling (정확한 알고리즘)

uniform 대신 flow burst 피크에서 클립을 뽑는다. 기존 `app/services/vision/flow_features.py` 재사용:

```
입력: video_path, num_clips=4, clip_span_sec=2.0
1. flow_mag, src_fps, duration = extract_flow_magnitude(video_path)   # 기존 함수, 30fps 정규화
2. smoothed = savgol 또는 moving average(window=15)                    # 기존 _smooth 재사용
3. win = int(clip_span_sec * 30)  # 30fps 기준 60 샘플
4. 각 시점 t의 window_mean[t] = smoothed[t:t+win].mean()
5. 비최대 억제(NMS)로 겹치지 않는 top-(num_clips-1) 윈도우 선택:
   - 최고 window_mean 시점 선택 → 그 ±win 구간 마스킹 → 반복
6. 마지막 1클립은 영상 전체 uniform 중앙 클립 (전역 컨텍스트 보존)
7. 각 윈도우를 영상 시간으로 환산해 frames_per_clip 프레임을 윈도우 안에서 균등 샘플
출력: list[clip_frame_indices]
```

- 영상이 짧아 (duration < num_clips × clip_span) 윈도우가 모자라면 uniform fallback
- 함수명 제안: `sample_burst_guided_clips(video_path, *, num_clips, frames_per_clip, clip_span_sec) -> list[NDArray]`
- TDD: 합성 flow(평탄 + burst 2개)로 선택 윈도우가 burst 위치와 일치하는지 / NMS 겹침 없는지

## Person-Crop 전처리 (정확한 알고리즘)

```python
# torchvision 내장 — 신규 의존성 없음
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
detector = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()
# COCO person = label 1
```

```
클립당 1회 detection (클립 중앙 프레임):
1. 중앙 프레임 → detector → label==1 & score≥0.5 박스들
2. 박스 선택: 면적 최대 (climber가 주 피사체 가정. 다인 영상에서 오선택 가능 — 한계로 기록)
3. 마진 30% 확장 + 정사각형화 (긴 변 기준) + 프레임 경계 clamp
4. 해당 클립의 모든 프레임을 같은 박스로 crop → 224×224 resize
5. detection 실패 시: 직전 클립 박스 재사용 → 그것도 없으면 full frame fallback
   (fallback 여부를 manifest에 기록 — 분석 시 분리 가능하게)
```

---

## Task 순서 (ablation으로 요인 분리)

### Task 0 — Preflight

```bash
pwd  # hola-climbing-ai
ls data/flow_dataset/gcs_flow_v1_reviewed_round3 | wc -l   # 419
ls data/gcs_cache/videos/original | wc -l                  # 425
uv sync --group ml && uv run python -c "import torch; print(torch.backends.mps.is_available())"
# transformers 추가 후: uv run python -c "from transformers import VideoMAEModel; print('ok')"
```

### Task 1 — burst-guided sampling + person-crop 구현 (TDD)

- [x] `sample_burst_guided_clips` 구현 + 합성 flow 테스트 (`tests/unit/test_build_video_encoder_dataset.py` 확장)
- [x] `make_person_crop_fn` 구현 + 합성 프레임 테스트 (박스 마진/clamp/fallback)
- [x] `build_video_encoder_dataset.py` CLI에 `--sampling burst|uniform`, `--person-crop` 플래그 추가
- [x] reuse 안전장치 추가: 기존 `.npz`의 encoder/weights/num_frames/num_clips/sampling/person_crop metadata가 현재 실행 조건과 일치할 때만 재사용

### Task 2 — A0: r3d_18 + burst + crop (요인 분리의 핵심, 가장 싸다)

- [x] dataset 생성:
  ```bash
  uv run python scripts/build_video_encoder_dataset.py \
    --labels data/review/labels_gcs_flow_reviewed_round3.csv \
    --videos-dir data/gcs_cache/videos/original \
    --encoder-model r3d_18 --num-clips 4 --sampling burst --person-crop \
    --out data/video_encoder_dataset/gcs_r3d18_burst_crop
  ```
- [x] 학습/평가 (동일 프로토콜) + 회귀 비교 (round3 miss 65 / 고확신 20)
- [x] **판독**: A0가 0.63 → 0.75+ 점프하면 sampling/crop이 주범이었다는 증거 → M1/M2 기대 상향.
  A0가 0.65 부근 정체면 encoder 표현력 자체가 병목 → M1 결과가 결정적

### Task 3 — A1: VideoMAE + burst + crop

- [x] `make_videomae_embedding_fn` 추가 (M1 명세 그대로)
- [x] dataset `gcs_videomae_burst_crop` 생성 → 학습/평가/회귀
- [ ] (선택) A1-nocrop 1회 — crop 기여도 분리. A1 본 실험이 0.85 미만이라 생략

### Task 4 — A2: DINOv2 + burst + crop (A1이 0.85 미만일 때만)

- [x] `make_dinov2_embedding_fn` 추가 (M2 명세)
- [x] dataset `gcs_dinov2_burst_crop` 생성 → 학습/평가/회귀

### Task 5 — 판정 + 기록

- [x] 성공 기준 대조표 작성 (A0/A1/A2 × bal_acc/고확신회복/신규오답)
- [x] encoder-only ≥0.85 달성 시: flow round3 prob + encoder prob 2-feature logistic stacking 실험 1회. 조건 미달로 생략
- [x] vault `30_Decisions/` 결정 기록 + 본 plan에 Execution Result 섹션 추가
- [x] 어느 구성도 기각 기준이면: frozen 트랙 종료 선언, 제품 측 유보 UX를 공식 경로로

## Execution Result 2026-06-11

419개 round3 라벨(`static=210`, `dynamic=209`)과 동일 `group-kfold` 프로토콜로 실행했다. 운영 artifact는 변경하지 않았고, 비교 기준은 `flow_gcs_reviewed_round3_v1` RF balanced accuracy `0.8449`, baseline miss 65건, 고확신 baseline miss 20건이다.

| Ablation | best model | gk bal_acc | dynamic recall | static specificity | 고확신 회복 /20 | 회복 | 신규 오답 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 `r3d_18` + burst + crop | SVM | `0.6940` | `0.6880` | `0.7000` | 11 | 43 | 106 | 0.75 미만 |
| A1 VideoMAE + burst + crop | logreg | `0.6849` | `0.6602` | `0.7095` | 10 | 36 | 103 | 0.75 미만 |
| A2 DINOv2 + burst + crop | logreg | `0.6135` | `0.6413` | `0.5857` | 9 | 33 | 130 | 0.75 미만 |

Dataset 생성 결과:

| Dataset | rows | num_frames | clips | crop fallback |
|---|---:|---:|---:|---:|
| `gcs_r3d18_burst_crop` | 419 | 16 | 1676 | 668 (`39.9%`) |
| `gcs_videomae_burst_crop` | 419 | 16 | 1676 | 668 (`39.9%`) |
| `gcs_dinov2_burst_crop` | 419 | 8 | 1676 | 653 (`39.0%`) |

Artifacts:

- Dataset: `data/video_encoder_dataset/gcs_r3d18_burst_crop`, `data/video_encoder_dataset/gcs_videomae_burst_crop`, `data/video_encoder_dataset/gcs_dinov2_burst_crop`
- Metrics: `models/reports/video_encoder_gcs_*_burst_crop_metrics.json`
- Regression summaries: `models/reports/video_encoder_gcs_*_burst_crop_probe_summary.json`
- 통합 요약: `models/reports/video_encoder_burst_crop_v2_summary.json`

판정:

- 세 encoder 모두 encoder-only balanced accuracy가 `0.75` 미만이라 사전 등록 기각 기준에 해당한다.
- 고확신 miss는 9~11건 회복하지만 신규 오답이 103~130건으로 회복 수보다 훨씬 많다.
- encoder-only `0.85` 이상 조건을 만족한 구성이 없어 flow+encoder stacking은 실행하지 않았다.
- frozen pretrained embedding 트랙은 MVP 경로에서 종료한다. 다음 정확도 개선은 fine-tuning 또는 제품 측 유보 UX/재촬영 UX가 우선이다.

## Review 2026-06-11 (오케스트레이터 사후 검토) — 종료 판정 승인 + 원인 확정 + 노이즈 정정

**판정 승인.** 추가로 마지막 남은 카드까지 데이터로 닫았다:

**① 선택적 rescue stacking — 음성 (이번 검토에서 계산).**
"flow가 static 확신(prob<0.3)할 때만 encoder 의견으로 뒤집기" 규칙을 round3 flow RF × A0 SVM 예측으로 전수 시뮬레이션:

| encoder 임계 | dyn FN 회복 | static 신규오답 | net |
|---|---|---|---|
| 0.6 | 2 | 14 | **−12** |
| 0.7 | 2 | 4 | −2 |
| 0.8 | 1 | 0 | +1 |

원인: flow-static-확신 구간에서 encoder prob 중앙값이 진짜 dynamic 0.49 vs 진짜 static 0.44 — **필요한 곳에서 무정보**.
→ **1차 probe의 "고확신 miss 8~11건 회복" 해석을 정정한다: 직교 신호가 아니라 노이즈 분류기의 우연 적중이었다.**

**② 왜 안 됐나 — 원인 확정:**
- burst+crop은 진짜 결함이었음 (A0: 0.63→0.694, +6.4%p) — 단, 부차 요인
- **encoder 용량은 병목이 아님**: VideoMAE(0.6849) ≈ r3d_18(0.6940). 강한 표현이 0 추가 이득
- 남는 설명: K400 frozen 임베딩의 분산은 장면/외형/행동카테고리에 쓰이는데, 암장 영상은 장면이 모두 비슷해
  필요한 신호(미세한 시간적 움직임 품질)가 nuisance(클라이머 외형·벽·카메라)에 묻힘.
  **419개 영상 단위 binary 라벨로는 그 부분공간을 읽어낼 수 없다**
- crop fallback 39.9% — person detection이 클립 2/5에서 실패 (와이드샷/블러). crop 수정도 절반만 작동
- **근본 원인 (flow→encoder 전체 아크 공통): supervision 부족.** flow가 0.84인 이유는 학습 없이
  대상량을 직접 측정하기 때문. "학습이 필요한" 모든 접근은 419 비트에서 죽는다

**③ 유보(abstention) 천장 — round3 419개 재계산:**

| 유보 구간 | 커버리지 | 노출 정확도 |
|---|---|---|
| 0.25~0.75 | 65.4% | **90.5%** |
| 0.20~0.80 | 56.8% | 90.8% |

→ 제품 측 유보 UX의 정확한 기대치: **노출 정확도 ~90.5%, 유보율 ~35%**. 95%는 현 데이터 체제에서 불가.

**④ 남은 대안 (우선순위):**
1. **Track A (MVP, 즉시)**: 유보 UX — `prob 0.25~0.75` 구간을 "혼합/판정 보류"로 노출. Spring/프론트 작업
2. **Track B (분석율의 다음 실질 점프)**: **segment-level 라벨** — supervision 체제 전환 (419 비트 → 수천 구간 라벨).
   6기술 GT 확보 겸용 (사용자 체감 본체). 이후 flow+pose 시계열 위 소형 temporal 모델로 재도전
3. **Track C (조건부 보류)**: fine-tuning — 재방문 조건: segment 라벨 확보 **또는** 영상 1000개+. 그 전에는 금지

## 금지 사항

- 분류기/하이퍼파라미터 실험 (capacity probe 음성 — 2026-06-10 확정)
- 라벨 변경/재리뷰 (round3 소진)
- 평가 프로토콜 변경 (419 라벨, group-kfold seed 동결)
- 운영 artifact/게이트 변경 — 이 plan 전체가 오프라인. 승격은 별도 결정 필요
- MVP(6/25) 전 착수 — 운영 안정화·E2E 검증·유보 UX가 우선

## 참고

- 1차 probe + 전체 소거 이력: `docs/superpowers/plans/2026-06-10-roi-flow-direction-decomposition.md`
- vault: `10_Projects/hola-climbing-ai/2026-06-10-flow-accuracy-improvement-plan.md`
- 결정: `30_Decisions/2026-06-10-hola-r3d18-encoder-probe-not-promoted.md`
- 실행 환경: 영상 디코드 + torch 추론은 샌드박스 밖에서 (MediaPipe는 본 plan에서 불필요)
