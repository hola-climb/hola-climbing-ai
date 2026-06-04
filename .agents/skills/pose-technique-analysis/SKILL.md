---
name: pose-technique-analysis
description: "MediaPipe Pose로 클라이밍 영상의 33개 키포인트를 추출하고, 규칙 기반 분류기로 하이스텝·플래깅·훅(토/힐)·락오프·다이노·코디네이션 6가지 기술을 구간별로 분류한다. 출력은 Spring의 AnalysisIngestRequest.segments[] 형식. 'pose 추출', '클라이밍 기술 인식', '동작 분석', 'MediaPipe', '구간 분할', '하이스텝/플래깅/훅/락오프/다이노/코디네이션 감지' 요청 시 반드시 사용."
---

# Pose & Technique Analysis

클라이밍 영상에서 **키포인트 추출 + 6가지 기술 구간 분류 + Spring 호환 segments 출력**을 수행하는 스킬. vision-engineer 전용.

## 언제 사용하는가

- 영상 프레임 시퀀스로부터 33개 키포인트 추출
- 시계열에서 6가지 클라이밍 기술 감지 + 구간(start/end ms) 산출
- 기술별 confidence 산출
- `is_dynamic` 플래그 결정 (다이노/코디네이션은 true)
- Spring `POST /api/analysis/videos/{videoId}` 콜백의 `segments` 필드를 채울 데이터 생성

## 핵심 원칙

1. **휴리스틱 우선.** 4주 데드라인. MediaPipe 키포인트 + 각도/속도 규칙으로 MVP. 정확도 한계 시에만 PyTorch 분류기 검토 (architect 협의).
2. **결정론적.** 같은 영상 = 같은 segments. 비결정성 도입 금지.
3. **임계값은 사용자(도메인 전문가) 검토 대상.** 모든 임계값은 `_workspace/02_vision_technique_rules.md` 에 표로 명문화. 사용자가 실제 영상으로 검증 후 조정.
4. **Spring schema 강제.** 출력의 모든 필드명·타입은 `AnalysisSegmentPayload` 와 완전 일치 (이 문서 끝의 "Spring 호환 출력 schema" 참조).

## 분류할 6가지 기술

| 기술 | 영문 라벨 (`technique` 필드 값) | `is_dynamic` | 핵심 단서 |
|------|--------------------------------|-------------|----------|
| 하이스텝 | `high_step` | false | 무릎이 골반 위 (정적) |
| 플래깅 | `flagging` | false | 한 발이 반대편으로 (정적) |
| 훅 | `toe_hook` 또는 `heel_hook` | false | 발끝/발뒤꿈치를 홀드에 거는 자세 (정적) |
| 락오프 | `lock_off` | false | 한 손 정지 + 팔꿈치 굴곡 (정적) |
| 다이노 | `dyno` | **true** | 양 발이 동시에 벽에서 이탈 (동적) |
| 코디네이션 | `coordination` | **true** | 손/발 다발 동시 이동 (동적, 다이노 제외) |

> `technique` 값은 위 영문 snake_case 7종 중 하나. Spring 측은 enum이 아니라 자유 문자열로 수신하지만, 우리는 위 목록을 strictly 사용.

## MediaPipe Pose 통합

### 키포인트 인덱스 (33개 중 사용하는 것)

| 인덱스 | 부위 | 인덱스 | 부위 |
|--------|------|--------|------|
| 0 | nose | 23 | left_hip |
| 11 | left_shoulder | 24 | right_hip |
| 12 | right_shoulder | 25 | left_knee |
| 13 | left_elbow | 26 | right_knee |
| 14 | right_elbow | 27 | left_ankle |
| 15 | left_wrist | 28 | right_ankle |
| 16 | right_wrist | 29 | left_heel |
| | | 30 | right_heel |
| | | 31 | left_foot_index |
| | | 32 | right_foot_index |

### 초기화 (영상 모드)

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

### 프레임 처리

```python
def extract_landmarks(frames):
    for idx, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            yield idx, [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
        else:
            yield idx, None
```

## 6가지 기술 규칙 (정밀화)

> 모든 좌표는 MediaPipe 정규화 좌표 (`0~1`). `y`는 위쪽이 작음.
> 모든 임계값은 `_workspace/02_vision_technique_rules.md` 에 표로 저장 → 사용자 검토.

### 공통 유틸

```python
def body_height(lm):
    """nose-ankle 거리. 정규화 신체 크기 계산용."""
    nose_y = lm[0][1]
    ankle_y = min(lm[27][1], lm[28][1])  # 더 아래(=큰 y)
    return abs(ankle_y - nose_y)

def velocity(lm_prev, lm_curr, idx):
    if lm_prev is None or lm_curr is None: return None
    dx = lm_curr[idx][0] - lm_prev[idx][0]
    dy = lm_curr[idx][1] - lm_prev[idx][1]
    return (dx**2 + dy**2) ** 0.5

def angle(a, b, c):
    """세 점이 이루는 각도 (b가 꼭짓점). 도 단위."""
    import math
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    cos = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        (ba[0]**2+ba[1]**2)**0.5 * (bc[0]**2+bc[1]**2)**0.5 + 1e-9)
    return math.degrees(math.acos(max(-1, min(1, cos))))
```

### 1. 하이스텝 (`high_step`, `is_dynamic=false`)

**기준:** 무릎이 골반보다 신체 키의 일정 비율 이상 위로 올라가 있는 정적 구간.

| 파라미터 | 값 | 사유 |
|---------|-----|------|
| `knee_above_hip_ratio` | 0.10 (신체 키 대비) | 평균 클라이밍 하이스텝은 무릎이 골반 위 10~20cm. 신체 키 1.7m 기준 약 6~12% → 보수적으로 10% |
| `min_duration_frames` | 30 (1초 @ 30fps) | 순간적 무릎 들기는 일반 보행. 1초 이상 유지 → 하이스텝 의도 |
| `ankle_visibility_min` | 0.5 | 가시성 낮으면 false positive 증가 |
| `confidence_base` | 0.7 | 휴리스틱 기본 신뢰도 |

```python
def detect_high_step(seq, fps=30, body_h_avg=0.5):
    HIP_L, HIP_R, KNEE_L, KNEE_R, ANK_L, ANK_R = 23, 24, 25, 26, 27, 28
    threshold = 0.10 * body_h_avg
    events = []
    for side, hip, knee, ank in [("L", HIP_L, KNEE_L, ANK_L), ("R", HIP_R, KNEE_R, ANK_R)]:
        streak = 0
        for idx, lm in enumerate(seq):
            if lm is None or lm[ank][3] < 0.5:
                if streak >= 30:
                    events.append(_make_segment("high_step", idx - streak, idx, fps, conf=0.7))
                streak = 0
                continue
            if lm[knee][1] < lm[hip][1] - threshold:
                streak += 1
            else:
                if streak >= 30:
                    events.append(_make_segment("high_step", idx - streak, idx, fps, conf=0.7))
                streak = 0
    return events
```

### 2. 플래깅 (`flagging`, `is_dynamic=false`)

**기준:** 한 발의 x 좌표가 골반 중심선을 기준으로 반대편 다리와 같은 쪽(X자 교차) 또는 한 발이 들려 다른 발이 반대편으로.

| 파라미터 | 값 | 사유 |
|---------|-----|------|
| `crossing_threshold` | 0.02 (정규화 x 차이) | 양 발이 골반 중심선에서 같은 쪽일 때 (정상은 반대) |
| `min_duration_frames` | 20 (0.67초) | 짧게도 가능. 매치/풋체인지보다는 길게 |
| `confidence_base` | 0.6 | 가장 false positive 많은 케이스 |

```python
def detect_flagging(seq, fps=30):
    HIP_L, HIP_R, ANK_L, ANK_R = 23, 24, 27, 28
    events, streak, start_idx = [], 0, 0
    for idx, lm in enumerate(seq):
        if lm is None: streak = 0; continue
        hip_cx = (lm[HIP_L][0] + lm[HIP_R][0]) / 2
        l_side = lm[ANK_L][0] - hip_cx  # 음수 정상
        r_side = lm[ANK_R][0] - hip_cx  # 양수 정상
        # 둘 다 같은 부호 = 한쪽으로 다리가 몰림 → 플래깅
        crossed = (l_side > 0.02 and r_side > 0.02) or (l_side < -0.02 and r_side < -0.02)
        if crossed:
            if streak == 0: start_idx = idx
            streak += 1
        else:
            if streak >= 20:
                events.append(_make_segment("flagging", start_idx, idx, fps, conf=0.6))
            streak = 0
    return events
```

### 3. 훅 (`toe_hook` / `heel_hook`, `is_dynamic=false`)

**기준:** 발끝(31/32) 또는 발뒤꿈치(29/30)가 발목(27/28)보다 위쪽으로 회전. 즉 발이 거꾸로 잡혀있는 자세.

| 파라미터 | 값 | 사유 |
|---------|-----|------|
| `toe_above_ankle_y` | -0.04 (정규화) | 토훅: foot_index.y < ankle.y - 0.04 (발끝이 위) |
| `heel_above_toe_y` | -0.03 | 힐훅: heel.y < foot_index.y - 0.03 (뒤꿈치가 앞끝보다 위) |
| `min_duration_frames` | 25 (0.83초) | 훅은 정적 유지가 핵심 |
| `confidence_base` | 0.65 | 정확한 식별은 ML 필요. 휴리스틱은 신호만 |

```python
def detect_hook(seq, fps=30):
    pairs = [("L", 27, 29, 31), ("R", 28, 30, 32)]  # ankle, heel, toe
    events = []
    for side, ank, heel, toe in pairs:
        for hook_type, predicate in [
            ("toe_hook",  lambda lm: lm[toe][1]  < lm[ank][1] - 0.04),
            ("heel_hook", lambda lm: lm[heel][1] < lm[toe][1] - 0.03),
        ]:
            streak, start_idx = 0, 0
            for idx, lm in enumerate(seq):
                if lm is None or lm[ank][3] < 0.5: streak = 0; continue
                if predicate(lm):
                    if streak == 0: start_idx = idx
                    streak += 1
                else:
                    if streak >= 25:
                        events.append(_make_segment(hook_type, start_idx, idx, fps, conf=0.65))
                    streak = 0
    return events
```

### 4. 락오프 (`lock_off`, `is_dynamic=false`)

**기준:** 한 손이 정지(속도 ~0) + 그 손의 팔꿈치가 굴곡(각도 ≤ 90°) + 다른 손은 이동 또는 자유.

| 파라미터 | 값 | 사유 |
|---------|-----|------|
| `static_wrist_max_speed` | 0.005 / frame | 거의 정지 |
| `moving_wrist_min_speed` | 0.015 | 다른 손은 활발 |
| `elbow_angle_max` | 95° | 90° 이하면 명확. 95°로 약간 여유 |
| `min_duration_frames` | 15 (0.5초) | 짧아도 의미 있음 |
| `confidence_base` | 0.7 | |

```python
def detect_lock_off(seq, fps=30):
    SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
    events = []
    for static, moving, sh, el, wr in [("L", "R", SH_L, EL_L, WR_L),
                                        ("R", "L", SH_R, EL_R, WR_R)]:
        streak, start_idx = 0, 0
        for idx in range(1, len(seq)):
            prev, curr = seq[idx-1], seq[idx]
            if prev is None or curr is None: streak = 0; continue
            v_static = velocity(prev, curr, wr)
            v_moving = velocity(prev, curr, WR_L if moving == "L" else WR_R)
            elbow_ang = angle(curr[sh], curr[el], curr[wr])
            if v_static < 0.005 and v_moving > 0.015 and elbow_ang <= 95:
                if streak == 0: start_idx = idx
                streak += 1
            else:
                if streak >= 15:
                    events.append(_make_segment("lock_off", start_idx, idx, fps, conf=0.7))
                streak = 0
    return events
```

### 5. 다이노 (`dyno`, `is_dynamic=true`)

**기준:** 양 발목이 동시에 큰 y-방향 이동(상승) → 짧은 공중 구간 (visibility 일시 감소 또는 y가 위로 가속) → 다음 정지.

| 파라미터 | 값 | 사유 |
|---------|-----|------|
| `ankle_lift_speed` | 0.02 / frame | 발이 위로 빠르게 |
| `both_feet_window` | 5 frames (~0.17s) | 양 발이 거의 동시 |
| `airborne_min_frames` | 3 | 짧은 비행 |
| `confidence_base` | 0.75 | 다이노는 시각적으로 명확 |

```python
def detect_dyno(seq, fps=30):
    ANK_L, ANK_R = 27, 28
    events = []
    i = 1
    while i < len(seq):
        prev, curr = seq[i-1], seq[i]
        if prev is None or curr is None: i += 1; continue
        vL = (prev[ANK_L][1] - curr[ANK_L][1])  # 양수면 위로
        vR = (prev[ANK_R][1] - curr[ANK_R][1])
        if vL > 0.02 and vR > 0.02:
            # 양 발 동시 이탈 → 공중 구간 탐색
            start = i
            airborne = 0
            j = i
            while j < len(seq) and j < i + 30:
                lm = seq[j]
                if lm is None or lm[ANK_L][3] < 0.4 or lm[ANK_R][3] < 0.4:
                    airborne += 1
                else:
                    break
                j += 1
            if airborne >= 3:
                events.append(_make_segment("dyno", start, j, fps, conf=0.75, is_dynamic=True))
                i = j + 1
                continue
        i += 1
    return events
```

### 6. 코디네이션 (`coordination`, `is_dynamic=true`)

**기준:** 짧은 윈도우 내에 손/발 중 2개 이상의 키포인트가 동시에 큰 이동. 단, 다이노로 분류된 구간은 제외.

| 파라미터 | 값 | 사유 |
|---------|-----|------|
| `window_frames` | 15 (0.5s) | 짧은 시간 |
| `min_movers` | 2 (손 2 / 발 2 / 손+발 조합) | "여럿이 같이" |
| `mover_threshold_speed` | 0.025 | 충분히 큰 이동 |
| `confidence_base` | 0.55 | 가장 모호한 클래스. low |

```python
def detect_coordination(seq, fps=30, dyno_ranges=None):
    KEYS = [15, 16, 27, 28]  # wrists + ankles
    events = []
    for start in range(0, len(seq) - 15, 5):  # 5프레임씩 슬라이딩
        end = start + 15
        if dyno_ranges and any(r[0] <= start < r[1] for r in dyno_ranges):
            continue
        movers = 0
        for k in KEYS:
            max_v = 0
            for i in range(start + 1, end):
                v = velocity(seq[i-1], seq[i], k) if seq[i-1] and seq[i] else 0
                if v: max_v = max(max_v, v)
            if max_v > 0.025: movers += 1
        if movers >= 2:
            events.append(_make_segment("coordination", start, end, fps, conf=0.55, is_dynamic=True))
    return _merge_overlapping(events)
```

## 후처리

1. **중복 제거 + 병합** — 같은 기술의 연속 segment는 병합. 다른 기술 간 시간 겹침은 더 높은 confidence 유지.
2. **너무 짧은 segment 필터** — 50ms 이하는 제거.
3. **sequence_index 부여** — 시간순으로 0부터 정렬.

```python
def finalize_segments(all_events, fps):
    all_events.sort(key=lambda e: e["start_time_ms"])
    for i, e in enumerate(all_events):
        e["sequence_index"] = i
    return all_events
```

## Spring 호환 출력 schema

각 segment dict는 정확히 다음 키만 가져야 한다 (Spring `AnalysisSegmentPayload` 일치):

```python
{
    "sequence_index": int,     # 0부터 시계열 순
    "start_time_ms": int,      # 시작 ms
    "end_time_ms": int,        # 종료 ms
    "technique": str,          # 위 7개 라벨 중 하나
    "is_dynamic": bool,
    "confidence": float,       # 0.0~1.0
}
```

최종 콜백 페이로드 (vision-engineer는 segments만 채우고, 나머지는 pipeline/callback이 결합):

```python
{
    "status": "done",                       # 또는 "failed"
    "model_version": "rule_v1",             # 휴리스틱 버전 식별
    "segments": [seg1, seg2, ...]
}
```

`model_version` 핀: 휴리스틱 변경 시 버전 올림 (`rule_v1` → `rule_v2`). ML 도입 시 `lstm_v1` 같은 prefix 변경.

## 정확도 측정 프로토콜

라벨링 데이터로 검증 (qa-engineer 협업):

1. `/Users/minjoun/Workspace/projects/Hola-Climbing/labels.csv` 에서 라벨 채워진 행만 사용
2. 각 영상에 대해 모델 추론 → ground truth segments와 IoU 계산
3. 기술별 precision/recall + segment-level IoU 평균
4. `_workspace/02_vision_accuracy.md` 에 baseline 기록

## Apple Silicon 주의사항

- MediaPipe 0.10.14+ arm64 네이티브 휠 제공
- `opencv-python-headless` 사용 (GUI 의존성 제거)

## 산출물 체크리스트

- [ ] `_workspace/02_vision_pose_extractor.md` — MediaPipe 초기화 + 프레임 처리
- [ ] `_workspace/02_vision_technique_rules.md` — **6가지 규칙 + 임계값 표 (사용자 검토 마커 포함)**
- [ ] `_workspace/02_vision_classifier.md` — 분류기 구조 + Spring schema 출력
- [ ] `_workspace/02_vision_accuracy.md` — baseline 정확도
- [ ] 코드: `app/services/vision/` (pose.py, techniques/{high_step,flagging,hook,lock_off,dyno,coordination}.py, classifier.py)
