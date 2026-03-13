# Contextual Video Ad Insertion System (맥락 기반 영상 광고 자동 편성 시스템)

## 📌 프로젝트 개요
드라마/VOD 영상을 분석하여 맥락에 맞는 광고를 자동으로 삽입하는 **멀티모달 AI 파이프라인**입니다.
YOLO-World(시각)로 화면 속 사물을 탐지하고, Gemini Vision으로 장면을 자연어로 묘사한 뒤,
실제 DB의 광고 이름(`ad_inventory.ad_name`)과 **코사인 유사도**로 매칭합니다.
librosa 묵음 감지와 씬 전환 타이밍을 결합하여 시청 경험을 최대한 방해하지 않는 시점에 광고를 편성합니다.

### 핵심 차별점
1. **실존 광고만 추천**: Gemini가 광고를 "상상"하는 방식을 제거하고, PostgreSQL DB의 `ad_inventory` 테이블에 실제로 존재하는 광고만 추천합니다.
2. **코사인 유사도 매칭**: 장면 설명 텍스트와 `ad_name`을 Gemini `gemini-embedding-001`로 임베딩하여 의미론적 유사도로 매칭합니다.
3. **타이밍 3조건**: 묵음 2초↑ / 씬 전환 후 3초↓ / 오브젝트 밀도 0.3↓ — 조건 중 `MIN_CONDITIONS`개 이상 충족 시에만 광고 삽입 후보로 등록합니다. (드라마처럼 묵음이 적은 영상은 `MIN_CONDITIONS=1` 권장)
4. **YOLO-World 오픈 보캐블러리**: COCO 80개 클래스 제한 없이 의류·가구·뷰티 등 텍스트로 탐지 대상을 자유롭게 지정합니다.
5. **원클릭 마스터 스크립트**: 영상 경로만 넘기면 5단계 파이프라인이 자동으로 돌아갑니다.

---

## ⚙️ 파이프라인 워크플로우

`run_pipeline.py`를 실행하면 아래 5단계가 순차적으로 실행됩니다.

### [STEP 1] 영상 전처리
- `split_media.py`: scenedetect로 씬 분할 + ffmpeg으로 오디오 추출
- 씬별 시작/종료 시간을 `scene_timestamps.csv`로 저장
  → STEP 2 타임스탬프 계산 및 STEP 5 씬 전환 타이밍 조건에 활용

### [STEP 2] 객체 탐지 — YOLO-World
- `vision_analyzer.py`: `yolov8s-worldv2.pt` + `set_classes(WORLD_CLASSES)` 로 씬별 사물 탐지
- 탐지 대상 클래스: COCO 기본 클래스 + 의류(dress, shoes...), 뷰티(cosmetics, perfume...), 가구(lamp, cabinet...) 등 확장
- 타임스탬프 계산: `scene_start_sec + (frame_count / fps)` — 원본 영상 기준 절대 시간
- 결과: `all_scenes_candidates.csv`

**Gemini 호출 전 필터링 기준값** (`gemini_matcher.py`에서 조정 가능):
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `MIN_APPEARANCES` | 3 | 최소 등장 프레임 수 |
| `MIN_AREA_RATIO_PCT` | 1.5 | 최소 화면 비중 (%) |
| `MIN_SCORE` | 0.003 | 최소 광고 적합도 점수 |
| `TOP_PER_SCENE` | 3 | 씬당 최대 처리 객체 수 |
| `TOP_GLOBAL` | 15 | 전체 최대 Gemini 호출 수 |

### [STEP 3] 장면 설명 생성 — Gemini Vision
- `gemini_matcher.py` (`GeminiSceneDescriber`): 씬 대표 크롭 이미지 + 감지 객체 목록 → Gemini에 전달
- "이 사물이 등장하는 장면 어떤 상황이야?" → 한국어 자연어 장면 설명 텍스트 생성
- 씬 내 객체들의 화면 비중 합산으로 **오브젝트 밀도** 계산 (0~1 범위)
- 결과: `scene_descriptions.csv` (씬이름 / 등장시간 / 감지사물목록 / 오브젝트밀도 / 장면설명)

### [STEP 4] 대사 추출 — Whisper STT
- `audio_analyzer.py`: Whisper(`base` 모델)로 오디오 → 타임라인별 대사 텍스트 추출
- 결과: `transcript.csv`

### [STEP 5] 광고 매칭 + 타임테이블 편성
- `timetable_generator.py` (`AdTimetableGenerator`):
  1. `src/database/ad_inventory.py`로 PostgreSQL DB에서 광고 목록 로드
  2. Gemini `gemini-embedding-001`로 장면 설명 + 광고 이름 전체 임베딩
  3. 코사인 유사도 행렬 계산 → 씬별 유사도 최상위 광고 선택
  4. librosa로 오디오 묵음 구간(2초↑) 감지
  5. 타이밍 3조건 체크 → `MIN_CONDITIONS`개 이상 충족 시 타임테이블 등록
  6. fallback 장면 설명("화면에 ~이(가) 등장하는 장면") 자동 제외 — 의미 없는 임베딩으로 인한 오매칭 방지
  7. 광고 진입/종료 시간을 MM:SS (HH:MM:SS) 포맷으로 저장
- 결과: `final_ad_timetable.csv`

**타이밍 기준값** (`timetable_generator.py`에서 조정 가능):
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `MIN_SILENCE_SEC` | 2.0 | 묵음 최소 길이 (초) |
| `SILENCE_TOP_DB` | 40 | librosa 묵음 민감도 (dB) |
| `SCENE_TRANSITION_SEC` | 3.0 | 씬 전환 후 허용 시간 (초) |
| `MAX_OBJECT_DENSITY` | 0.3 | 오브젝트 밀도 상한 |
| `MIN_CONDITIONS` | 1 | 최소 충족 조건 수 (드라마 권장: 1) |
| `MIN_SIMILARITY` | 0.25 | 코사인 유사도 하한 |

### [STEP 6] webapp — 실제 광고 에셋 서빙
- `final_ad_timetable.csv`의 `매칭 광고 ID`로 DB의 `resource_path`를 조회하여 실제 광고 파일을 직접 재생
- 별도의 배너 생성 없이 DB에 등록된 원본 광고 에셋을 사용

---

## 🗄 DB 연결 정보

- 연결: `src/database/ad_inventory.py`에서 관리
- DSN: `postgresql://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME`
- 사용 테이블: `ad_inventory` (197개 광고)

**ad_inventory 주요 컬럼:**
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `ad_id` | VARCHAR(200) | 광고 고유 ID |
| `ad_name` | TEXT | 매칭 기준 텍스트 (예: "CJ제일제당 - 햇반") |
| `ad_type` | VARCHAR(50) | `video_clip` \| `banner` |
| `resource_path` | TEXT | 광고 파일 경로 |
| `duration_sec` | DOUBLE | 광고 길이 (초) |

---

## 🚀 환경 설정

### 1. API 키 / DB 설정
프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 입력하세요.
```env
GEMINI_API_KEY=발급받은_Gemini_API_키
DB_DSN=postgresql://유저:비밀번호@호스트:포트/DB명
```

### 2. 의존 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 🛠 실행 방법

```bash
# 기본 샘플 영상(바탕화면의 SampleVideo.mp4)으로 실행
python run_pipeline.py

# 특정 영상 파일 지정
python run_pipeline.py 드라마영상.mp4
```

**결과물 저장 위치**
`data/processed/{영상파일명}_{날짜시간}/` 폴더에 모든 결과가 저장됩니다.

---

## 📂 프로젝트 구조

```text
PROJECT_2/
├── run_pipeline.py             # 파이프라인 원클릭 실행 마스터 스크립트
├── .env                        # Gemini API 키 (git 제외)
├── .gitignore
├── README.md
│
├── data/
│   └── processed/
│       └── {영상명}_{날짜시간}/
│           ├── audio/
│           │   ├── extracted_audio.wav
│           │   └── transcript.csv               # Whisper STT 결과
│           ├── vision_results/
│           │   ├── all_scenes_candidates.csv    # YOLO-World 탐지 결과
│           │   ├── scene_timestamps.csv          # 씬별 시작/종료 시간
│           │   └── scene_descriptions.csv        # Gemini 장면 설명 + 오브젝트 밀도
│           ├── generated_ad_banners/             # 나노바나나 배너 이미지
│           └── final_ad_timetable.csv            # 🎉 최종 광고 타임테이블
│
└── src/
    ├── preprocessing/
    │   └── split_media.py                        # 씬 분할 + 오디오 추출
    ├── database/
    │   └── ad_inventory.py                       # PostgreSQL DB 연결 및 광고 목록 조회
    ├── analysis/
    │   ├── vision_analyzer.py                    # YOLO-World 객체 탐지 + 타임스탬프
    │   ├── gemini_matcher.py                     # Gemini Vision 장면 설명 생성
    │   ├── audio_analyzer.py                     # Whisper STT
    │   └── timetable_generator.py                # 코사인 유사도 매칭 + 타이밍 로직
    └── webapp/
        ├── app.py                                # Flask 웹 서버
        ├── templates/index.html
        └── static/
            ├── script.js                         # 광고 오버레이 + 타임라인 렌더링
            └── style.css
```
