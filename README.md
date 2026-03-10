# Contextual Video Ad Insertion System (맥락 기반 영상 광고 자동 편성 시스템)

## 📌 프로젝트 개요
드라마/VOD 영상을 분석하여 맥락에 맞는 광고를 자동으로 삽입하는 **멀티모달 AI 파이프라인**입니다.
YOLO-World(시각)로 화면 속 사물을 탐지하고, Gemini Vision으로 장면을 자연어로 묘사한 뒤,
실제 DB의 광고 이름(ad_inventory.ad_name)과 **코사인 유사도**로 매칭합니다.
librosa 묵음 감지와 씬 전환 타이밍을 결합하여 시청 경험을 최대한 방해하지 않는 시점에 광고를 편성합니다.

### 핵심 차별점
1. **실존 광고만 추천**: Gemini가 광고를 "상상"하는 방식을 제거하고, PostgreSQL DB의 `ad_inventory` 테이블에 실제로 존재하는 광고만 추천합니다.
2. **코사인 유사도 매칭**: 장면 설명 텍스트와 `ad_name`을 Gemini `text-embedding-004`로 임베딩하여 의미론적 유사도로 매칭합니다.
3. **타이밍 3조건**: 묵음 2초↑ / 씬 전환 후 3초↓ / 오브젝트 밀도 0.3↓ — 3조건 중 2개 이상 충족 시에만 광고 삽입 후보로 등록합니다.
4. **원클릭 마스터 스크립트**: 영상 경로만 넘기면 6단계 파이프라인이 자동으로 돌아갑니다.

---

## ⚙️ 파이프라인 워크플로우

`run_pipeline.py`를 실행하면 아래 6단계가 순차적으로 실행됩니다.

1. **[STEP 1] 영상 전처리**
   - `split_media.py`: scenedetect로 씬 분할 + ffmpeg으로 오디오 추출
   - 씬별 시작/종료 시간을 `scene_timestamps.csv`로 저장 (타이밍 조건에 활용)

2. **[STEP 2] 객체 탐지 — YOLO-World**
   - `vision_analyzer.py`: YOLO-World(`yolov8s-worldv2.pt`)로 씬별 사물 탐지
   - COCO 80개 제한 없이 의류·가구·뷰티 등 텍스트로 클래스 자유 지정
   - `scene_start_sec + (frame_count / fps)` 방식으로 정확한 절대 타임스탬프 계산
   - 결과: `all_scenes_candidates.csv`

3. **[STEP 3] 장면 설명 생성 — Gemini Vision**
   - `gemini_matcher.py`: 씬 대표 크롭 이미지 + 감지 객체 목록을 Gemini에 전달
   - "이 사물이 등장하는 장면 어떤 상황이야?" → 자연어 장면 설명 텍스트 생성
   - 오브젝트 밀도(0~1) 계산 — 타이밍 조건 3에 활용
   - 결과: `scene_descriptions.csv`

4. **[STEP 4] 대사 추출 — Whisper STT**
   - `audio_analyzer.py`: Whisper로 오디오 → 타임라인별 대사 텍스트 추출
   - 결과: `transcript.csv`

5. **[STEP 5] 광고 매칭 + 타임테이블 편성**
   - `timetable_generator.py`:
     - PostgreSQL `ad_inventory` DB에서 광고 목록(ad_name) 로드
     - Gemini `text-embedding-004`로 장면 설명 + 광고 이름 임베딩
     - 코사인 유사도 행렬로 씬별 최적 광고 매칭
     - librosa로 묵음 구간 감지 (2초 이상)
     - 타이밍 3조건 체크 → 2개 이상 충족 시 타임테이블 등록
   - 결과: `final_ad_timetable.csv`

6. **[STEP 6] 배너 이미지 생성 — 나노바나나**
   - `nanobanana_generator.py`: 타임테이블 기반 광고 배너 이미지 자동 렌더링
   - `--skip-ad` 옵션으로 건너뛸 수 있음

---

## 🚀 환경 설정

### 1. API 키 설정
프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 입력하세요.
```env
GEMINI_API_KEY=발급받은_Gemini_API_키
```

### 2. DB 연결
PostgreSQL DB에 `ad_inventory` 테이블이 있어야 합니다.
```
postgresql://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME
```

### 3. 의존 패키지
```bash
pip install ultralytics google-genai psycopg2-binary librosa scikit-learn python-dotenv
```

---

## 🛠 실행 방법

```bash
# 기본 샘플 영상(바탕화면의 SampleVideo.mp4)으로 실행
python run_pipeline.py

# 특정 영상 파일 지정
python run_pipeline.py 드라마영상.mp4

# 배너 이미지 생성 단계 건너뜀
python run_pipeline.py 드라마영상.mp4 --skip-ad
```

**결과물 저장 위치**
`data/processed/{영상파일명}_{날짜시간}/` 폴더에 모든 결과가 저장됩니다.

---

## 📂 프로젝트 구조

```text
PROJECT_2/
├── run_pipeline.py        # 파이프라인 원클릭 실행 마스터 스크립트
├── .env                   # Gemini API 키 (git 제외)
├── .gitignore
├── README.md
│
├── data/
│   └── processed/
│       └── {영상명}_{날짜시간}/
│           ├── audio/
│           │   ├── extracted_audio.wav
│           │   └── transcript.csv          # Whisper STT 결과
│           ├── vision_results/
│           │   ├── all_scenes_candidates.csv   # YOLO-World 탐지 결과
│           │   ├── scene_timestamps.csv         # 씬별 시작/종료 시간
│           │   └── scene_descriptions.csv       # Gemini 장면 설명 + 오브젝트 밀도
│           ├── generated_ad_banners/        # 나노바나나 배너 이미지
│           └── final_ad_timetable.csv       # 🎉 최종 광고 타임테이블
│
└── src/
    ├── preprocessing/
    │   └── split_media.py
    └── analysis/
        ├── vision_analyzer.py       # YOLO-World 객체 탐지
        ├── gemini_matcher.py        # Gemini 장면 설명 생성
        ├── audio_analyzer.py        # Whisper STT
        ├── timetable_generator.py   # 코사인 유사도 매칭 + 타이밍 로직
        └── nanobanana_generator.py  # 배너 이미지 생성
```
