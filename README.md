# Contextual Video Ad Insertion System (맥락 기반 영상 광고 자동 편성 시스템)

## 📌 프로젝트 개요
수동으로 하드코딩된 PPL 배너를 배치하는 구시대적 방식을 넘어선 **진정한 멀티모달(시각+청각) AI 융합 파이프라인**입니다. 이 시스템은 비디오의 화면 속 객체(물건)와 주인공의 대사(맥락)를 동시에 분석하여 주인공이 화를 내고 있다면 '매운 떡볶이'를, 화면에 우드 체어가 잡히면 '프리미엄 원목 가구' 배너를 가장 자연스러운 타이밍(5~10초)에 팝업 시키는 **지능형 VOD 자동 편성 시스템**입니다.

### 핵심 차별점
1. **시청 경험 최적화 (5~10초 플래시 배너)**: 씬 전체 길이에 무작정 띄우지 않습니다. AI가 맥락이 바뀌는 임팩트 있는 시점을 잡아 5~10초만 광고를 짧게 노출하여 스팸을 방지합니다.
2. **Vision + Audio + LLM 융합**: YOLOv8(시각)으로 화면의 사물을 뽑고, Whisper(청각)로 대사를 뽑아, 최신 Gemini-Flash 모델이 "이 물건과 이 대사"에 맞는 최고의 광고 상품 스케줄표를 스스로 상상해서 짜냅니다.
3. **원클릭 마스터 스크립트 기반**: 영상만 던져주면 복잡한 5단계 파이프라인 모듈이 톱니바퀴처럼 맞물려 통합 편성본(`final_ad_timetable.csv`)을 뱉어냅니다.

---

## ⚙️ 주요 파이프라인 워크플로우 (Data Pipeline)

전체 자동화 마스터 스크립트인 `run_pipeline.py`를 실행하면 아래 5단계가 순차적으로 돌아갑니다.

1. **[STEP 1] 영상 전처리 (Preprocessing)**
   - `split_media.py`: 영상을 화면 전환(Scene) 단위로 분할(scenedetect)하고, 오디오 파일(`.wav`)을 분리 추출합니다.

2. **[STEP 2] 프레임별 타겟 사물 인식 (YOLO Vision)**
   - `vision_analyzer.py`: 분할된 씬에서 YOLOv8 모델을 이용해 광고 가능한 PPL 사물(가방, 의자 등)을 감지하고 해당 프레임을 크롭(Crop) 저장합니다.

3. **[STEP 3] 시각 사물의 세부 디자인 추출 (Gemini Vision)**
   - `gemini_matcher.py`: 크롭된 사물 이미지를 Gemini Vision API에 던져 상품의 "대표 색상, 주요 재질, 스타일" 같은 디테일 키워드(`ad_recommendations.csv`)를 뽑아냅니다.

4. **[STEP 4] 원본 대사 텍스트 추출 (Whisper STT Audio)**
   - `audio_analyzer.py`: 오디오 파일(.wav)을 Whisper 모델에 넣어 타임라인별 한국어 대사 텍스트(`transcript.csv`)를 완벽하게 추출합니다.

5. **[STEP 5] 맥락+사물 융합 스케줄링 (Gemini LLM Timetable)**
   - `timetable_generator.py`: '시각 키워드'와 '오디오 대사'를 하나로 묶어 Gemini에게 줍니다. "화면 속 우드 체어와 비자금 은닉 대사에 가장 알맞은 광고를 추천해!" 라고 지시해 최종 편성표(`final_ad_timetable.csv`)를 생성합니다.

---

## 🚀 필수 환경 설정 (Getting Started)

### 1. API 키 발급 및 보안 설정
이 프로젝트는 Gemini API를 백본으로 사용합니다. 깃허브 해킹 방지를 위해 `.env` 파일에 발급받은 키를 등록해야 합니다.

프로젝트 최상단 루트 디렉토리에 `.env` 파일을 만들고 아래 코드를 입력하세요.
```env
GEMINI_API_KEY=AIzaSy_여러분의_비밀_API_키를_여기에_붙여넣으세요
```

### 2. `.gitignore` 확인
대용량 멀티미디어 파일과 보안 파일이 올라가지 않도록 이미 `.gitignore`가 완벽하게 구성되어 있습니다 (`.env`, `*.mp4`, `*.wav`).

---

## 🛠 단 1줄로 실행하는 방법 (How to Run)

터미널에서 원클릭 마스터 스크립트 명령어를 실행하면 끝납니다!
```bash
# 기본 샘플 영상(SampleVideo.mp4)으로 테스트할 때
python run_pipeline.py          

# 새로운 드라마/유튜브 영상으로 테스트할 때
python run_pipeline.py 새로운_다운로드_영상.mp4 
```

**✅ 결과물 저장 위치**
분석이 끝나면 덮어쓰기 방지를 위해 `data/processed/{영상파일명_현재날짜시간}/` 이름으로 아주 깔끔한 독립 폴더가 생성되며, 그 내부에 모든 결과 CSV와 크롭 캡처 사진들이 보관됩니다!

---

## 📂 프로젝트 구조 (Directory Structure)

```text
PROJECT_2/
├── run_pipeline.py        # [마스터 봇] 파이프라인 원클릭 실행 스크립트
├── .env                   # [보안] 개인 API 키 보관소 (숨김 파일)
├── .gitignore             # [보안] 대용량 미디어 및 API 키 GitHub 업로드 방지
├── README.md              # 프로젝트 가이드
│
├── data/                  # [결과물] 분석 데이터 저장소
│   └── processed/         # 📁 생성된 타임스탬프 고유 폴더 (예: SampleVideo_20260304_153839)
│       ├── audio/             # 오디오 및 STT(transcript.csv) 저장
│       ├── scenes/            # 분할된 클립 영상 모음
│       ├── vision_results/    # 크롭된 캡처 사진들 및 (ad_recommendations.csv)
│       └── final_ad_timetable.csv  # 🎉 [최종 결과물] 광고 배치 스케줄 
│
└── src/                   # 메인 소스코드 (Modular)
    ├── preprocessing/     # 비디오/오디오 분리 모듈
    └── analysis/          # YOLO / Whisper / Gemini AI 로직 모음
```
