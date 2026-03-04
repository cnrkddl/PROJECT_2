# Contextual Video Ad Insertion System (맥락 기반 영상 광고 자동 편성 시스템)

## 📌 프로젝트 개요
이 프로젝트는 단순히 영상 중간에 임의로 광고를 배치하는 것을 넘어, **AI가 영상의 흐름(대사)과 시각적 요소(등장 사물)를 통섭적으로 이해하여 가장 자연스럽고 효과적인 PPL/배너 광고를 자동 편성**하는 지능형 파이프라인 시스템입니다.

### 핵심 목적
1. **시청 경험 최적화 & 광고 효율 극대화**: 영상의 맥락과 동떨어진 무분별한 중간 광고 대신, 극중 상황(로맨스, 오피스 등)이나 장면에 등장하는 소품(가방, 카페 등)과 직접 이어지는 상품을 적절한 타이밍에 노출하여 사용자 거부감을 최소화합니다.
2. **멀티모달 AI(Vision + Audio + LLM) 융합**: 화면 속 객체 인식(YOLO)부터 객체 디자인/재질 분석(Gemini Vision), 대사 인식(Whisper STT), 그리고 최종 광고 타임테이블 판단(Gemini LLM)에 이르기까지 일련의 과정을 100% 자동화합니다.

---

## ⚙️ 주요 시스템 워크플로우 (Data Pipeline)

전체 자동화 마스터 스크립트인 `run_pipeline.py`를 통해 다음 5단계가 순차적으로 실행되며 분석 시점마다 **동적 폴더**(`data/processed/{영상명}_{시간}`)에 결과가 분리 저장됩니다.

1. **[STEP 1] 영상 전처리 (Preprocessing)**
   - `split_media.py`: 업로드된 원본 영상을 화면 전환(Scene) 단위로 분할(scenedetect)하고, 오디오(대사) 포맷을 분리 추출합니다.

2. **[STEP 2] 프레임별 객체 인식 (YOLO Vision)**
   - `vision_analyzer.py`: 분할된 각 씬(Scene)에서 YOLO 모델을 이용해 광고 가능한 사물(가방, 의자, 노트북 등)을 감지하고, 해당 객체가 가장 크게/선명하게 잡힌 프레임을 크롭(Crop)하여 저장합니다.

3. **[STEP 3] 객체별 키워드 속성 추출 (Gemini Vision)**
   - `gemini_matcher.py`: YOLO가 찾은 사물 크롭 이미지를 Gemini API(Vision)에 전달하여, 상품의 "대표 색상, 주요 재질, 디자인/스타일" 같은 세부 속성 키워드를 추출합니다.

4. **[STEP 4] 대사 텍스트 추출 (Whisper STT Audio)**
   - `audio_analyzer.py`: 영상에서 추출된 오디오 파일(.wav)을 Whisper 모델에 넣어 타임라인별 한국어 대사 텍스트(STT)를 추출합니다.

5. **[STEP 5] 맥락+사물 융합 최종 타임테이블 제작 (Gemini LLM)**
   - `timetable_generator.py`: 추출된 '씬별 시각(Vision) 속성 데이터'와 '타임라인별 대사(Audio) 트랜스크립트'를 한 번에 Gemini LLM에 전달하여, 영상의 맥락을 분석하고 어떤 시간에 어떤 형태의 광고(배너, 팝업 등)를 띄우는 것이 가장 자연스러운지 최종 `final_ad_timetable.csv`를 생성합니다.

---

## 🚀 기술 스택 및 차별점 (Core Strengths)

- **Vision**: `YOLOv8` (객체 인식), `Gemini Vision API` (객체 디테일 속성 분석)
- **Audio / NLP**: `OpenAI Whisper` (음성 인식/ASR)
- **LLM / Context**: `Gemini Flash API` (멀티모달 융합 및 타임테이블 스케줄링)
- **Media Processing**: `scenedetect`, `moviepy`, `OpenCV`
- **Output Management**: 매 분석마다 `데이터/시간` 기반의 고유한 결과 폴더 생성을 통해 멀티태스킹 파이프라인 관리

단순한 키워드 매칭이 아니라, 영상 전체의 "분위기(Context)"와 "화면 속 상품(Vision)"이 결합된 종합 광고 편성 AI 기술이라는 점이 가장 큰 차별점입니다.

---

## 📂 폴더 구조 및 모듈 안내

```text
PROJECT_2/
├── README.md              # 프로젝트 소개 (현재 파일)
├── run_pipeline.py        # [마스터 파일] 전체 파이프라인 1~5단계 순차 실행 스크립트
├── data/                  # 데이터 저장소
│   └── processed/         # 처리된 비디오/오디오/분석 결과가 {비디오명}_{타임스탬프} 폴더 단위로 누적
├── src/                   # 메인 소스코드
│   ├── preprocessing/     # 비디오 분할 및 오디오 추출 모듈 (split_media.py)
│   └── analysis/          # 핵심 AI 분석 로직
│       ├── vision_analyzer.py       # (STEP 2) YOLO 기반 객체 크롭
│       ├── gemini_matcher.py        # (STEP 3) Gemini Vision 키워드 추출
│       ├── audio_analyzer.py        # (STEP 4) Whisper STT 음성 인식
│       └── timetable_generator.py   # (STEP 5) 멀티모달 최종 편성 스케줄링
└── SampleVideo.mp4        # 테스트용 메인 영상 파일 (기본값)
```

## 🛠 실행 방법

터미널에서 아래 명령을 실행하면 `SampleVideo.mp4`를 기준으로 전체 파이프라인이 자동 실행됩니다.
```bash
python run_pipeline.py          # 기본 영상으로 실행
python run_pipeline.py test.mp4 # 다른 영상 지정해서 실행
```
모든 결과는 `data/processed/{영상명}_{생성시간}/` 폴더에 깔끔하게 자동 정리됩니다.
