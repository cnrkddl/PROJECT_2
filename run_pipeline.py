import os
import sys

# 프로젝트의 src 폴더를 파이썬 경로에 추가하여 모듈 임포트 가능하도록 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocessing.split_media import extract_audio, detect_and_split_scenes
from src.analysis.vision_analyzer import VisionAnalyzer
from src.analysis.gemini_matcher import GeminiAdMatcher
from src.analysis.audio_analyzer import AudioAnalyzer
from src.analysis.timetable_generator import DynamicTimetableGenerator

def run_contextual_ad_pipeline(video_file_path: str):
    """
    모든 파이프라인(전처리 -> Vision -> Gemini Vision -> Audio STT -> Gemini Timetable)을
    순서대로 자동 실행하는 마스터(Runner) 스크립트입니다.
    """
    if not os.path.exists(video_file_path):
        print(f"❌ 오류: 입력하신 비디오 파일을 찾을 수 없습니다. 경로를 확인해주세요: {video_file_path}")
        return

    print("="*60)
    print(f"🚀 [Contextual Video Ad Pipeline] 전체 자동화 시작")
    print(f"▶️ 대상 파일: {video_file_path}")
    print("="*60)

    # -------------------------------------------------------------------------
    # 결과가 저장될 내부 경로 세팅
    # -------------------------------------------------------------------------
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    AUDIO_DIR = os.path.join(PROCESSED_DIR, "audio")
    SCENE_DIR = os.path.join(PROCESSED_DIR, "scenes")
    VISION_OUT_DIR = os.path.join(PROCESSED_DIR, "vision_results")
    
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SCENE_DIR, exist_ok=True)
    os.makedirs(VISION_OUT_DIR, exist_ok=True)

    # 파이프라인 파일명 지정
    output_audio = os.path.join(AUDIO_DIR, "extracted_audio.wav")
    yolo_candidates_csv = os.path.join(VISION_OUT_DIR, "all_scenes_candidates.csv")
    final_vision_csv = os.path.join(VISION_OUT_DIR, "ad_recommendations.csv")
    transcript_csv = os.path.join(AUDIO_DIR, "transcript.csv")
    final_timetable_csv = os.path.join(PROCESSED_DIR, "final_ad_timetable.csv")

    # -------------------------------------------------------------------------
    # STEP 1. 영상 전처리 (오디오 추출 및 씬 분할)
    # -------------------------------------------------------------------------
    print("\n[STEP 1/5] 🎬 원본 영상 전처리 진행 중...")
    extract_audio(video_file_path, output_audio)
    detect_and_split_scenes(video_file_path, SCENE_DIR)

    # -------------------------------------------------------------------------
    # STEP 2. 프레임별 객체 인식 (YOLO Vision)
    # -------------------------------------------------------------------------
    print("\n[STEP 2/5] 👁️ 영상 내 광고 가능 사물 인식(YOLO) 추출 중...")
    analyzer = VisionAnalyzer()
    all_candidates = []
    
    # 생성된 씬 폴더 안의 모든 mp4 스캔
    scene_files = [f for f in os.listdir(SCENE_DIR) if f.endswith(".mp4")]
    scene_files.sort()
    
    for scene_file in scene_files:
        test_path = os.path.join(SCENE_DIR, scene_file)
        top_candidates = analyzer.analyze_scene(test_path, VISION_OUT_DIR, sample_rate=15)
        if top_candidates:
            all_candidates.extend(top_candidates)
            
    # 후보군을 임시 CSV로 저장
    if all_candidates:
        import csv
        with open(yolo_candidates_csv, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_candidates[0].keys())
            writer.writeheader()
            writer.writerows(all_candidates)
    else:
        print("  -> 경고: 시각적 모델이 영상 내에서 광고할 만한 상품을 하나도 찾지 못했습니다.")

    # -------------------------------------------------------------------------
    # STEP 3. 객체별 키워드 속성 추출 (Gemini Vision)
    # -------------------------------------------------------------------------
    print("\n[STEP 3/5] 💎 시각 객체의 세부 디자인/재질 속성 분석 중...")
    if os.path.exists(yolo_candidates_csv):
        matcher = GeminiAdMatcher()
        matcher.process_candidates(yolo_candidates_csv, final_vision_csv)
    else:
        print("  -> YOLO 추출 결과가 없으므로 API Vision 단계를 통과합니다.")

    # -------------------------------------------------------------------------
    # STEP 4. 대사 텍스트 추출 (Whisper STT Audio)
    # -------------------------------------------------------------------------
    print("\n[STEP 4/5] 🗣️ 영상 대사(음성) 텍스트 추출 중 (Whisper)...")
    if os.path.exists(output_audio):
        audio_analyzer = AudioAnalyzer("base")
        audio_analyzer.extract_transcript(output_audio, transcript_csv)
    else:
        print("  -> 추출된 오디오 파일이 없어 STT 단계를 건너뜁니다.")

    # -------------------------------------------------------------------------
    # STEP 5. 맥락+사물 융합 최종 광고 타임테이블 제작 (Gemini LLM)
    # -------------------------------------------------------------------------
    print("\n[STEP 5/5] 🧠 멀티모달(대사+시각) 융합 최종 타임테이블 편성 중...")
    if os.path.exists(transcript_csv):
        table_generator = DynamicTimetableGenerator()
        table_generator.generate_timetable(transcript_csv, final_vision_csv, final_timetable_csv)
    else:
        print("  -> 정리된 대사 텍스트(STT)가 없으므로 타임테이블 스케줄링을 건너뜁니다.")

    print("\n" + "="*60)
    print("✅ 모든 파이프라인 처리가 성공적으로 완료되었습니다!")
    print(f"👉 시각속성 결과(VISION): {final_vision_csv}")
    print(f"👉 광고달력 결과(TIMETABLE): {final_timetable_csv}")
    print("="*60)

if __name__ == "__main__":
    # 터미널에서 `python run_pipeline.py filename.mp4` 와 같이 인자를 받아서 실행 가능
    if len(sys.argv) > 1:
        target_video = sys.argv[1]
    else:
        # 인자 없이 실행하면 기본 위치의 SampleVideo.mp4 실행
        target_video = os.path.join(BASE_DIR, "SampleVideo.mp4")
        
    run_contextual_ad_pipeline(target_video)
