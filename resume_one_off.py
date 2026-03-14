import os
import sys
import shutil

# 프로젝트의 루트 폴더를 BASE_DIR로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.analysis.gemini_matcher import GeminiSceneDescriber
from src.analysis.audio_analyzer import AudioAnalyzer
from src.analysis.timetable_generator import AdTimetableGenerator


def resume_pipeline(run_folder_name: str):
    print("="*60)
    print(f"🔄 [Resume Pipeline] {run_folder_name} 이어하기 시작")
    print("="*60)

    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", run_folder_name)
    AUDIO_DIR = os.path.join(PROCESSED_DIR, "audio")
    VISION_OUT_DIR = os.path.join(PROCESSED_DIR, "vision_results")

    # 파이프라인 파일명 지정
    output_audio          = os.path.join(AUDIO_DIR,      "extracted_audio.wav")
    yolo_candidates_csv   = os.path.join(VISION_OUT_DIR, "all_scenes_candidates.csv")
    scene_timestamps_csv  = os.path.join(VISION_OUT_DIR, "scene_timestamps.csv")
    scene_descriptions_csv = os.path.join(VISION_OUT_DIR, "scene_descriptions.csv")
    transcript_csv        = os.path.join(AUDIO_DIR,      "transcript.csv")
    final_timetable_csv   = os.path.join(PROCESSED_DIR,  "final_ad_timetable.csv")

    # -------------------------------------------------------------------------
    # STEP 3. 장면 설명 생성 (Gemini Vision)
    # -------------------------------------------------------------------------
    print("\n[STEP 3/5] 💎 Gemini Vision 장면 설명 생성 중...")
    if os.path.exists(yolo_candidates_csv):
        describer = GeminiSceneDescriber()
        describer.process_candidates(yolo_candidates_csv, scene_descriptions_csv)

        print("  -> 🗑️  분석 완료된 크롭 이미지 삭제 중...")
        for entry in os.listdir(VISION_OUT_DIR):
            entry_path = os.path.join(VISION_OUT_DIR, entry)
            if os.path.isdir(entry_path) and entry.startswith("crops_"):
                shutil.rmtree(entry_path)
                print(f"  -> 크롭 폴더 삭제 완료: {entry_path}")
    else:
        print("  -> YOLO 추출 결과가 없으므로 장면 설명 생성을 건너뜁니다.")

    # -------------------------------------------------------------------------
    # STEP 4. 대사 텍스트 추출 (Whisper STT Audio)
    # -------------------------------------------------------------------------
    if not os.path.exists(transcript_csv):
        print("\n[STEP 4/5] 🗣️ 영상 대사(음성) 텍스트 추출 중 (Whisper)...")
        if os.path.exists(output_audio):
            audio_analyzer = AudioAnalyzer("base")
            audio_analyzer.extract_transcript(output_audio, transcript_csv)
        else:
            print("  -> 추출된 오디오 파일이 없어 STT 단계를 건너뜁니다.")
    else:
        print("\n[SKIP] Step 4 결과가 이미 존재하므로 건너뜁니다.")

    # -------------------------------------------------------------------------
    # STEP 5. 광고 매칭 + 타임테이블 생성 (DB 코사인 유사도 매칭)
    # -------------------------------------------------------------------------
    if not os.path.exists(final_timetable_csv):
        print("\n[STEP 5/5] 🧠 광고 매칭 및 타임테이블 편성 중...")
        if os.path.exists(scene_descriptions_csv):
            table_generator = AdTimetableGenerator()
            table_generator.generate_timetable(
                scene_descriptions_csv=scene_descriptions_csv,
                transcript_csv=transcript_csv,
                scene_timestamps_csv=scene_timestamps_csv,
                audio_path=output_audio,
                output_csv_path=final_timetable_csv,
            )
        else:
            print("  -> 장면 설명 파일이 없으므로 타임테이블 생성을 건너뜁니다.")
    else:
        print("\n[SKIP] Step 5 결과가 이미 존재하므로 건너뜁니다.")

    print("\n" + "="*60)
    print("✅ 이어하기 처리가 완료되었습니다!")
    print(f"👉 장면 설명 결과(VISION): {scene_descriptions_csv}")
    print(f"👉 광고 타임테이블(TIMETABLE): {final_timetable_csv}")
    print("="*60)


if __name__ == "__main__":
    RUN_FOLDER = "언더커버 미쓰홍.E16.260308.720p-NEXT_20260309_204753"
    resume_pipeline(RUN_FOLDER)
