import os
import sys
import csv as _csv
import shutil
import datetime

# 프로젝트의 루트 폴더(run_pipeline.py가 위치한 폴더)를 BASE_DIR로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.preprocessing.split_media import extract_audio, detect_and_split_scenes
from src.analysis.vision_analyzer import VisionAnalyzer
from src.analysis.gemini_matcher import GeminiSceneDescriber
from src.analysis.audio_analyzer import AudioAnalyzer
from src.analysis.timetable_generator import AdTimetableGenerator
from src.analysis.nanobanana_generator import NanoBananaGenerator


def run_contextual_ad_pipeline(video_file_path: str, skip_ad_generation: bool = False):
    """
    전체 파이프라인을 순서대로 자동 실행하는 마스터 스크립트입니다.

    [파이프라인 흐름]
    STEP 1: 영상 전처리 — 씬 분할 + 오디오 추출 (scenedetect + ffmpeg)
    STEP 2: 객체 탐지  — YOLO-World로 씬별 사물 인식 + 정확한 타임스탬프 계산
    STEP 3: 장면 설명  — Gemini Vision으로 씬별 자연어 장면 설명 텍스트 생성
    STEP 4: 대사 추출  — Whisper STT로 음성을 텍스트로 변환
    STEP 5: 광고 매칭  — Gemini 임베딩 + 코사인 유사도로 ad_inventory 광고 매칭
                         librosa 묵음 감지 + 씬 전환 타이밍으로 삽입 시점 결정
    STEP 6: 배너 생성  — 나노바나나로 광고 배너 이미지 렌더링
    """
    if not os.path.exists(video_file_path):
        print(f"❌ 오류: 입력하신 비디오 파일을 찾을 수 없습니다: {video_file_path}")
        return

    print("="*60)
    print("🚀 [Contextual Video Ad Pipeline] 전체 자동화 시작")
    print(f"▶️  대상 파일: {video_file_path}")
    print("="*60)

    # ── 결과 저장 경로 세팅 ────────────────────────────────
    video_filename = os.path.splitext(os.path.basename(video_file_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"{video_filename}_{timestamp}"

    PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed", run_folder_name)
    AUDIO_DIR      = os.path.join(PROCESSED_DIR, "audio")
    SCENE_DIR      = os.path.join(PROCESSED_DIR, "scenes")
    VISION_OUT_DIR = os.path.join(PROCESSED_DIR, "vision_results")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SCENE_DIR, exist_ok=True)
    os.makedirs(VISION_OUT_DIR, exist_ok=True)

    # ── 파이프라인 파일명 정의 ─────────────────────────────
    output_audio          = os.path.join(AUDIO_DIR,      "extracted_audio.wav")
    yolo_candidates_csv   = os.path.join(VISION_OUT_DIR, "all_scenes_candidates.csv")
    scene_timestamps_csv  = os.path.join(VISION_OUT_DIR, "scene_timestamps.csv")
    scene_descriptions_csv = os.path.join(VISION_OUT_DIR, "scene_descriptions.csv")
    transcript_csv        = os.path.join(AUDIO_DIR,      "transcript.csv")
    final_timetable_csv   = os.path.join(PROCESSED_DIR,  "final_ad_timetable.csv")

    # ─────────────────────────────────────────────────────────
    # STEP 1. 영상 전처리 — 오디오 추출 + 씬 분할
    # ─────────────────────────────────────────────────────────
    print("\n[STEP 1/6] 🎬 원본 영상 전처리 진행 중...")
    extract_audio(video_file_path, output_audio)
    scene_list = detect_and_split_scenes(video_file_path, SCENE_DIR)

    # 씬별 시작/종료 시간을 CSV로 저장
    # — STEP 2에서 정확한 타임스탬프 계산에 사용
    # — STEP 5에서 씬 전환 타이밍 조건 체크에 사용
    if scene_list:
        # scenedetect는 총 씬 수 기준으로 자리수를 자동 결정 (1000개↑ → 4자리)
        digits = len(str(len(scene_list)))
        with open(scene_timestamps_csv, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = _csv.DictWriter(f, fieldnames=['씬 이름', '시작 시간 (초)', '종료 시간 (초)'])
            writer.writeheader()
            for i, (start_time, end_time) in enumerate(scene_list, start=1):
                writer.writerow({
                    '씬 이름':        f"scene-{i:0{digits}d}",
                    '시작 시간 (초)': round(start_time.get_seconds(), 2),
                    '종료 시간 (초)': round(end_time.get_seconds(), 2),
                })
        print(f"  -> 씬 타임스탬프 저장 완료: {scene_timestamps_csv}")

    # ─────────────────────────────────────────────────────────
    # STEP 2. 객체 탐지 — YOLO-World
    # ─────────────────────────────────────────────────────────
    print("\n[STEP 2/6] 👁️  YOLO-World 객체 탐지 중...")
    analyzer = VisionAnalyzer()
    all_candidates = []

    scene_files = sorted(f for f in os.listdir(SCENE_DIR) if f.endswith(".mp4"))

    # scene_list에서 씬 시작 시간(초)을 딕셔너리로 변환
    # — analyze_scene의 scene_start_sec 파라미터에 전달하여 정확한 타임스탬프 계산
    scene_start_sec_map = {}
    if scene_list:
        digits = len(str(len(scene_list)))
        for i, (start_time, _) in enumerate(scene_list, start=1):
            scene_name = f"scene-{i:0{digits}d}"
            scene_start_sec_map[scene_name] = round(start_time.get_seconds(), 2)

    for scene_file in scene_files:
        scene_name = os.path.splitext(scene_file)[0]
        test_path  = os.path.join(SCENE_DIR, scene_file)
        scene_start = scene_start_sec_map.get(scene_name, 0.0)

        candidates = analyzer.analyze_scene(
            test_path, VISION_OUT_DIR,
            sample_rate=15,
            scene_start_sec=scene_start,   # 원본 영상 기준 절대 시작 시간 전달
        )
        if candidates:
            all_candidates.extend(candidates)

    # YOLO 결과 임시 CSV 저장
    if all_candidates:
        with open(yolo_candidates_csv, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = _csv.DictWriter(f, fieldnames=all_candidates[0].keys())
            writer.writeheader()
            writer.writerows(all_candidates)
        print(f"  -> YOLO 후보 {len(all_candidates)}개 저장: {yolo_candidates_csv}")
    else:
        print("  -> 경고: 영상 내에서 광고할 만한 사물을 찾지 못했습니다.")

    # 씬 파일 삭제 (저장공간 절약 — 크롭 이미지는 VISION_OUT_DIR에 유지)
    print("  -> 🗑️  분석 완료된 씬 파일 삭제 중...")
    if os.path.exists(SCENE_DIR):
        shutil.rmtree(SCENE_DIR)
        print(f"  -> scenes 폴더 삭제 완료")

    # ─────────────────────────────────────────────────────────
    # STEP 3. 장면 설명 생성 — Gemini Vision
    # ─────────────────────────────────────────────────────────
    # 씬 대표 크롭 이미지를 Gemini에 전달하여 자연어 장면 설명 텍스트를 생성합니다.
    # 생성된 텍스트는 STEP 5에서 ad_inventory.ad_name과 코사인 유사도 매칭에 사용됩니다.
    print("\n[STEP 3/6] 💬 Gemini Vision 장면 설명 생성 중...")
    if os.path.exists(yolo_candidates_csv):
        describer = GeminiSceneDescriber()
        describer.process_candidates(yolo_candidates_csv, scene_descriptions_csv)

        # Gemini 분석 완료 후 크롭 이미지 삭제 (저장공간 절약)
        print("  -> 🗑️  분석 완료된 크롭 이미지 삭제 중...")
        for entry in os.listdir(VISION_OUT_DIR):
            entry_path = os.path.join(VISION_OUT_DIR, entry)
            if os.path.isdir(entry_path) and entry.startswith("crops_"):
                shutil.rmtree(entry_path)
        print("  -> 크롭 이미지 삭제 완료")
    else:
        print("  -> YOLO 결과가 없으므로 장면 설명 생성을 건너뜁니다.")

    # ─────────────────────────────────────────────────────────
    # STEP 4. 대사 추출 — Whisper STT
    # ─────────────────────────────────────────────────────────
    print("\n[STEP 4/6] 🗣️  Whisper STT 대사 추출 중...")
    if os.path.exists(output_audio):
        audio_analyzer = AudioAnalyzer("base")
        audio_analyzer.extract_transcript(output_audio, transcript_csv)
    else:
        print("  -> 오디오 파일이 없어 STT 단계를 건너뜁니다.")

    # ─────────────────────────────────────────────────────────
    # STEP 5. 광고 매칭 + 타임테이블 생성
    # ─────────────────────────────────────────────────────────
    # 1) Gemini 임베딩 + 코사인 유사도로 ad_inventory DB에서 최적 광고 매칭
    # 2) librosa 묵음 감지 + 씬 전환 타이밍 3조건으로 삽입 시점 결정
    print("\n[STEP 5/6] 🧠 광고 매칭 및 타임테이블 편성 중...")
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

    # ─────────────────────────────────────────────────────────
    # STEP 6. 배너 이미지 생성 — 나노바나나
    # ─────────────────────────────────────────────────────────
    if skip_ad_generation:
        print("\n[SKIP] --skip-ad 옵션에 의해 STEP 6(배너 이미지 생성)를 건너뜁니다.")
    else:
        print("\n[STEP 6/6] 🍌 나노바나나 배너 이미지 렌더링 중...")
        generated_images_dir = os.path.join(PROCESSED_DIR, "generated_ad_banners")
        if os.path.exists(final_timetable_csv):
            nano_generator = NanoBananaGenerator()
            nano_generator.process_timetable(final_timetable_csv, generated_images_dir)
        else:
            print("  -> 최종 타임테이블이 없으므로 배너 이미지 생성을 건너뜁니다.")

    print("\n" + "="*60)
    print("✅ 모든 파이프라인 처리가 완료되었습니다!")
    print(f"👉 장면 설명 결과: {scene_descriptions_csv}")
    print(f"👉 광고 타임테이블: {final_timetable_csv}")
    print("="*60)


if __name__ == "__main__":
    target_video = None
    skip_ad = False

    for arg in sys.argv[1:]:
        if arg == "--skip-ad":
            skip_ad = True
        else:
            target_video = arg

    if not target_video:
        # 인자 없이 실행하면 바탕화면의 SampleVideo.mp4 실행 (기본 테스트 위치)
        desktop_dir = os.path.dirname(BASE_DIR)
        target_video = os.path.join(desktop_dir, "SampleVideo.mp4")

    run_contextual_ad_pipeline(target_video, skip_ad_generation=skip_ad)
