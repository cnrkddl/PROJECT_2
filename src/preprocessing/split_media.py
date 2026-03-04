import os
from moviepy import VideoFileClip
from scenedetect import detect, ContentDetector, split_video_ffmpeg

def extract_audio(video_path: str, output_audio_path: str):
    """
    비디오 파일에서 오디오 스트림만 추출하여 지정된 경로에 WAV 파일로 저장합니다.
    (나중에 오디오(대사) 기반 Whisper STT 처리에 사용됩니다.)
    """
    print(f"[{video_path}] 에서 오디오 추출을 시작합니다...")
    try:
        # 1. moviepy를 사용하여 비디오 클립 로드
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        
        # 오디오 트랙이 존재하지 않는 비디오일 경우 에러 방지 처리
        if audio_clip is None:
             print("비디오에 오디오 트랙이 포함되어 있지 않습니다.")
             return False
             
        # 2. Whisper 모델에서 가장 잘 인식할 수 있는 .wav 포맷으로 오디오만 우선 저장
        # (불필요한 로그 출력을 끄기 위해 logger=None 설정)
        audio_clip.write_audiofile(output_audio_path, logger=None)
        
        # 3. 메모리 누수(Memory leak) 방지를 위해 사용이 끝난 클립 리소스 안전하게 닫기
        audio_clip.close()
        video_clip.close()
        
        print(f"오디오 추출이 완료되었습니다. 저장 경로: {output_audio_path}")
        return True
    except Exception as e:
        print(f"오디오 추출 중 오류 발생: {e}")
        return False

def detect_and_split_scenes(video_path: str, output_dir: str):
    """
    영상 내에서 의미 있는 화면(Scene) 전환을 감지하고,
    해당 전환점을 기준으로 원본 영상을 분할하여 각각 독립된 mp4 파일로 저장합니다.
    """
    print(f"[{video_path}] 에서 씬(Scene) 감지 및 분할을 시작합니다...")
    try:
        # 1. scenedetect의 ContentDetector를 사용하여 영상 내 씬 전환 시점 감지
        # threshold(기본값 27.0): 값이 낮을수록 미세한 화면 변화에도 씬을 나누고, 높을수록 아주 큰 화면 변화에만 씬을 나눕니다.
        scene_list = detect(video_path, ContentDetector(threshold=27.0))
        
        # 감지된 씬이 없는 경우 (예: 화면 변화가 없는 고정 카메라 영상이거나 길이가 너무 짧은 경우)
        if not scene_list:
            print("감지된 화면 전환이 없거나, 영상이 너무 짧습니다.")
            return []
            
        print(f"총 {len(scene_list)}개의 화면 전환(씬)이 발견되었습니다. 영상을 분할합니다...")
        
        # 2. 씬별로 저장될 파일명 템플릿 지정 (예: 분할된 영상은 scene-001.mp4, scene-002.mp4 로 저장됨)
        output_template = os.path.join(output_dir, "scene-$SCENE_NUMBER.mp4")
        
        # 3. ffmpeg 백엔드를 호출하여, 원본 영상을 위에서 얻은 씬 리스트(scene_list) 기준으로 실제 디스크에 자르고 저장
        split_video_ffmpeg(video_path, scene_list, output_file_template=output_template, show_progress=True)
        
        print(f"씬(Scene) 단위 영상 분할이 성공적으로 완료되었습니다. 저장 디렉토리: {output_dir}")
        return scene_list
        
    except Exception as e:
        print(f"씬 분할 과정 중 오류 발생 부분: {e}")
        return []

if __name__ == "__main__":
    # 프로젝트 최상단 루트 디렉토리 절대경로 계산
    # (현재 파일 split_media.py 위치가 프로젝트 폴더 안의 src/preprocessing 이므로, 3단계 상위 폴더로 올라감)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TEST_VIDEO = os.path.join(BASE_DIR, "SampleVideo.mp4")
    
    import datetime
    import os
    video_filename = os.path.splitext(os.path.basename(TEST_VIDEO))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"{video_filename}_{timestamp}"
    
    # 전처리된 결과 데이터들이 저장될 목적지 폴더 경로 설정 (data/processed/ 하위)
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", run_folder_name)
    AUDIO_DIR = os.path.join(PROCESSED_DIR, "audio")
    SCENE_DIR = os.path.join(PROCESSED_DIR, "scenes")
    
    # 결과물을 저장할 폴더가 혹시 없다면 프로그램 실행 중 자동으로 생성하도록 옵션 지정 (exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SCENE_DIR, exist_ok=True)
    
    print("=== [Contextual Video Ad Insertion] 전처리(Preprocessing) 파이프라인 시작 ===")
    
    # [Step 1] 오디오(대사, 효과음 등) 파트 전처리 실행
    # (이렇게 분리된 오디오 파일은 향후 src/analysis/ 모듈에서 STT/NLP 맥락 분석 모델에 사용됩니다.)
    output_audio = os.path.join(AUDIO_DIR, "extracted_audio.wav")
    extract_audio(TEST_VIDEO, output_audio)
    
    # [Step 2] 비디오(화면 흐름 등) 파트 전처리 실행 (화면 컷 전환 기준)
    # (이렇게 분할된 영상 클립들은 향후 Vision 멀티모달 모델을 통해 '이 장면이 광고가 들어갈 골든타임인지' 스코어링하는 기준점이 됩니다.)
    detect_and_split_scenes(TEST_VIDEO, SCENE_DIR)
    
    print("=== 전처리(Preprocessing) 파이프라인 무사히 종료되었습니다! ===")
