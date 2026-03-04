import os
import csv
import whisper
import warnings

# Whisper 내부의 FP16 관련 경고 무시 (CPU 환경 등에서 자주 발생)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class AudioAnalyzer:
    def __init__(self, model_size="base"):
        """
        Whisper 모델을 로드하여 오디오 분석 객체를 초기화합니다.
        
        :param model_size: 모델 크기 ('tiny', 'base', 'small', 'medium', 'large')
                           테스트용으로는 'base'나 'small'이 적당합니다.
        """
        print(f"Loading Whisper model ({model_size})... 이 작업은 최초 실행 시 모델 다운로드로 수 분이 걸릴 수 있습니다.")
        self.model = whisper.load_model(model_size)
    
    def extract_transcript(self, audio_path: str, output_csv: str):
        """
        오디오 파일에서 STT(Speech-to-Text)를 수행하고 결과를 CSV로 저장합니다.
        
        :param audio_path: 추출된 WAV 오디오 파일 경로
        :param output_csv: 대사 텍스트를 저장할 CSV 파일 경로
        :return: 분석된 텍스트 세그먼트 리스트
        """
        if not os.path.exists(audio_path):
            print(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            return []
            
        print(f"\n[{os.path.basename(audio_path)}] STT(음성 인식) 분석 시작...")
        
        # Whisper를 통해 음성을 한국어로 인식
        result = self.model.transcribe(audio_path, language="ko")
        
        segments = result.get('segments', [])
        transcript_data = []
        
        for segment in segments:
            start_time = round(segment['start'], 2)
            end_time = round(segment['end'], 2)
            text = segment['text'].strip()
            
            # 대사가 있는 경우에만 저장
            if text:
                transcript_data.append({
                    '시작 시간 (초)': start_time,
                    '종료 시간 (초)': end_time,
                    '대사 (Text)': text
                })
            
        # 결과를 CSV로 깔끔하게 저장
        if transcript_data:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=transcript_data[0].keys())
                writer.writeheader()
                writer.writerows(transcript_data)
            print(f"🎉 STT 대사 추출 및 저장 완료: {output_csv}")
        else:
            print("인식된 대사가 없습니다.")
            
        return transcript_data

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # split_media.py에서 만들어둔 오디오 파일 경로
    AUDIO_PATH = os.path.join(BASE_DIR, "data", "processed", "audio", "extracted_audio.wav")
    
    # 결과를 저장할 경로
    OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "audio", "transcript.csv")
    
    analyzer = AudioAnalyzer("base") # 더 높은 정확도를 원하시면 "small" 로 변경 가능
    analyzer.extract_transcript(AUDIO_PATH, OUTPUT_CSV)
