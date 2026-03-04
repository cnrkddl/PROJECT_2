import os
import csv
import json
from dotenv import load_dotenv
from google import genai

# .env 파일에서 환경변수 로드
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class DynamicTimetableGenerator:
    def __init__(self):
        """
        로컬 키워드 사전의 한계를 극복하기 위해,
        전체 대사를 Gemini LLM에게 한 번만 전달하여 
        동적으로 영상의 맥락을 분석하고 광고를 추천하는 클래스입니다.
        """
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = 'models/gemini-flash-latest'

    def generate_timetable(self, transcript_csv_path: str, output_csv_path: str):
        if not os.path.exists(transcript_csv_path):
            print(f"STT 결과 파일을 찾을 수 없습니다: {transcript_csv_path}")
            return
            
        print("전체 대사 데이터를 읽고 있습니다...")
        transcript_lines = []
        with open(transcript_csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript_lines.append(f"[{row['시작 시간 (초)']}s ~ {row['종료 시간 (초)']}s] {row['대사 (Text)']}")
                
        # 대사가 너무 길면 API 제한이 있을 수 있으니 하나의텍스트로 병합
        full_transcript = "\n".join(transcript_lines)
        
        print("Gemini AI에게 영상 전체 흐름 분석 및 동적 광고 편성을 요청합니다...")
        
        prompt = f"""
        당신은 영상 맥락 분석 및 광고 편성(Contextual Ad Insertion) 전문가입니다.
        아래는 어떤 드라마/영상의 타임라인별 대사(Transcript)입니다.

        이 대사의 흐름을 분석하여, 내용이나 분위기(Context)가 전환되는 의미 있는 구간들로 나누어 주세요.
        그리고 각 구간의 맥락(예: 갈등, 로맨스, 병원, 범죄, 일상 등 자유롭게 유추)을 정의하고, 
        그 상황에 가장 잘 어울릴 법한 기발한 광고 상품을 1개씩 추천해 주세요.

        출력은 반드시 아래 JSON 배열 형식으로만 작성해 주세요. (마크다운 코드블록 금지, 순수 JSON만 출력)
        [
            {{
                "start_time": "0.0",
                "end_time": "40.0",
                "context": "유추한 분위기/맥락",
                "ad_product": "추천 광고 상품명",
                "ad_format": "배너 형태 (예: 하단 팝업)",
                "reason": "왜 이 광고를 추천했는지 사유"
            }},
            ...
        ]

        [대본 데이터]
        {full_transcript}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text.strip()
            
            # JSON 파싱을 위해 마크다운 잔재 제거
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            timetable_data = json.loads(response_text)
            
            # CSV로 변경
            formatted_timetable = []
            for item in timetable_data:
                formatted_timetable.append({
                    '광고 진입 시간 (초)': f"{item.get('start_time', '')}s",
                    '광고 종료 시간 (초)': f"{item.get('end_time', '')}s",
                    '영상 분위기 (Context)': item.get('context', ''),
                    '추천 광고 상품명': item.get('ad_product', ''),
                    '추천 UI/UX 노출 형태': item.get('ad_format', ''),
                    '광고 매칭 사유': item.get('reason', '')
                })
                
            # CSV 저장
            if formatted_timetable:
                os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=formatted_timetable[0].keys())
                    writer.writeheader()
                    writer.writerows(formatted_timetable)
                print(f"\n🎉 [AI 동적 편성 완료] 최종 영상 광고 타임테이블 저장: {output_csv_path}")
                
                # 터미널에 요약 출력
                for entry in formatted_timetable[:5]:
                    print(f"⏰ {entry['광고 진입 시간 (초)']}~{entry['광고 종료 시간 (초)']} | [{entry['영상 분위기 (Context)']}]")
                    print(f" 👉 띄울 광고: {entry['추천 광고 상품명']} ({entry['추천 UI/UX 노출 형태']})")
                    print(f" 👉 사유: {entry['광고 매칭 사유']}\n")

        except Exception as e:
            print(f"Gemini API 호출 및 파싱 중 오류 발생: {e}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 텍스트가 모두 쪼개져 있는 STT 원본 파일을 입력으로 사용
    TRANSCRIPT_CSV = os.path.join(BASE_DIR, "data", "processed", "audio", "transcript.csv")
    
    # 최종 타임테이블 아웃풋
    TIMETABLE_CSV = os.path.join(BASE_DIR, "data", "processed", "final_ad_timetable.csv")
    
    generator = DynamicTimetableGenerator()
    generator.generate_timetable(TRANSCRIPT_CSV, TIMETABLE_CSV)
