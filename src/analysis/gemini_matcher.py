import os
import csv
import json
import requests
from dotenv import load_dotenv
from google import genai
from typing import List, Dict

# .env 파일에서 환경변수 로드
load_dotenv()

# 이제 GitHub에 올라가도 안전하게 로컬 환경변수에서만 키를 가져옵니다.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class GeminiAdMatcher:
    def __init__(self):
        """
        Gemini Vision 모델과 연동하여
        이미지에서 상품 키워드를 뽑는 class
        """
        # Gemini API 초기화 (최신 SDK)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        # 이미지 입력을 지원하는 가장 최신의 빠르고 저렴한 모델
        self.model_name = 'models/gemini-flash-latest'
        
    def extract_keywords_from_image(self, image_path: str, object_name: str) -> Dict:
        """
        크롭된 이미지를 Gemini에 전송하여 '색상', '재질', '디자인/스타일' 특징을 추출합니다.
        
        :param image_path: YOLO로 크롭된 상품 이미지 경로
        :param object_name: YOLO가 판별한 객체 이름 (예: chair, handbag)
        :return: 추출된 키워드 딕셔너리 (예: {'color': '블랙', 'material': '가죽', 'style': '모던'})
        """
        if not os.path.exists(image_path):
            return {"color": "", "material": "", "style": ""}
            
        print(f"[{object_name}] 이미지 분석 중... (Gemini Vision)")
        
        # 이미지 파일 로드 (최신 SDK 파일 업로드 방식)
        img_file = self.client.files.upload(file=image_path)
        
        # LLM에게 지시할 프롬프트
        prompt = f"""
        당신은 광고 상품 추천을 위한 AI 비전 분석가입니다.
        제가 보내준 이미지는 영상 속에서 캡처한 '{object_name}' (상품) 입니다.
        
        이 상품을 쇼핑몰에서 검색하기 위해 가장 중요한 3가지 시각적 특징을 한국어 단어로만 추출해주세요:
        1. 색상 (Color)
        2. 재질 (Material) (잘 모르겠으면 '일반'이라고 작성)
        3. 디자인 또는 분위기 (Style) (예: 모던한, 클래식한, 빈티지)
        
        답변은 반드시 아래 JSON 형식으로만 작성해주세요:
        {{"color": "색상단어", "material": "재질단어", "style": "스타일단어"}}
        """
        
        try:
            # Gemini에게 이미지와 프롬프트 전송
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img_file, prompt]
            )
            response_text = response.text.strip()
            
            # 마크다운 ```json ... ``` 형태가 섞여있을 수 있으므로 제거
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            keywords = json.loads(response_text)
            
            # API 할당량/정리를 위해 업로드한 이미지 삭제
            self.client.files.delete(name=img_file.name)
            
            return keywords
            
        except Exception as e:
            print(f"Gemini API 호출 중 오류 발생: {e}")
            return {"color": "", "material": "", "style": ""}

    def process_candidates(self, candidates_csv_path: str, output_csv_path: str, scene_timestamps_csv_path: str = None):
        """
        YOLO로 추출한 기존 CSV 파일을 읽고,
        각 후보 이미지마다 Gemini (키워드 추출) 흐름을 타서
        세부 정보가 가득 담긴 최종 하나의 CSV 파일로 내보끜다.
        """
        if not os.path.exists(candidates_csv_path):
            print(f"후보군 CSV 파일을 찾을 수 없습니다: {candidates_csv_path}")
            return

        # 씬 타임스탬프 로드 (씨 이름 -> (start_sec, end_sec) 딕셔너리)
        scene_time_map = {}
        if scene_timestamps_csv_path and os.path.exists(scene_timestamps_csv_path):
            with open(scene_timestamps_csv_path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scene_time_map[row['씬 이름']] = (
                        row['시작 시간 (초)'],
                        row['종료 시간 (초)']
                    )
            print(f"  -> 씬 타임스탬프 {len(scene_time_map)}개 로드 완료")
            
        # 기존 CSV 읽기
        candidates = []
        with open(candidates_csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append(row)
                
        print(f"잘 {len(candidates)}개의 광고 후보의 세부 속성을 분석합니다...")
        
        final_recommendations = []
        
        for cand in candidates:
            scene = cand.get('씬 이름 (Scene)')
            obj_name = cand.get('상품 종류 (Object)')
            img_path = cand.get('크롭 이미지 경로')
            original_score = cand.get('광고 적합도 점수')
            
            # 씬 이름에서 시작/종료 시간 조회
            start_sec, end_sec = scene_time_map.get(scene, ('', ''))
            
            # 1. Gemini 비전 분석으로 "디자인/재질/색상" 키워드 추출
            gemini_keywords = self.extract_keywords_from_image(img_path, obj_name)
            
            # 2. 분석 결과를 리스트에 저장 (시작/종료 시간 포함)
            final_recommendations.append({
                '씬 이름 (Scene)': scene,
                '시작 시간 (초)': start_sec,
                '종료 시간 (초)': end_sec,
                '상품 종류 (Object)': obj_name,
                '광고 적합도 점수': original_score,
                '대표 색상 (Color)': gemini_keywords.get('color', ''),
                '주요 재질 (Material)': gemini_keywords.get('material', ''),
                '디자인/스타일 (Style)': gemini_keywords.get('style', ''),
                '크롭 이미지 경로': img_path
            })
                    
        # 결과를 단일 CSV 하나로 깔끔하게 저장
        if final_recommendations:
            with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=final_recommendations[0].keys())
                writer.writeheader()
                writer.writerows(final_recommendations)
            print(f"\n🎉 [Gemini 비전 분석 완료] 추출된 매칭 속성 리스트 저장 완료: {output_csv_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import os
    processed_base = os.path.join(BASE_DIR, "data", "processed")
    if os.path.exists(processed_base):
        run_folders = [os.path.join(processed_base, d) for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))]
        latest_run = max(run_folders, key=os.path.getmtime) if run_folders else processed_base
    else:
        latest_run = processed_base

    # YOLO에서 뽑아둔 all_scenes_candidates.csv 파일 경로
    VISION_OUT_DIR = os.path.join(latest_run, "vision_results")
    candidates_csv = os.path.join(VISION_OUT_DIR, "all_scenes_candidates.csv")
    
    # 딥 다이브 매칭 결과를 저장할 최종 파일 (단일 CSV)
    final_output_csv = os.path.join(VISION_OUT_DIR, "ad_recommendations.csv")
    
    matcher = GeminiAdMatcher()
    matcher.process_candidates(candidates_csv, final_output_csv)

#python src/analysis/gemini_matcher.py

