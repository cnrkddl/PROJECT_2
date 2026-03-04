import os
import csv
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
NANOBANANA_API_KEY = os.environ.get("NANOBANANA_API_KEY")

class NanoBananaGenerator:
    def __init__(self):
        """
        타임테이블에 적혀있는 추천 광고 상품과 맥락(Context)을 바탕으로,
        나노바나나 API(NanoBanana / Gemini Flash Image 등)에 프롬프트를 전송하여
        실제 노출용 배너 이미지를 생성하는 클래스입니다.
        """
        self.api_key = NANOBANANA_API_KEY
        
        # TODO: 실제 사용하시는 나노바나나 서비스의 공식 API 엔드포인트 URL로 변경해주세요.
        # (예: https://api.nanobanana.ai/v1/images/generate 또는 Google Cloud Vertex API 엔드포인트)
        self.endpoint = "https://api.nanobanana.ai/v1/images/generate"

    def generate_ad_banner(self, prompt: str, output_path: str) -> bool:
        """
        단일 프롬프트를 API로 전송하고 결과를 이미지(PNG/JPG)로 다운받아 저장합니다.
        """
        if not self.api_key:
            print("❌ 오류: NANOBANANA_API_KEY가 설정되지 않았습니다. .env 파일에 키를 추가해주세요.")
            return False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # TODO: 나노바나나 공식 API 문서에 맞춰 파라미터(payload) 이름을 수정해야 할 수 있습니다. (예: aspect_ratio 등)
        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 512,  # 배너 형태를 위해 가로로 긴 비율 요청
            "style": "advertising"
        }
        
        try:
            print(f"🎨 나노바나나 API 이미지 생성 요청 중... (프롬프트 요약: {prompt[:30]}...)")
            
            # API 호출
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status() 
            
            result = response.json()
            
            # 응답 결과에서 이미지 URL 추출 (API 응답 JSON 구조에 따라 수정 필요)
            image_url = result.get('image_url') 
            
            if image_url:
                # 생성된 이미지 다운로드 및 로컬 저장
                img_data = requests.get(image_url).content
                with open(output_path, 'wb') as handler:
                    handler.write(img_data)
                return True
            else:
                print("❌ 오류: 성공적으로 호출되었으나, 응답에서 이미지 URL을 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ NanoBanana API 호출 실패: {e}")
            return False

    def process_timetable(self, timetable_csv_path: str, output_dir: str):
        """
        생성된 final_ad_timetable.csv를 순회하며 각각의 광고 배너 이미지를 자동 생성합니다.
        """
        if not os.path.exists(timetable_csv_path):
            print(f"타임테이블 파일을 찾을 수 없습니다: {timetable_csv_path}")
            return
            
        print("\n========================================================")
        print("🍌 나노바나나 배너 이미지 자동 생성 파이프라인 시작 🍌")
        print("========================================================\n")

        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        with open(timetable_csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader, start=1):
                product_name = row.get('추천 광고 상품명', '')
                ad_format = row.get('추천 UI/UX 노출 형태', '')
                context = row.get('영상 분위기 (Context)', '')
                reason = row.get('광고 매칭 사유', '')
                
                if not product_name:
                    continue
                    
                # 이미지 생성을 위한 고해상도 영문 프롬프트 엔지니어링 수행
                # 나노바나나(혹은 미드저니/달리)가 찰떡같이 알아들을 수 있도록 구체적으로 묘사
                prompt = (
                    f"A high-quality, professional advertising banner for '{product_name}'. "
                    f"Design format: {ad_format}. "
                    f"The mood and background strictly match the context: {context}. "
                    f"Concept reason: {reason}. "
                    f"Cinematic lighting, modern marketing layout, textless, highly attractive, 4k resolution."
                )
                
                # 파일명 생성: ad_banner_01_맥심커피.png 형태
                safe_product_name = "".join([c for c in product_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                output_filename = os.path.join(output_dir, f"ad_banner_{idx:02d}_{safe_product_name.replace(' ', '_')}.png")
                
                success = self.generate_ad_banner(prompt, output_filename)
                
                if success:
                    print(f"✅ [{product_name}] 배너 이미지 생성 완료! -> {output_filename}\n")
                    success_count += 1
                else:
                    print(f"⚠️ [{product_name}] 배너 이미지 생성 실패\n")
                    
        print(f"\n🎉 나노바나나 파이프라인 종료! (총 {success_count}개의 배너 이미지 생성 완료)")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # -------------------------------------------------------------
    # 테스트 로직: 가장 최신(방금 분석이 끝난) processed 폴더를 찾아 해당 타임테이블을 읽음
    # -------------------------------------------------------------
    processed_base = os.path.join(BASE_DIR, "data", "processed")
    if os.path.exists(processed_base):
        run_folders = [os.path.join(processed_base, d) for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))]
        latest_run = max(run_folders, key=os.path.getmtime) if run_folders else processed_base
    else:
         latest_run = processed_base
         
    timetable_csv = os.path.join(latest_run, "final_ad_timetable.csv")
    
    # 생성된 이미지들이 저장될 캡처본 폴더
    output_images_dir = os.path.join(latest_run, "generated_ad_banners")
    
    generator = NanoBananaGenerator()
    generator.process_timetable(timetable_csv, output_images_dir)
