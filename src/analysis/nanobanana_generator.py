import os
import csv
from dotenv import load_dotenv
from google import genai

# .env 파일 로드
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class NanoBananaGenerator:
    def __init__(self):
        """
        타임테이블에 적혀있는 추천 광고 상품과 맥락(Context)을 바탕으로,
        나노바나나 (Gemini 3.1 Flash Image Preview) 버전을 호출하여
        실제 노출용 배너 이미지를 생성하는 클래스입니다.
        """
        self.api_key = GEMINI_API_KEY
        # Google GenAI 클라이언트 초기화
        self.client = genai.Client(api_key=self.api_key)

    def generate_ad_banner(self, prompt: str, output_path: str) -> bool:
        """
        단일 프롬프트를 API로 전송하고 결과를 이미지(PNG)로 다운받아 저장합니다.
        """
        if not self.api_key:
            print("❌ 오류: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 키를 추가해주세요.")
            return False

        try:
            print(f"🎨 NanoBanana(Gemini 3.1 Flash Image) 이미지 생성 중... (프롬프트 요약: {prompt[:30]}...)")
            
            # 구글 공식 문서에 안내된 NanoBanana 전용 모델명 사용
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[prompt]
            )
            
            # 응답 객체의 parts 중에서 이미지 데이터만 추출하여 저장
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    image.save(output_path)
                    return True
                    
            print("❌ 오류: 응답을 받았으나, 포함된 이미지 데이터가 없습니다.")
            return False
            
        except Exception as e:
            print(f"❌ NanoBanana API SDK 호출 실패: {e}")
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
                product_name = row.get('추천 광고 준상품명', '')
                if not product_name:
                    product_name = row.get('추천 광고 상품명', '')
                    
                ad_format = row.get('추천 UI/UX 노출 형태', '')
                context = row.get('영상 분위기 (Context)', '')
                reason = row.get('광고 매칭 사유', '')
                
                if not product_name:
                    continue
                    
                # 이미지 생성을 위한 고해상도 영문 프롬프트 엔지니어링 
                prompt = (
                    f"A highly professional pop-up advertising banner for '{product_name}'. "
                    f"Design format requirement: {ad_format}. "
                    f"The mood strictly matches the scene context: {context}. "
                    f"Concept reason: {reason}. "
                    f"Include beautiful cinematic lighting, modern advertising layout, commercial product photography style, textless, 4k ultra realistic."
                )
                
                # 파일명 생성: ad_banner_01_맥심커피.png 형태 (Pillow/as_image 은 PNG 저장 지원)
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
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 만약 터미널에서 인자로 경로를 넘겼다면 그것을 사용
    if len(sys.argv) == 3:
        timetable_csv = sys.argv[1]
        output_images_dir = sys.argv[2]
    else:
        # 기본 동작: 가장 최신(방금 분석이 끝난) processed 폴더를 찾아 해당 타임테이블 읽기
        processed_base = os.path.join(BASE_DIR, "data", "processed")
        if os.path.exists(processed_base):
            run_folders = [os.path.join(processed_base, d) for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))]
            latest_run = max(run_folders, key=os.path.getmtime) if run_folders else processed_base
        else:
             latest_run = processed_base
             
        timetable_csv = os.path.join(latest_run, "final_ad_timetable.csv")
        output_images_dir = os.path.join(latest_run, "generated_ad_banners")
        
    generator = NanoBananaGenerator()
    generator.process_timetable(timetable_csv, output_images_dir)
