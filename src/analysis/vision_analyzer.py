import cv2
import os
import csv
from collections import defaultdict
from ultralytics import YOLO

# 광고로 매칭하기 좋은(Shoppable) COCO 데이터셋 클래스 ID 필터링
SHOPPABLE_CLASSES = {
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
    44: 'spoon', 45: 'bowl', 49: 'orange', 50: 'broccoli', 51: 'carrot', 
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'
}


class VisionAnalyzer:
    def __init__(self, model_version="yolov8n.pt"):
        """
        VisionAnalyzer 초기화
        :param model_version: ultralytics YOLO 모델 버전 (기본값: 가장 가벼운 nano 버전)
        """
        print(f"Loading YOLO model ({model_version})...")
        self.model = YOLO(model_version)
        
    def analyze_scene(self, video_path: str, output_dir: str, sample_rate: int = 15):
        """
        특정 씬(Scene) 영상에서 객체를 인식하고, 광고 후보군을 추출합니다.
        가장 잘 나온(가장 크게 잡힌) 순간의 이미지(Crop)도 함께 저장합니다.
        
        :param video_path: 분석할 씬 영상 경로
        :param output_dir: 크롭된 이미지와 CSV 결과를 저장할 디렉토리 경로
        :param sample_rate: 몇 프레임마다 분석할 것인지
        :return: 정렬된 최고 후보군 리스트 (Top N)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []
            
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_area = frame_width * frame_height
        
        # 결과를 저장할 폴더명 (실제 생성은 이미지를 저장할 때 수행)
        scene_name = os.path.splitext(os.path.basename(video_path))[0]
        crops_dir = os.path.join(output_dir, f"crops_{scene_name}")
        
        frame_count = 0
        
        # 객체별 통계를 저장할 딕셔너리
        object_stats = defaultdict(lambda: {'appearances': 0, 'max_area_ratio': 0.0, 'conf_sum': 0.0, 'best_crop_path': None})
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # sample_rate 단위로 프레임 건너뛰기
            if frame_count % sample_rate == 0:
                results = self.model(frame, verbose=False)
                
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    # 1차 필터링: 쇼핑/광고 가능한 카테고리만 대상 & 신뢰도 50% 이상
                    if cls_id in SHOPPABLE_CLASSES and conf > 0.5:
                        class_name = SHOPPABLE_CLASSES[cls_id]
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        box_area = (x2 - x1) * (y2 - y1)
                        area_ratio = box_area / total_area
                        
                        # 2차: 노이즈 필터링 (화면의 0.5%도 차지하지 않는 객체 제외)
                        if area_ratio < 0.005:
                            continue
                            
                        stats = object_stats[class_name]
                        stats['appearances'] += 1
                        stats['conf_sum'] += conf
                        
                        # 이 객체가 지금까지 본 것 중 가장 크게(선명하게) 잡혔다면, 이미지를 크롭해서 저장!
                        if area_ratio > stats['max_area_ratio']:
                            stats['max_area_ratio'] = area_ratio
                            
                            # 이미지 크롭 (안전하게 범위 제한)
                            crop_x1, crop_y1 = max(0, x1), max(0, y1)
                            crop_x2, crop_y2 = min(int(frame_width), x2), min(int(frame_height), y2)
                            
                            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                                cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                os.makedirs(crops_dir, exist_ok=True)
                                crop_filename = os.path.join(crops_dir, f"{class_name}_best.jpg")
                                cv2.imwrite(crop_filename, cropped_img)
                                stats['best_crop_path'] = crop_filename
                            
            frame_count += 1
            
        cap.release()
        
        # [최종 스코어링 및 랭킹 정렬]
        candidates = []
        for obj_name, stats in object_stats.items():
            avg_conf = stats['conf_sum'] / stats['appearances']
            score = stats['appearances'] * stats['max_area_ratio'] * avg_conf
            
            candidates.append({
                'scene': scene_name,
                'object': obj_name,
                'appearances': stats['appearances'],
                'max_area_ratio': round(stats['max_area_ratio'], 4),
                'avg_confidence': round(avg_conf, 4),
                'ad_suitability_score': round(score, 4),
                'crop_image_path': stats['best_crop_path']
            })
            
        # CSV 출력용 한글 컬럼명 매핑
        korean_candidates = []
        for cand in candidates:
            korean_candidates.append({
                '씬 이름 (Scene)': cand['scene'],
                '상품 종류 (Object)': cand['object'],
                '등장 프레임 수': cand['appearances'],
                '화면 내 최대 비중 (%)': round(cand['max_area_ratio'] * 100, 2), # %로 보기 편하게 변환
                'AI 인식 신뢰도 (%)': round(cand['avg_confidence'] * 100, 2),
                '광고 적합도 점수': cand['ad_suitability_score'],
                '크롭 이미지 경로': cand['crop_image_path']
            })
            
        return korean_candidates

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import os
    processed_base = os.path.join(BASE_DIR, "data", "processed")
    if os.path.exists(processed_base):
        run_folders = [os.path.join(processed_base, d) for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))]
        latest_run = max(run_folders, key=os.path.getmtime) if run_folders else processed_base
    else:
        latest_run = processed_base

    # 테스트 씬 경로 설정
    SCENES_DIR = os.path.join(latest_run, "scenes")
    
    # 분석 결과를 저장할 비전 아웃풋 디렉토리
    VISION_OUT_DIR = os.path.join(latest_run, "vision_results")
    os.makedirs(VISION_OUT_DIR, exist_ok=True)
    
    # 씬 목록 중 아무거나 여러 개 테스트해보기 (다양한 씬 추가)
    test_scenes = [f for f in os.listdir(SCENES_DIR) if f.endswith(".mp4")]
    test_scenes.sort() # 번호순(scene-001, 002...) 정렬    
    analyzer = VisionAnalyzer()
    
    # 모든 씬에서 추출된 광고 후보군을 모을 리스트
    all_candidates = []
    
    for scene_file in test_scenes:
        test_path = os.path.join(SCENES_DIR, scene_file)
        if os.path.exists(test_path):
            print(f"\n[{scene_file}] 분석 시작...")
            top_candidates = analyzer.analyze_scene(test_path, VISION_OUT_DIR, sample_rate=10)
            
            if not top_candidates:
                print(" -> 광고로 쓸만한 상품 무(無)")
            else:
                for i, cand in enumerate(top_candidates, 1):
                    # 한글 키워드로 출력 (터미널)
                    print(f" {i}위: {cand['상품 종류 (Object)']} (Score: {cand['광고 적합도 점수']}) -> 이미지: {cand['크롭 이미지 경로']}")
                # 추출된 후보군을 전체 리스트에 합침
                all_candidates.extend(top_candidates)
                
    # 모든 분석이 끝난 후 하나의 통합 CSV 파일로 저장
    if all_candidates:
        master_csv_path = os.path.join(VISION_OUT_DIR, "all_scenes_candidates.csv")
        # 엑셀에서 한글이 깨지지 않도록 utf-8-sig 인코딩 사용
        with open(master_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_candidates[0].keys())
            writer.writeheader()
            writer.writerows(all_candidates)
        print(f"\n✅ 통합 CSV 결과 저장 완료: {master_csv_path}")
    else:
        print("\n✅ 분석 완료. 추출된 광고 후보군이 없습니다.")

