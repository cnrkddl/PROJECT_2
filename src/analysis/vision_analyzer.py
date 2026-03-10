import cv2
import os
from collections import defaultdict
from ultralytics import YOLO

# YOLO-World에 전달할 탐지 대상 클래스 (텍스트로 자유 지정 — COCO 제한 없음)
WORLD_CLASSES = [
    # 기존 COCO 쇼핑/광고 관련 클래스
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
    # YOLO-World 확장 클래스 (드라마 속 주요 사물)
    'dress', 'shirt', 'pants', 'shoes', 'sneakers', 'jacket', 'coat',
    'sofa', 'lamp', 'cabinet', 'drawer', 'mirror',
    'coffee', 'beer', 'juice', 'bread', 'sandwich',
    'watch', 'sunglasses', 'ring', 'necklace', 'earring',
    'car', 'motorcycle', 'bicycle',
    'flower', 'candle', 'cosmetics', 'perfume',
]


class VisionAnalyzer:
    def __init__(self, model_version="yolov8s-worldv2.pt"):
        """
        YOLO-World 모델 초기화.
        COCO 80개 클래스 제한 없이 텍스트로 탐지 대상을 지정합니다.
        """
        print(f"Loading YOLO-World model ({model_version})...")
        self.model = YOLO(model_version)
        self.model.set_classes(WORLD_CLASSES)

    def analyze_scene(self, video_path: str, output_dir: str,
                      sample_rate: int = 15,
                      scene_start_sec: float = 0.0) -> list:
        """
        씬 영상에서 객체를 인식하고 타임스탬프를 정확히 계산합니다.

        :param video_path: 분석할 씬 영상 경로
        :param output_dir: 크롭 이미지를 저장할 디렉토리
        :param sample_rate: 몇 프레임마다 분석할지
        :param scene_start_sec: 원본 영상 기준 이 씬의 시작 시간(초) — 타임스탬프 정확도를 위해 필수
        :return: 객체별 통계 딕셔너리 리스트
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_area = frame_width * frame_height

        scene_name = os.path.splitext(os.path.basename(video_path))[0]
        crops_dir = os.path.join(output_dir, f"crops_{scene_name}")

        frame_count = 0
        object_stats = defaultdict(lambda: {
            'appearances': 0,
            'max_area_ratio': 0.0,
            'conf_sum': 0.0,
            'best_crop_path': None,
            'best_timestamp_sec': scene_start_sec,
        })

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # scene_start_sec + 현재 프레임의 상대 시간 = 원본 기준 절대 시간
                current_sec = scene_start_sec + (frame_count / fps)
                results = self.model(frame, verbose=False)

                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()

                    if conf < 0.3:
                        continue

                    class_name = self.model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / total_area

                    # 화면의 0.5% 미만 → 노이즈 제거
                    if area_ratio < 0.005:
                        continue

                    stats = object_stats[class_name]
                    stats['appearances'] += 1
                    stats['conf_sum'] += conf

                    # 가장 크게 잡힌 프레임의 이미지 + 타임스탬프 갱신
                    if area_ratio > stats['max_area_ratio']:
                        stats['max_area_ratio'] = area_ratio
                        stats['best_timestamp_sec'] = current_sec

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

        candidates = []
        for obj_name, stats in object_stats.items():
            avg_conf = stats['conf_sum'] / stats['appearances']
            score = stats['appearances'] * stats['max_area_ratio'] * avg_conf

            candidates.append({
                '씬 이름 (Scene)':        scene_name,
                '상품 종류 (Object)':      obj_name,
                '등장 프레임 수':           stats['appearances'],
                '화면 내 최대 비중 (%)':    round(stats['max_area_ratio'] * 100, 2),
                'AI 인식 신뢰도 (%)':       round(avg_conf * 100, 2),
                '광고 적합도 점수':          round(score, 4),
                '최고 등장 시간 (초)':       round(stats['best_timestamp_sec'], 2),
                '크롭 이미지 경로':          stats['best_crop_path'],
            })

        return candidates
