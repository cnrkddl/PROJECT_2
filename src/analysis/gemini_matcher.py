"""
gemini_matcher.py — 씬 장면 설명 생성 모듈

[역할 변경 이력]
- 구(AS-IS): 크롭 이미지에서 색상/재질/스타일 키워드 추출 (Gemini에 "이 물건 어떻게 생겼어?" 질문)
- 신(TO-BE): 씬 대표 이미지 + 감지 객체 목록으로 자연어 장면 설명 텍스트 생성
             (Gemini에 "이 장면 어떤 상황이야?" 질문)

[생성된 장면 설명의 용도]
timetable_generator.py에서 PostgreSQL ad_inventory 테이블의 ad_name과
코사인 유사도(Cosine Similarity)를 계산하여 가장 잘 맞는 광고를 매칭하는 데 사용됩니다.

[파이프라인 위치]
STEP 2 (YOLO-World 탐지) → [현재 모듈] STEP 3 (장면 설명 생성) → STEP 4 (Whisper STT) → STEP 5 (광고 매칭 + 타임테이블)
"""

import os
import csv
import cv2
from dotenv import load_dotenv
from google import genai
from collections import defaultdict
from typing import List, Dict

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ──────────────────────────────────────────────────────────
# Gemini 호출 전 필터링 기준값
# — 이 값들을 조정하여 Gemini API 호출 횟수를 제어할 수 있습니다
# ──────────────────────────────────────────────────────────
MIN_APPEARANCES     = 3     # 최소 등장 프레임 수 (이 미만은 스쳐 지나간 것으로 판단)
MIN_AREA_RATIO_PCT  = 1.5   # 최소 화면 비중 % (이 미만은 너무 작아 광고 가치 없음)
MIN_SCORE           = 0.003 # 최소 광고 적합도 점수 (appearances × area × confidence)
TOP_PER_SCENE       = 3     # 씬당 상위 N개 객체만 유지 (씬 내 노이즈 제거)
TOP_GLOBAL          = 15    # 전체 최대 처리할 씬 수 (API 비용 상한선)
DEDUP_SAME_OBJECT   = True  # 동일 객체가 여러 씬에 나올 때 가장 점수 높은 것만 유지


class GeminiSceneDescriber:
    """
    YOLO-World가 감지한 씬별 객체 정보를 바탕으로,
    Gemini Vision을 사용해 자연어 장면 설명 텍스트를 생성하는 클래스입니다.

    최종 출력 scene_descriptions.csv에는 각 씬에 대해:
    - 씬 이름, 등장 시간(초)
    - 감지된 사물 목록 (쉼표 구분)
    - 오브젝트 밀도 (광고 타이밍 결정 보조 지표)
    - 장면 설명 (Gemini 생성 자연어 텍스트)
    가 저장됩니다.
    """

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = 'models/gemini-flash-latest'

    def describe_scene(self, crop_image_path: str, objects_in_scene: List[str]) -> str:
        """
        씬의 대표 크롭 이미지와 감지된 객체 목록을 Gemini에 전달하여
        자연어 장면 설명 텍스트를 생성합니다.

        [이미지 선택 기준]
        - YOLO가 저장한 크롭 이미지 중 광고 적합도 점수가 가장 높은 객체의 이미지를 사용합니다.
        - 씬 전체 화면이 아닌 크롭 이미지를 사용하는 이유:
          씬 영상 파일은 YOLO 분석 후 삭제되지만 크롭 이미지는 이 단계까지 유지됩니다.

        [이미지가 없는 경우]
        - 객체 이름만으로 간단한 텍스트를 반환합니다 (Gemini 호출 생략).

        :param crop_image_path: YOLO가 저장한 대표 크롭 이미지 경로
        :param objects_in_scene: 해당 씬에서 감지된 객체 이름 목록 (맥락 힌트)
        :return: 장면 설명 텍스트 (한국어, 2~3문장)
        """
        # ── 이미지 없으면 객체 목록 기반 폴백 텍스트 반환 ──
        if not os.path.exists(crop_image_path):
            return f"화면에 {', '.join(objects_in_scene)}이(가) 등장하는 장면"

        objects_hint = ", ".join(objects_in_scene) if objects_in_scene else "정보 없음"
        print(f"  [Gemini 장면 설명] {os.path.basename(crop_image_path)} | 사물: {objects_hint}")

        # ── Gemini에게 장면 설명 요청 ──────────────────────
        # 핵심 포인트: "이 물건은 뭐야?" (구 방식) → "이 장면은 어떤 상황이야?" (신 방식)
        # 생성된 설명이 ad_name("CJ제일제당 - 햇반", "G마켓 - 패션상품")과 코사인 유사도 계산에 쓰이므로
        # 사물의 특성과 장면 맥락을 풍부하게 포함하도록 유도합니다.
        prompt = f"""당신은 영상 장면 분석가입니다.
이 이미지는 드라마/영상에 등장한 사물의 캡처입니다.
YOLO가 감지한 이 장면의 주요 사물: {objects_hint}

이 사물들이 등장하는 장면의 분위기, 공간, 상황을 2~3문장으로 자연스럽게 한국어로 묘사해주세요.
광고 매칭에 활용할 수 있도록 사물의 특성과 장면 맥락을 포함해주세요.
마크다운, JSON 없이 순수 텍스트만 출력하세요.

예시: "거실 소파 옆 테이블에 노트북이 놓여 있는 아늑한 실내 장면. 따뜻한 조명과 원목 가구가 편안한 분위기를 연출하고 있다."
"""
        try:
            img_file = self.client.files.upload(file=crop_image_path)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img_file, prompt]
            )
            self.client.files.delete(name=img_file.name)
            return response.text.strip()
        except Exception as e:
            print(f"  [Gemini 오류] {e}")
            # 오류 시에도 파이프라인이 멈추지 않도록 폴백 텍스트 반환
            return f"화면에 {', '.join(objects_in_scene)}이(가) 등장하는 장면"

    def _filter_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Gemini API를 호출하기 전에 후보군을 3단계로 필터링합니다.
        목적: 의미 없는 객체(스쳐 지나감, 너무 작음)에 대한 불필요한 API 비용 절감

        [1단계] 절대 기준 필터
            - 등장 프레임 수 MIN_APPEARANCES 미만 → 스쳐 지나간 것으로 제거
            - 화면 비중 MIN_AREA_RATIO_PCT 미만 → 너무 작아 광고 가치 없음으로 제거
            - 광고 적합도 점수 MIN_SCORE 미만 → 복합 점수 기준 미달로 제거

        [2단계] 씬당 상위 TOP_PER_SCENE개 유지
            - 한 씬에 탐지된 객체가 많아도 가장 중요한 3개만 처리
            - 씬 내 중요도 낮은 배경 객체 제거

        [3단계] 동일 객체 전체 중복 제거 + 전체 TOP_GLOBAL캡
            - 예: "cell phone"이 20개 씬에 나와도 점수가 가장 높은 1개만 처리
            - 최종적으로 가장 중요한 TOP_GLOBAL개 씬만 Gemini에 전달
        """
        original_count = len(candidates)

        # ── 1단계: 절대 기준 ──────────────────────────────
        step1 = [
            c for c in candidates
            if int(c.get('등장 프레임 수', 0)) >= MIN_APPEARANCES
            and float(c.get('화면 내 최대 비중 (%)', 0)) >= MIN_AREA_RATIO_PCT
            and float(c.get('광고 적합도 점수', 0)) >= MIN_SCORE
        ]
        print(f"  [필터 1/3] 절대 기준 ({MIN_APPEARANCES}회↑, {MIN_AREA_RATIO_PCT}%↑, 점수{MIN_SCORE}↑): "
              f"{original_count}개 → {len(step1)}개")

        # ── 2단계: 씬당 상위 TOP_PER_SCENE개 ──────────────
        by_scene: Dict[str, List] = defaultdict(list)
        for c in step1:
            by_scene[c.get('씬 이름 (Scene)', '')].append(c)

        step2 = []
        for scene_cands in by_scene.values():
            top = sorted(scene_cands,
                         key=lambda x: float(x.get('광고 적합도 점수', 0)),
                         reverse=True)[:TOP_PER_SCENE]
            step2.extend(top)
        print(f"  [필터 2/3] 씬당 상위 {TOP_PER_SCENE}개: {len(step1)}개 → {len(step2)}개")

        # ── 3단계: 동일 객체 중복 제거 ────────────────────
        if DEDUP_SAME_OBJECT:
            best_per_object: Dict[str, Dict] = {}
            for c in step2:
                obj = c.get('상품 종류 (Object)', '')
                cur_score = float(c.get('광고 적합도 점수', 0))
                if obj not in best_per_object or \
                   cur_score > float(best_per_object[obj].get('광고 적합도 점수', 0)):
                    best_per_object[obj] = c
            step3 = list(best_per_object.values())
        else:
            step3 = step2
        print(f"  [필터 3/3] 객체 중복 제거: {len(step2)}개 → {len(step3)}개")

        # ── 최종 전역 캡 ────────────────────────────────
        final = sorted(step3,
                       key=lambda x: float(x.get('광고 적합도 점수', 0)),
                       reverse=True)[:TOP_GLOBAL]
        saved_calls = original_count - len(final)
        print(f"\n  ✅ 필터링 완료: {original_count}개 → Gemini 호출 {len(final)}번 "
              f"(API 호출 {saved_calls}회 절감)\n")
        return final

    def process_candidates(self, candidates_csv_path: str, output_csv_path: str):
        """
        YOLO 결과 CSV(all_scenes_candidates.csv)를 읽고,
        필터링 통과한 씬에 대해 Gemini 장면 설명을 생성합니다.

        [처리 흐름]
        1. YOLO 결과 CSV 읽기
        2. 3단계 필터링으로 Gemini 호출 대상 압축
        3. 같은 씬의 객체들을 씬 단위로 그룹화
           (예: scene-040에서 cup, laptop 2개가 있으면 한 번의 Gemini 호출로 처리)
        4. 씬마다 대표 크롭 이미지 선택 (점수 최고 객체의 이미지)
        5. Gemini에 장면 설명 요청
        6. scene_descriptions.csv 저장

        [출력 CSV 컬럼 설명]
        - 씬 이름 (Scene): scene-001 형식
        - 등장 시간 (초): 해당 씬에서 객체가 가장 잘 잡힌 원본 영상 기준 절대 시간
        - 감지된 사물 목록: "cup, laptop, cell phone" 형식 (쉼표 구분)
        - 오브젝트 밀도: 씬 내 객체들의 화면 비중 합산을 0~1 범위로 정규화
                        타이밍 결정 기준 중 하나 (0.3 이하 = 화면이 한산한 시점)
        - 장면 설명: Gemini가 생성한 한국어 자연어 텍스트 (ad_name 코사인 유사도 매칭에 사용)
        """
        if not os.path.exists(candidates_csv_path):
            print(f"후보군 CSV 파일을 찾을 수 없습니다: {candidates_csv_path}")
            return

        # ── YOLO 결과 CSV 읽기 ────────────────────────────
        candidates = []
        with open(candidates_csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append(row)

        print(f"\n[GeminiSceneDescriber] 전체 후보: {len(candidates)}개")

        # ── Gemini 호출 전 필터링 ─────────────────────────
        filtered = self._filter_candidates(candidates)
        if not filtered:
            print("필터링 후 처리할 후보가 없습니다. MIN_APPEARANCES 또는 MIN_AREA_RATIO_PCT를 낮춰보세요.")
            return

        # ── 씬 단위로 그룹화 ─────────────────────────────
        # 동일 씬에 여러 객체가 필터링을 통과한 경우, 한 번의 Gemini 호출로 함께 처리
        by_scene: Dict[str, List] = defaultdict(list)
        for c in filtered:
            by_scene[c.get('씬 이름 (Scene)', '')].append(c)

        scene_descriptions = []
        for scene_name, scene_cands in by_scene.items():
            # ── 씬 내 객체 이름 목록 수집 ──────────────────
            objects = [c.get('상품 종류 (Object)', '') for c in scene_cands]

            # ── 대표 크롭 이미지: 점수가 가장 높은 객체의 이미지 선택 ──
            best_cand = max(scene_cands, key=lambda x: float(x.get('광고 적합도 점수', 0)))
            crop_path = best_cand.get('크롭 이미지 경로', '')

            # ── 등장 시간: 씬 내에서 가장 이른 최고 등장 시각 ──
            # '최고 등장 시간 (초)' 필드는 vision_analyzer.py에서 추가된 정확한 절대 타임스탬프
            timestamps = [float(c.get('최고 등장 시간 (초)', 0)) for c in scene_cands
                          if c.get('최고 등장 시간 (초)', '')]
            best_timestamp = min(timestamps) if timestamps else 0.0

            # ── 오브젝트 밀도 계산 ────────────────────────
            # 씬 내 객체들의 화면 비중(%) 합산 후 0~1 범위로 정규화
            # 이 값이 0.3 이하이면 화면이 한산한 상태 → 광고 삽입 타이밍 후보로 분류
            object_density = sum(
                float(c.get('화면 내 최대 비중 (%)', 0)) for c in scene_cands
            ) / 100.0

            # ── Gemini 장면 설명 생성 ───────────────────────
            description = self.describe_scene(crop_path, objects)

            scene_descriptions.append({
                '씬 이름 (Scene)': scene_name,
                '등장 시간 (초)':   round(best_timestamp, 2),
                '감지된 사물 목록': ', '.join(objects),
                '오브젝트 밀도':    round(object_density, 4),
                '장면 설명':        description,
            })

        # ── 결과 CSV 저장 ─────────────────────────────────
        if scene_descriptions:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=scene_descriptions[0].keys())
                writer.writeheader()
                writer.writerows(scene_descriptions)
            print(f"\n🎉 장면 설명 생성 완료: {output_csv_path}  ({len(scene_descriptions)}개 씬)")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    processed_base = os.path.join(BASE_DIR, "data", "processed")
    if os.path.exists(processed_base):
        run_folders = [os.path.join(processed_base, d)
                       for d in os.listdir(processed_base)
                       if os.path.isdir(os.path.join(processed_base, d))]
        latest_run = max(run_folders, key=os.path.getmtime) if run_folders else processed_base
    else:
        latest_run = processed_base

    VISION_OUT_DIR   = os.path.join(latest_run, "vision_results")
    candidates_csv   = os.path.join(VISION_OUT_DIR, "all_scenes_candidates.csv")
    descriptions_csv = os.path.join(VISION_OUT_DIR, "scene_descriptions.csv")

    describer = GeminiSceneDescriber()
    describer.process_candidates(candidates_csv, descriptions_csv)
