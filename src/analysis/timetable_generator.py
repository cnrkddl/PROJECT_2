"""
timetable_generator.py — 광고 타임테이블 생성 모듈

실제 DB의 ad_inventory.ad_name과 장면 설명을 코사인 유사도로 매칭
             → 반드시 실존하는 광고만 추천, 유사도 점수로 매칭 근거 명시

[파이프라인 위치]
STEP 3 (장면 설명 생성) → STEP 4 (Whisper STT) → [현재 모듈] STEP 5 (광고 매칭 + 타임테이블)

[광고 삽입 타이밍 기준 — 3조건 중 2개 이상 충족 필요]
  조건 1 (1순위): 묵음 2초 이상 — librosa로 감지 (대사 공백 = 광고 삽입 최적 타이밍)
  조건 2 (2순위): 씬 전환 후 3초 이내 — scenedetect 결과 활용 (장면 전환 직후)
  조건 3 (보조):  오브젝트 밀도 0.3 이하 — 화면이 한산한 시점 (스토리 흐름 덜 방해)
"""

import os
import csv
import librosa
import numpy as np
from dotenv import load_dotenv
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

from src.database.ad_inventory import load_ad_inventory

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ──────────────────────────────────────────────────────────
# 타이밍 기준값
# ──────────────────────────────────────────────────────────
MIN_SILENCE_SEC      = 2.0   # 묵음 판단 최소 길이(초) — 이 이상 조용해야 조건 충족
SILENCE_TOP_DB       = 40    # librosa 묵음 감지 민감도 (낮을수록 더 많이 묵음으로 판단)
SCENE_TRANSITION_SEC = 3.0   # 씬 전환 후 N초 이내 — 이 안에 들어오면 조건 충족
MAX_OBJECT_DENSITY   = 0.3   # 오브젝트 밀도 상한 — 이하면 화면이 한산하다고 판단
MIN_CONDITIONS       = 2     # 3조건 중 최소 충족 개수 (미달 시 타임테이블 제외)

# ──────────────────────────────────────────────────────────
# 코사인 유사도 기준값
# ──────────────────────────────────────────────────────────
MIN_SIMILARITY = 0.25   # 이 미만의 유사도는 매칭으로 인정하지 않음
TOP_K_ADS      = 1      # 씬당 추천 광고 수 (현재는 1위만)


class AdTimetableGenerator:
    """
    장면 설명(scene_descriptions.csv)과 DB의 광고 이름(ad_inventory.ad_name)을
    Gemini 임베딩 기반 코사인 유사도로 매칭하고,
    librosa 묵음 감지 + 씬 전환 타이밍으로 광고 삽입 시점을 결정합니다.
    """

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.embedding_model = 'models/gemini-embedding-001'

    @staticmethod
    def _fmt_time(sec: float) -> str:
        """초 → HH:MM:SS 또는 MM:SS 포맷으로 변환"""
        h = int(sec) // 3600
        m = (int(sec) % 3600) // 60
        s = int(sec) % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # ──────────────────────────────────────────────────────
    # [2단계] Gemini 임베딩 생성
    # ──────────────────────────────────────────────────────
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 목록을 Gemini text-embedding-004 모델로 임베딩합니다.
        한국어 의미론적 유사도 계산에 최적화되어 있습니다.

        반환: shape (N, 768) numpy 배열
        오류 발생 시 해당 텍스트는 zero vector로 처리 (파이프라인 중단 방지)
        """
        vectors = []
        for text in texts:
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text
                )
                vectors.append(response.embeddings[0].values)
            except Exception as e:
                print(f"  [임베딩 오류] '{text[:30]}...' → {e}")
                vectors.append([0.0] * 768)
        return np.array(vectors, dtype=np.float32)

    # ──────────────────────────────────────────────────────
    # [3단계] 묵음 구간 감지
    # ──────────────────────────────────────────────────────
    def _detect_silence_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        librosa로 오디오 파일에서 묵음 구간(MIN_SILENCE_SEC 이상)을 감지합니다.

        [동작 방식]
        1. librosa.effects.split()으로 소리가 있는 구간을 추출
        2. 소리 있는 구간들 사이의 간격(= 묵음 구간)을 계산
        3. MIN_SILENCE_SEC(2초) 이상인 묵음 구간만 반환

        반환: [(묵음시작초, 묵음종료초), ...] 리스트
        """
        if not os.path.exists(audio_path):
            print(f"  [묵음 감지] 오디오 파일 없음: {audio_path}")
            return []

        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            # 소리가 있는 구간 추출 (SILENCE_TOP_DB 이상의 음량)
            non_silent = librosa.effects.split(y, top_db=SILENCE_TOP_DB)

            silence_segments = []
            prev_end_sec = 0.0

            for start_frame, end_frame in non_silent:
                start_sec = start_frame / sr
                # 이전 소리 구간 종료 ~ 현재 소리 구간 시작 = 묵음 구간
                gap = start_sec - prev_end_sec
                if gap >= MIN_SILENCE_SEC:
                    silence_segments.append((round(prev_end_sec, 2), round(start_sec, 2)))
                prev_end_sec = end_frame / sr

            print(f"  -> 묵음 구간 {len(silence_segments)}개 감지 ({MIN_SILENCE_SEC}초 이상)")
            return silence_segments
        except Exception as e:
            print(f"  [librosa 오류] {e}")
            return []

    # ──────────────────────────────────────────────────────
    # [4단계] 타이밍 3조건 체크
    # ──────────────────────────────────────────────────────
    def _check_timing_conditions(self,
                                  timestamp_sec: float,
                                  object_density: float,
                                  silence_segments: List[Tuple[float, float]],
                                  scene_start_times: List[float]) -> Tuple[int, List[str]]:
        """
        특정 시점(timestamp_sec)이 광고 삽입 타이밍 3조건을 몇 개 충족하는지 체크합니다.

        [조건 1] 묵음 2초 이상
            - timestamp_sec 기준 ±2초 윈도우 내에 묵음 구간이 존재하는지 확인
            - 묵음 구간이 timestamp와 겹치면 충족

        [조건 2] 씬 전환 후 3초 이내
            - 가장 가까운 씬 시작 시각으로부터 timestamp_sec까지의 거리가
              SCENE_TRANSITION_SEC(3초) 이하이면 충족

        [조건 3] 오브젝트 밀도 0.3 이하 (보조 조건)
            - scene_descriptions.csv에서 계산된 오브젝트 밀도가 MAX_OBJECT_DENSITY 이하이면 충족
            - 단독으로는 광고 삽입을 결정하지 않고, 다른 조건과 함께 쓰임

        반환: (충족 조건 수, 충족된 조건 이름 리스트)
        """
        conditions_met = []
        window = 2.0  # 묵음 판단 시 타임스탬프 주변 ±2초 윈도우

        # ── 조건 1: 묵음 구간 ─────────────────────────────
        for sil_start, sil_end in silence_segments:
            # 묵음 구간이 타임스탬프 ±window 범위와 겹치는지 확인
            if sil_start <= (timestamp_sec + window) and sil_end >= (timestamp_sec - window):
                conditions_met.append("묵음2초↑")
                break

        # ── 조건 2: 씬 전환 후 3초 이내 ──────────────────
        for scene_start in scene_start_times:
            diff = timestamp_sec - scene_start
            if 0 <= diff <= SCENE_TRANSITION_SEC:
                conditions_met.append("씬전환3초↓")
                break

        # ── 조건 3: 오브젝트 밀도 낮음 ──────────────────
        if object_density <= MAX_OBJECT_DENSITY:
            conditions_met.append("밀도0.3↓")

        return len(conditions_met), conditions_met

    # ──────────────────────────────────────────────────────
    # [메인] 타임테이블 생성
    # ──────────────────────────────────────────────────────
    def generate_timetable(self,
                            scene_descriptions_csv: str,
                            transcript_csv: str,
                            scene_timestamps_csv: str,
                            audio_path: str,
                            output_csv_path: str):
        """
        전체 타임테이블 생성 메인 함수.

        [처리 흐름]
        1. DB에서 ad_inventory 로드 (ad_id, ad_name, duration_sec 등)
        2. 장면 설명(scene_descriptions.csv) 로드
        3. Gemini text-embedding-004로 장면 설명 + 광고 이름 전부 임베딩
        4. 코사인 유사도 행렬 계산 → 씬별 최적 광고(1위) 선택
        5. librosa로 묵음 구간 감지
        6. 씬 타임스탬프 CSV에서 씬 전환 시각 목록 추출
        7. 각 씬-광고 후보에 대해 타이밍 3조건 체크
           → MIN_CONDITIONS(2)개 이상 충족한 경우만 타임테이블 등록
        8. 최종 결과 CSV 저장

        [입력 파일]
        - scene_descriptions_csv: gemini_matcher.py 출력 (씬이름, 등장시간, 사물목록, 오브젝트밀도, 장면설명)
        - transcript_csv:          audio_analyzer.py 출력 (대사 텍스트, 타임스탬프)
        - scene_timestamps_csv:    run_pipeline.py에서 저장 (씬이름, 시작초, 종료초)
        - audio_path:              추출된 오디오 WAV 파일 (묵음 감지용)

        [출력 CSV 컬럼]
        - 광고 진입 시간 (초), 광고 종료 시간 (초)
        - 씬 이름, 장면 설명 요약
        - 매칭 광고 ID, 광고 이름, 광고 유형
        - 코사인 유사도 점수 (0~1, 높을수록 잘 맞음)
        - 충족 타이밍 조건 (어떤 조건들을 만족했는지)
        """

        # ── 1단계: DB 광고 목록 로드 ─────────────────────
        # DB 연결 로직은 src/database/ad_inventory.py에서 관리
        print("\n[STEP 5-1] DB에서 광고 목록 로드 중...")
        ads = load_ad_inventory()
        if not ads:
            print("  -> 광고 목록이 비어있어 타임테이블 생성을 중단합니다.")
            return

        # ── 2단계: 장면 설명 CSV 로드 ────────────────────
        print("\n[STEP 5-2] 장면 설명 데이터 로드 중...")
        if not os.path.exists(scene_descriptions_csv):
            print(f"  -> 장면 설명 파일 없음: {scene_descriptions_csv}")
            return

        scenes = []
        with open(scene_descriptions_csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenes.append(row)
        print(f"  -> 장면 {len(scenes)}개 로드")

        # ── 씬 타임스탬프 로드 (조건 2: 씬 전환 시각 목록) ─
        scene_start_times = []
        if os.path.exists(scene_timestamps_csv):
            with open(scene_timestamps_csv, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        scene_start_times.append(float(row.get('시작 시간 (초)', 0)))
                    except ValueError:
                        pass
        print(f"  -> 씬 전환 시각 {len(scene_start_times)}개 로드")

        # ── 3단계: 임베딩 생성 ───────────────────────────
        print("\n[STEP 5-3] Gemini 임베딩 생성 중...")
        scene_texts = [row.get('장면 설명', '') for row in scenes]
        ad_texts    = [ad['ad_name'] for ad in ads]

        print(f"  -> 장면 설명 {len(scene_texts)}개 임베딩 중...")
        scene_embeddings = self._embed_texts(scene_texts)   # shape: (씬 수, 768)

        print(f"  -> 광고 이름 {len(ad_texts)}개 임베딩 중...")
        ad_embeddings = self._embed_texts(ad_texts)          # shape: (광고 수, 768)

        # ── 4단계: 코사인 유사도 계산 ────────────────────
        print("\n[STEP 5-4] 코사인 유사도 매칭 중...")
        # similarity_matrix: shape (씬 수, 광고 수)
        # similarity_matrix[i][j] = i번째 씬과 j번째 광고의 유사도
        similarity_matrix = cosine_similarity(scene_embeddings, ad_embeddings)

        # 씬별 Top-K 광고 인덱스 추출
        # argsort로 내림차순 정렬 후 상위 TOP_K_ADS개만 선택
        top_ad_indices = np.argsort(-similarity_matrix, axis=1)[:, :TOP_K_ADS]

        # ── 5단계: 묵음 구간 감지 ───────────────────────
        print("\n[STEP 5-5] 오디오 묵음 구간 감지 중...")
        silence_segments = self._detect_silence_segments(audio_path)

        # ── 6단계: 타이밍 조건 체크 + 타임테이블 구성 ──
        print("\n[STEP 5-6] 타이밍 조건 체크 및 타임테이블 구성 중...")
        timetable = []

        for i, scene in enumerate(scenes):
            scene_name      = scene.get('씬 이름 (Scene)', '')
            timestamp_sec   = float(scene.get('등장 시간 (초)', 0))
            object_density  = float(scene.get('오브젝트 밀도', 1.0))
            scene_desc      = scene.get('장면 설명', '')

            # 타이밍 3조건 체크
            cond_count, cond_names = self._check_timing_conditions(
                timestamp_sec, object_density, silence_segments, scene_start_times
            )

            # 조건 미달 시 이 씬은 타임테이블에서 제외
            if cond_count < MIN_CONDITIONS:
                print(f"  [SKIP] {scene_name} ({self._fmt_time(timestamp_sec)}) — 조건 {cond_count}/{MIN_CONDITIONS}: {cond_names}")
                continue

            # 코사인 유사도 최상위 광고 선택
            best_ad_idx  = top_ad_indices[i][0]
            best_sim     = float(similarity_matrix[i][best_ad_idx])
            best_ad      = ads[best_ad_idx]

            # 유사도 임계값 미달 시 제외
            if best_sim < MIN_SIMILARITY:
                print(f"  [SKIP] {scene_name} — 유사도 {best_sim:.3f} < {MIN_SIMILARITY}")
                continue

            ad_duration   = float(best_ad.get('duration_sec') or 10.0)
            ad_end_sec    = round(timestamp_sec + ad_duration, 2)

            print(f"  ✅ {scene_name} ({self._fmt_time(timestamp_sec)}) | 조건: {cond_names} | "
                  f"유사도: {best_sim:.3f} | 광고: {best_ad['ad_name']}")

            timetable.append({
                '광고 진입 시간 (초)':   timestamp_sec,
                '광고 종료 시간 (초)':   ad_end_sec,
                '광고 진입 시간':         self._fmt_time(timestamp_sec),
                '광고 종료 시간':         self._fmt_time(ad_end_sec),
                '씬 이름':               scene_name,
                '장면 설명 요약':         scene_desc[:80] + ('...' if len(scene_desc) > 80 else ''),
                '매칭 광고 ID':           best_ad['ad_id'],
                '광고 이름':              best_ad['ad_name'],
                '광고 유형':              best_ad.get('ad_type', ''),
                '코사인 유사도':           round(best_sim, 4),
                '충족 타이밍 조건':        ' / '.join(cond_names),
            })

        # ── 7단계: CSV 저장 ──────────────────────────────
        if timetable:
            # 광고 진입 시간 기준 오름차순 정렬
            timetable.sort(key=lambda x: x['광고 진입 시간 (초)'])
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=timetable[0].keys())
                writer.writeheader()
                writer.writerows(timetable)
            print(f"\n🎉 [타임테이블 생성 완료] {output_csv_path}  ({len(timetable)}개 광고 편성)")

            # 터미널 요약 출력
            for entry in timetable[:5]:
                print(f"  ⏰ {entry['광고 진입 시간']} ~ {entry['광고 종료 시간']}"
                      f" | [{entry['충족 타이밍 조건']}]"
                      f" | {entry['광고 이름']} (유사도: {entry['코사인 유사도']})")
        else:
            print("\n⚠️  타이밍 조건을 충족하고 유사도가 높은 광고 삽입 구간이 없습니다.")
            print("   MIN_CONDITIONS 또는 MIN_SIMILARITY 값을 조정해보세요.")


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

    SCENE_DESCRIPTIONS_CSV = os.path.join(latest_run, "vision_results", "scene_descriptions.csv")
    TRANSCRIPT_CSV         = os.path.join(latest_run, "audio", "transcript.csv")
    SCENE_TIMESTAMPS_CSV   = os.path.join(latest_run, "vision_results", "scene_timestamps.csv")
    AUDIO_PATH             = os.path.join(latest_run, "audio", "extracted_audio.wav")
    TIMETABLE_CSV          = os.path.join(latest_run, "final_ad_timetable.csv")

    generator = AdTimetableGenerator()
    generator.generate_timetable(
        SCENE_DESCRIPTIONS_CSV,
        TRANSCRIPT_CSV,
        SCENE_TIMESTAMPS_CSV,
        AUDIO_PATH,
        TIMETABLE_CSV,
    )
