"""
ad_inventory.py — PostgreSQL DB 연결 및 광고 목록 조회 모듈

timetable_generator.py에서 코사인 유사도 매칭에 사용할
ad_inventory 테이블의 광고 목록을 가져오는 역할을 담당합니다.
"""

import psycopg2
from typing import List, Dict

# ──────────────────────────────────────────────────────────
# DB 연결 정보
# ──────────────────────────────────────────────────────────
DB_DSN = "postgresql://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME"


def load_ad_inventory() -> List[Dict]:
    """
    PostgreSQL ad_inventory 테이블에서 광고 목록을 가져옵니다.

    [ad_inventory 테이블 스키마]
    - ad_id         : 광고 고유 ID
    - ad_name       : 광고 이름 (코사인 유사도 매칭의 기준 텍스트)
                      예시: "CJ제일제당 - 햇반", "G마켓 - H.O.T. 스타일"
    - ad_type       : 광고 유형 ('video_clip' | 'banner')
    - resource_path : 광고 파일 경로
    - duration_sec  : 광고 길이 (초)

    반환: 광고 딕셔너리 리스트
    """
    ads = []
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("""
            SELECT ad_id, ad_name, ad_type, resource_path, duration_sec
            FROM ad_inventory
            WHERE ad_name IS NOT NULL AND ad_name != ''
        """)
        for row in cur.fetchall():
            ads.append({
                'ad_id':         row[0],
                'ad_name':       row[1],
                'ad_type':       row[2],
                'resource_path': row[3],
                'duration_sec':  row[4],
            })
        cur.close()
        conn.close()
        print(f"  -> DB에서 광고 {len(ads)}개 로드 완료")
    except Exception as e:
        print(f"  [DB 오류] {e}")
    return ads
