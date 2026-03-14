import os
import csv
import json
import psycopg2
from flask import Flask, render_template, send_file, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DSN = os.environ.get("DB_DSN")


def get_latest_run_dir():
    processed_base = os.path.join(BASE_DIR, 'data', 'processed')
    if not os.path.exists(processed_base):
        return None
    run_folders = [
        os.path.join(processed_base, d)
        for d in os.listdir(processed_base)
        if os.path.isdir(os.path.join(processed_base, d))
    ]
    return max(run_folders, key=os.path.getmtime) if run_folders else None


def fetch_ad_asset(ad_id: str) -> dict | None:
    """DB에서 ad_id로 광고 에셋 정보를 조회합니다."""
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute(
            "SELECT resource_path, ad_type FROM ad_inventory WHERE ad_id = %s",
            (ad_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {'resource_path': row[0], 'ad_type': row[1]}
    except Exception as e:
        print(f"[DB 오류] {e}")
    return None


def parse_timetable():
    run_dir = get_latest_run_dir()
    if not run_dir:
        return []

    timetable_csv = os.path.join(run_dir, 'final_ad_timetable.csv')
    if not os.path.exists(timetable_csv):
        return []

    timetable = []
    with open(timetable_csv, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ad_id = row.get('매칭 광고 ID', '').strip()
            asset = fetch_ad_asset(ad_id) if ad_id else None

            timetable.append({
                'start_time': float(row.get('광고 진입 시간 (초)', 0)),
                'end_time':   float(row.get('광고 종료 시간 (초)', 0)),
                'ad_id':      ad_id,
                'ad_name':    row.get('광고 이름', '').strip(),
                'ad_type':    row.get('광고 유형', '').strip(),
                'similarity': row.get('코사인 유사도', ''),
                'scene_desc': row.get('장면 설명 요약', '').strip(),
                'asset_url':  f'/ad-asset/{ad_id}' if asset else None,
            })

    print(f"Total ads in timetable: {len(timetable)}")
    return timetable


@app.route('/')
def index():
    timetable = parse_timetable()
    return render_template('index.html', timetable_json=json.dumps(timetable))


@app.route('/video')
def serve_video():
    return send_file('/Users/bagjimin/Desktop/sampleVideo.mp4')


@app.route('/ad-asset/<ad_id>')
def serve_ad_asset(ad_id):
    asset = fetch_ad_asset(ad_id)
    if asset and asset['resource_path'] and os.path.exists(asset['resource_path']):
        return send_file(asset['resource_path'])
    return jsonify({'error': 'Asset not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)
