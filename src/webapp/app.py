import os
import csv
import json
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

# Base paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'SampleVideo_20260305_171923')
BANNERS_DIR = os.path.join(DATA_DIR, 'generated_ad_banners')
VIDEO_PATH = os.path.join(DATA_DIR, 'SampleVideo.mp4')
TIMETABLE_CSV = os.path.join(DATA_DIR, 'final_ad_timetable.csv')

def parse_timetable():
    timetable = []
    if not os.path.exists(TIMETABLE_CSV):
        return timetable
    
    with open(TIMETABLE_CSV, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_time_str = row.get('광고 진입 시간 (초)', '').replace('s', '').strip()
            end_time_str = row.get('광고 종료 시간 (초)', '').replace('s', '').strip()
            item_name = row.get('추천 광고 상품명', '').strip()
            ui_type = row.get('추천 UI/UX 노출 형태', '').strip()
            
            if not start_time_str or not end_time_str or not item_name:
                continue
                
            start_time = float(start_time_str)
            end_time = float(end_time_str)
            
            # Find the corresponding banner image
            banner_image = None
            if os.path.exists(BANNERS_DIR):
                for filename in os.listdir(BANNERS_DIR):
                    if filename.endswith(".png"):
                        # Basic matching logic: checking if part of the recommendation title is in the filename
                        # Since the filename is generated like: ad_banner_01_프리미엄_인체공학_사무용_의자.png
                        # We can just extract the name part or check for substring
                        safe_name = item_name.replace(" ", "_")
                        safe_name = safe_name.replace("/", "")
                        if safe_name in filename or item_name in filename.replace("_", " "):
                            banner_image = filename
                            break
            
            reason = row.get('광고 매칭 사유', '').strip()

            if banner_image:
                print(f"Matched banner: {banner_image} for item: {item_name}")
                timetable.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'item_name': item_name,
                    'ui_type': ui_type,
                    'reason': reason,
                    'image_url': f'/banners/{banner_image}'
                })
            else:
                print(f"WARNING: Could not find banner for {item_name}")
                
    print(f"Total banners matched: {len(timetable)}")
    return timetable

@app.route('/')
def index():
    timetable = parse_timetable()
    return render_template('index.html', timetable_json=json.dumps(timetable))

@app.route('/video')
def serve_video():
    return send_from_directory('/Users/bagjimin/Desktop', 'sampleVideo.mp4')

@app.route('/banners/<filename>')
def serve_banner(filename):
    return send_from_directory(BANNERS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
