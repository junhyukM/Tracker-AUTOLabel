import cv2
import time
import json
import sys
import os
from datetime import datetime

# 설정 파일 로드
with open("config_label.json", 'r', encoding='UTF-8') as file:
    config_json = json.load(file)

# 비디오 캡처
cap = cv2.VideoCapture(config_json["source"])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 설정값 로딩
scale = config_json['resolution_ratio']
SAVE_INTERVAL = config_json.get("save_interval_sec", 1.0)
CLASS_ID = 0

# 해상도 정보
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_width, target_height = 1920, 1080
if original_width > target_width or original_height > target_height:
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    processing_scale = min(width_scale, height_scale)
else:
    processing_scale = 1.0

print(f"원본 해상도: {original_width}x{original_height}")
print(f"처리 스케일: {processing_scale}")
print(f"스케일링 후 해상도: {int(original_width * processing_scale)}x{int(original_height * processing_scale)}")

# 저장 디렉토리
SAVE_DIR = "AUTO_label"
SAVE_DIR_images = f"{SAVE_DIR}\\images"
SAVE_DIR_labels = f"{SAVE_DIR}\\labels"
os.makedirs(SAVE_DIR_images, exist_ok=True)
os.makedirs(SAVE_DIR_labels, exist_ok=True)
# 클래스 이름 정의 (예: drone)
CLASS_NAMES = [config_json["classes"]]  # 혹은 여러 개: ["person", "car", "drone"]
# 저장 경로
CLASSES_FILE = os.path.join(SAVE_DIR_labels, "classes.txt")
# 파일 생성
with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
    for class_name in CLASS_NAMES:
        f.write(f"{class_name}\n")

# 상태 변수
tracker = None
bbox = None
selecting = False
tracking = False
frame = None
scaled_frame = None
last_save_time = 0


def select_roi(event, x, y, flags, param):
    global tracker, bbox, selecting, tracking, frame, scaled_frame, processing_scale, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        tracking = False
        bbox = (x, y, 0, 0)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        bbox = (bbox[0], bbox[1], x - bbox[0], y - bbox[1])
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if bbox[2] > 0 and bbox[3] > 0:
            original_x = int(bbox[0] / scale)
            original_y = int(bbox[1] / scale)
            original_w = int(bbox[2] / scale)
            original_h = int(bbox[3] / scale)
            scaled_x = int(original_x * processing_scale)
            scaled_y = int(original_y * processing_scale)
            scaled_w = int(original_w * processing_scale)
            scaled_h = int(original_h * processing_scale)

            scaled_width = int(original_width * processing_scale)
            scaled_height = int(original_height * processing_scale)
            scaled_x = max(0, scaled_x)
            scaled_y = max(0, scaled_y)
            scaled_w = min(scaled_width - scaled_x, scaled_w)
            scaled_h = min(scaled_height - scaled_y, scaled_h)

            if scaled_w <= 0 or scaled_h <= 0:
                print("유효하지 않은 ROI 크기입니다.")
                return

            scaled_bbox = (scaled_x, scaled_y, scaled_w, scaled_h)
            tracker = cv2.legacy.TrackerCSRT_create()
            try:
                success = tracker.init(scaled_frame, scaled_bbox)
                if success:
                    tracking = True
                    print(f"트래커 초기화 성공: {scaled_bbox}")
                else:
                    print("트래커 초기화 실패")
            except cv2.error as e:
                print(f"트래커 오류: {e}")


def main():
    global tracker, bbox, selecting, tracking, frame, scaled_frame, processing_scale, scale, last_save_time

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", select_roi)

    while True:
        ret, frame_original = cap.read()
        if not ret:
            break

        frame = frame_original.copy()
        if processing_scale < 1.0:
            scaled_width = int(original_width * processing_scale)
            scaled_height = int(original_height * processing_scale)
            scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))
        else:
            scaled_frame = frame.copy()

        if tracking and tracker is not None:
            success, track_bbox = tracker.update(scaled_frame)
            if success:
                x, y, w, h = [int(v) for v in track_bbox]
                p1x_scaled, p1y_scaled = x, y
                p2x_scaled, p2y_scaled = x + w, y + h

                # 스케일 복원 (scaled → original)
                p1x = int(p1x_scaled / processing_scale)
                p1y = int(p1y_scaled / processing_scale)
                p2x = int(p2x_scaled / processing_scale)
                p2y = int(p2y_scaled / processing_scale)

                # 디스플레이용 시각화
                cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), (0, 255, 255), 4)

                # 저장 주기 확인
                current_time = time.time()
                if current_time - last_save_time >= SAVE_INTERVAL:
                    last_save_time = current_time

                    # ✅ YOLOv8 형식 중심 좌표 및 크기 계산
                    x1 = min(p1x, p2x)
                    y1 = min(p1y, p2y)
                    x2 = max(p1x, p2x)
                    y2 = max(p1y, p2y)

                    bbox_cx = ((x1 + x2) / 2) / original_width
                    bbox_cy = ((y1 + y2) / 2) / original_height
                    bbox_width = (x2 - x1) / original_width
                    bbox_height = (y2 - y1) / original_height

                    # 파일명 기준 시각 생성
                    date_time = datetime.utcnow()
                    timestamp = date_time.strftime('%Y%m%d_%H%M%S')

                    # 저장 경로 설정
                    image_path = os.path.join(SAVE_DIR_images, f"{timestamp}.jpg")
                    label_path = os.path.join(SAVE_DIR_labels, f"{timestamp}.txt")

                    # 이미지 저장 (원본)
                    cv2.imwrite(image_path, frame_original)

                    # YOLO 라벨 저장
                    with open(label_path, 'w') as f:
                        f.write(f"{CLASS_ID} {bbox_cx:.6f} {bbox_cy:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

                    print(f"[{timestamp}] 저장 완료: {image_path}, {label_path}")

        display_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        if selecting and bbox:
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(display_frame, p1, p2, (255, 0, 0), 2)
            cv2.putText(display_frame, f"ROI: {p1} to {p2}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow('Tracking', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracking = False
            tracker = None
            bbox = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
