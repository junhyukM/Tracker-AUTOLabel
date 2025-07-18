'''
CSRT_mqtt2 코드가 아무래도 4k 영상은 tracking 시 딜레이 발생하는 것 같아서
FHD로 스케일링하는 코드 추가, Desktop으로는 어느정도 실시간에 적용 가능해보이긴함..
'''
import multiprocessing
import cv2
import time
from datetime import datetime
import multiprocessing as mp
import threading
import json
import sys
import os

# json 파일 불러오기
with open("config.json", 'r', encoding='UTF-8') as file:
    config_json = json.load(file)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(config_json["source"])

# cap = cv2.VideoCapture(config_json["source"], cv2.CAP_DSHOW)  # Windows에서는 DirectShow 사용
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화

# ROI 선택 및 출력 영상의 크기 비율 (0.5 = 50% 크기)
scale = config_json['resolution_ratio']
tracker = None
bbox = None
selecting = False
tracking = False
frame = None
scaled_frame = None

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FHD 해상도 기준 (1920 x 1080)
target_width = 1920
target_height = 1080

# 원본 영상 해상도가 FHD 이상인 경우 스케일링 비율 계산
processing_scale = 1.0
if original_width > target_width or original_height > target_height:
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    processing_scale = min(width_scale, height_scale)  # 비율 유지를 위해 더 작은 비율 선택
else:
    processing_scale = 1.0  # 이미 FHD 이하면 스케일링 안 함

print(f"원본 해상도: {original_width}x{original_height}")
print(f"처리 스케일: {processing_scale}")
print(f"스케일링 후 해상도: {int(original_width * processing_scale)}x{int(original_height * processing_scale)}")

# 메시지 큐 생성
detect_queue = mp.Queue()


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
            # 디스플레이 화면에서 선택된 ROI를 스케일링된 영상 좌표로 변환

            # 1. 먼저 디스플레이 좌표를 원본 이미지 좌표로 변환
            original_x = int(bbox[0] / scale)
            original_y = int(bbox[1] / scale)
            original_w = int(bbox[2] / scale)
            original_h = int(bbox[3] / scale)

            # 2. 원본 이미지 좌표를 스케일링된 영상 좌표로 변환
            scaled_x = int(original_x * processing_scale)
            scaled_y = int(original_y * processing_scale)
            scaled_w = int(original_w * processing_scale)
            scaled_h = int(original_h * processing_scale)

            # 좌표가 스케일링된 이미지 경계 내에 있는지 확인
            scaled_width = int(original_width * processing_scale)
            scaled_height = int(original_height * processing_scale)

            # 경계를 벗어나는 좌표 보정
            if scaled_x < 0:
                scaled_x = 0
            if scaled_y < 0:
                scaled_y = 0
            if scaled_x + scaled_w > scaled_width:
                scaled_w = scaled_width - scaled_x
            if scaled_y + scaled_h > scaled_height:
                scaled_h = scaled_height - scaled_y

            # 유효한 ROI인지 확인
            if scaled_w <= 0 or scaled_h <= 0:
                print("유효하지 않은 ROI 크기입니다. 다시 선택해주세요.")
                return

            scaled_bbox = (scaled_x, scaled_y, scaled_w, scaled_h)

            # 트래커 초기화 (스케일링된 영상에서 동작)
            tracker = cv2.legacy.TrackerCSRT_create()
            try:
                success = tracker.init(scaled_frame, scaled_bbox)
                if success:
                    tracking = True
                    print(f"트래커 초기화 성공(스케일링 영상 기준): {scaled_bbox}")
                    print(f"스케일링된 이미지 크기: {scaled_frame.shape[1]}x{scaled_frame.shape[0]}")
                else:
                    print("트래커 초기화 실패")
            except cv2.error as e:
                print(f"트래커 초기화 오류: {e}")
                print(f"ROI: {scaled_bbox}")
                print(f"이미지 크기: {scaled_frame.shape[1]}x{scaled_frame.shape[0]}")


def mqtt_publisher(detect_queue, broker_address="localhost", broker_port=1883, topic="tracking/data"):
    """
    MQTT 메시지 발행 프로세스
    지정된 브로커 주소로 실제 메시지를 발행함

    매개변수:
    - detect_queue: 전송할 메시지가 있는 큐
    - broker_address: MQTT 브로커 주소 (기본값: localhost)
    - broker_port: MQTT 브로커 포트 (기본값: 1883)
    - topic: 발행할 MQTT 토픽 (기본값: tracking/data)
    """
    import paho.mqtt.client as mqtt
    import json
    import time

    # MQTT 클라이언트 생성
    client = mqtt.Client()

    # 연결 콜백 함수 정의
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"MQTT 브로커 {broker_address}:{broker_port}에 연결됨")
        else:
            print(f"MQTT 브로커 연결 실패, 반환 코드: {rc}")

    # 발행 콜백 함수 정의
    def on_publish(client, userdata, mid):
        print(f"메시지 ID {mid} 발행 완료")

    # 콜백 함수 등록
    client.on_connect = on_connect
    client.on_publish = on_publish

    # 브로커 연결 시도
    try:
        print(f"MQTT 브로커 {broker_address}:{broker_port}에 연결 시도 중...")
        client.connect(broker_address, broker_port, 60)
        client.loop_start()
    except Exception as e:
        print(f"MQTT 브로커 연결 오류: {e}")
        return

    print(f"MQTT 발행자 시작됨 (토픽: {topic})")

    # 메시지 발행 루프
    try:
        while True:
            if not detect_queue.empty():
                message = detect_queue.get()

                # JSON 형식으로 변환
                try:
                    # 이미 JSON 문자열인 경우 그대로 사용, 아니면 객체를 JSON으로 변환
                    if isinstance(message, str):
                        try:
                            # 유효한 JSON인지 확인
                            json.loads(message)
                            payload = message
                        except json.JSONDecodeError:
                            # 일반 문자열이면 JSON으로 변환
                            payload = json.dumps({"data": message})
                    else:
                        payload = json.dumps(message)

                    # 메시지 발행
                    result = client.publish(topic, payload)
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print(f"메시지 발행 성공: {message}")
                    else:
                        print(f"메시지 발행 실패, 코드: {result.rc}")

                except Exception as e:
                    print(f"메시지 변환 또는 발행 오류: {e}")

            time.sleep(0.01)  # CPU 사용량 절약

    except KeyboardInterrupt:
        print("MQTT 발행자 종료 중...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("MQTT 발행자 종료됨")


def main():
    global tracker, bbox, selecting, tracking, frame, scaled_frame, detect_queue, processing_scale, scale

    # 멀티프로세싱 설정 (Windows 호환성)
    if sys.platform.lower() == "win32" or os.name.lower() == "nt":
        from asyncio import set_event_loop_policy, WindowsSelectorEventLoopPolicy
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    # 마우스 콜백 설정
    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", select_roi)

    # json 파일 불러오기
    with open("config.json", 'r', encoding='UTF-8') as file:
        config_json = json.load(file)

    # MQTT 발행 스레드 시작 (별도 프로세스 대신 스레드 사용)
    mqtt_thread = threading.Thread(target=mqtt_publisher,
                                   args=(detect_queue, config_json['mqtt_server'], 1883, config_json['detect_topic']))
    mqtt_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
    mqtt_thread.start()

    send_time = time.time()

    while True:
        ret, frame_original = cap.read()

        if not ret:
            break

        # 전역 변수 frame 업데이트
        frame = frame_original.copy()

        # 고해상도 영상을 FHD 수준으로 스케일링 (트래킹 계산용)
        if processing_scale < 1.0:
            scaled_width = int(original_width * processing_scale)
            scaled_height = int(original_height * processing_scale)
            scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))
        else:
            scaled_frame = frame.copy()

        if tracking and tracker is not None:
            # 스케일링된 영상에서 트래킹 수행
            success, track_bbox = tracker.update(scaled_frame)

            if success:
                x, y, w, h = [int(v) for v in track_bbox]

                # 스케일링된 좌표계에서의 좌표
                p1x_scaled = x
                p1y_scaled = y
                p2x_scaled = x + w
                p2y_scaled = y + h
                cx_scaled = int((p1x_scaled + p2x_scaled) / 2)
                cy_scaled = int((p1y_scaled + p2y_scaled) / 2)

                # 스케일링된 좌표를 원본 크기로 변환
                p1x = int(p1x_scaled / processing_scale)
                p1y = int(p1y_scaled / processing_scale)
                p2x = int(p2x_scaled / processing_scale)
                p2y = int(p2y_scaled / processing_scale)
                cx = int(cx_scaled / processing_scale)
                cy = int(cy_scaled / processing_scale)

                # 원본 프레임에 사각형 및 중심점 표시
                cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), (0, 255, 255), 4)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                # MQTT 메시지 준비
                detect_time = time.time()
                # 유닉스 타임스탬프를 UTC 시간 기준의 datetime
                date_time = datetime.utcfromtimestamp(detect_time)
                # 초 부분을 소수 셋째 자리까지 포함
                formatted_time = date_time.strftime('%Y-%m-%d %H:%M:%S') + f".{date_time.microsecond // 1000:03d}"

                # 정규화 처리
                normalized_p1x = p1x / original_width
                normalized_p1y = p1y / original_height
                normalized_p2x = p2x / original_width
                normalized_p2y = p2y / original_height
                normalized_cx = cx / original_width
                normalized_cy = cy / original_height

                # MQTT 메시지 구성
                message = {
                    "version": "1.0.0",
                    "current_time": formatted_time,
                    "camera_node": 0,
                    "obj_no": 1,
                    "count": 0,
                    "box_left": normalized_p1x,
                    "box_right": normalized_p2x,
                    "box_top": normalized_p1y,
                    "box_bottom": normalized_p2y,
                    "center_x": normalized_cx,
                    "center_y": normalized_cy,
                }

                # 메시지를 JSON 문자열로 변환
                detect_message = json.dumps(message)

                # 메시지 발행 빈도 제한 (0.1초 간격)
                if time.time() - send_time > 0.1:
                    send_time = time.time()
                    detect_queue.put(detect_message)
            else:
                # 트래킹 실패 시 표시
                cv2.putText(frame, "Tracking failed!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 화면에 표시할 프레임 생성 (디스플레이용 스케일링)
        display_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        # ROI 선택 중일 때 사각형 표시
        if selecting and bbox:
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(display_frame, p1, p2, (255, 0, 0), 2)

            # 현재 좌표 표시
            cv2.putText(display_frame, f"ROI: {p1} to {p2}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 트래킹 상태 및 해상도 정보 표시
        # status = "Tracking" if tracking else "Not tracking"
        # resolution_info = f"Original: {original_width}x{original_height}, Processing: {scaled_frame.shape[1]}x{scaled_frame.shape[0]}"
        #
        # cv2.putText(display_frame, status, (10, display_frame.shape[0] - 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if tracking else (0, 0, 255), 1)
        # cv2.putText(display_frame, resolution_info, (10, display_frame.shape[0] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Tracking', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # r 키를 누르면 트래킹 리셋
            tracking = False
            tracker = None
            bbox = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Windows에서 multiprocessing 지원 활성화
    multiprocessing.freeze_support()
    main()