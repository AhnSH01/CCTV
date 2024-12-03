import cv2
import requests
import base64
import time
import threading

# 웹캡처 설정
cap = cv2.VideoCapture(0)

# API 엔드포인트 설정
api_url = "http://localhost:3000/stream"
# api_url = "http://172.23.126.70:3000/stream"

# API 요청 함수
def send_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode()
    payload = {"image": jpg_as_text}
    try:
        response = requests.post(api_url, json=payload)
    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")

last_update_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 캡처할 수 없습니다.")
        break

    frame = cv2.resize(frame, (320, 240))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # 쓰레드로 이미지 전송
    threading.Thread(target=send_frame, args=(frame,)).start()

    # 일정 시간마다 화면 갱신
    if time.time() - last_update_time >= 0.1:  # 100ms마다 화면을 갱신
        cv2.imshow('Webcam', frame)
        last_update_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)  # 100ms 간격으로 프레임 전송

cap.release()
cv2.destroyAllWindows()