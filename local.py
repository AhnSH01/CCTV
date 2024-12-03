from flask import Flask, Response, request, jsonify
import threading
import cv2
import numpy as np
import os
import datetime
from PIL import ImageFont, ImageDraw, Image
import requests
import base64

app = Flask(__name__)

# 업로드된 이미지 저장 폴더 설정
UPLOAD_FOLDER = './uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 얼굴 인식 모델 및 레이블 로드
model_path = 'face_recognizer.yml'
label_dict_path = 'label_dict.npy'

if not os.path.exists(model_path) or not os.path.exists(label_dict_path):
    print("모델 또는 레이블 파일이 없습니다. 먼저 모델을 학습하세요.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
label_dict = np.load(label_dict_path, allow_pickle=True).item()

# ESP32 스트림 활성화 플래그
ESP32_STREAM_ACTIVE = False
frame_lock = threading.Lock()
current_frame = None

# 스트림 종료 플래그
stop_stream = [False]

# 한글 텍스트 표시 함수
def putText_korean(img, text, position, font_size=20, font_color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # 한글 폰트 경로 설정 (예: 맑은 고딕)
    fontpath = "C:/Windows/Fonts/malgun.ttf"  # 시스템에 맞게 경로 수정
    font = ImageFont.truetype(fontpath, font_size)
    draw.text(position, text, font=font, fill=font_color)
    return np.array(img_pil)

# 홈 페이지
@app.route('/', methods=['GET'])
def home():
    return "Flask 서버가 실행 중입니다.", 200

# ESP32 스트림 수신 및 브라우저용 스트리밍 엔드포인트 통합
@app.route('/stream', methods=['POST', 'GET'])
def stream():
    global current_frame, ESP32_STREAM_ACTIVE

    if request.method == 'POST':
        # ESP32에서 전송된 JPEG 데이터를 수신
        ESP32_STREAM_ACTIVE = True
        data = request.json['image']
        image_data = base64.b64decode(data)

        # 데이터가 비어있는지 확인
        if not image_data:
            print("Received empty image data")
            return jsonify({"status": "error", "message": "Empty image data"}), 400

        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            with frame_lock:
                current_frame = process_frame(frame)  # 얼굴 인식 및 처리 추가

                # 프레임 저장
                file_path = os.path.join(UPLOAD_FOLDER, 'frame.jpg')
                cv2.imwrite(file_path, current_frame)

        return jsonify({"status": "success", "message": "Frame received"}), 200

    elif request.method == 'GET':
        # 실시간 스트리밍 데이터 반환
        def generate():
            while True:
                if ESP32_STREAM_ACTIVE:
                    with frame_lock:
                        if current_frame is None:
                            continue
                        _, buffer = cv2.imencode('.jpg', current_frame)
                        frame_data = buffer.tobytes()
                else:
                    frame_path = os.path.join(UPLOAD_FOLDER, 'frame.jpg')
                    if os.path.exists(frame_path):
                        with open(frame_path, 'rb') as f:
                            frame_data = f.read()
                    else:
                        continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 스트림 종료 엔드포인트
@app.route('/stop', methods=['POST'])
def stop():
    global ESP32_STREAM_ACTIVE
    ESP32_STREAM_ACTIVE = False
    cv2.destroyAllWindows()
    return jsonify({"status": "success", "message": "ESP32 stream stopped"}), 200

# 프레임 처리 함수 (얼굴 인식 및 박스 그리기)
def process_frame(frame):
    global recognizer, label_dict
    approved = False
    approved_timer = None
    not_approved = False
    not_approved_timer = None
    message_duration = 3

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))
        label, confidence = recognizer.predict(face_img)

        if confidence < 50:
            user_name = label_dict[label]
            text = f"{user_name} ({confidence:.2f})"
            color = (0, 255, 0)
            
            approved = True
            approved_timer = datetime.datetime.now()

            # 승인 데이터 전송 (옵션)
            # data = {"user_name": user_name, "confidence": confidence, "timestamp": now}
            # try:
            #     response = requests.post("http://localhost:3000/data", json=data)
            #     print("Data sent to server:", response.json())
            # except Exception as e:
            #     print("Failed to send data to server:", e)
        else:
            text = "Unknown"
            color = (0, 0, 255)
            not_approved = True
            not_approved_timer = datetime.datetime.now()

        # 얼굴에 박스 및 텍스트 추가
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = putText_korean(frame, text, (x, y - 30), font_size=20, font_color=color)
        
    # 승인 메시지 표시
    if approved:
        elapsed_time = (datetime.datetime.now() - approved_timer).total_seconds()
        if elapsed_time < message_duration:
            frame = putText_korean(
                frame, 
                "사용자 승인됨", 
                (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)), 
                font_size=40, 
                font_color=(0, 255, 0)
            )
        else:
            approved = False

    # 미승인 메시지 표시
    if not_approved:
        elapsed_time = (datetime.datetime.now() - not_approved_timer).total_seconds()
        if elapsed_time < message_duration:
            frame = putText_korean(
                frame, 
                "사용자 승인안됨", 
                (int(frame.shape[1]/2) - 120, int(frame.shape[0]/2)), 
                font_size=40, 
                font_color=(0, 0, 255)
            )
        else:
            not_approved = False
            
    # 화면에 현재 시간 표시
    frame = putText_korean(frame, "Webcam " + nowDatetime, (10, 35), font_size=20, font_color=(255, 255, 255))

    return frame

# 앱 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)