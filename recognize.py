import cv2
import datetime
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import requests

# 한글 텍스트 표시를 위한 함수
def putText_korean(img, text, position, font_size=20, font_color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    fontpath = "C:/Windows/Fonts/malgun.ttf"  # 시스템에 맞는 폰트 경로
    font = ImageFont.truetype(fontpath, font_size)
    draw.text(position, text, font=font, fill=font_color)
    return np.array(img_pil)

# 얼굴 인식 모델 및 레이블 딕셔너리 로드
model_path = 'face_recognizer.yml'
label_dict_path = 'label_dict.npy'

if not os.path.exists(model_path) or not os.path.exists(label_dict_path):
    print("모델 또는 레이블 파일이 없습니다. 먼저 모델을 학습하세요.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
label_dict = np.load(label_dict_path, allow_pickle=True).item()

# 카메라 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# 승인 메시지 관련 변수
approved = False
approved_timer = None
approved_duration = 3

# 무한 루프
while True:
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    ret, frame = capture.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            # 데이터를 Flask 서버로 전송
            data = {"user_name": user_name, "confidence": confidence, "timestamp": nowDatetime}
            try:
                response = requests.post("http://localhost:3000/data", json=data)
                print("Data sent to server:", response.json())
            except Exception as e:
                print("Failed to send data to server:", e)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = putText_korean(frame, text, (x, y - 30), font_size=20, font_color=color)

    if approved:
        elapsed_time = (datetime.datetime.now() - approved_timer).total_seconds()
        if elapsed_time < approved_duration:
            frame = putText_korean(frame, "사용자 승인됨", (200, 200), font_size=40, font_color=(0, 255, 0))
        else:
            approved = False

    frame = putText_korean(frame, f"Webcam {nowDatetime}", (10, 35), font_size=20)
    cv2.imshow("output", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
