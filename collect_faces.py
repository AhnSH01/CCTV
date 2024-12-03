import cv2
import datetime
import numpy as np
import os

# 얼굴 인식 모델 및 레이블 딕셔너리 로드
model_path = 'face_recognizer.yml'
label_dict_path = 'label_dict.npy'

if not os.path.exists(model_path) or not os.path.exists(label_dict_path):
    print("모델 또는 레이블 파일이 없습니다. 먼저 모델을 학습하세요.")
    exit()

# 얼굴 인식기 및 레이블 딕셔너리 초기화
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
label_dict = np.load(label_dict_path, allow_pickle=True).item()

# 카메라 영상을 받아올 객체 선언 및 설정
capture = cv2.VideoCapture(0)  # 필요에 따라 카메라 인덱스를 조정하세요
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# Haar Cascade 검출기 객체 선언
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print("Haar Cascade 파일을 찾을 수 없습니다.")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # 필요 시 사용

# 녹화 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 영상을 기록할 코덱 설정
is_record = False                          # 녹화 상태 초기화
video = None                               # 비디오 객체 초기화

print("프로그램을 시작합니다. 'r' 키: 녹화 시작/중지, 'c' 키: 캡처, 'q' 키: 종료")

# 승인 메시지 관련 변수
approved = False
approved_timer = None
approved_duration = 3  # 메시지를 표시할 시간 (초)

# 무한 루프 시작
while True:
    # 현재 시각을 불러와 문자열로 저장
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    nowDatetime_path = now.strftime('%Y-%m-%d_%H_%M_%S')  # 파일 이름에서 ':' 제거

    ret, frame = capture.read()  # 프레임 읽기
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로 변환

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.5, 
        minNeighbors=5, 
        minSize=(50, 50)
    )

    # 얼굴 영역에 사각형 표시 및 얼굴 인식 수행
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))

        # 얼굴 인식 수행
        label, confidence = recognizer.predict(face_img)

        if confidence < 50:
            user_name = label_dict.get(label, "Unknown")
            text = f"{user_name} ({confidence:.2f})"
            color = (0, 255, 0)  # 녹색

            # 승인된 사용자일 경우 승인 메시지 표시
            approved = True
            approved_timer = datetime.datetime.now()
        else:
            text = "승인되지 않은 사용자"
            color = (0, 0, 255)  # 빨간색

        # 얼굴 영역에 사각형과 이름 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 승인 메시지 표시
    if approved:
        elapsed_time = (datetime.datetime.now() - approved_timer).total_seconds()
        if elapsed_time < approved_duration:
            cv2.putText(
                frame, 
                "사용자 승인됨", 
                (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 255, 0), 
                3
            )
        else:
            approved = False

    # 녹화 중이면 빨간색 점을 표시
    if is_record:
        # 화면 상단 우측에 빨간색 점 표시 (해상도에 따라 조정 필요)
        cv2.circle(img=frame, center=(620, 15), radius=5, color=(0, 0, 255), thickness=-1)

    # 화면에 현재 시간 표시
    cv2.putText(
        frame, 
        "Webcam " + nowDatetime, 
        (10, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (255, 255, 255), 
        2
    )

    # 프레임 출력
    cv2.imshow("output", frame)

    key = cv2.waitKey(1) & 0xFF  # 키 입력 대기

    if key == ord('r'):
        if not is_record:
            is_record = True  # 녹화 시작
            video_filename = "record_" + nowDatetime_path + ".avi"
            video = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print("녹화를 시작합니다.")
        else:
            is_record = False  # 녹화 중지
            if video is not None:
                video.release()
                video = None
            print("녹화를 종료합니다.")
    elif key == ord('c'):
        # 캡처 이미지 저장
        image_filename = "capture_" + nowDatetime_path + ".png"
        cv2.imwrite(image_filename, frame)
        print(f"이미지를 저장했습니다: {image_filename}")
    elif key == ord('q'):
        print("프로그램을 종료합니다.")
        break

    if is_record and video is not None:
        video.write(frame)  # 녹화 중이면 프레임 저장

# 자원 해제
capture.release()
if video is not None:
    video.release()
cv2.destroyAllWindows()
