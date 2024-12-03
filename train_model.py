import cv2
import numpy as np
import os

def get_images_and_labels(dataset_path):
    image_paths = []
    labels = []
    label_dict = {}
    current_label = 0

    # 사용자 폴더를 순회하며 이미지와 레이블을 가져옵니다.
    for user_folder_name in os.listdir(dataset_path):
        user_folder_path = os.path.join(dataset_path, user_folder_name)
        if not os.path.isdir(user_folder_path):
            continue

        # 레이블 딕셔너리에 사용자 이름과 레이블을 매핑
        label_dict[current_label] = user_folder_name

        for image_name in os.listdir(user_folder_path):
            image_path = os.path.join(user_folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image_paths.append(image)
            labels.append(current_label)

        current_label += 1

    return image_paths, labels, label_dict

dataset_path = "dataset"

print("얼굴 이미지와 레이블을 불러오는 중...")
images, labels, label_dict = get_images_and_labels(dataset_path)

if len(images) == 0:
    print("이미지가 없습니다. 먼저 데이터를 수집하세요.")
    exit()

print("모델을 학습하는 중...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))

# 학습된 모델 저장
model_path = 'face_recognizer.yml'
recognizer.save(model_path)

# 레이블 딕셔너리 저장
label_dict_path = 'label_dict.npy'
np.save(label_dict_path, label_dict)

print("모델 학습 및 저장이 완료되었습니다.")
