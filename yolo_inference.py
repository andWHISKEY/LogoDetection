import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 모델 로드
MODEL_PATH = "best.pt"  # YOLO 모델 파일
model = YOLO(MODEL_PATH)

def show_image(image, title="Detected Logo"):
    """
    OpenCV imshow()를 대체하여 Matplotlib으로 이미지를 표시하는 함수.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR -> RGB 변환
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def detect_and_crop_logo(image_path, conf_threshold=0.2, margin=20):
    """
    YOLO 모델을 사용하여 로고를 감지하고, 바운딩 박스를 기준으로 Crop한 이미지를 반환하는 함수.
    바운딩 박스보다 마진(여백)을 추가하여 로고가 짤리지 않도록 처리.

    :param image_path: 분석할 이미지 파일 경로
    :param conf_threshold: 신뢰도(Confidence) 임계값
    :param margin: 바운딩 박스에 추가할 여백 (기본값: 20픽셀)
    :return: Crop된 로고 이미지 리스트 (복수 개)
    """
    # 이미지 로드
    img = cv2.imread(image_path)

    if img is None:
        print(f"🚨 [오류] 이미지 로드 실패: {image_path}")
        return []

    img_h, img_w, _ = img.shape  # 원본 이미지 크기 확인

    # YOLO 모델로 객체 탐지 실행
    results = model.predict(image_path, conf=conf_threshold)

    cropped_logos = []  # Crop된 이미지 리스트

    for i, result in enumerate(results):
        boxes = result.boxes  # 탐지된 BBox 리스트

        for j, box in enumerate(boxes):
            confidence = box.conf.item()  # 신뢰도 점수
            if confidence < conf_threshold:
                continue  # 신뢰도 낮은 객체는 무시

            # 원본 BBox 좌표 가져오기
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            # 마진 적용 (이미지 경계를 벗어나지 않도록 조정)
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_w, x_max + margin)
            y_max = min(img_h, y_max + margin)

            # 이미지 Crop
            cropped_logo = img[y_min:y_max, x_min:x_max]

            if cropped_logo.size == 0:
                continue  # 잘못된 Crop 방지

            cropped_logos.append(cropped_logo)  # 리스트에 추가

    return cropped_logos  # Crop된 이미지 리스트 반환

# 사용 예시
if __name__ == "__main__":
    TEST_IMAGE = "sample.jpeg"  # 테스트할 이미지 경로
    cropped_images = detect_and_crop_logo(TEST_IMAGE, margin=20)  # 마진 20픽셀 적용

    for i, cropped in enumerate(cropped_images):
        show_image(cropped, title=f"Logo {i+1}")
