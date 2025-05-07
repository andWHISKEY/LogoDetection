📌 이 프로젝트는 YOLOv11s 모델을 사용하여 이미지 속 로고를 감지하고, 바운딩 박스 기준으로 로고 이미지를 Crop하여 반환하는 기능을 제공합니다.

📌 YOLOv11 로고 감지 모델 (best.pt)
YOLOv11s 모델을 기반으로 학습된 best.pt를 사용하여 로고를 탐지하고, Crop된 로고 이미지를 반환하는 프로젝트입니다.

1️⃣ 프로젝트 구조
```text
📂 YOLO_Logo_Detection
 ├── 📄 best.pt              # 학습된 YOLOv11s 모델 파일
 ├── 📄 best_ver2.pt         # 학습된 YOLOv11s 모델 파일일 - 혹시 best.pt가 성능이 안좋을 경우 이 모델로 교체하면 잘 나올지도 몰라 추가해놨습니다.
 ├── 📄 sample.jpeg          # inference용 샘플이미지
 ├── 📄 yolo_inference.py    # YOLO 모델을 사용하여 로고 감지 및 Crop 수행
 ├── 📄 requirements.txt     # 필수 라이브러리 목록
 ├── 📄 README.md            # 프로젝트 설명 문서 (현재 파일)
```
2️⃣ 필수 라이브러리 설치
아래 명령어를 실행하여 필요한 라이브러리를 설치가능합니다.

pip install -r requirements.txt

3️⃣ YOLO 모델 실행 방법

python yolo_inference.py

4️⃣ yolo_inference.py 코드 설명

YOLO 모델을 사용하여 이미지 속 로고를 감지한 후, 바운딩 박스 기준으로 Crop된 이미지를 반환합니다.Crop된 이미지는 바운딩 박스보다 마진(여백)을 추가하여, 로고가 짤리지 않도록 처리했습니당.

문의가 있다면 서혜교/ markhyegyo@gmail.com 로 연락주세요. 자리에 바로 오셔도 좋습니다!
