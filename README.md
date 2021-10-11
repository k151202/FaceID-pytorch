# FaceID-pytorch

Face recognition &amp; identification with facenet-pytorch

## 프로젝트 주제

- FaceNet기반의 얼굴인식 모델을 이용하여 Face Identification을 구현하는 프로젝트 입니다. 이 프로젝트에서는 tensorflow기반인 facenet을 pytorch의 파라미터로 컨버팅한 facenet-pytorch 패키지를 사용합니다.
- Reference : https://github.com/timesler/facenet-pytorch

## 프로젝트 구현 과정

1. 모델 분석 및 설계
   - 구글링 및 논문을 통해 face detection algorithm에 대한 학습
2. 데이터 수집 및 전처리
   - 눈, 코, 입이 제대로 나오지 않은 사진은 제외
   - 얼굴이 가려져있거나 두사람이상 나온 사진 제외
3. Fine-tuning
   - AI hub 한국인 얼굴 데이터 & 해외 유명인사 얼굴 사진을 훈련시킨 뒤 테스트
4. 모델 테스트
   - inference.py로 훈련된 모델을 테스트
   - hyper parameter을 바꾸어가며 적절한 threshold, margin값을 지정
   - test.py에서 추론 테스트
5. 실시간 인식 테스트
   - OpenCV를 이용해 비디오 파일(eg. youtube video)을 테스트
   - 웹캠을 통한 실시간 얼굴 검출(detection) 및 인식(recognition) 테스트
6. 안드로이드 앱 포팅
   - pre-trained 모델을 tensorflow-lite 버전으로 컨버팅
   - 용량이 큰 문제로 축소버전인 MobileFaceNet의 tensorflow-lite 버전으로 구현함
   - 안드로이드 관련 코드는 추후 업로드 예정

## 의존성

```
torch = "*"
numpy = "*"
pandas = "*"
torchvision = "*"
requests = "*"
tensorboard = "*"
pillow = "*"
opencv-python = "*"
python_version = "3.8"
```
