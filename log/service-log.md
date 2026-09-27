# 서비스 로그

데이터셋, 학습 결과, 파이프라인 설정, 최적화, Jetson Nano 배포 설정, 한계를 정리한다. README에는 동작과 사용 방법만 둔다.

## 1. 데이터셋

Roboflow에서 내보낸 두 데이터셋을 쓴다. 레포에는 넣지 않았다.

| 데이터셋 | 클래스 | 이미지 |
|---|---|---|
| `kkk8-detection` v2 | 인형 8종 | 383 |
| `agvdetect` v1 | `AGV` | 40 |

`scripts/prepare_dataset.py`가 두 데이터셋을 8클래스 하나로 합친다.

- `DOG`는 `DARK_GRAY`로 이름만 바꾼다(라벨 번호 유지).
- 쓰지 않는 `RATEL` 박스는 뺀다.
- `AGV`를 마지막 클래스로 붙인다.
- 결과 클래스: `[BLACK, DARK_GRAY, GRAY_BIG, GRAY_SMALL, PINK, SKYBLUE, WHITE, AGV]`
- 분할은 Roboflow의 train/valid/test를 그대로 쓴다.

## 2. 학습과 모델

- YOLOv5 v7.0, `yolov5s.pt`에서 시작, 입력 640, 100 epoch, batch 32.
- 데이터 412장(train 371 / val 17 / test 24).
- `best.pt`를 ONNX(opset 12, simplify)로 내보냈다. 가중치는
  [Releases v1.0.0](https://github.com/donghyeoni/agv-grid-localization/releases/tag/v1.0.0)에 있다.

**Test set** (이미지 24장, 객체 144개):

| 클래스 | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **all** | **0.989** | **0.989** | **0.995** | **0.654** |
| BLACK | 0.986 | 1.000 | 0.995 | 0.649 |
| DARK_GRAY | 0.985 | 0.950 | 0.993 | 0.644 |
| GRAY_BIG | 0.978 | 1.000 | 0.995 | 0.674 |
| GRAY_SMALL | 0.986 | 1.000 | 0.995 | 0.728 |
| PINK | 0.990 | 1.000 | 0.995 | 0.700 |
| SKYBLUE | 0.997 | 1.000 | 0.995 | 0.571 |
| WHITE | 1.000 | 0.961 | 0.995 | 0.493 |
| AGV | 0.993 | 1.000 | 0.995 | 0.771 |

| 파일 | sha256 |
|---|---|
| `best.pt` | `a95cf8666fca77300db619931fd2a7157d489169cf18afe4c16749ffc9940347` |
| `best.onnx` | `640ade596d3b3b0e671b593f7b354eae3f60b6b44a46fdcf8a69befaf4cc7c3f` |

## 3. 파이프라인 설정

| 단계 | 설정 |
|---|---|
| 판 검출 | 가장자리 30px 마스킹, 흑백값 170 이하를 0으로(`THRESH_TOZERO`), 외곽 윤곽 중 면적 최대, `approxPolyDP`(둘레의 2%)로 꼭짓점 4개 |
| 꼭짓점 정렬 | x+y 최소 = 좌상, 최대 = 우하, y−x 최소 = 우상, 최대 = 좌하 |
| 검출 입력 | 비율 유지 리사이즈 후 회색(114) 패딩으로 640×640, RGB, 0–1 정규화 |
| 검출 후처리 | 신뢰도(객체성 × 클래스 점수) 0.25 이상, 클래스별 NMS IoU 0.45 |
| 칸 배정 | 판 꼭짓점 → `[0, 8]²` 원근 변환, 박스 중심의 격자 좌표를 내림해 (행, 열), 번호 = 행 × 8 + 열 |

## 4. 최적화

결과를 원본 코드와 비트 단위로 같게 유지하며 줄였다. 합성 영상과 모델 출력을 흉내 낸 가짜 세션으로 원본과 출력을 비교했다.

- 판 검출 2,700건, 칸 배정 900건: numpy 1.26, 2.4에서 모두 같았다.
- 검출 후처리 200건(검출 33,836개): 아래 NMS 변경 전 코드로 numpy 1.26에서 같았다. numpy 2.4에서는 원본의 좌표 복원이
  float32라 다르다(아래 좌표 복원 정밀도).

| 변경 | 내용 |
|---|---|
| 판 검출 | 컬러 영상 마스킹 → 흑백 변환 → `np.where` 임계값을, 흑백 변환 → 흑백 마스킹 → `cv2.threshold`로 |
| 검출 전처리 | 패딩을 `cv2.copyMakeBorder`로, 입력 텐서를 한 번에 연속 메모리 float32로 만든 뒤 제자리 정규화 |
| 검출 후처리 | 객체성이 임계값 미만인 후보를 먼저 제거(클래스 점수 ≤ 1이라 결과 같음), 박스 변환·좌표 복원을 배열 연산으로 |
| 칸 배정 | 박스마다 `perspectiveTransform`을 부르던 것을 한 번 호출로 |
| 카메라 데모 | 프레임 전체 BGR→RGB 변환을 없애고, 판 검출은 BGR→흑백, 검출은 640 리사이즈 뒤에 색 순서 변환 |

1280×720 영상 한 장, 개발 PC(CPU), numpy 1.26, 모델 추론 시간 제외:

| 단계 | 원본 | 최적화 |
|---|---|---|
| 판 검출 | 1.54 ms | 0.55 ms |
| 검출 전처리·후처리 | 13.3 ms | 8.9 ms |
| 카메라 프레임 (BGR 입력) | 16.3 ms | 9.7 ms |

같이 바꾼 동작:

- **좌표 복원 정밀도.** 원본은 박스 좌표를 원본 영상 크기로 되돌릴 때 numpy 1에서는 float64, numpy 2에서는 float32로
  계산했다. float64로 고정했다.
- **NMS.** 클래스를 가리지 않던 NMS를 YOLOv5 `detect.py`·`val.py`와 같은 클래스별 NMS로 바꿨다. 서로 다른 클래스의
  박스는 겹쳐도 둘 다 남는다. 모든 박스가 한 클래스인 입력 300건에서는 이전과 출력이 같았다.
- **`--border-thickness 0`.** 영상 전체가 검게 마스킹되던 것을 고쳤다(`[-0:]` 슬라이스).
- **판 미검출.** `localize.py`가 판을 찾지 못하면 종료하던 것을, 카메라 데모처럼 검출 결과를 칸 없이 보고하게 했다.

## 5. Jetson Nano 배포

| 항목 | 값 |
|---|---|
| JetPack | 4.6 |
| Python | 3.6 |
| onnxruntime-gpu | 1.10.0 (Jetson용 wheel) |
| ONNX opset | 12 (JetPack 4.x onnxruntime·TensorRT 빌드가 지원하는 최고 버전) |
| 기기 의존성 | onnxruntime, numpy, OpenCV (PyTorch 없음) |
| CSI 카메라 | `nvarguscamerasrc` 1280×720 30fps GStreamer 파이프라인 |

- 입력 크기는 내보낼 때 고정된다(기본 640). `scripts/export_onnx.py --img 416`처럼 작게 내보낼 수 있다.
- 첫 추론은 세션 준비 때문에 느리고, 두 번째 프레임부터 처리 시간이 일정해진다.

## 6. 파일 구성

| 파일 | 역할 |
|---|---|
| [`localize.py`](../localize.py) | 영상 한 장 처리: 판 검출 + 물체 검출 + 칸 배정, 표·JSON·주석 영상 출력 |
| [`src/preprocessing.py`](../src/preprocessing.py) | 영상 읽기, 가장자리 마스킹, 임계값 |
| [`src/plate_detection.py`](../src/plate_detection.py) | 가장 큰 윤곽, 꼭짓점 정렬, `find_plate` |
| [`src/grid.py`](../src/grid.py) | 원근에 맞춘 격자 그리기 |
| [`src/grid_mapping.py`](../src/grid_mapping.py) | 박스 중심 → 칸(행, 열, 번호) |
| [`src/yolo_onnx.py`](../src/yolo_onnx.py) | ONNX Runtime YOLOv5 추론(레터박스, 후처리, 클래스별 NMS) |
| [`deploy/camera_demo.py`](../deploy/camera_demo.py) | Jetson Nano 실시간 카메라 데모(CSI, USB) |
| [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py) | Roboflow 데이터셋 두 개를 8클래스로 병합 |
| [`scripts/train.py`](../scripts/train.py) | YOLOv5 v7.0 학습 래퍼 |
| [`scripts/export_onnx.py`](../scripts/export_onnx.py) | `best.pt` → ONNX(opset 12, simplify) |

## 7. 한계

- 판 검출은 밝기 임계값 기반이다. 기본값(임계값 170, 테두리 30px)은 원래 촬영 환경에 맞춘 값이라 조명이 다르면 다시
  맞춰야 한다. 카메라 데모는 이 두 값이 코드 상수다.
- AGV 데이터셋(`agvdetect` v1)은 train/valid/test를 합쳐 이미지 40장이다.
- 칸은 박스 중심 한 점으로 정한다.

## 8. 참여

- [donghyeoni](https://github.com/donghyeoni)
- [Kyung Sejin (kkkksj)](https://github.com/kkkksj)
