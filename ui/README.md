# 캐리어 배치 최적화 도구

사진 속 짐을 인식하고, 전처리된 결과 위에서 객체를 확인·삭제한 뒤 레이어 형태로 확인할 수 있는 PySide6 기반 GUI 프로젝트입니다.

설치 필요!!
pip install PySide6 opencv-python numpy ultralytics
pip install matplotlib koreanize-matplotlib

---

## 1. 프로젝트 개요

이 프로젝트는 사용자가 사진을 불러오면 다음 순서로 처리합니다.

1. **사진 입력**
2. **전처리**
   - ChArUco 마커 기반 원근 보정
   - 책상(배경) 영역 자동 crop
3. **세그멘테이션**
   - YOLO segmentation 모델로 객체 인식
   - 객체별 `label`, `confidence`, `mask` 생성
4. **후처리 UI**
   - 낮은 신뢰도 항목 강조
   - 리스트 선택 삭제
   - 이미지 우클릭으로 해당 위치 mask 삭제
5. **레이어 뷰**
   - 남은 객체 mask를 오버레이하여 최종 확인

---

## 2. 주요 기능

### 2.1 사진 선택
- 최대 4장의 이미지를 썸네일 형태로 표시
- 클릭한 이미지를 세그멘테이션 단계로 전달

### 2.2 전처리
- **원근 보정**: ChArUco 보드를 사용해 top-view 이미지 생성
- **자동 crop**: 밝은 책상 영역을 contour 기반으로 검출하여 잘라냄

### 2.3 세그멘테이션
- YOLO segmentation 모델로 객체 인식
- 각 객체에 대해 다음 정보 생성
  - `label`
  - `confidence`
  - `mask`

### 2.4 객체 검토 및 삭제
- confidence가 낮은 항목은 경고 스타일로 강조
- 리스트에서 선택 삭제 가능
- 이미지에서 **우클릭**하면 클릭한 위치의 mask를 직접 삭제 가능

### 2.5 레이어 뷰
- 최종적으로 남은 객체들을 전처리된 이미지 위에 오버레이하여 시각화

---

## 3. 화면 구성

### 3.1 Home
프로젝트 소개 화면입니다.  
`시작하기` 버튼으로 사진 입력 화면으로 이동합니다.

### 3.2 Photo Input
이미지를 선택하고 썸네일로 확인하는 화면입니다.

### 3.3 Segmentation
전처리 결과 이미지 위에 세그멘테이션 mask를 보여줍니다.

- 전체 mask 표시
- 항목 hover 시 해당 mask 강조
- 낮은 신뢰도 항목 강조
- 선택 항목 삭제 / 우클릭 삭제 지원

### 3.4 Layer View
삭제 후 남아 있는 객체 mask들을 최종 오버레이 형태로 보여줍니다.

---

## 4. 폴더 / 파일 구성

```bash
.
├── main.py
├── model.py
├── preprocess.py
├── preprocess_batch.py
├── crop.py
├── get_perspective_image.py
├── ui_main.py
├── ui_photo.py
├── ui_segment.py
├── ui_layer.py
├── images/
│   └── start.jpg
├── data/
├── output/
└── README.md