# 🏆 Segmentation 모델 평가 프레임워크

## 폴더 구조
설치 필요 -> pip install opencv-python numpy matplotlib ultralytics koreanize-matplotlib
```
eval_dataset/
├── image_001/
│   ├── image_001.jpg
│   └── gt_masks/   ← 폴더 명은 맞춰줘야 됨 
│       ├── laptop.png   ← 흰색=객체, 검정=배경
│       └── mouse.png
├── image_002/
│   ├── image_002.jpg
│   └── gt_masks/
│       ├── laptop.png   ← 흰색=객체, 검정=배경
│       └── mouse.png
```

## 새 모델 추가 방법
**STEP 1 셀**에서 함수 하나만 추가하고 `MODELS`에 등록하세요.
```python
def my_model(img_path, W, H):
    return [
        {'label': 'laptop', 'confidence': 0.92, 'mask': np.ndarray(bool, shape H×W)},  ← label 이름, confidence 확률(없는 모델이면 1), mask (H,W) array.
    ] 

MODELS = {  
    'YOLO11x': model_yolo,  ← 이름과 정보를 넣어주어야 함 ('이름' : 정보)
    'MyModel': my_model,      정보에는 label, confidence, mask가 필요함.
}
```

<img width="498" height="232" alt="image" src="https://github.com/user-attachments/assets/eefb6bba-0a7b-460b-a854-b632eb3f44da" />

## 결과
**모델 성능 비교는 1. 정답률을 먼저 확인하고 2. score 확인하여 순위를 정하면 좋을듯**
1. Confusion Matrix
<img width="1004" height="769" alt="image" src="https://github.com/user-attachments/assets/d2b1b0cf-ed60-422c-95bd-8120fa004f37" />

2. 모델 별 Precision, Recall
<img width="1085" height="355" alt="image" src="https://github.com/user-attachments/assets/011b37c0-a0ac-4bc1-becc-30900f3fa3c7" />

3. 최종 결과
<img width="611" height="551" alt="image" src="https://github.com/user-attachments/assets/46aee49d-1882-4609-bfe0-1fd81c3ef45b" />