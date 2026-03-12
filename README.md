# 🏆 Segmentation 모델 평가 프레임워크

## Rating
Test Data Set 기반으로 모델 성능(yolo, sam2) 평가하기. 

정답률, Score(정답 * IOU * confidence), 사이즈(IOU * confidence) 
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
