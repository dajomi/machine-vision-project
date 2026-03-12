# 🏆 Segmentation 모델 평가 프레임워크

## Rating
Test Data Set 기반으로 모델 성능(yolo, sam2) 평가하기. 


입력 : Test DataSet, 모델


출력 : 정답률, Score(정답 * IOU * confidence), 사이즈(IOU * confidence) 
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

## UI
1. 기본 화면
2. 사진 입력 -> 사진 선택
3. label과 음영(or 음영만), 지우기 가능
4. MaxRect 결과 도출해주기.
