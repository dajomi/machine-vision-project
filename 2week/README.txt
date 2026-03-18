input : 사진

640x640
linux - docker 
colab

output : label과 object detection.
output : 혼동행렬

1. 모델 학습은 window 환경에서 진행. 
2. 적용은 jetson nano 2gb developer kit에 linux 기반으로
3. 위 방법 적용이 가능한 ssd-mobilenet


pip install torch torchvision torchaudio
pip install pillow lxml tqdm matplotlib


dataset/
├─ train/
│  ├─ images/
│  │  ├─ img001.jpg
│  │  ├─ img002.jpg
│  └─ Annotations/
│     ├─ img001.xml
│     ├─ img002.xml
├─ val/
│  ├─ images/
│  │  ├─ img001.jpg
│  │  ├─ img002.jpg
│  └─ Annotations/
│     ├─ img001.xml
│     ├─ img002.xml
├─ test/
│   ├─ images/
│  │  ├─ img001.jpg
│  │  ├─ img002.jpg
│   └─ Annotations/
│     ├─ img001.xml
│     ├─ img002.xml
└─ classes.txt
