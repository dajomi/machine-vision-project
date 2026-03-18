# SSD-MobileNetV2 → ONNX → Jetson Nano 실행 가이드


ssd\_mobilenet\_v2\_common을 공통으로 사용한다고 함.

# 컴퓨터에서 학습

> 필요한거 : YOUR\\\_DATASET, ssd\\\_mobilenet\\\_v2\\\_common

* python 01\_train\_ssd\_mobilenet\_v2.py --data-root YOUR\_DATASET --epochs 20 --batch-size 4 --img-size 320



# 컴퓨터에서 성능 평가

> 필요한거 : YOUR\\\_DATASET, ssd\\\_mobilenet\\\_v2\\\_common, infer\\\_jetson\\\_onnx, checkpoints\\\_ssd\\\_mbv2/best.pth

* python 02\_eval\_detection.py --data-root YOUR\_DATASET --backend pytorch --checkpoint checkpoints\_ssd\_mbv2/best.pth







조절할 후처리 파라미터(이거 해야만 onnx 변환이 그나마 잘 되는 편)

> --checkpoint, --output, --opset, --score-thresh, --nms-thresh, --detections-per-img, --topk-candidates


> 경로, 출력, opset 버전?, threshold, NMS IoU threshold, 최대 검출 수, top-k 후보 수



# 컴퓨터에서 onnx 변환

> 필요한거 : ssd\\\_mobilenet\\\_v2\\\_common, checkpoints\\\_ssd\\\_mbv2/best.pth
>  성능 좋은거는 windows\\\_onnx로 checkpoints\\\_ssd\\\_mbv2 폴더 자체를 옮겨주기

* python 03\_export\_ssd\_mobilenet\_v2\_onnx.py --checkpoint best.pth --output ssd\_mobilenetv2\_320\_post.onnx --opset 13



# 리눅스에서 작동(bash)

> 필요한거 : infer\\\_jetson\\\_onnx.py, ssd\\\_mobilenetv2\\\_320\\\_raw.onnx, ssd\\\_mobilenetv2\\\_320\\\_raw.onnx.data

python infer\_jetson\_onnx.py --onnx ssd\_mobilenetv2\_320\_post.onnx --image img001.jpg --labels labels.txt --input-size 320 --output result.jpg






## 1\) 데이터셋 구조

```text
YOUR\\\_DATASET/
  JPEGImages/
    IMG\\\_0001.jpg
    IMG\\\_0002.jpg
  Annotations/
    IMG\\\_0001.xml
    IMG\\\_0002.xml
  ImageSets/
    Main/
      train.txt
      val.txt
      test.txt
  labels.txt
```

### labels.txt 예시

```text
scratch
dent
dirt

smash
```

## 2\) Windows에서 학습

```bash
python train\\\_ssd\\\_mobilenet\\\_v2.py --data-root YOUR\\\_DATASET --epochs 20 --batch-size 4 --img-size 320
```

학습 결과:

* `checkpoints\\\_ssd\\\_mbv2/best.pth`
* `checkpoints\\\_ssd\\\_mbv2/latest.pth`

## 3\) best.pth → ONNX 변환

```bash
python export\\\_ssd\\\_mobilenet\\\_v2\\\_onnx.py --checkpoint checkpoints\\\_ssd\\\_mbv2/best.pth --output ssd\\\_mobilenetv2\\\_320\\\_raw.onnx
```

생성 ONNX 출력:

* `class\\\_logits` : `\\\[1, num\\\_anchors, num\\\_classes]`
* `bbox\\\_regression` : `\\\[1, num\\\_anchors, 4]`
* `anchors` : `\\\[num\\\_anchors, 4]`

## 4\) Windows에서 성능 평가

### PyTorch checkpoint 기준

```bash
python eval\\\_detection.py --data-root YOUR\\\_DATASET --backend pytorch --checkpoint checkpoints\\\_ssd\\\_mbv2/best.pth
```

### ONNX 기준

```bash
python eval\\\_detection.py --data-root YOUR\\\_DATASET --backend onnx --onnx ssd\\\_mobilenetv2\\\_320\\\_raw.onnx
```

출력:

* class별 Recall
* class별 Precision
* class별 AP50
* 전체 mAP50



## 5\) Jetson Nano 패키지 설치

```bash
sudo apt update
sudo apt install -y python3-pip libopenblas-base libjpeg-dev
pip3 install numpy opencv-python onnxruntime
```

> Jetson Nano 2GB는 메모리가 작아서 배치 1, input 320 권장.



\###

## 6\) Jetson Nano에 복사할 파일

* `ssd\\\\\\\_mobilenetv2\\\\\\\_320\\\\\\\_raw.onnx`
* `infer\\\\\\\_jetson\\\\\\\_onnx.py`
* `labels.txt`
* 필요 시 테스트 이미지



\###

## 7\) Jetson Nano 추론

```bash
python3 infer\\\_jetson\\\_onnx.py \\\\
  --onnx ssd\\\_mobilenetv2\\\_320\\\_raw.onnx \\\\
  --image img001.jpg \\\\
  --labels labels.txt \\\\
  --input-size 320 \\\\
  --output result.jpg
```

콘솔 출력 예:

```text
{'label': 'scratch', 'score': 0.92, 'position\\\_xyxy': \\\[101.3, 120.5, 221.7, 262.1]}
{'label': 'dent', 'score': 0.81, 'position\\\_xyxy': \\\[310.2, 88.4, 401.8, 170.9]}
```

## 8\) 사용자 요구 output 대응

입력:

* 사진: `img001.jpg`
* annotation: `img001.xml`

출력:

* label
* object detection position `(xmin, ymin, xmax, ymax)`
* Recall
* Precision
* AP

즉,

* **1장 추론 결과**는 `infer\\\_jetson\\\_onnx.py`
* **테스트셋 전체 성능**은 `eval\\\_detection.py`
에서 확인.

## 9\) `.pt` / `.pth` / `.onnx` 정리

|파일|의미|여기서 역할|
|-|-|-|
|`.pt`|PyTorch 저장 형식(모델/스크립트 등)|지금 구조에서는 직접 사용 안 함|
|`.pth`|PyTorch state\_dict checkpoint|학습 결과 `best.pth`|
|`.onnx`|프레임워크 독립 추론 형식|Jetson Nano 배포용|

## 10\) 권장 흐름

1. Windows에서 `best.pth` 학습 완료
2. `best.pth`를 ONNX로 export
3. Jetson Nano로 ONNX 복사
4. Jetson에서 ONNX Runtime + 후처리 코드로 추론
5. Windows에서 test set 기준 Recall / Precision / AP 계산

## 11\) 주의

* `labels.txt` 순서는 학습/평가/Jetson 추론에서 **항상 동일**해야 함.
* `train.txt`, `val.txt`, `test.txt`에는 **확장자 없는 이미지 ID**만 들어가야 함. 예: `IMG\\\_0001`
* XML의 `<name>`은 반드시 `labels.txt` 항목과 정확히 일치해야 함.
* Jetson Nano 2GB에서는 640보다 **320 입력**이 현실적.
