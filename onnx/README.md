**## 1\\) 데이터셋 구조**



**```text**

**YOUR\\\_DATASET/**

&#x20; **JPEGImages/**

&#x20;   **IMG\\\_0001.jpg**

&#x20;   **IMG\\\_0002.jpg**

&#x20; **Annotations/**

&#x20;   **IMG\\\_0001.xml**

&#x20;   **IMG\\\_0002.xml**

&#x20; **ImageSets/**

&#x20;   **Main/**

&#x20;     **train.txt**

&#x20;     **val.txt**

&#x20;     **test.txt**

&#x20; **labels.txt**

**```**



**### labels.txt 예시**



**```text**

**normal**

**scratch**

**dent**

**contamination**

**```**





**##** 모델 실행

python model.py --data-root dataset

&#x20;        (파일이름)                  (폴더)



\## onnx 변환

python export\_detectnet\_onnx.py --checkpoint checkpoints/best.pth

&#x20;                  (파일 이름)                                    (폴더)/(이름).pth



\# model.py의 output 구조

{

&#x20;   "model\_state\_dict": model.state\_dict(),

&#x20;   "classes": \["scratch", "dent", "dirt", "smash"]

&#x20;   "img\_size": ..., \[int = 320]

&#x20;   "normalization": {
		"range": \[0.0, 1.0], 

&#x09;	"mean": \[0.5, 0.5, 0.5], 

&#x09;	"std": \[0.5, 0.5, 0.5]

&#x09;},

&#x20;   "detectnet\_export": {

&#x09;	"input\_name": "input\_0", 

&#x09;	"scores\_name": "scores", 

&#x09;	"boxes\_name": "boxes"

&#x09;}

}

