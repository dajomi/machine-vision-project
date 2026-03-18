"""
infer_jetson_onnx.py
Jetson Nano / 일반 환경에서 SSD-MobileNetV2 ONNX 추론

Input  : 이미지 파일, ONNX 모델, labels.txt
Output : 콘솔 검출 결과 + 바운딩 박스 시각화 이미지

사용법:
    python infer_jetson_onnx.py \
        --onnx         ssd_mobilenetv2_320_raw.onnx \
        --image        img001.jpg \
        --labels       labels.txt \
        --input-size   320 \
        --score-thresh 0.25 \
        --nms-thresh   0.45 \
        --output       result.jpg
"""

# ============================================================
# 표준 라이브러리
# ============================================================
import argparse
from pathlib import Path
from typing import Tuple

# ============================================================
# 서드파티 라이브러리
# ============================================================
import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# 상수 (ImageNet 정규화)
# ============================================================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 클래스별 시각화 색상 (BGR)
_COLORS = [
    (0,   255,   0),   # scratch  - 초록
    (255,   0,   0),   # dent     - 파랑
    (0,     0, 255),   # smash    - 빨강
    (0,   255, 255),   # dirt     - 노랑
    (255,   0, 255),   # 예비
]


# ============================================================
# 유틸리티
# ============================================================

def load_labels(label_path: str) -> list:
    """labels.txt 로드 → 클래스 이름 리스트 반환"""
    return [
        line.strip()
        for line in Path(label_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def preprocess_image(
    image_bgr: np.ndarray,
    input_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    BGR 이미지를 모델 입력 텐서로 변환

    Returns
    -------
    x         : [1, 3, input_size, input_size]  float32
    orig_size : (원본 너비, 원본 높이)
    """
    h0, w0  = image_bgr.shape[:2]
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x       = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    x       = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]
    return x, (w0, h0)


# ============================================================
# 후처리 헬퍼 함수
# ============================================================

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _decode_boxes(
    deltas:  np.ndarray,
    anchors: np.ndarray,
    weights: Tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> np.ndarray:
    """박스 회귀 델타 → 절대 좌표 [x1, y1, x2, y2] 변환"""
    wx, wy, ww, wh = weights
    anchors = anchors.astype(np.float32)
    deltas  = deltas.astype(np.float32)

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw =  anchors[:, 2] - anchors[:, 0]
    ah =  anchors[:, 3] - anchors[:, 1]

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = np.clip(deltas[:, 2] / ww, None, np.log(1000.0 / 16.0))
    dh = np.clip(deltas[:, 3] / wh, None, np.log(1000.0 / 16.0))

    px = dx * aw + ax
    py = dy * ah + ay
    pw = np.exp(dw) * aw
    ph = np.exp(dh) * ah

    return np.stack(
        [px - pw * 0.5, py - ph * 0.5, px + pw * 0.5, py + ph * 0.5],
        axis=1,
    )


def _clip_boxes(boxes: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """박스 좌표를 이미지 경계 내로 클리핑"""
    if len(boxes) == 0:
        return boxes
    h, w   = size_hw
    max_x  = max(w - 1, 0)
    max_y  = max(h - 1, 0)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max_x)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max_y)
    return boxes


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> np.ndarray:
    """Non-Maximum Suppression (numpy 구현)"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i   = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / np.clip(areas[i] + areas[order[1:]] - inter, 1e-8, None)
        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return np.asarray(keep, dtype=np.int64)


# ============================================================
# 후처리 메인 함수
# ============================================================

def postprocess(
    class_logits:    np.ndarray,
    bbox_regression: np.ndarray,
    anchors:         np.ndarray,
    orig_size:       Tuple[int, int],
    input_size:      int,
    score_thresh:    float = 0.25,
    nms_thresh:      float = 0.45,
    topk:            int   = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ONNX raw 출력 → (boxes, scores, labels)

    Returns
    -------
    boxes  : [N, 4]  float32  (원본 이미지 좌표, xyxy)
    scores : [N]     float32
    labels : [N]     int64    (1-indexed, 배경=0 제외)
    """
    probs  = _softmax(class_logits, axis=-1)[0]  # [num_anchors, num_classes]
    deltas = bbox_regression[0]                   # [num_anchors, 4]

    boxes = _decode_boxes(deltas, anchors)
    boxes = _clip_boxes(boxes, (input_size, input_size))

    out_boxes, out_scores, out_labels = [], [], []
    num_classes = probs.shape[-1]

    for cls_idx in range(1, num_classes):  # 0 = 배경 skip
        cls_scores = probs[:, cls_idx]
        mask       = cls_scores > score_thresh
        if not np.any(mask):
            continue

        b = boxes[mask]
        s = cls_scores[mask]

        order = np.argsort(-s)[:topk]
        b, s  = b[order], s[order]

        keep = _nms(b, s, iou_thresh=nms_thresh)
        out_boxes.append(b[keep])
        out_scores.append(s[keep])
        out_labels.append(np.full(len(keep), cls_idx, dtype=np.int64))

    if not out_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,),   dtype=np.float32),
            np.zeros((0,),   dtype=np.int64),
        )

    boxes  = np.concatenate(out_boxes,  axis=0)
    scores = np.concatenate(out_scores, axis=0)
    labels = np.concatenate(out_labels, axis=0)

    # score 내림차순 topk 정렬
    order          = np.argsort(-scores)[:topk]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    # 원본 이미지 좌표로 스케일 복원
    orig_w, orig_h = orig_size
    boxes[:, [0, 2]] *= orig_w / float(input_size)
    boxes[:, [1, 3]] *= orig_h / float(input_size)
    boxes = _clip_boxes(boxes, (orig_h, orig_w))

    return boxes, scores, labels


# ============================================================
# 시각화
# ============================================================

def draw_detections(
    image_bgr:   np.ndarray,
    boxes:       np.ndarray,
    scores:      np.ndarray,
    labels:      np.ndarray,
    class_names: list,
) -> np.ndarray:
    """검출 결과를 이미지에 오버레이"""
    out = image_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int).tolist()
        color = _COLORS[(label - 1) % len(_COLORS)]
        text  = f"{class_names[label - 1]}  {score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(th + 4, y1 - 2)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(
            out, text, (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ============================================================
# 인수 파싱
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SSD-MobileNetV2 ONNX 추론 (Jetson / PC)")
    p.add_argument("--onnx",         required=True,             help="ONNX 모델 파일 경로")
    p.add_argument("--image",        required=True,             help="입력 이미지 경로")
    p.add_argument("--labels",       required=True,             help="labels.txt 경로")
    p.add_argument("--input-size",   type=int,   default=320,   help="모델 입력 해상도 (기본: 320)")
    p.add_argument("--score-thresh", type=float, default=0.25,  help="검출 score 임계값 (기본: 0.25)")
    p.add_argument("--nms-thresh",   type=float, default=0.45,  help="NMS IoU 임계값 (기본: 0.45)")
    p.add_argument("--topk",         type=int,   default=200,   help="최대 검출 수 (기본: 200)")
    p.add_argument("--output",       default="result.jpg",      help="결과 이미지 저장 경로")
    p.add_argument("--no-cuda",      action="store_true",       help="CUDA 비활성화")
    return p.parse_args()


# ============================================================
# 메인
# ============================================================

def main():
    args = parse_args()

    # ── 레이블 / 이미지 로드 ──────────────────────────────────
    class_names = load_labels(args.labels)
    image_bgr   = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {args.image}")

    # ── 전처리 ────────────────────────────────────────────────
    x, orig_size = preprocess_image(image_bgr, args.input_size)

    # ── ONNX Runtime 세션 ─────────────────────────────────────
    providers = ["CPUExecutionProvider"]
    if not args.no_cuda:
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

    sess = ort.InferenceSession(args.onnx, providers=providers)
    print(f"실행 프로바이더: {sess.get_providers()[0]}")

    # ── 추론 ──────────────────────────────────────────────────
    class_logits, bbox_regression, anchors = sess.run(
        None, {sess.get_inputs()[0].name: x}
    )

    # ── 후처리 ────────────────────────────────────────────────
    boxes, scores, labels = postprocess(
        class_logits,
        bbox_regression,
        anchors,
        orig_size=orig_size,
        input_size=args.input_size,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.topk,
    )

    # ── 결과 출력 ─────────────────────────────────────────────
    print(f"\n검출된 객체: {len(boxes)}개")
    print("-" * 60)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = (float(v) for v in box)
        print(
            f"  [{i+1:>2d}] {class_names[label - 1]:<12s}  "
            f"score={score:.4f}  "
            f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
    print("-" * 60)

    # ── 시각화 저장 ───────────────────────────────────────────
    vis = draw_detections(image_bgr, boxes, scores, labels, class_names)
    cv2.imwrite(args.output, vis)
    print(f"\n결과 이미지 저장: {args.output}")


if __name__ == "__main__":
    main()
