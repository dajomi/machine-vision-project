"""
Jetson Nano ONNX inference for raw SSD-MobileNetV2 outputs.

Input:
  - image: img001.jpg (e.g. 640x480)
  - onnx : ssd_mobilenetv2_320_raw.onnx
  - labels.txt

Output:
  - label, score, bounding box position(xmin,ymin,xmax,ymax)
  - annotated image
  - console summary
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_labels(label_path: str):
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_image(image_bgr: np.ndarray, input_size: int):
    h0, w0 = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x, (w0, h0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_boxes_numpy(deltas: np.ndarray, anchors: np.ndarray, weights=(10.0, 10.0, 5.0, 5.0)):
    wx, wy, ww, wh = weights
    anchors = anchors.astype(np.float32)
    deltas = deltas.astype(np.float32)

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    dw = np.clip(dw, a_min=None, a_max=np.log(1000.0 / 16.0))
    dh = np.clip(dh, a_min=None, a_max=np.log(1000.0 / 16.0))

    px = dx * aw + ax
    py = dy * ah + ay
    pw = np.exp(dw) * aw
    ph = np.exp(dh) * ah

    x1 = px - 0.5 * pw
    y1 = py - 0.5 * ph
    x2 = px + 0.5 * pw
    y2 = py + 0.5 * ph
    return np.stack([x1, y1, x2, y2], axis=1)


def clip_boxes(boxes, size_hw):
    h, w = size_hw
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def nms_numpy(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.clip(areas[i] + areas[order[1:]] - inter, 1e-8, None)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.asarray(keep, dtype=np.int64)


def postprocess(class_logits, bbox_regression, anchors, orig_size, input_size, score_thresh=0.25, nms_thresh=0.45, topk=200):
    probs = softmax(class_logits, axis=-1)[0]
    deltas = bbox_regression[0]
    anchors = anchors

    boxes = decode_boxes_numpy(deltas, anchors)
    boxes = clip_boxes(boxes, (input_size, input_size))

    out_boxes, out_scores, out_labels = [], [], []
    num_classes = probs.shape[-1]

    for cls_idx in range(1, num_classes):
        cls_scores = probs[:, cls_idx]
        keep = cls_scores > score_thresh
        if not np.any(keep):
            continue

        b = boxes[keep]
        s = cls_scores[keep]
        order = np.argsort(-s)[:topk]
        b = b[order]
        s = s[order]
        keep_nms = nms_numpy(b, s, iou_thresh=nms_thresh)

        out_boxes.append(b[keep_nms])
        out_scores.append(s[keep_nms])
        out_labels.append(np.full(len(keep_nms), cls_idx, dtype=np.int64))

    if not out_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = np.concatenate(out_boxes, axis=0)
    scores = np.concatenate(out_scores, axis=0)
    labels = np.concatenate(out_labels, axis=0)

    order = np.argsort(-scores)[:topk]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    orig_w, orig_h = orig_size
    scale_x = orig_w / float(input_size)
    scale_y = orig_h / float(input_size)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    boxes = clip_boxes(boxes, (orig_h, orig_w))

    return boxes, scores, labels


def draw_detections(image_bgr, boxes, scores, labels, class_names):
    out = image_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int).tolist()
        text = f"{class_names[label - 1]} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--nms-thresh", type=float, default=0.45)
    ap.add_argument("--output", default="result.jpg")
    args = ap.parse_args()

    class_names = load_labels(args.labels)
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(args.image)

    x, orig_size = preprocess_image(image_bgr, args.input_size)

    providers = ["CPUExecutionProvider"]
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    sess = ort.InferenceSession(args.onnx, providers=providers)
    class_logits, bbox_regression, anchors = sess.run(None, {sess.get_inputs()[0].name: x})

    boxes, scores, labels = postprocess(
        class_logits,
        bbox_regression,
        anchors,
        orig_size=orig_size,
        input_size=args.input_size,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
    )

    for box, score, label in zip(boxes, scores, labels):
        print({
            "label": class_names[label - 1],
            "score": float(score),
            "position_xyxy": [float(v) for v in box.tolist()],
        })

    vis = draw_detections(image_bgr, boxes, scores, labels, class_names)
    cv2.imwrite(args.output, vis)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
