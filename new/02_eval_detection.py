"""
02_eval_detection.py
테스트 셋 전체에 대한 Recall / Precision / AP50 / mAP50 평가

사용법:
    # PyTorch 체크포인트 기준
    python 02_eval_detection.py \
        --data-root YOUR_DATASET \
        --backend pytorch \
        --checkpoint checkpoints_ssd_mbv2/best.pth

    # ONNX (후처리 포함 버전)
    python 02_eval_detection.py \
        --data-root YOUR_DATASET \
        --backend onnx \
        --onnx ssd_mobilenetv2_320_post.onnx
"""

# ============================================================
# 표준 라이브러리
# ============================================================
import argparse
from pathlib import Path

# ============================================================
# 서드파티 라이브러리
# ============================================================
import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

# ============================================================
# 내부 모듈
# ============================================================
from ssd_mobilenet_v2_common import (
    evaluate_detections,
    find_image_path,
    load_checkpoint_model,
    load_ground_truth_from_voc,
    load_labels,
    read_split_ids,
)
from infer_jetson_onnx import postprocess, preprocess_image


# ============================================================
# 추론 함수
# ============================================================

@torch.no_grad()
def infer_pytorch(model, image_bgr: np.ndarray, device: str) -> dict:
    """PyTorch 모델 단일 이미지 추론"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    out = model([x.to(device)])[0]
    return {
        "boxes":  out["boxes"].cpu().numpy().astype(np.float32),
        "scores": out["scores"].cpu().numpy().astype(np.float32),
        "labels": out["labels"].cpu().numpy().astype(np.int64),
    }


def infer_onnx(
    sess,
    image_bgr: np.ndarray,
    input_size: int,
    score_thresh: float,
    nms_thresh: float,
) -> dict:
    """ONNX Runtime 단일 이미지 추론"""
    x, orig_size = preprocess_image(image_bgr, input_size)
    class_logits, bbox_regression, anchors = sess.run(
        None, {sess.get_inputs()[0].name: x}
    )
    boxes, scores, labels = postprocess(
        class_logits,
        bbox_regression,
        anchors,
        orig_size=orig_size,
        input_size=input_size,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )
    return {"boxes": boxes, "scores": scores, "labels": labels}


# ============================================================
# 인수 파싱
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SSD-MobileNetV2 Object Detection 평가")
    p.add_argument("--data-root",    required=True,             help="데이터셋 루트 경로")
    p.add_argument("--backend",      required=True,             choices=["pytorch", "onnx"])
    p.add_argument("--checkpoint",   default="",                help=".pth 경로 (backend=pytorch 필수)")
    p.add_argument("--onnx",         default="",                help="ONNX 모델 경로 (backend=onnx 필수)")
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--nms-thresh",   type=float, default=0.45)
    p.add_argument("--split",        default="test",            help="평가 split 이름 (기본: test)")
    return p.parse_args()


# ============================================================
# 메인
# ============================================================

def main():
    args = parse_args()

    data_root  = Path(args.data_root)
    ann_dir    = data_root / "Annotations"
    img_dir    = data_root / "JPEGImages"
    split_dir  = data_root / "ImageSets" / "Main"
    label_file = data_root / "labels.txt"

    class_names, class_to_idx, _ = load_labels(label_file)
    test_ids      = read_split_ids(split_dir / f"{args.split}.txt")
    gt_by_image   = load_ground_truth_from_voc(ann_dir, test_ids, class_to_idx)
    pred_by_image: dict = {}

    # ── PyTorch 추론 ──────────────────────────────────────────
    if args.backend == "pytorch":
        if not args.checkpoint:
            raise ValueError("--checkpoint 가 필요합니다 (backend=pytorch)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _, _ = load_checkpoint_model(args.checkpoint, device=device)
        print(f"[PyTorch] 모델 로드 완료  device={device}")

        for image_id in tqdm(test_ids, desc="[PyTorch] 추론"):
            img_path  = find_image_path(img_dir, image_id)
            image_bgr = cv2.imread(str(img_path))
            pred_by_image[image_id] = infer_pytorch(model, image_bgr, device)

    # ── ONNX 추론 ────────────────────────────────────────────
    else:
        if not args.onnx:
            raise ValueError("--onnx 가 필요합니다 (backend=onnx)")
        sess       = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_size = int(sess.get_inputs()[0].shape[-1])
        print(f"[ONNX] 모델 로드 완료  input_size={input_size}")

        for image_id in tqdm(test_ids, desc="[ONNX] 추론"):
            img_path  = find_image_path(img_dir, image_id)
            image_bgr = cv2.imread(str(img_path))
            pred_by_image[image_id] = infer_onnx(
                sess, image_bgr, input_size, args.score_thresh, args.nms_thresh,
            )

    # ── 평가 결과 출력 ────────────────────────────────────────
    rows, mAP50 = evaluate_detections(pred_by_image, gt_by_image, class_names, iou_thresh=0.5)

    col_w  = max(len(n) for n in class_names) + 2
    header = f"{'class':{col_w}s} {'GT':>6s} {'Pred':>6s} {'Recall':>10s} {'Precision':>10s} {'AP50':>10s}"
    sep    = "-" * len(header)
    bar    = "=" * len(header)

    print(f"\n{bar}")
    print("  Per-class metrics  (IoU = 0.5)")
    print(bar)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['class']:{col_w}s} {r['num_gt']:6d} {r['num_pred']:6d} "
            f"{r['recall']:10.4f} {r['precision']:10.4f} {r['ap50']:10.4f}"
        )
    print(sep)
    print(f"{'mAP50':>{col_w + 6 + 6 + 10 + 10 + 2}s} {mAP50:10.4f}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
