import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from ssd_mobilenet_v2_common import (
    evaluate_detections,
    find_image_path,
    load_checkpoint_model,
    load_ground_truth_from_voc,
    load_labels,
    read_split_ids,
)
from infer_jetson_onnx import preprocess_image, postprocess


@torch.no_grad()
def infer_pytorch(model, image_bgr, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    out = model([x.to(device)])[0]
    return {
        "boxes": out["boxes"].detach().cpu().numpy().astype(np.float32),
        "scores": out["scores"].detach().cpu().numpy().astype(np.float32),
        "labels": out["labels"].detach().cpu().numpy().astype(np.int64),
    }


def infer_onnx(sess, image_bgr, input_size, score_thresh, nms_thresh):
    x, orig_size = preprocess_image(image_bgr, input_size)
    class_logits, bbox_regression, anchors = sess.run(None, {sess.get_inputs()[0].name: x})
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--backend", choices=["pytorch", "onnx"], required=True)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--onnx", type=str, default="")
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--nms-thresh", type=float, default=0.45)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ann_dir = data_root / "Annotations"
    img_dir = data_root / "JPEGImages"
    split_dir = data_root / "ImageSets" / "Main"
    label_file = data_root / "labels.txt"

    class_names, class_to_idx, _ = load_labels(label_file)
    test_ids = read_split_ids(split_dir / "test.txt")
    gt_by_image = load_ground_truth_from_voc(ann_dir, test_ids, class_to_idx)
    pred_by_image = {}

    if args.backend == "pytorch":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for backend=pytorch")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _, _ = load_checkpoint_model(args.checkpoint, device=device)
        for image_id in tqdm(test_ids, desc="Eval PyTorch"):
            img_path = find_image_path(img_dir, image_id)
            image_bgr = cv2.imread(str(img_path))
            pred_by_image[image_id] = infer_pytorch(model, image_bgr, device)
    else:
        if not args.onnx:
            raise ValueError("--onnx is required for backend=onnx")
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_size = sess.get_inputs()[0].shape[-1]
        for image_id in tqdm(test_ids, desc="Eval ONNX"):
            img_path = find_image_path(img_dir, image_id)
            image_bgr = cv2.imread(str(img_path))
            pred_by_image[image_id] = infer_onnx(sess, image_bgr, int(input_size), args.score_thresh, args.nms_thresh)

    rows, mAP50 = evaluate_detections(pred_by_image, gt_by_image, class_names, iou_thresh=0.5)

    print("\n=== Per-class metrics (IoU=0.5) ===")
    print(f"{'class':20s} {'GT':>6s} {'Pred':>6s} {'Recall':>10s} {'Precision':>10s} {'AP50':>10s}")
    for r in rows:
        print(f"{r['class']:20s} {r['num_gt']:6d} {r['num_pred']:6d} {r['recall']:10.4f} {r['precision']:10.4f} {r['ap50']:10.4f}")
    print(f"\nmAP50 = {mAP50:.4f}")


if __name__ == "__main__":
    main()
