import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


# =========================================================
# Config
# =========================================================
class CFG:
    TEST_IMG_DIR = Path("dataset/test/images")
    TEST_ANN_DIR = Path("dataset/test/Annotations")
    CLASS_FILE   = Path("dataset/classes.txt")
    MODEL_PATH   = Path("checkpoints/best.pth")

    BATCH_SIZE   = 4
    NUM_WORKERS  = 0
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    IOU_THRESH   = 0.5
    SCORE_THRESH = 0.05   # 평가용이므로 너무 높게 두지 말 것


# =========================================================
# Utils
# =========================================================
def collate_fn(batch):
    return tuple(zip(*batch))


def load_classes(class_file: Path):
    with open(class_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    class_to_idx = {name: i + 1 for i, name in enumerate(classes)}  # 0 background
    idx_to_class = {i + 1: name for i, name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class


def compute_iou(box1, box2):
    """
    box: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_ap(recall, precision):
    """
    VOC-style AP (all-point interpolation)
    recall, precision: list or tensor sorted by score
    """
    mrec = [0.0] + list(recall) + [1.0]
    mpre = [0.0] + list(precision) + [0.0]

    # precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


# =========================================================
# Dataset
# =========================================================
class VOCTestDataset(Dataset):
    def __init__(self, img_dir, ann_dir, class_to_idx):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.class_to_idx = class_to_idx

        self.image_files = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

    def __len__(self):
        return len(self.image_files)

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in self.class_to_idx:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])

        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = self.ann_dir / f"{img_path.stem}.xml"

        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image)

        boxes, labels = self.parse_voc_xml(xml_path)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": img_path.stem,
        }

        return image_tensor, target


# =========================================================
# Model
# =========================================================
def build_model(num_classes):
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        num_classes=num_classes
    )
    return model


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_map_recall(model, loader, idx_to_class, iou_thresh=0.5, score_thresh=0.05):
    model.eval()

    # GT 저장
    gt_by_class = defaultdict(lambda: defaultdict(list))
    gt_count_by_class = defaultdict(int)

    # prediction 저장
    pred_by_class = defaultdict(list)

    # -----------------------------------------------------
    # gather GT / predictions
    # -----------------------------------------------------
    for images, targets in loader:
        images = [img.to(CFG.DEVICE) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"]

            gt_boxes = target["boxes"].tolist()
            gt_labels = target["labels"].tolist()

            for box, label in zip(gt_boxes, gt_labels):
                gt_by_class[label][image_id].append({
                    "box": box,
                    "matched": False
                })
                gt_count_by_class[label] += 1

            pred_boxes = output["boxes"].detach().cpu().tolist()
            pred_scores = output["scores"].detach().cpu().tolist()
            pred_labels = output["labels"].detach().cpu().tolist()

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < score_thresh:
                    continue
                pred_by_class[label].append({
                    "image_id": image_id,
                    "box": box,
                    "score": score
                })

    # -----------------------------------------------------
    # class-wise metric
    # -----------------------------------------------------
    results = {}
    ap_list = []

    class_ids = sorted(idx_to_class.keys())

    for cls_id in class_ids:
        class_name = idx_to_class[cls_id]
        preds = pred_by_class[cls_id]
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        n_gt = gt_count_by_class[cls_id]

        tp = []
        fp = []

        for pred in preds:
            image_id = pred["image_id"]
            pred_box = pred["box"]

            gts = gt_by_class[cls_id][image_id]

            best_iou = 0.0
            best_gt_idx = -1

            for i, gt in enumerate(gts):
                iou = compute_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_thresh and best_gt_idx >= 0 and not gts[best_gt_idx]["matched"]:
                tp.append(1)
                fp.append(0)
                gts[best_gt_idx]["matched"] = True
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = []
        fp_cum = []
        s1 = 0
        s2 = 0
        for a, b in zip(tp, fp):
            s1 += a
            s2 += b
            tp_cum.append(s1)
            fp_cum.append(s2)

        if len(tp_cum) > 0:
            precision = [t / max(t + f, 1e-12) for t, f in zip(tp_cum, fp_cum)]
            recall = [t / max(n_gt, 1e-12) for t in tp_cum]
            ap = compute_ap(recall, precision)
            final_precision = precision[-1]
            final_recall = recall[-1]
            total_tp = tp_cum[-1]
            total_fp = fp_cum[-1]
        else:
            precision = []
            recall = []
            ap = 0.0
            final_precision = 0.0
            final_recall = 0.0
            total_tp = 0
            total_fp = 0

        fn = n_gt - total_tp

        results[class_name] = {
            "gt_count": n_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": fn,
            "precision": final_precision,
            "recall": final_recall,
            "ap50": ap,
        }

        # GT가 있는 클래스만 mAP에 포함
        if n_gt > 0:
            ap_list.append(ap)

    map50 = sum(ap_list) / len(ap_list) if len(ap_list) > 0 else 0.0
    return results, map50


# =========================================================
# Main
# =========================================================
def main():
    classes, class_to_idx, idx_to_class = load_classes(CFG.CLASS_FILE)
    num_classes = len(classes) + 1  # background 포함

    print("Classes:", classes)
    print("Device:", CFG.DEVICE)
    print("Model:", CFG.MODEL_PATH)

    test_dataset = VOCTestDataset(
        CFG.TEST_IMG_DIR,
        CFG.TEST_ANN_DIR,
        class_to_idx
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(num_classes).to(CFG.DEVICE)

    ckpt = torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    results, map50 = evaluate_map_recall(
        model,
        test_loader,
        idx_to_class=idx_to_class,
        iou_thresh=CFG.IOU_THRESH,
        score_thresh=CFG.SCORE_THRESH
    )

    print("\n" + "=" * 80)
    print("Test Result")
    print("=" * 80)
    print(f"{'Class':20s} {'GT':>6s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Precision':>12s} {'Recall':>12s} {'AP@0.5':>12s}")
    print("-" * 80)

    for cls_name, metric in results.items():
        print(
            f"{cls_name:20s} "
            f"{metric['gt_count']:6d} "
            f"{metric['tp']:6d} "
            f"{metric['fp']:6d} "
            f"{metric['fn']:6d} "
            f"{metric['precision']:12.4f} "
            f"{metric['recall']:12.4f} "
            f"{metric['ap50']:12.4f}"
        )

    print("-" * 80)
    print(f"{'mAP@0.5':20s} {map50:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()