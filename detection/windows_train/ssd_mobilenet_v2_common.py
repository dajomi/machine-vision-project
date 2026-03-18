import random
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.transforms import functional as F


# =========================================================
# General utils
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_labels(label_file: Path):
    if not label_file.exists():
        raise FileNotFoundError(f"labels.txt not found: {label_file}")

    with open(label_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        raise ValueError("labels.txt is empty")

    class_to_idx = {name: i + 1 for i, name in enumerate(classes)}
    idx_to_class = {i + 1: name for i, name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class


def read_split_ids(split_file: Path):
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"{split_file.name} is empty")
    return ids


def find_image_path(img_dir: Path, image_id: str):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def validate_dataset(img_dir: Path, ann_dir: Path, split_ids, class_to_idx, split_name: str):
    missing_images = []
    missing_xmls = []
    unknown_labels = []
    invalid_boxes = []

    for image_id in split_ids:
        img_path = find_image_path(img_dir, image_id)
        xml_path = ann_dir / f"{image_id}.xml"

        if img_path is None:
            missing_images.append(image_id)

        if not xml_path.exists():
            missing_xmls.append(image_id)
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox = obj.find("bndbox")

            if name_tag is None or name_tag.text is None:
                continue

            cls_name = name_tag.text.strip()
            if cls_name not in class_to_idx:
                unknown_labels.append((image_id, cls_name))

            if bndbox is not None:
                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    if xmax <= xmin or ymax <= ymin:
                        invalid_boxes.append((image_id, xmin, ymin, xmax, ymax))
                except Exception:
                    invalid_boxes.append((image_id, "parse_error"))

    if missing_images:
        raise FileNotFoundError(f"[{split_name}] missing images: {missing_images[:10]}")
    if missing_xmls:
        raise FileNotFoundError(f"[{split_name}] missing xmls: {missing_xmls[:10]}")
    if unknown_labels:
        raise ValueError(f"[{split_name}] unknown labels: {unknown_labels[:10]}")
    if invalid_boxes:
        raise ValueError(f"[{split_name}] invalid boxes: {invalid_boxes[:10]}")

    print(f"[OK] {split_name}: {len(split_ids)} samples")


# =========================================================
# Dataset
# =========================================================
class VOCDataset(Dataset):
    def __init__(self, img_dir, ann_dir, split_ids, class_to_idx, train=True, hflip_prob=0.5):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.image_ids = split_ids
        self.class_to_idx = class_to_idx
        self.train = train
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.image_ids)

    def parse_xml(self, xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        iscrowd = []

        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox = obj.find("bndbox")

            if name_tag is None or name_tag.text is None or bndbox is None:
                continue

            cls_name = name_tag.text.strip()
            if cls_name not in self.class_to_idx:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
            iscrowd.append(0)

        return boxes, labels, iscrowd

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = find_image_path(self.img_dir, image_id)
        xml_path = self.ann_dir / f"{image_id}.xml"

        image = Image.open(img_path).convert("RGB")
        width, _ = image.size

        boxes, labels, iscrowd = self.parse_xml(xml_path)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        if self.train and random.random() < self.hflip_prob:
            image = F.hflip(image)
            if boxes.shape[0] > 0:
                xmin = width - boxes[:, 2]
                xmax = width - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax

        image = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image, target


# =========================================================
# SSD-MobileNetV2 model
# =========================================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class SSDMobileNetV2Backbone(nn.Module):
    def __init__(self, img_size=320, pretrained=True, trainable=True):
        super().__init__()
        self.img_size = img_size
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights).features

        self.stage1 = nn.Sequential(*backbone[:7])
        self.stage2 = nn.Sequential(*backbone[7:14])
        self.stage3 = nn.Sequential(*backbone[14:])

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

        self.extra1 = nn.Sequential(
            ConvBNReLU(1280, 512, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(512, 512, kernel_size=3, stride=2, padding=1),
        )
        self.extra2 = nn.Sequential(
            ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, padding=1),
        )
        self.extra3 = nn.Sequential(
            ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, padding=1),
        )

        self.out_channels = self._get_out_channels()

    def _get_out_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, self.img_size, self.img_size)
            features = self.forward(x)
            return [feat.shape[1] for feat in features.values()]

    def forward(self, x):
        out = OrderedDict()
        x = self.stage1(x)
        out["0"] = x
        x = self.stage2(x)
        out["1"] = x
        x = self.stage3(x)
        out["2"] = x
        x = self.extra1(x)
        out["3"] = x
        x = self.extra2(x)
        out["4"] = x
        x = self.extra3(x)
        out["5"] = x
        return out


def build_model(num_classes: int, img_size=320, pretrained_backbone=True, trainable_backbone=True):
    backbone = SSDMobileNetV2Backbone(
        img_size=img_size,
        pretrained=pretrained_backbone,
        trainable=trainable_backbone,
    )

    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 3],
            [2],
            [2],
        ],
        min_ratio=0.15,
        max_ratio=0.90,
    )

    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(img_size, img_size),
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=200,
        topk_candidates=400,
    )
    return model


def load_checkpoint_model(checkpoint_path: str, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 320))
    model = build_model(
        num_classes=len(classes) + 1,
        img_size=img_size,
        pretrained_backbone=False,
        trainable_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes, img_size, ckpt


# =========================================================
# ONNX raw wrapper
# =========================================================
class SSDRawExportWrapper(nn.Module):
    """
    Input : preprocessed tensor [B,3,H,W] (already resized to img_size and normalized)
    Output: cls_logits [B,N,C], bbox_regression [B,N,4], anchors [N,4]
    """
    def __init__(self, model: SSD):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        features = self.model.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        head_outputs = self.model.head(features)
        image_sizes = [(int(x.shape[-2]), int(x.shape[-1])) for _ in range(int(x.shape[0]))]
        image_list = ImageList(x, image_sizes)
        anchors = self.model.anchor_generator(image_list, features)
        return head_outputs["cls_logits"], head_outputs["bbox_regression"], anchors[0]


# =========================================================
# Evaluation helpers (IoU=0.5 mAP style)
# =========================================================
def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(x11, x21)
    inter_y1 = np.maximum(y11, y21)
    inter_x2 = np.minimum(x12, x22)
    inter_y2 = np.minimum(y12, y22)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, (x12 - x11)) * np.maximum(0.0, (y12 - y11))
    area2 = np.maximum(0.0, (x22 - x21)) * np.maximum(0.0, (y22 - y21))
    union = area1 + area2 - inter
    return inter / np.clip(union, 1e-8, None)


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_detections(pred_by_image: Dict[str, dict], gt_by_image: Dict[str, dict], class_names: List[str], iou_thresh=0.5):
    results = []
    aps = []

    for cls_idx, cls_name in enumerate(class_names, start=1):
        gt_records = {}
        npos = 0
        for image_id, gt in gt_by_image.items():
            mask = gt["labels"] == cls_idx
            boxes = gt["boxes"][mask]
            gt_records[image_id] = {
                "boxes": boxes,
                "detected": np.zeros(len(boxes), dtype=bool),
            }
            npos += len(boxes)

        preds = []
        for image_id, pred in pred_by_image.items():
            mask = pred["labels"] == cls_idx
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                preds.append((image_id, float(score), box))
        preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (image_id, _, box_pred) in enumerate(preds):
            gt_entry = gt_records[image_id]
            gt_boxes = gt_entry["boxes"]
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            ious = box_iou_np(np.asarray([box_pred], dtype=np.float32), gt_boxes.astype(np.float32))[0]
            max_idx = int(np.argmax(ious))
            max_iou = float(ious[max_idx])

            if max_iou >= iou_thresh and not gt_entry["detected"][max_idx]:
                tp[i] = 1
                gt_entry["detected"][max_idx] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(npos, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
        ap = voc_ap(recalls, precisions) if len(preds) > 0 else 0.0
        aps.append(ap)

        final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
        final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0

        results.append({
            "class": cls_name,
            "num_gt": int(npos),
            "num_pred": int(len(preds)),
            "recall": final_recall,
            "precision": final_precision,
            "ap50": float(ap),
        })

    mAP50 = float(np.mean(aps)) if aps else 0.0
    return results, mAP50


def load_ground_truth_from_voc(ann_dir: Path, image_ids: List[str], class_to_idx: Dict[str, int]):
    gt_by_image = {}
    for image_id in image_ids:
        xml_path = ann_dir / f"{image_id}.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            if name not in class_to_idx:
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin", default="0"))
            ymin = float(b.findtext("ymin", default="0"))
            xmax = float(b.findtext("xmax", default="0"))
            ymax = float(b.findtext("ymax", default="0"))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_idx[name])
        gt_by_image[image_id] = {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, dtype=np.int64),
        }
    return gt_by_image
