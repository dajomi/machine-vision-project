"""
ssd_mobilenet_v2_common.py
공통 유틸리티 / 데이터셋 / 모델 / 평가 모듈
"""

# ============================================================
# 표준 라이브러리
# ============================================================
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 서드파티 라이브러리
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.transforms import functional as TF


# ============================================================
# 1. 공통 유틸리티
# ============================================================

def set_seed(seed: int) -> None:
    """재현성을 위한 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """DataLoader용 collate 함수 (가변 크기 타겟 처리)"""
    return tuple(zip(*batch))


def load_labels(label_file: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    labels.txt 로드

    Returns
    -------
    classes      : ['scratch', 'dent', ...]
    class_to_idx : {'scratch': 1, 'dent': 2, ...}  (0 = 배경)
    idx_to_class : {1: 'scratch', 2: 'dent', ...}
    """
    label_file = Path(label_file)
    if not label_file.exists():
        raise FileNotFoundError(f"labels.txt 없음: {label_file}")

    classes = [l.strip() for l in label_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not classes:
        raise ValueError("labels.txt 가 비어 있습니다.")

    class_to_idx = {name: i + 1 for i, name in enumerate(classes)}
    idx_to_class = {i + 1: name for i, name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class


def read_split_ids(split_file: Path) -> List[str]:
    """ImageSets/Main/train.txt 등에서 이미지 ID 목록 로드"""
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"split 파일 없음: {split_file}")

    ids = [l.strip() for l in split_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not ids:
        raise ValueError(f"{split_file.name} 이 비어 있습니다.")
    return ids


def find_image_path(img_dir: Path, image_id: str) -> Optional[Path]:
    """이미지 ID로 실제 파일 경로 탐색 (.jpg / .jpeg / .png / .bmp 순)"""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = Path(img_dir) / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


# ============================================================
# 2. 데이터셋 무결성 검사
# ============================================================

def validate_dataset(
    img_dir: Path,
    ann_dir: Path,
    split_ids: List[str],
    class_to_idx: Dict[str, int],
    split_name: str,
) -> None:
    """
    데이터셋 무결성 검사
    - 이미지 / XML 파일 존재 여부
    - labels.txt 에 없는 클래스명
    - bndbox 좌표 유효성 (xmax > xmin, ymax > ymin)
    """
    missing_images, missing_xmls, unknown_labels, invalid_boxes = [], [], [], []

    for image_id in split_ids:
        img_path = find_image_path(img_dir, image_id)
        xml_path = Path(ann_dir) / f"{image_id}.xml"

        if img_path is None:
            missing_images.append(image_id)
        if not xml_path.exists():
            missing_xmls.append(image_id)
            continue

        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox   = obj.find("bndbox")

            if name_tag is None or not (name_tag.text or "").strip():
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

    errors = []
    if missing_images:
        errors.append(f"[{split_name}] 이미지 없음: {missing_images[:10]}")
    if missing_xmls:
        errors.append(f"[{split_name}] XML 없음: {missing_xmls[:10]}")
    if unknown_labels:
        errors.append(f"[{split_name}] 미등록 라벨: {unknown_labels[:10]}")
    if invalid_boxes:
        errors.append(f"[{split_name}] 잘못된 박스: {invalid_boxes[:10]}")

    if errors:
        raise ValueError("\n".join(errors))

    print(f"[OK] {split_name}: {len(split_ids)}개 샘플 검증 완료")


# ============================================================
# 3. 데이터셋
# ============================================================

class VOCDataset(Dataset):
    """Pascal VOC 형식 객체 검출 데이터셋"""

    def __init__(
        self,
        img_dir: Path,
        ann_dir: Path,
        split_ids: List[str],
        class_to_idx: Dict[str, int],
        train: bool = True,
        hflip_prob: float = 0.5,
    ):
        self.img_dir      = Path(img_dir)
        self.ann_dir      = Path(ann_dir)
        self.image_ids    = split_ids
        self.class_to_idx = class_to_idx
        self.train        = train
        self.hflip_prob   = hflip_prob

    def __len__(self) -> int:
        return len(self.image_ids)

    def _parse_xml(self, xml_path: Path) -> Tuple[List, List, List]:
        """XML annotation에서 boxes / labels / iscrowd 파싱"""
        root = ET.parse(xml_path).getroot()
        boxes, labels, iscrowd = [], [], []

        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox   = obj.find("bndbox")

            if name_tag is None or not (name_tag.text or "").strip() or bndbox is None:
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

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = find_image_path(self.img_dir, image_id)
        if img_path is None:
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {self.img_dir / image_id}.*\n"
                "  → validate_dataset()을 먼저 실행해 데이터셋을 검증하세요."
            )
        xml_path = self.ann_dir / f"{image_id}.xml"

        image = Image.open(img_path).convert("RGB")
        width, _ = image.size

        boxes, labels, iscrowd = self._parse_xml(xml_path)

        if boxes:
            boxes_t   = torch.tensor(boxes,   dtype=torch.float32)
            labels_t  = torch.tensor(labels,  dtype=torch.int64)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            area_t    = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t   = torch.zeros((0, 4), dtype=torch.float32)
            labels_t  = torch.zeros((0,),   dtype=torch.int64)
            iscrowd_t = torch.zeros((0,),   dtype=torch.int64)
            area_t    = torch.zeros((0,),   dtype=torch.float32)

        # 좌우 반전 증강 (학습 시)
        if self.train and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if boxes_t.shape[0] > 0:
                xmin_new       = width - boxes_t[:, 2]
                xmax_new       = width - boxes_t[:, 0]
                boxes_t[:, 0]  = xmin_new
                boxes_t[:, 2]  = xmax_new

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area":     area_t,
            "iscrowd":  iscrowd_t,
        }
        return TF.to_tensor(image), target


# ============================================================
# 4. 백본 / 모델 정의
# ============================================================

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU6 블록"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class SSDMobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 기반 SSD 백본 (6개 피처맵 스케일)

    Stage 분할
    ----------
    stage1 : features[0:7]   stride  8 → 32ch
    stage2 : features[7:14]  stride 16 → 96ch
    stage3 : features[14:]   stride 32 → 1280ch
    extra1~3 : 추가 다운샘플 레이어
    """

    def __init__(self, img_size: int = 320, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        self.img_size = img_size

        weights  = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights).features

        self.stage1 = nn.Sequential(*backbone[:7])    # 32ch
        self.stage2 = nn.Sequential(*backbone[7:14])  # 96ch
        self.stage3 = nn.Sequential(*backbone[14:])   # 1280ch

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

        self.extra1 = nn.Sequential(
            ConvBNReLU(1280, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256,  256, kernel_size=3, stride=2, padding=1),
        )
        self.extra2 = nn.Sequential(
            ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, padding=1),
        )
        self.extra3 = nn.Sequential(
            ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, padding=1),
        )

        self.out_channels = self._compute_out_channels()

    def _compute_out_channels(self) -> List[int]:
        """각 피처맵 채널 수 자동 계산"""
        with torch.no_grad():
            dummy    = torch.zeros(1, 3, self.img_size, self.img_size)
            features = self.forward(dummy)
            return [f.shape[1] for f in features.values()]

    def forward(self, x: torch.Tensor) -> OrderedDict:
        out = OrderedDict()
        x = self.stage1(x);  out["0"] = x
        x = self.stage2(x);  out["1"] = x
        x = self.stage3(x);  out["2"] = x
        x = self.extra1(x);  out["3"] = x
        x = self.extra2(x);  out["4"] = x
        x = self.extra3(x);  out["5"] = x
        return out


def build_model(
    num_classes: int,
    img_size: int = 320,
    pretrained_backbone: bool = True,
    trainable_backbone: bool = True,
    score_thresh: float = 0.01,
    nms_thresh: float = 0.45,
    detections_per_img: int = 200,
    topk_candidates: int = 400,
) -> SSD:
    """SSD-MobileNetV2 모델 생성"""
    backbone = SSDMobileNetV2Backbone(
        img_size=img_size,
        pretrained=pretrained_backbone,
        trainable=trainable_backbone,
    )
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        min_ratio=0.15,
        max_ratio=0.90,
    )
    return SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(img_size, img_size),
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        topk_candidates=topk_candidates,
    )


def load_checkpoint_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[SSD, List[str], int, dict]:
    """
    .pth 체크포인트 로드

    Returns
    -------
    model, classes, img_size, checkpoint_dict
    """
    ckpt     = torch.load(checkpoint_path, map_location=device)
    classes  = ckpt["classes"]
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


# ============================================================
# 5. ONNX 내보내기용 래퍼 (후처리 없음 / Raw)
# ============================================================

class SSDRawExportWrapper(nn.Module):
    """
    ONNX 내보내기용 래퍼 — 후처리 미포함

    Input
    -----
    x : [B, 3, H, W]  정규화된 float32 텐서

    Outputs
    -------
    class_logits    : [B, num_anchors, num_classes]
    bbox_regression : [B, num_anchors, 4]
    anchors         : [num_anchors, 4]
    """

    def __init__(self, model: SSD):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        features = self.model.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs = self.model.head(feature_list)
        image_sizes  = [(int(x.shape[-2]), int(x.shape[-1]))] * int(x.shape[0])
        image_list   = ImageList(x, image_sizes)
        anchors      = self.model.anchor_generator(image_list, feature_list)

        return (
            head_outputs["cls_logits"],
            head_outputs["bbox_regression"],
            anchors[0],
        )


# ============================================================
# 6. 평가 유틸리티 (AP50 / mAP50)
# ============================================================

def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """두 박스 배열 간 IoU 행렬 계산 (numpy)"""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0],   boxes2[:, 1],   boxes2[:, 2],   boxes2[:, 3]

    inter_w = np.maximum(0.0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0.0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter   = inter_w * inter_h

    area1 = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    area2 = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)
    union = area1 + area2 - inter
    return inter / np.clip(union, 1e-8, None)


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC 방식 11-point interpolated AP 계산"""
    mrec = np.concatenate(([0.0], rec,  [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detections(
    pred_by_image: Dict[str, dict],
    gt_by_image:   Dict[str, dict],
    class_names:   List[str],
    iou_thresh:    float = 0.5,
) -> Tuple[List[dict], float]:
    """
    Per-class AP50 및 mAP50 계산

    Recall 위주 평가 정책
    --------------------
    - GT가 한 개도 없는 클래스는 mAP 계산에서 제외 (패널티 방지)
    - 예측이 0개인 클래스도 recall=0, ap50=0 으로 정상 집계

    Returns
    -------
    rows  : 클래스별 dict 리스트
            {'class', 'num_gt', 'num_pred', 'recall', 'precision', 'ap50'}
    mAP50 : float  (GT가 있는 클래스들의 평균만 포함)
    """
    results, aps = [], []

    for cls_idx, cls_name in enumerate(class_names, start=1):
        # ── Ground truth 수집 ─────────────────────────────────
        gt_records: Dict[str, dict] = {}
        npos = 0
        for image_id, gt in gt_by_image.items():
            mask     = gt["labels"] == cls_idx
            gt_boxes = gt["boxes"][mask]
            gt_records[image_id] = {
                "boxes":    gt_boxes,
                "detected": np.zeros(len(gt_boxes), dtype=bool),
            }
            npos += len(gt_boxes)

        # ── Prediction 수집 → score 내림차순 정렬 ────────────
        preds = [
            (image_id, float(score), box)
            for image_id, pred in pred_by_image.items()
            for box, score in zip(
                pred["boxes"][pred["labels"] == cls_idx],
                pred["scores"][pred["labels"] == cls_idx],
            )
        ]
        preds.sort(key=lambda t: t[1], reverse=True)

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (image_id, _, box_pred) in enumerate(preds):
            gt_entry = gt_records.get(image_id)
            if gt_entry is None or len(gt_entry["boxes"]) == 0:
                fp[i] = 1
                continue

            ious    = box_iou_np(
                np.asarray([box_pred], dtype=np.float32),
                gt_entry["boxes"].astype(np.float32),
            )[0]
            max_idx = int(np.argmax(ious))
            max_iou = float(ious[max_idx])

            if max_iou >= iou_thresh and not gt_entry["detected"][max_idx]:
                tp[i] = 1
                gt_entry["detected"][max_idx] = True
            else:
                fp[i] = 1

        cum_tp     = np.cumsum(tp)
        cum_fp     = np.cumsum(fp)
        recalls    = cum_tp / max(npos, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
        ap         = voc_ap(recalls, precisions) if preds else 0.0

        # GT가 있는 클래스만 mAP 집계에 포함 (recall 위주 평가 정책)
        if npos > 0:
            aps.append(ap)

        results.append({
            "class":     cls_name,
            "num_gt":    int(npos),
            "num_pred":  int(len(preds)),
            "recall":    float(recalls[-1])    if len(recalls)    > 0 else 0.0,
            "precision": float(precisions[-1]) if len(precisions) > 0 else 0.0,
            "ap50":      float(ap),
        })

    mAP50 = float(np.mean(aps)) if aps else 0.0
    return results, mAP50


def load_ground_truth_from_voc(
    ann_dir:      Path,
    image_ids:    List[str],
    class_to_idx: Dict[str, int],
) -> Dict[str, dict]:
    """VOC XML annotation에서 ground truth 로드"""
    gt_by_image = {}

    for image_id in image_ids:
        xml_path = Path(ann_dir) / f"{image_id}.xml"
        root     = ET.parse(xml_path).getroot()
        boxes, labels = [], []

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_to_idx:
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin", "0"))
            ymin = float(b.findtext("ymin", "0"))
            xmax = float(b.findtext("xmax", "0"))
            ymax = float(b.findtext("ymax", "0"))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_idx[name])

        gt_by_image[image_id] = {
            "boxes":  np.asarray(boxes,  dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    return gt_by_image
