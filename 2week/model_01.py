import os
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F


# =========================================================
# Hyperparameters / Config
# =========================================================
class CFG:
    # -------- paths --------
    DATA_ROOT = Path("dataset")
    TRAIN_IMG_DIR = DATA_ROOT / "train" / "images"
    TRAIN_ANN_DIR = DATA_ROOT / "train" / "Annotations"
    VAL_IMG_DIR   = DATA_ROOT / "val" / "images"
    VAL_ANN_DIR   = DATA_ROOT / "val" / "Annotations"
    CLASS_FILE    = DATA_ROOT / "classes.txt"
    SAVE_DIR      = Path("checkpoints")

    # -------- training --------
    NUM_EPOCHS    = 50
    BATCH_SIZE    = 8
    NUM_WORKERS   = 0           # Windows는 0 또는 2부터 시작 추천
    LEARNING_RATE = 0.001
    WEIGHT_DECAY  = 1e-4
    MOMENTUM      = 0.9

    # -------- lr scheduler --------
    USE_SCHEDULER = True
    STEP_SIZE     = 15
    GAMMA         = 0.1

    # -------- model --------
    PRETRAINED    = True
    TRAINABLE_BACKBONE_LAYERS = 6   # 0~6
    IMAGE_MIN_SIZE = 320
    IMAGE_MAX_SIZE = 320

    # -------- aug --------
    HFLIP_PROB    = 0.5

    # -------- save / eval --------
    SAVE_EVERY    = 5
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    SEED          = 42


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_classes(class_file: Path):
    with open(class_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    class_to_idx = {name: i + 1 for i, name in enumerate(classes)}  # 0 is background
    return classes, class_to_idx


# =========================================================
# VOC Dataset
# =========================================================
class VOCDataset(Dataset):
    def __init__(self, img_dir, ann_dir, class_to_idx, train=True, hflip_prob=0.5):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.class_to_idx = class_to_idx
        self.train = train
        self.hflip_prob = hflip_prob

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
        iscrowd = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in self.class_to_idx:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # invalid box skip
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
            iscrowd.append(0)

        return boxes, labels, iscrowd

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = self.ann_dir / (img_path.stem + ".xml")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        boxes, labels, iscrowd = self.parse_voc_xml(xml_path)

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

        image_id = torch.tensor([idx])

        # augmentation: horizontal flip
        if self.train and random.random() < self.hflip_prob:
            image = F.hflip(image)
            if boxes.shape[0] > 0:
                xmin = width - boxes[:, 2]
                xmax = width - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax

        image = F.to_tensor(image)   # [0,1] float tensor

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        return image, target


# =========================================================
# Model
# =========================================================
def build_model(num_classes: int):
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if CFG.PRETRAINED else None

    model = ssdlite320_mobilenet_v3_large(
        weights=weights,
        num_classes=num_classes,   # includes background
        trainable_backbone_layers=CFG.TRAINABLE_BACKBONE_LAYERS,
    )

    return model


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        total_loss += loss_value
        pbar.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch):
    # torchvision detection 모델은 eval()일 때 예측만 반환함
    # 그래서 검증 손실 계산 대신 예측 개수만 간단히 확인
    model.eval()
    total_preds = 0

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    for images, _ in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)
        total_preds += sum(len(out["boxes"]) for out in outputs)

    avg_preds = total_preds / max(len(loader.dataset), 1)
    return avg_preds


def main():
    set_seed(CFG.SEED)
    CFG.SAVE_DIR.mkdir(parents=True, exist_ok=True)

    classes, class_to_idx = load_classes(CFG.CLASS_FILE)
    num_classes = len(classes) + 1   # + background

    print("Classes:", classes)
    print("Num classes (with background):", num_classes)
    print("Device:", CFG.DEVICE)

    train_dataset = VOCDataset(
        CFG.TRAIN_IMG_DIR, CFG.TRAIN_ANN_DIR, class_to_idx,
        train=True, hflip_prob=CFG.HFLIP_PROB
    )
    val_dataset = VOCDataset(
        CFG.VAL_IMG_DIR, CFG.VAL_ANN_DIR, class_to_idx,
        train=False, hflip_prob=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(num_classes=num_classes).to(CFG.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CFG.LEARNING_RATE,
        momentum=CFG.MOMENTUM,
        weight_decay=CFG.WEIGHT_DECAY
    )

    scheduler = None
    if CFG.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=CFG.STEP_SIZE,
            gamma=CFG.GAMMA
        )

    best_train_loss = float("inf")

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, CFG.DEVICE, epoch)
        val_avg_preds = validate_one_epoch(model, val_loader, CFG.DEVICE, epoch)

        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_avg_preds={val_avg_preds:.2f}")

        # save latest
        latest_path = CFG.SAVE_DIR / "latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
            "cfg": {k: v for k, v in CFG.__dict__.items() if not k.startswith("__")}
        }, latest_path)

        # save best
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_path = CFG.SAVE_DIR / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": classes,
                "best_train_loss": best_train_loss,
                "cfg": {k: v for k, v in CFG.__dict__.items() if not k.startswith("__")}
            }, best_path)

        # periodic save
        if epoch % CFG.SAVE_EVERY == 0:
            save_path = CFG.SAVE_DIR / f"epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": classes,
                "cfg": {k: v for k, v in CFG.__dict__.items() if not k.startswith("__")}
            }, save_path)

    print("Training finished.")


if __name__ == "__main__":
    main()