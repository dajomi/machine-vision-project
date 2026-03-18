import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssd_mobilenet_v2_common import (
    VOCDataset,
    build_model,
    collate_fn,
    load_labels,
    read_split_ids,
    set_seed,
    validate_dataset,
)


class CFG:
    DATA_ROOT = Path("YOUR_DATASET")
    IMG_DIR = DATA_ROOT / "JPEGImages"
    ANN_DIR = DATA_ROOT / "Annotations"
    SPLIT_DIR = DATA_ROOT / "ImageSets" / "Main"
    LABEL_FILE = DATA_ROOT / "labels.txt"
    SAVE_DIR = Path("checkpoints_ssd_mbv2")

    NUM_EPOCHS = 20
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    USE_SCHEDULER = True
    STEP_SIZE = 15
    GAMMA = 0.1

    PRETRAINED_BACKBONE = True
    TRAINABLE_BACKBONE = True
    IMG_SIZE = 320
    HFLIP_PROB = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    SAVE_EVERY = 5


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

        total_loss += losses.item()
        pbar.set_postfix(loss=f"{losses.item():.4f}")
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch):
    model.train()  # torchvision SSD returns loss only in train mode
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        pbar.set_postfix(val_loss=f"{losses.item():.4f}")
    return total_loss / max(len(loader), 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(CFG.DATA_ROOT))
    p.add_argument("--save-dir", type=str, default=str(CFG.SAVE_DIR))
    p.add_argument("--epochs", type=int, default=CFG.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=CFG.LR)
    p.add_argument("--weight-decay", type=float, default=CFG.WEIGHT_DECAY)
    p.add_argument("--momentum", type=float, default=CFG.MOMENTUM)
    p.add_argument("--img-size", type=int, default=CFG.IMG_SIZE, choices=[320, 640])
    p.add_argument("--hflip-prob", type=float, default=CFG.HFLIP_PROB)
    p.add_argument("--num-workers", type=int, default=CFG.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=CFG.SEED)
    p.add_argument("--save-every", type=int, default=CFG.SAVE_EVERY)
    p.add_argument("--step-size", type=int, default=CFG.STEP_SIZE)
    p.add_argument("--gamma", type=float, default=CFG.GAMMA)
    p.add_argument("--no-scheduler", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    img_dir = data_root / "JPEGImages"
    ann_dir = data_root / "Annotations"
    split_dir = data_root / "ImageSets" / "Main"
    label_file = data_root / "labels.txt"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    classes, class_to_idx, _ = load_labels(label_file)
    num_classes = len(classes) + 1

    train_ids = read_split_ids(split_dir / "train.txt")
    val_ids = read_split_ids(split_dir / "val.txt")

    validate_dataset(img_dir, ann_dir, train_ids, class_to_idx, "train")
    validate_dataset(img_dir, ann_dir, val_ids, class_to_idx, "val")

    train_dataset = VOCDataset(img_dir, ann_dir, train_ids, class_to_idx, train=True, hflip_prob=args.hflip_prob)
    val_dataset = VOCDataset(img_dir, ann_dir, val_ids, class_to_idx, train=False, hflip_prob=0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    model = build_model(
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained_backbone=True,
        trainable_backbone=True,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = None if args.no_scheduler else torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_loss = float("inf")

    print("Classes:", classes)
    print("Num classes(with background):", num_classes)
    print("Device:", device)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device, epoch)
        if scheduler is not None:
            scheduler.step()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
            "val_loss": val_loss,
            "img_size": args.img_size,
        }
        torch.save(ckpt, save_dir / "latest.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, save_dir / "best.pth")

        if epoch % args.save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch}.pth")

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()
