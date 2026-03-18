"""
01_train_ssd_mobilenet_v2.py
SSD-MobileNetV2 학습 스크립트

사용법:
    python 01_train_ssd_mobilenet_v2.py \
        --data-root YOUR_DATASET \
        --epochs 20 \
        --batch-size 4 \
        --img-size 320
"""

# ============================================================
# 표준 라이브러리
# ============================================================
import argparse
from pathlib import Path

# ============================================================
# 서드파티 라이브러리
# ============================================================
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# 내부 모듈
# ============================================================
from ssd_mobilenet_v2_common import (
    VOCDataset,
    build_model,
    collate_fn,
    load_labels,
    read_split_ids,
    set_seed,
    validate_dataset,
)


# ============================================================
# 기본 설정값
# ============================================================
class CFG:
    DATA_ROOT    = Path("YOUR_DATASET")
    SAVE_DIR     = Path("checkpoints_ssd_mbv2")

    NUM_EPOCHS   = 20
    BATCH_SIZE   = 4
    NUM_WORKERS  = 0
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    MOMENTUM     = 0.9

    # StepLR 스케줄러
    STEP_SIZE    = 15
    GAMMA        = 0.1

    IMG_SIZE     = 320      # 320 또는 640
    HFLIP_PROB   = 0.5
    SEED         = 42
    SAVE_EVERY   = 5        # N epoch 마다 중간 체크포인트 저장


# ============================================================
# 학습 루프
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)

    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


# ============================================================
# 검증 루프
# ============================================================

@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch: int) -> float:
    # torchvision SSD는 train 모드에서만 loss를 반환
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}", leave=False)

    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())
        total_loss += loss.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


# ============================================================
# 인수 파싱
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SSD-MobileNetV2 학습")
    p.add_argument("--data-root",    type=str,   default=str(CFG.DATA_ROOT))
    p.add_argument("--save-dir",     type=str,   default=str(CFG.SAVE_DIR))
    p.add_argument("--epochs",       type=int,   default=CFG.NUM_EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=CFG.BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=CFG.LR)
    p.add_argument("--weight-decay", type=float, default=CFG.WEIGHT_DECAY)
    p.add_argument("--momentum",     type=float, default=CFG.MOMENTUM)
    p.add_argument("--img-size",     type=int,   default=CFG.IMG_SIZE, choices=[320, 640])
    p.add_argument("--hflip-prob",   type=float, default=CFG.HFLIP_PROB)
    p.add_argument("--num-workers",  type=int,   default=CFG.NUM_WORKERS)
    p.add_argument("--seed",         type=int,   default=CFG.SEED)
    p.add_argument("--save-every",   type=int,   default=CFG.SAVE_EVERY)
    p.add_argument("--step-size",    type=int,   default=CFG.STEP_SIZE)
    p.add_argument("--gamma",        type=float, default=CFG.GAMMA)
    p.add_argument("--no-scheduler", action="store_true", help="LR 스케줄러 비활성화")
    return p.parse_args()


# ============================================================
# 메인
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── 경로 설정 ─────────────────────────────────────────────
    data_root  = Path(args.data_root)
    img_dir    = data_root / "JPEGImages"
    ann_dir    = data_root / "Annotations"
    split_dir  = data_root / "ImageSets" / "Main"
    label_file = data_root / "labels.txt"
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 레이블 / Split 로드 ───────────────────────────────────
    classes, class_to_idx, _ = load_labels(label_file)
    num_classes = len(classes) + 1  # 배경(0) 포함

    train_ids = read_split_ids(split_dir / "train.txt")
    val_ids   = read_split_ids(split_dir / "val.txt")

    # ── 데이터셋 무결성 검사 ──────────────────────────────────
    validate_dataset(img_dir, ann_dir, train_ids, class_to_idx, "train")
    validate_dataset(img_dir, ann_dir, val_ids,   class_to_idx, "val")

    # ── 데이터셋 / DataLoader ─────────────────────────────────
    train_ds = VOCDataset(img_dir, ann_dir, train_ids, class_to_idx, train=True,  hflip_prob=args.hflip_prob)
    val_ds   = VOCDataset(img_dir, ann_dir, val_ids,   class_to_idx, train=False, hflip_prob=0.0)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    pin_mem = device == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin_mem,
    )

    # ── 모델 ──────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained_backbone=True,
        trainable_backbone=True,
    ).to(device)

    # ── Optimizer / Scheduler ─────────────────────────────────
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = (
        None if args.no_scheduler
        else torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    )

    print(f"\n{'='*55}")
    print(f"  Classes       : {classes}")
    print(f"  Num classes   : {num_classes}  (배경 포함)")
    print(f"  Device        : {device}")
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")
    print(f"  Image size    : {args.img_size}×{args.img_size}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  LR            : {args.lr}")
    print(f"{'='*55}\n")

    # ── 학습 루프 ─────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss   = validate_one_epoch(model, val_loader, device, epoch)

        if scheduler is not None:
            scheduler.step()

        ckpt = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes":             classes,
            "val_loss":            val_loss,
            "img_size":            args.img_size,
        }

        torch.save(ckpt, save_dir / "latest.pth")

        if val_loss < best_val_loss:
            best_val_loss         = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, save_dir / "best.pth")
            best_mark = " ← best"
        else:
            best_mark = ""

        if epoch % args.save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pth")

        print(
            f"[Epoch {epoch:>3d}/{args.epochs}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"best={best_val_loss:.4f}{best_mark}"
        )

    print("\n학습 완료.")
    print(f"체크포인트 저장 위치: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
