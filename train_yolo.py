"""YOLO training pipeline for PASCAL VOC 2012.

Usage:
    python train_yolo.py --epochs 50 --lr 1e-3 --batch_size 16 --device auto

Key flags:
    --epochs        Training epochs (default: 50)
    --lr            Initial learning rate (default: 1e-3)
    --batch_size    Batch size (default: 16)
    --img_size      Input image size — must match VOCDetectionDataset (default: 640)
    --weight_decay  Adam weight decay (default: 5e-4)
    --save_path     Checkpoint path (default: yolo_best.pth)
    --voc_dir       Root directory for VOCdevkit (default: ./data/VOC)
    --device        auto | cuda | cpu
    --no-log        Disable file logging
    --plot          Save training loss curve to results/yolo/
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import complete_box_iou_loss

from datasets.voc import VOCDetectionDataset, detection_collate, NUM_CLASSES
from models.YOLO import YOLOv8
from utils.logger import TrainLogger


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class YOLOv8Loss(nn.Module):
    """Combined classification + regression loss for YOLOv8 heads.

    For each scale:
      - Classification : BCE with logits on assigned cells.
      - Regression     : CIoU loss on assigned cells.

    Label assignment uses a simplified IoU-based strategy:
      For each ground-truth box, find the grid cell whose centre falls
      inside the box and assign that cell as positive.

    Args:
        num_classes : Number of detection classes.
        img_size    : Input image size (square).
        cls_weight  : Weight on classification loss term.
        box_weight  : Weight on box regression loss term.
    """

    STRIDES = [8, 16, 32]  # P3, P4, P5

    def __init__(
        self,
        num_classes: int = 20,
        img_size:    int = 640,
        cls_weight:  float = 0.5,
        box_weight:  float = 7.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.cls_weight  = cls_weight
        self.box_weight  = box_weight
        self.bce         = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, outputs, targets: list[Tensor]) -> Tensor:
        """
        Args:
            outputs : list of 3 tuples (cls_logits, reg_preds), one per scale.
            targets : list of B tensors, each (M_i, 5) [x1,y1,x2,y2,cls].
        """
        total_cls = torch.tensor(0.0, device=outputs[0][0].device)
        total_box = torch.tensor(0.0, device=outputs[0][0].device)
        n_pos     = 0

        for (cls_logits, reg_preds), stride in zip(outputs, self.STRIDES):
            B, C, H, W = cls_logits.shape
            gs = stride  # grid size in pixels

            cls_tgt = torch.zeros_like(cls_logits)   # (B, C, H, W)
            reg_tgt = torch.zeros_like(reg_preds)     # (B, 4, H, W)
            mask    = torch.zeros(B, H, W, dtype=torch.bool,
                                  device=cls_logits.device)

            for b, tgt in enumerate(targets):
                if tgt.shape[0] == 0:
                    continue
                for box in tgt:
                    x1, y1, x2, y2, cls = box
                    cx = ((x1 + x2) / 2) / gs
                    cy = ((y1 + y2) / 2) / gs
                    gx = int(cx.clamp(0, W - 1))
                    gy = int(cy.clamp(0, H - 1))

                    cls_tgt[b, int(cls), gy, gx] = 1.0
                    reg_tgt[b, :, gy, gx] = torch.stack([
                        cx - gx,
                        cy - gy,
                        (x2 - x1) / gs,
                        (y2 - y1) / gs,
                    ])
                    mask[b, gy, gx] = True

            # Classification loss on all cells
            total_cls += self.bce(cls_logits, cls_tgt)

            # Box regression loss only on positive cells
            if mask.any():
                reg_pred_pos = reg_preds.permute(0, 2, 3, 1)[mask]  # (K, 4)
                reg_tgt_pos  = reg_tgt.permute(0, 2, 3, 1)[mask]    # (K, 4)

                # Convert to xyxy for CIoU
                def to_xyxy(t: Tensor) -> Tensor:
                    cx_, cy_ = t[:, 0], t[:, 1]
                    w_,  h_  = t[:, 2].exp(), t[:, 3].exp()
                    return torch.stack([cx_ - w_/2, cy_ - h_/2,
                                        cx_ + w_/2, cy_ + h_/2], dim=1)

                total_box += complete_box_iou_loss(
                    to_xyxy(reg_pred_pos),
                    to_xyxy(reg_tgt_pos),
                    reduction="mean",
                )
                n_pos += mask.sum().item()

        loss = self.cls_weight * total_cls + self.box_weight * total_box
        return loss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(args) -> tuple[DataLoader, DataLoader]:
    train_ds = VOCDetectionDataset(args.voc_dir, split="train",
                                   image_size=args.img_size, download=True)
    val_ds   = VOCDetectionDataset(args.voc_dir, split="val",
                                   image_size=args.img_size, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=detection_collate,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=detection_collate,
                              pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:   YOLOv8,
    loader:  DataLoader,
    loss_fn: YOLOv8Loss,
    opt:     torch.optim.Optimizer,
    device:  torch.device,
    logger:  TrainLogger,
    epoch:   int,
    epochs:  int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    header = f"{'Epoch':>6} {'Batch':>7} {'Loss':>10}"
    sep    = "-" * len(header)
    if epoch == 1:
        logger._w(f"\n{header}\n{sep}")

    for i, (imgs, targets) in enumerate(loader, 1):
        imgs    = imgs.to(device)
        targets = [t.to(device) for t in targets]

        opt.zero_grad()
        outputs = model(imgs)
        loss    = loss_fn(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        total_loss += loss.item()
        if i % 20 == 0 or i == n_batches:
            logger._w(f"{epoch:>6} {i:>4}/{n_batches:<3} {loss.item():>10.4f}")

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model:   YOLOv8,
    loader:  DataLoader,
    loss_fn: YOLOv8Loss,
    device:  torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs    = imgs.to(device)
        targets = [t.to(device) for t in targets]
        outputs = model(imgs)
        total_loss += loss_fn(outputs, targets).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_yolo_training(args, device: torch.device, logger: TrainLogger) -> YOLOv8:
    train_loader, val_loader = get_loaders(args)
    model   = YOLOv8(num_classes=NUM_CLASSES).to(device)
    loss_fn = YOLOv8Loss(num_classes=NUM_CLASSES, img_size=args.img_size)
    opt     = torch.optim.AdamW(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    logger._w(f"\nYOLOv8n | VOC 2012 | img={args.img_size} | "
              f"bs={args.batch_size} | lr={args.lr} | epochs={args.epochs}")
    logger._w(f"Train samples: {len(train_loader.dataset)}  "
              f"Val samples: {len(val_loader.dataset)}\n")

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt,
                                     device, logger, epoch, args.epochs)
        val_loss   = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        logger._w(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            logger._w(f"  → saved checkpoint (val_loss={val_loss:.4f})")

    if args.plot:
        os.makedirs("results/yolo", exist_ok=True)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(train_losses, label="Train")
            ax.plot(val_losses,   label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("YOLOv8n | VOC 2012")
            ax.legend()
            fig.savefig("results/yolo/yolo_loss.png", bbox_inches="tight")
            plt.close(fig)
            logger._w("Loss curve saved to results/yolo/yolo_loss.png")
        except Exception as e:
            logger._w(f"Plot failed: {e}")

    logger._w(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    logger._w(f"Checkpoint: {args.save_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8n on PASCAL VOC 2012")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--img_size",     type=int,   default=640)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--save_path",    type=str,   default="yolo_best.pth")
    p.add_argument("--voc_dir",      type=str,   default="./data/VOC")
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    logger = TrainLogger(experiment="YOLOv8n_VOC2012", enabled=args.log)
    run_yolo_training(args, device, logger)