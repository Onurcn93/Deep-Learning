"""YOLO training pipeline for PASCAL VOC 2012.

Usage:
    python train_yolo.py --epochs 50 --lr 1e-3 --batch_size 16 --device auto

Key flags:
    --epochs              Training epochs (default: 50)
    --lr                  Initial learning rate for neck/heads (default: 1e-3)
    --batch_size          Batch size (default: 16)
    --img_size            Input image size (default: 640)
    --weight_decay        AdamW weight decay (default: 5e-4)
    --save_path           Checkpoint path (default: yolo_best.pth)
    --voc_dir             Root directory for VOCdevkit (default: ./data/VOC)
    --device              auto | cuda | cpu
    --pretrained_backbone Use ImageNet-pretrained ResNet50 backbone (default: True)
    --freeze_backbone     Freeze backbone weights entirely (default: False)
    --backbone_lr_mult    Backbone LR = lr × this factor (default: 0.1)
    --eval_map_every      Compute mAP@0.5 on val every N epochs (default: 1, 0 = off)
    --no-log              Disable file logging
    --plot                Save training loss curve to results/yolo/
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import complete_box_iou_loss, box_iou

from datasets.voc import VOCDetectionDataset, detection_collate, NUM_CLASSES, VOC_CLASSES, NUM_CLASSES_PERSON
from models.YOLO import YOLOv8
from utils.detection import decode_predictions, compute_map
from utils.logger import TrainLogger


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class YOLOv8Loss(nn.Module):
    """Classification + regression loss with Task-Aligned Label Assignment (TAL).

    For each scale:
      - TAL assigns the top-k highest-alignment cells per GT as positives.
        Alignment score = cls_score^alpha × IoU^beta, restricted to cells
        whose centre falls inside the GT box.
      - Classification : focal BCE on all cells; positives get soft IoU-weighted
        class targets, negatives get 0.
      - Regression     : CIoU on positive cells only.

    Args:
        num_classes  : Number of detection classes.
        img_size     : Input image size (square).
        cls_weight   : Weight on classification loss.
        box_weight   : Weight on box regression loss.
        topk         : Max positive cells assigned per GT per scale (default 10).
        alpha        : cls-score exponent in TAL alignment metric.
        beta         : IoU exponent in TAL alignment metric.
        focal_gamma  : Focal loss γ — down-weights easy negatives.
    """

    STRIDES = [8, 16, 32]

    def __init__(
        self,
        num_classes:  int   = 20,
        img_size:     int   = 640,
        cls_weight:   float = 0.5,
        box_weight:   float = 7.5,
        topk:         int   = 10,
        alpha:        float = 0.5,
        beta:         float = 6.0,
        focal_gamma:  float = 1.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.cls_weight  = cls_weight
        self.box_weight  = box_weight
        self.topk        = topk
        self.alpha       = alpha
        self.beta        = beta
        self.focal_gamma = focal_gamma

    # ── helpers ─────────────────────────────────────────────────────────────

    def _decode_preds(self, reg_preds: Tensor, stride: int) -> Tensor:
        """(B,4,H,W) reg predictions → (B,H*W,4) absolute xyxy boxes."""
        B, _, H, W = reg_preds.shape
        device = reg_preds.device
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        gx = gx.reshape(1, 1, H, W)
        gy = gy.reshape(1, 1, H, W)
        cx = (reg_preds[:, 0:1] + gx) * stride
        cy = (reg_preds[:, 1:2] + gy) * stride
        w  = reg_preds[:, 2:3].clamp(-8, 8).exp() * stride
        h  = reg_preds[:, 3:4].clamp(-8, 8).exp() * stride
        boxes = torch.cat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        return boxes.permute(0, 2, 3, 1).reshape(B, H * W, 4)

    @staticmethod
    def _to_xyxy(t: Tensor) -> Tensor:
        """Grid-relative [cx_off, cy_off, log_w, log_h] → virtual xyxy for CIoU."""
        cx, cy = t[:, 0], t[:, 1]
        w, h   = t[:, 2].clamp(-8, 8).exp(), t[:, 3].clamp(-8, 8).exp()
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    def _focal_bce(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Focal-weighted BCE averaged over all elements."""
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t   = targets * logits.detach().sigmoid() + (1 - targets) * (1 - logits.detach().sigmoid())
        return (bce * (1 - p_t).pow(self.focal_gamma)).mean()

    # ── Task-Aligned Label Assignment ────────────────────────────────────────

    def _tal_assign(
        self,
        cls_logits: Tensor,
        reg_preds:  Tensor,
        targets:    list[Tensor],
        stride:     int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Assign GT boxes to grid cells.

        Returns:
            cls_tgt  (B, N, C)  IoU-weighted class targets; 0 = background.
            reg_tgt  (B, N, 4)  [cx_off, cy_off, log_w, log_h] for positive cells.
            mask_pos (B, N)     True for positive cells.
        """
        B, C, H, W = cls_logits.shape
        N      = H * W
        device = cls_logits.device

        # Grid cell origins and centres in pixel space
        gy_idx, gx_idx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        gx_f = gx_idx.reshape(-1)           # (N,) integer col index
        gy_f = gy_idx.reshape(-1)           # (N,) integer row index
        cx_c = (gx_f + 0.5) * stride        # cell-centre x (pixels)
        cy_c = (gy_f + 0.5) * stride        # cell-centre y (pixels)

        pred_boxes = self._decode_preds(reg_preds, stride)              # (B, N, 4)
        cls_scores = (cls_logits.detach().sigmoid()
                      .permute(0, 2, 3, 1).reshape(B, N, C))            # (B, N, C)

        cls_tgt  = torch.zeros(B, N, C, device=device)
        reg_tgt  = torch.zeros(B, N, 4, device=device)
        mask_pos = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            tgt = targets[b]                # (M, 5) [x1,y1,x2,y2,cls]
            if tgt.shape[0] == 0:
                continue

            gt_boxes = tgt[:, :4]           # (M, 4) absolute xyxy
            gt_cls   = tgt[:, 4].long()     # (M,)
            M        = gt_boxes.shape[0]

            # ── candidate cells: centre must fall inside GT box ──────────
            inside = (
                (cx_c.unsqueeze(0) >= gt_boxes[:, 0:1]) &   # (M, N)
                (cx_c.unsqueeze(0) <= gt_boxes[:, 2:3]) &
                (cy_c.unsqueeze(0) >= gt_boxes[:, 1:2]) &
                (cy_c.unsqueeze(0) <= gt_boxes[:, 3:4])
            )

            # ── IoU between current predicted boxes and all GTs ──────────
            iou_mn = box_iou(gt_boxes, pred_boxes[b].detach())          # (M, N)

            # ── per-GT class score at each cell ──────────────────────────
            # cls_scores[b]: (N, C) → index GT classes → (N, M) → T → (M, N)
            gt_cls_sc = cls_scores[b][:, gt_cls].T                      # (M, N)

            # ── alignment score (zero outside GT box) ────────────────────
            align = (gt_cls_sc.clamp(0).pow(self.alpha) *
                     iou_mn.clamp(0).pow(self.beta) *
                     inside.float())                                     # (M, N)

            # ── top-k cells per GT ────────────────────────────────────────
            k = min(self.topk, N)
            _, topk_idx = align.topk(k, dim=1)                          # (M, k)
            cand = torch.zeros(M, N, dtype=torch.bool, device=device)
            cand.scatter_(1, topk_idx, True)
            cand &= inside                                               # (M, N)

            if not cand.any():
                continue

            # ── conflict resolution: each cell keeps the GT with best IoU ─
            best_gt = (iou_mn * cand.float()).argmax(dim=0)             # (N,)
            winner  = torch.zeros(M, N, dtype=torch.bool, device=device)
            winner.scatter_(0, best_gt.unsqueeze(0), True)
            cand   &= winner

            cell_pos = cand.any(dim=0)                                   # (N,)
            if not cell_pos.any():
                continue

            pos_idx  = cell_pos.nonzero(as_tuple=True)[0]               # (K,)
            asgn_gt  = best_gt[pos_idx]                                  # (K,)
            asgn_cls = gt_cls[asgn_gt]                                   # (K,)
            asgn_iou = iou_mn[asgn_gt, pos_idx].clamp(0, 1)            # (K,)

            # ── soft class targets (IoU-weighted) ────────────────────────
            cls_tgt[b][pos_idx, asgn_cls] = asgn_iou
            mask_pos[b][pos_idx] = True

            # ── regression targets ────────────────────────────────────────
            ab  = gt_boxes[asgn_gt]
            gcx = (ab[:, 0] + ab[:, 2]) / 2
            gcy = (ab[:, 1] + ab[:, 3]) / 2
            gw  = ab[:, 2] - ab[:, 0]
            gh  = ab[:, 3] - ab[:, 1]
            reg_tgt[b][pos_idx] = torch.stack([
                gcx / stride - gx_f[pos_idx],
                gcy / stride - gy_f[pos_idx],
                torch.log(gw / stride + 1e-6),
                torch.log(gh / stride + 1e-6),
            ], dim=1)

        return cls_tgt, reg_tgt, mask_pos

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, outputs, targets: list[Tensor]) -> Tensor:
        """
        Args:
            outputs : list of 3 tuples (cls_logits, reg_preds), one per scale.
            targets : list of B tensors, each (M_i, 5) [x1,y1,x2,y2,cls].
        """
        device    = outputs[0][0].device
        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)

        for (cls_logits, reg_preds), stride in zip(outputs, self.STRIDES):
            B, C, H, W = cls_logits.shape
            N = H * W

            cls_tgt, reg_tgt, mask_pos = self._tal_assign(
                cls_logits, reg_preds, targets, stride)

            # focal BCE on all cells (positives: soft IoU targets; negatives: 0)
            cls_flat   = cls_logits.permute(0, 2, 3, 1).reshape(B, N, C)
            total_cls += self._focal_bce(cls_flat, cls_tgt)

            # CIoU on positive cells only
            if mask_pos.any():
                reg_flat   = reg_preds.permute(0, 2, 3, 1).reshape(B, N, 4)
                total_box += complete_box_iou_loss(
                    self._to_xyxy(reg_flat[mask_pos]),
                    self._to_xyxy(reg_tgt[mask_pos]),
                    reduction='mean',
                )

        return self.cls_weight * total_cls + self.box_weight * total_box


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(args) -> tuple[DataLoader, DataLoader]:
    train_ds = VOCDetectionDataset(args.voc_dir, split="train",
                                   image_size=args.img_size, download=True,
                                   person_only=args.person_only)
    val_ds   = VOCDetectionDataset(args.voc_dir, split="val",
                                   image_size=args.img_size, download=True,
                                   person_only=args.person_only)
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
def eval_map(
    model:       YOLOv8,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int   = NUM_CLASSES,
    img_size:    int   = 640,
    conf_thresh: float = 0.25,
    iou_thresh:  float = 0.45,
) -> dict:
    model.eval()
    all_preds, all_tgts = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        raw     = model.decode(outputs, img_size=img_size, conf_thresh=conf_thresh)
        preds   = decode_predictions(raw.cpu(), conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        all_preds.extend(preds)
        all_tgts.extend([t.cpu() for t in targets])
    return compute_map(all_preds, all_tgts, num_classes=num_classes, iou_thresh=0.5)


def _log_map_table(results: dict, logger: TrainLogger, class_names: list[str]) -> None:
    logger._w(f"\n  {'Class':<20} {'AP':>8}")
    logger._w(f"  {'-'*30}")
    for c, name in enumerate(class_names):
        ap = results.get(f"AP_{c}", 0.0)
        logger._w(f"  {name:<20} {ap * 100:>7.2f}%")
    logger._w(f"  {'─'*30}")
    logger._w(f"  {'mAP@0.5':<20} {results['mAP'] * 100:>7.2f}%")


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
    num_classes  = NUM_CLASSES_PERSON if args.person_only else NUM_CLASSES
    class_names  = ["person"] if args.person_only else VOC_CLASSES

    train_loader, val_loader = get_loaders(args)
    model   = YOLOv8(
        num_classes=num_classes,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    loss_fn = YOLOv8Loss(num_classes=num_classes, img_size=args.img_size)

    if args.freeze_backbone:
        for p in model.backbone_parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(
            model.head_parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        opt = torch.optim.AdamW([
            {"params": model.backbone_parameters(),
             "lr": args.lr * args.backbone_lr_mult},
            {"params": model.head_parameters(),
             "lr": args.lr},
        ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    backbone_mode = ("frozen" if args.freeze_backbone else
                     f"lr×{args.backbone_lr_mult}") if args.pretrained_backbone else "scratch"
    mode_str = "person-only (1 class)" if args.person_only else f"{num_classes} classes"
    logger._w(f"\nYOLOv8 (ResNet50) | VOC 2012 | {mode_str} | img={args.img_size} | "
              f"bs={args.batch_size} | lr={args.lr} | epochs={args.epochs} | "
              f"backbone={backbone_mode}")
    logger._w(f"Train: {len(train_loader.dataset)}  "
              f"Val: {len(val_loader.dataset)}")

    # ── epoch summary table ───────────────────────────────────────────────
    HDR = (f"\n  {'Epoch':>9}  {'Train':>8}  {'Val':>8}  "
           f"{'mAP@0.5':>10}  {'LR':>9}  {'Time':>7}")
    SEP = "  " + "─" * 63
    logger._w(HDR)
    logger._w(SEP)

    last_map_results = None

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt,
                                     device, logger, epoch, args.epochs)
        val_loss   = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        map_val = None
        run_map = (args.eval_map_every > 0 and
                   (epoch % args.eval_map_every == 0 or epoch == args.epochs))
        if run_map:
            map_results      = eval_map(model, val_loader, device,
                                        num_classes=num_classes,
                                        img_size=args.img_size)
            map_val          = map_results["mAP"]
            last_map_results = map_results

        elapsed  = time.time() - t0
        map_col  = f"{map_val * 100:.2f}%" if map_val is not None else "–"
        ckpt_col = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            ckpt_col = "  ✓"

        epoch_str = f"{epoch}/{args.epochs}"
        logger._w(
            f"  {epoch_str:>9}  "
            f"{train_loss:>8.4f}  "
            f"{val_loss:>8.4f}  "
            f"{map_col:>10}  "
            f"{scheduler.get_last_lr()[-1]:>9.2e}  "
            f"{elapsed:>6.1f}s"
            f"{ckpt_col}"
        )

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

    logger._w(SEP)
    logger._w(f"\nBest val loss: {best_val_loss:.4f}  |  Checkpoint: {args.save_path}")

    logger._w("\n─── Final mAP@0.5 (val) ───")
    # reuse last computed mAP rather than re-running the full eval pass
    final_map = last_map_results or eval_map(model, val_loader, device,
                                             num_classes=num_classes,
                                             img_size=args.img_size)
    _log_map_table(final_map, logger, class_names)

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
    p.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use ImageNet-pretrained ResNet50 backbone (default: True)")
    p.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Freeze backbone weights — only train neck + heads")
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="Backbone LR = lr × this factor (default: 0.1)")
    p.add_argument("--eval_map_every", type=int, default=1,
                   help="Compute mAP@0.5 on val every N epochs (0 = disabled)")
    p.add_argument("--person_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Detect person class only — num_classes=1 (default: True)")
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