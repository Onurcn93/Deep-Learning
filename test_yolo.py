"""YOLO evaluation: mAP@0.5 on PASCAL VOC 2012 val set.

Usage:
    python test_yolo.py --model_path weights/yolo_best.pth --img_size 640 --device auto

Key flags:
    --model_path   Path to trained YOLOv8 checkpoint (required)
    --img_size     Input image size used during training (default: 640)
    --conf_thresh  Confidence threshold for predictions (default: 0.25)
    --iou_thresh   NMS IoU threshold (default: 0.45)
    --map_iou      IoU threshold for mAP computation (default: 0.5)
    --voc_dir      Root directory for VOCdevkit (default: ./data/VOC)
    --device       auto | cuda | cpu
    --batch_size   Evaluation batch size (default: 8)
    --person_only  Evaluate person-only checkpoint (default: True)
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from datasets.voc import (VOCDetectionDataset, detection_collate,
                           VOC_CLASSES, NUM_CLASSES, NUM_CLASSES_PERSON)
from models.YOLO import YOLOv8
from utils.detection import decode_predictions, compute_map


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_yolo_test(
    model:       YOLOv8,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int   = NUM_CLASSES,
    img_size:    int   = 640,
    conf_thresh: float = 0.25,
    iou_thresh:  float = 0.45,
    map_iou:     float = 0.5,
) -> dict:
    model.eval()

    all_predictions: list[dict]         = []
    all_targets:     list[torch.Tensor] = []

    for imgs, targets in loader:
        imgs = imgs.to(device)

        outputs  = model(imgs)
        raw      = model.decode(outputs, img_size=img_size, conf_thresh=conf_thresh)
        preds    = decode_predictions(raw.cpu(), conf_thresh=conf_thresh,
                                      iou_thresh=iou_thresh)
        all_predictions.extend(preds)
        all_targets.extend([t.cpu() for t in targets])

    results = compute_map(all_predictions, all_targets,
                          num_classes=num_classes, iou_thresh=map_iou)
    return results


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(results: dict, class_names: list[str], map_iou: float) -> None:
    print(f"\n{'='*50}")
    print(f"  YOLOv8n — PASCAL VOC 2012 val  (IoU@{map_iou:.2f})")
    print(f"{'='*50}")
    print(f"  {'mAP':<30} {results['mAP']*100:.2f}%")
    print(f"{'-'*50}")
    print(f"  {'Class':<30} {'AP':>8}")
    print(f"{'-'*50}")
    for c, name in enumerate(class_names):
        key = f"AP_{c}"
        if key in results:
            print(f"  {name:<30} {results[key]*100:>7.2f}%")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8n on PASCAL VOC 2012 val")
    p.add_argument("--model_path",  type=str,   required=True,
                   help="Path to trained YOLOv8 checkpoint")
    p.add_argument("--img_size",    type=int,   default=640)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--iou_thresh",  type=float, default=0.45)
    p.add_argument("--map_iou",     type=float, default=0.5,
                   help="IoU threshold for mAP (default: 0.5 = mAP@50)")
    p.add_argument("--voc_dir",     type=str,   default="./data/VOC")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--person_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate person-only checkpoint (default: True)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    num_classes = NUM_CLASSES_PERSON if args.person_only else NUM_CLASSES
    class_names = ["person"] if args.person_only else VOC_CLASSES

    model = YOLOv8(num_classes=num_classes).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    print(f"Loaded checkpoint: {args.model_path}")

    val_ds = VOCDetectionDataset(args.voc_dir, split="val",
                                 image_size=args.img_size, download=True,
                                 person_only=args.person_only)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=detection_collate)
    print(f"Val samples: {len(val_ds)}")

    results = run_yolo_test(
        model, loader, device,
        num_classes=num_classes,
        img_size=args.img_size,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        map_iou=args.map_iou,
    )
    print_results(results, class_names, args.map_iou)
