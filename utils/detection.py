"""Detection utilities: IoU, NMS, and mAP — wrapping torchvision.ops."""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import box_iou, batched_nms


# ---------------------------------------------------------------------------
# NMS helper
# ---------------------------------------------------------------------------

def apply_nms(
    boxes:      Tensor,
    scores:     Tensor,
    class_ids:  Tensor,
    iou_thresh: float = 0.45,
) -> Tensor:
    """Batched NMS per class.

    Args:
        boxes      : (N, 4) float, xyxy absolute coords.
        scores     : (N,) float confidence scores.
        class_ids  : (N,) int class indices.
        iou_thresh : IoU threshold for suppression.

    Returns:
        kept : (K,) long indices into the input tensors.
    """
    return batched_nms(boxes, scores, class_ids, iou_thresh)


# ---------------------------------------------------------------------------
# Prediction decoding
# ---------------------------------------------------------------------------

def decode_predictions(
    raw:         Tensor,
    conf_thresh: float = 0.25,
    iou_thresh:  float = 0.45,
) -> list[dict]:
    """Decode a batch of raw YOLO head outputs into per-image predictions.

    Args:
        raw         : (B, N, 5 + num_classes) — tx,ty,tw,th,conf,cls_probs.
                      Boxes must already be in absolute xyxy pixel coords.
        conf_thresh : Minimum objectness × class score to keep.
        iou_thresh  : NMS IoU threshold.

    Returns:
        List of B dicts, each with keys:
            'boxes'   : (K, 4) float xyxy
            'scores'  : (K,) float
            'labels'  : (K,) long
    """
    results = []
    for pred in raw:                          # pred: (N, 5+C)
        obj_conf  = pred[:, 4]
        cls_probs = pred[:, 5:]
        cls_scores, cls_ids = cls_probs.max(dim=1)
        scores = obj_conf * cls_scores

        mask = scores > conf_thresh
        if mask.sum() == 0:
            results.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
            continue

        boxes_f  = pred[mask, :4]
        scores_f = scores[mask]
        labels_f = cls_ids[mask]

        kept = apply_nms(boxes_f, scores_f, labels_f, iou_thresh)
        results.append({
            "boxes":  boxes_f[kept],
            "scores": scores_f[kept],
            "labels": labels_f[kept],
        })
    return results


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def compute_map(
    predictions: list[dict],
    targets:     list[Tensor],
    num_classes: int,
    iou_thresh:  float = 0.5,
) -> dict[str, float]:
    """Compute mAP@iou_thresh over a dataset.

    Args:
        predictions : List of N dicts from decode_predictions.
        targets     : List of N tensors (M_i, 5) — each row [x1,y1,x2,y2,cls].
        num_classes : Number of classes.
        iou_thresh  : IoU threshold for a true positive.

    Returns:
        dict with 'mAP' (float) and per-class APs under key 'AP_<cls_idx>'.
    """
    # Collect all detections and ground truths per class
    all_tp:     dict[int, list] = {c: [] for c in range(num_classes)}
    all_scores: dict[int, list] = {c: [] for c in range(num_classes)}
    n_gt:       dict[int, int]  = {c: 0  for c in range(num_classes)}

    for pred, tgt in zip(predictions, targets):
        gt_boxes  = tgt[:, :4]   # (M, 4)
        gt_labels = tgt[:, 4].long()

        for c in range(num_classes):
            gt_mask  = gt_labels == c
            gt_c     = gt_boxes[gt_mask]
            n_gt[c] += gt_c.shape[0]

            det_mask = pred["labels"] == c
            if det_mask.sum() == 0:
                continue

            det_boxes  = pred["boxes"][det_mask]
            det_scores = pred["scores"][det_mask]

            if gt_c.shape[0] == 0:
                all_tp[c].extend([0] * det_boxes.shape[0])
                all_scores[c].extend(det_scores.tolist())
                continue

            iou = box_iou(det_boxes, gt_c)          # (D, G)
            matched_gt = set()
            for d_idx in range(det_boxes.shape[0]):
                best_iou, best_g = iou[d_idx].max(0)
                if best_iou.item() >= iou_thresh and best_g.item() not in matched_gt:
                    all_tp[c].append(1)
                    matched_gt.add(best_g.item())
                else:
                    all_tp[c].append(0)
                all_scores[c].append(det_scores[d_idx].item())

    # Compute AP per class via 11-point interpolation
    aps = {}
    for c in range(num_classes):
        if n_gt[c] == 0:
            continue
        scores_c = torch.tensor(all_scores[c])
        tp_c     = torch.tensor(all_tp[c], dtype=torch.float32)

        if scores_c.numel() == 0:
            aps[c] = 0.0
            continue

        order    = scores_c.argsort(descending=True)
        tp_c     = tp_c[order]
        cum_tp   = tp_c.cumsum(0)
        cum_fp   = (1 - tp_c).cumsum(0)

        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / (n_gt[c] + 1e-9)

        # 11-point interpolation
        ap = 0.0
        for thresh in torch.linspace(0, 1, 11):
            mask = recall >= thresh
            ap  += precision[mask].max().item() if mask.any() else 0.0
        aps[c] = ap / 11.0

    mean_ap = sum(aps.values()) / max(len(aps), 1)
    result  = {"mAP": mean_ap}
    result.update({f"AP_{c}": v for c, v in aps.items()})
    return result