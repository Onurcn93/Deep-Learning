"""PASCAL VOC 2012 dataset wrappers.

Three modes:
  - VOCClassification       : single-label classification (20 classes).
                              Label = class of the largest bounding-box object.
  - VOCBinaryClassification : binary classification — person present (1) or not (0).
  - VOCDetectionDataset     : object detection.
                              Returns image + list of (x1,y1,x2,y2,class_idx) boxes.
                              Pass person_only=True to restrict to person class only (num_classes=1).
                              Pass augment=True (train split) for detection-aware augmentation.

All are built on torchvision.datasets.VOCDetection so the raw XML
annotations are parsed once and shared.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]

# 20 PASCAL VOC classes (alphabetical, matching official ordering)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX  = {c: i for i, c in enumerate(VOC_CLASSES)}
NUM_CLASSES   = len(VOC_CLASSES)   # 20
NUM_CLASSES_PERSON = 1             # person-only detection mode


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

def classification_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def detection_transforms(image_size: int = 640) -> transforms.Compose:
    """Minimal image-only transform for detection (boxes rescaled separately)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Detection augmentation helpers (box-safe, train split only)
# ---------------------------------------------------------------------------

_BOX_MIN_SIDE       = 8     # minimum box side length (pixels) after any crop
_BOX_MIN_AREA_RATIO = 0.01  # minimum box area / image area after any crop

# Stateless transforms applied per-call
_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
_JITTER = transforms.ColorJitter(
    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
)


def _filter_boxes(boxes: list[list], img_w: int, img_h: int) -> list[list]:
    """Clip boxes to image boundary and drop those that become too small."""
    kept = []
    img_area = img_w * img_h
    for b in boxes:
        x1 = max(0.0, b[0]);  y1 = max(0.0, b[1])
        x2 = min(float(img_w), b[2]);  y2 = min(float(img_h), b[3])
        if x2 - x1 < _BOX_MIN_SIDE or y2 - y1 < _BOX_MIN_SIDE:
            continue
        if (x2 - x1) * (y2 - y1) < _BOX_MIN_AREA_RATIO * img_area:
            continue
        kept.append([x1, y1, x2, y2, b[4]])
    return kept


def _hflip(img: Image.Image, boxes: list[list]) -> tuple:
    """Horizontal flip — mirrors box x-coords around image centre."""
    w = img.width
    return img.transpose(Image.FLIP_LEFT_RIGHT), [
        [w - b[2], b[1], w - b[0], b[3], b[4]] for b in boxes
    ]


def _scale_crop(img: Image.Image, boxes: list[list], target: int) -> tuple:
    """Random scale (0.5–1.5×) then crop or pad-to-target.

    scale > 1  →  zoom in  (resize larger, random crop)
    scale < 1  →  zoom out (resize smaller, paste on black canvas)
    """
    scale = random.uniform(0.5, 1.5)
    new_w = int(target * scale)
    new_h = int(target * scale)
    img   = img.resize((new_w, new_h), _BILINEAR)

    # Scale boxes proportionally
    sx, sy = new_w / target, new_h / target
    boxes  = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy, b[4]] for b in boxes]

    if new_w >= target:                         # zoom-in: crop
        x0 = random.randint(0, new_w - target)
        y0 = random.randint(0, new_h - target)
        img   = img.crop((x0, y0, x0 + target, y0 + target))
        boxes = [[b[0]-x0, b[1]-y0, b[2]-x0, b[3]-y0, b[4]] for b in boxes]
    else:                                        # zoom-out: pad
        canvas = Image.new("RGB", (target, target), (0, 0, 0))
        x0 = random.randint(0, target - new_w)
        y0 = random.randint(0, target - new_h)
        canvas.paste(img, (x0, y0))
        img   = canvas
        boxes = [[b[0]+x0, b[1]+y0, b[2]+x0, b[3]+y0, b[4]] for b in boxes]

    return img, _filter_boxes(boxes, target, target)


# ---------------------------------------------------------------------------
# Classification dataset
# ---------------------------------------------------------------------------

class VOCClassification(Dataset):
    """PASCAL VOC 2012 — single-label image classification.

    Each image is labelled with the class of its largest bounding-box object
    (by pixel area).  Images with no valid annotation are skipped.

    Args:
        root       : Root directory where VOCdevkit will be downloaded/found.
        split      : 'train' or 'val'.
        image_size : Resize target (square).
        download   : Download dataset if not found.
    """

    def __init__(
        self,
        root:       str,
        split:      str  = "train",
        image_size: int  = 224,
        download:   bool = True,
    ) -> None:
        self._base = VOCDetection(
            root=root, year="2012",
            image_set=split,
            download=download,
        )
        self.transform = classification_transforms(train=(split == "train"),
                                                   image_size=image_size)
        # Pre-compute valid indices and labels
        self._samples: list[tuple[int, int]] = []
        for i in range(len(self._base)):
            label = self._dominant_class(self._base[i][1])
            if label >= 0:
                self._samples.append((i, label))

    @staticmethod
    def _dominant_class(annotation: dict) -> int:
        """Return class index of the largest object; -1 if none found."""
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        best_cls, best_area = -1, -1
        for obj in objects:
            name = obj.get("name", "")
            if name not in CLASS_TO_IDX:
                continue
            bb   = obj["bndbox"]
            area = (int(bb["xmax"]) - int(bb["xmin"])) * \
                   (int(bb["ymax"]) - int(bb["ymin"]))
            if area > best_area:
                best_area = area
                best_cls  = CLASS_TO_IDX[name]
        return best_cls

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        base_idx, label = self._samples[idx]
        img, _ = self._base[base_idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Binary classification dataset (person / no-person)
# ---------------------------------------------------------------------------

class VOCBinaryClassification(Dataset):
    """PASCAL VOC 2012 — binary classification: person present (1) or not (0).

    Args:
        root       : Root directory where VOCdevkit will be downloaded/found.
        split      : 'train' or 'val'.
        image_size : Resize target (square).
        download   : Download dataset if not found.
    """

    def __init__(
        self,
        root:       str,
        split:      str  = "train",
        image_size: int  = 224,
        download:   bool = True,
    ) -> None:
        self._base = VOCDetection(
            root=root, year="2012",
            image_set=split,
            download=download,
        )
        self.transform = classification_transforms(train=(split == "train"),
                                                   image_size=image_size)
        self._samples: list[tuple[int, int]] = [
            (i, self._has_person(self._base[i][1]))
            for i in range(len(self._base))
        ]

    @staticmethod
    def _has_person(annotation: dict) -> int:
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        return int(any(o.get("name") == "person" for o in objects))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        base_idx, label = self._samples[idx]
        img, _ = self._base[base_idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Detection dataset
# ---------------------------------------------------------------------------

class VOCDetectionDataset(Dataset):
    """PASCAL VOC 2012 — object detection.

    Returns:
        image  : FloatTensor (3, H, W) normalised.
        target : FloatTensor (N, 5) — each row is [x1, y1, x2, y2, class_idx]
                 in absolute pixel coordinates of the *resized* image.
                 Empty tensor (0, 5) if image has no valid objects.

    Args:
        root        : Root directory where VOCdevkit will be downloaded/found.
        split       : 'train' or 'val'.
        image_size  : Resize target (square). Boxes are rescaled accordingly.
        download    : Download dataset if not found.
        person_only : Only keep person class (num_classes=1).
        augment     : Apply detection augmentation (train split only).
    """

    def __init__(
        self,
        root:        str,
        split:       str  = "train",
        image_size:  int  = 640,
        download:    bool = True,
        person_only: bool = False,
        augment:     bool = False,
    ) -> None:
        self._base = VOCDetection(
            root=root, year="2012",
            image_set=split,
            download=download,
        )
        self.image_size  = image_size
        self.person_only = person_only
        self.augment     = augment
        self.img_transform = detection_transforms(image_size)   # val path

    # ── helpers ─────────────────────────────────────────────────────────────

    def _parse_boxes(
        self, annotation: dict, scale_x: float, scale_y: float
    ) -> list[list]:
        """Parse raw VOC annotation → list of [x1,y1,x2,y2,cls] scaled by sx/sy."""
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        boxes = []
        for obj in objects:
            name = obj.get("name", "")
            if self.person_only:
                if name != "person":
                    continue
                cls = 0.0
            else:
                if name not in CLASS_TO_IDX:
                    continue
                cls = float(CLASS_TO_IDX[name])
            bb  = obj["bndbox"]
            x1  = float(bb["xmin"]) * scale_x
            y1  = float(bb["ymin"]) * scale_y
            x2  = float(bb["xmax"]) * scale_x
            y2  = float(bb["ymax"]) * scale_y
            boxes.append([x1, y1, x2, y2, cls])
        return boxes

    def _mosaic(self, idx: int):
        """Combine 4 images into a single mosaic of size (image_size × image_size).

        A random centre point divides the canvas into 4 quadrants. Each image
        is resized to fill its quadrant; boxes are offset to canvas coordinates.
        """
        S  = self.image_size
        cx = random.randint(S // 4, 3 * S // 4)
        cy = random.randint(S // 4, 3 * S // 4)

        indices    = [idx] + random.choices(range(len(self._base)), k=3)
        canvas     = Image.new("RGB", (S, S), (0, 0, 0))
        all_boxes: list[list] = []

        # Quadrant layout: TL, TR, BL, BR
        placements = [
            (0,  0,  cx, cy),
            (cx, 0,  S,  cy),
            (0,  cy, cx, S),
            (cx, cy, S,  S),
        ]

        for i, (px1, py1, px2, py2) in enumerate(placements):
            pw, ph        = px2 - px1, py2 - py1
            img_i, ann_i  = self._base[indices[i]]
            orig_w, orig_h = img_i.size
            boxes_i       = self._parse_boxes(ann_i, pw / orig_w, ph / orig_h)
            img_i         = img_i.resize((pw, ph), _BILINEAR)
            canvas.paste(img_i, (px1, py1))
            boxes_i = [
                [b[0]+px1, b[1]+py1, b[2]+px1, b[3]+py1, b[4]]
                for b in boxes_i
            ]
            all_boxes.extend(boxes_i)

        all_boxes = _filter_boxes(all_boxes, S, S)
        target = (torch.tensor(all_boxes, dtype=torch.float32) if all_boxes
                  else torch.zeros((0, 5), dtype=torch.float32))
        return _NORMALIZE(canvas), target

    # ── dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        # ── Mosaic path (train only, 50 % probability) ──────────────────────
        if self.augment and random.random() < 0.5:
            return self._mosaic(idx)

        img, annotation = self._base[idx]
        orig_w, orig_h  = img.size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        boxes   = self._parse_boxes(annotation, scale_x, scale_y)

        # ── Augmented train path ─────────────────────────────────────────────
        if self.augment:
            img = img.resize((self.image_size, self.image_size), _BILINEAR)
            if random.random() < 0.5:
                img, boxes = _hflip(img, boxes)
            img = _JITTER(img)
            if random.random() < 0.5:
                img, boxes = _scale_crop(img, boxes, self.image_size)
            boxes  = _filter_boxes(boxes, self.image_size, self.image_size)
            target = (torch.tensor(boxes, dtype=torch.float32) if boxes
                      else torch.zeros((0, 5), dtype=torch.float32))
            return _NORMALIZE(img), target

        # ── Val path (no augmentation) ───────────────────────────────────────
        target = (torch.tensor(boxes, dtype=torch.float32) if boxes
                  else torch.zeros((0, 5), dtype=torch.float32))
        return self.img_transform(img), target


# ---------------------------------------------------------------------------
# Detection collate — variable-length box lists need padding
# ---------------------------------------------------------------------------

def detection_collate(batch):
    """Stack images; keep targets as a list (variable number of boxes)."""
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
