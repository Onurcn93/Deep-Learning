"""PASCAL VOC 2012 dataset wrappers.

Three modes:
  - VOCClassification       : single-label classification (20 classes).
                              Label = class of the largest bounding-box object.
  - VOCBinaryClassification : binary classification — person present (1) or not (0).
  - VOCDetectionDataset     : object detection.
                              Returns image + list of (x1,y1,x2,y2,class_idx) boxes.
                              Pass person_only=True to restrict to person class only (num_classes=1).

All are built on torchvision.datasets.VOCDetection so the raw XML
annotations are parsed once and shared.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection

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
        root       : Root directory where VOCdevkit will be downloaded/found.
        split      : 'train' or 'val'.
        image_size : Resize target (square). Boxes are rescaled accordingly.
        download   : Download dataset if not found.
    """

    def __init__(
        self,
        root:        str,
        split:       str  = "train",
        image_size:  int  = 640,
        download:    bool = True,
        person_only: bool = False,
    ) -> None:
        self._base = VOCDetection(
            root=root, year="2012",
            image_set=split,
            download=download,
        )
        self.image_size  = image_size
        self.person_only = person_only
        self.img_transform = detection_transforms(image_size)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, annotation = self._base[idx]

        orig_w, orig_h = img.size          # PIL image size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

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

        target = torch.tensor(boxes, dtype=torch.float32) if boxes \
                 else torch.zeros((0, 5), dtype=torch.float32)

        return self.img_transform(img), target


# ---------------------------------------------------------------------------
# Detection collate — variable-length box lists need padding
# ---------------------------------------------------------------------------

def detection_collate(batch):
    """Stack images; keep targets as a list (variable number of boxes)."""
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets