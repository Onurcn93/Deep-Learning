"""YOLOv8-style object detection model.

Architecture:
  Backbone : ResNet50 pretrained on ImageNet (torchvision)
             → P3 (stride /8, 512ch), P4 (stride /16, 1024ch), P5 (stride /32, 2048ch)
  Adapters : 1×1 Conv projections to neck channel widths
  Neck     : PANet (top-down + bottom-up feature pyramid) with C2f blocks
  Head     : Anchor-free decoupled detection heads on P3 / P4 / P5

Neck/head scaling (nano defaults):
  depth_multiple = 0.33
  width_multiple = 0.25

References:
  Jocher et al., "Ultralytics YOLOv8", 2023.
  He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1,
                 p: int | None = None) -> None:
        super().__init__()
        pad = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Conv1×1 → Conv3×3 with optional residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True,
                 e: float = 0.5) -> None:
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1      = Conv(in_ch,  hidden, 3)
        self.cv2      = Conv(hidden, out_ch, 3)
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions (YOLOv8 C3 replacement)."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1,
                 shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        self.hidden = int(out_ch * e)
        self.cv1    = Conv(in_ch,             2 * self.hidden, 1)
        self.cv2    = Conv((2 + n) * self.hidden, out_ch,      1)
        self.m      = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling — Fast (three sequential 5×5 MaxPool)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 5) -> None:
        super().__init__()
        hidden   = in_ch // 2
        self.cv1 = Conv(in_ch,      hidden, 1)
        self.cv2 = Conv(hidden * 4, out_ch, 1)
        self.m   = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        x  = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------

class DetHead(nn.Module):
    """Anchor-free decoupled head: separate cls and reg branches.

    cls branch → (B, num_classes, H, W)  class logits
    reg branch → (B, 4, H, W)            cx_off, cy_off, log_w, log_h
    """

    def __init__(self, in_ch: int, num_classes: int, reg_ch: int = 16) -> None:
        super().__init__()
        self.cls = nn.Sequential(
            Conv(in_ch, in_ch, 3),
            Conv(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        self.reg = nn.Sequential(
            Conv(in_ch,       reg_ch * 4, 3),
            Conv(reg_ch * 4,  reg_ch * 4, 3),
            nn.Conv2d(reg_ch * 4, 4, 1),
        )

    def forward(self, x: Tensor):
        return self.cls(x), self.reg(x)


# ---------------------------------------------------------------------------
# YOLOv8 with ResNet50 backbone
# ---------------------------------------------------------------------------

class YOLOv8(nn.Module):
    """YOLOv8-style detector with ResNet50 ImageNet backbone.

    Backbone : ResNet50 (pretrained on ImageNet by default).
               Extracts P3 / P4 / P5 at strides 8 / 16 / 32.
    Adapters : 1×1 Conv to project ResNet channels to neck widths.
    Neck     : PANet with C2f blocks (scaled by width_mult / depth_mult).
    Heads    : Three anchor-free DetHead modules.

    Args:
        num_classes         : Detection classes (20 for VOC, 80 for COCO).
        width_mult          : Neck/head channel multiplier (0.25 = nano).
        depth_mult          : C2f repeat multiplier (0.33 = nano).
        pretrained_backbone : Load ImageNet weights for ResNet50 (default True).

    Output (training):
        Three (cls_logits, reg_preds) tuples — P3, P4, P5.

    Output (inference via decode()):
        (B, N, 5 + num_classes) — decoded xyxy boxes + obj_conf + cls_probs.
    """

    def __init__(
        self,
        num_classes:         int   = 20,
        width_mult:          float = 0.25,
        depth_mult:          float = 0.33,
        pretrained_backbone: bool  = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        def ch(c: int) -> int:
            return max(round(c * width_mult), 1)

        def dep(n: int) -> int:
            return max(round(n * depth_mult), 1)

        # ── Backbone: ResNet50 ────────────────────────────────────────────
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        _r50    = resnet50(weights=weights)

        # Reuse ResNet50 layers directly — no wrapper needed
        self.backbone_stem = nn.Sequential(
            _r50.conv1, _r50.bn1, _r50.relu, _r50.maxpool,
            _r50.layer1,                        # /4,  256ch
        )
        self.backbone_p3 = _r50.layer2          # /8,  512ch  → P3 raw
        self.backbone_p4 = _r50.layer3          # /16, 1024ch → P4 raw
        self.backbone_p5 = _r50.layer4          # /32, 2048ch → P5 raw

        # ── Adapters: project ResNet channels → neck widths ───────────────
        self.adapt_p3 = Conv(512,  ch(256), 1)
        self.adapt_p4 = Conv(1024, ch(512), 1)
        self.adapt_p5 = Conv(2048, ch(512), 1)
        self.sppf     = SPPF(ch(512), ch(512))

        # ── Neck: top-down ────────────────────────────────────────────────
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.nc1 = C2f(ch(512) + ch(512), ch(512), dep(3), shortcut=False)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.nc2 = C2f(ch(512) + ch(256), ch(256), dep(3), shortcut=False)  # P3 out

        # ── Neck: bottom-up ───────────────────────────────────────────────
        self.dn1 = Conv(ch(256), ch(256), 3, 2)
        self.nc3 = C2f(ch(256) + ch(512), ch(512), dep(3), shortcut=False)  # P4 out

        self.dn2 = Conv(ch(512), ch(512), 3, 2)
        self.nc4 = C2f(ch(512) + ch(512), ch(512), dep(3), shortcut=False)  # P5 out

        # ── Detection heads ───────────────────────────────────────────────
        self.head_p3 = DetHead(ch(256), num_classes)
        self.head_p4 = DetHead(ch(512), num_classes)
        self.head_p5 = DetHead(ch(512), num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming-init only the non-backbone modules (adapters, neck, heads)."""
        backbone_ids = {
            id(p) for m in (
                self.backbone_stem, self.backbone_p3,
                self.backbone_p4,   self.backbone_p5,
            ) for p in m.parameters()
        }
        for m in self.modules():
            if any(id(p) in backbone_ids for p in m.parameters(recurse=False)):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def backbone_parameters(self) -> list:
        """All parameters belonging to the ResNet50 backbone."""
        return (
            list(self.backbone_stem.parameters()) +
            list(self.backbone_p3.parameters()) +
            list(self.backbone_p4.parameters()) +
            list(self.backbone_p5.parameters())
        )

    def head_parameters(self) -> list:
        """All parameters NOT in the backbone (adapters + neck + heads)."""
        backbone_ids = {id(p) for p in self.backbone_parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def forward(self, x: Tensor):
        # ── Backbone ──────────────────────────────────────────────────────
        x   = self.backbone_stem(x)
        p3r = self.backbone_p3(x)          # /8,  512ch
        p4r = self.backbone_p4(p3r)        # /16, 1024ch
        p5r = self.backbone_p5(p4r)        # /32, 2048ch

        # ── Adapt + SPPF ──────────────────────────────────────────────────
        p3 = self.adapt_p3(p3r)
        p4 = self.adapt_p4(p4r)
        p5 = self.sppf(self.adapt_p5(p5r))

        # ── Neck top-down ─────────────────────────────────────────────────
        n1 = self.nc1(torch.cat([self.up1(p5), p4], dim=1))
        n2 = self.nc2(torch.cat([self.up2(n1), p3], dim=1))

        # ── Neck bottom-up ────────────────────────────────────────────────
        n3 = self.nc3(torch.cat([self.dn1(n2), n1], dim=1))
        n4 = self.nc4(torch.cat([self.dn2(n3), p5], dim=1))

        # ── Heads ─────────────────────────────────────────────────────────
        return self.head_p3(n2), self.head_p4(n3), self.head_p5(n4)

    @torch.no_grad()
    def decode(
        self,
        outputs,
        img_size:    int   = 640,
        conf_thresh: float = 0.25,
    ) -> Tensor:
        """Decode head outputs → (B, N, 5 + num_classes) absolute xyxy boxes."""
        all_preds = []
        for cls_logits, reg_preds in outputs:
            B, _, H, W = cls_logits.shape
            gy, gx = torch.meshgrid(
                torch.arange(H, device=cls_logits.device),
                torch.arange(W, device=cls_logits.device),
                indexing="ij",
            )
            stride = img_size // H

            cx = (reg_preds[:, 0] + gx.unsqueeze(0)) * stride
            cy = (reg_preds[:, 1] + gy.unsqueeze(0)) * stride
            bw = reg_preds[:, 2].clamp(-8, 8).exp() * stride
            bh = reg_preds[:, 3].clamp(-8, 8).exp() * stride

            x1 = (cx - bw / 2).unsqueeze(-1)
            y1 = (cy - bh / 2).unsqueeze(-1)
            x2 = (cx + bw / 2).unsqueeze(-1)
            y2 = (cy + bh / 2).unsqueeze(-1)

            cls_conf = cls_logits.sigmoid()

            boxes = torch.cat([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
            obj   = torch.ones(B, H * W, 1, device=cls_logits.device)
            clspr = cls_conf.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

            all_preds.append(torch.cat([boxes, obj, clspr], dim=-1))

        return torch.cat(all_preds, dim=1)   # (B, total_anchors, 5+C)
