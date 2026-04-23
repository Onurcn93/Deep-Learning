"""YOLOv8n — nano-scale object detection model.

Architecture (Ultralytics YOLOv8, 2023):
  Backbone : CSPDarknet with C2f blocks and SPPF
  Neck     : PANet (top-down + bottom-up feature pyramid)
  Head     : Anchor-free decoupled detection heads on P3 / P4 / P5

Scaling factors for nano variant:
  depth_multiple  = 0.33   (number of bottleneck repeats)
  width_multiple  = 0.25   (channel widths)

References:
  Jocher et al., "Ultralytics YOLOv8", 2023.
  https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU activation."""

    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        k:      int = 1,
        s:      int = 1,
        p:      int | None = None,
    ) -> None:
        super().__init__()
        pad = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck: Conv1×1 → Conv3×3 with optional shortcut."""

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        shortcut: bool = True,
        e:        float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1      = Conv(in_ch,  hidden, 3)
        self.cv2      = Conv(hidden, out_ch, 3)
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions (YOLOv8 replacement for C3).

    Splits input channels, runs n Bottleneck blocks on one half, then
    concatenates all intermediate outputs before a final projection.

    Args:
        in_ch    : Input channels.
        out_ch   : Output channels.
        n        : Number of Bottleneck repeats.
        shortcut : Enable residual connections inside Bottleneck blocks.
        e        : Channel expansion ratio inside each Bottleneck.
    """

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        n:        int   = 1,
        shortcut: bool  = True,
        e:        float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden = int(out_ch * e)
        self.cv1    = Conv(in_ch,       2 * self.hidden, 1)
        self.cv2    = Conv((2 + n) * self.hidden, out_ch, 1)
        self.m      = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling — Fast.

    Applies three sequential MaxPool2d(5×5) operations (equivalent to
    5×5, 9×9, 13×13 receptive fields) and concatenates the outputs,
    capturing multi-scale context with a single pooling kernel size.

    Args:
        in_ch  : Input channels.
        out_ch : Output channels.
        k      : Kernel size of each MaxPool2d layer (default 5).
    """

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
    """Anchor-free decoupled detection head for one scale.

    Two separate branches — classification and regression — each consisting
    of two Conv layers followed by a final 1×1 convolution that outputs:
      - cls branch : (B, num_classes, H, W) — class logits
      - reg branch : (B, 4, H, W)           — (cx, cy, w, h) relative to cell

    Args:
        in_ch       : Input channels from neck feature map.
        num_classes : Number of object classes.
        reg_ch      : Hidden channels in regression branch.
    """

    def __init__(self, in_ch: int, num_classes: int, reg_ch: int = 16) -> None:
        super().__init__()
        # Classification branch
        self.cls = nn.Sequential(
            Conv(in_ch, in_ch,       3),
            Conv(in_ch, in_ch,       3),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        # Regression branch
        self.reg = nn.Sequential(
            Conv(in_ch,  reg_ch * 4, 3),
            Conv(reg_ch * 4, reg_ch * 4, 3),
            nn.Conv2d(reg_ch * 4, 4, 1),
        )

    def forward(self, x: Tensor):
        return self.cls(x), self.reg(x)


# ---------------------------------------------------------------------------
# YOLOv8 (nano scale)
# ---------------------------------------------------------------------------

class YOLOv8(nn.Module):
    """YOLOv8n — nano variant of YOLOv8 for PASCAL VOC / COCO detection.

    Backbone produces three feature maps (P3, P4, P5).
    PANet neck fuses them top-down then bottom-up.
    Three DetHead modules predict at each scale.

    Args:
        num_classes  : Number of detection classes (20 for VOC, 80 for COCO).
        width_mult   : Channel width multiplier (0.25 = nano).
        depth_mult   : Block depth multiplier   (0.33 = nano).

    Output (training):
        List of three (cls_logits, reg_preds) tuples, one per scale.
        cls_logits : (B, num_classes, H_i, W_i)
        reg_preds  : (B, 4, H_i, W_i)  — cx,cy,w,h in grid-relative coords

    Output (inference, via decode()):
        (B, N, 5 + num_classes) — decoded absolute xyxy boxes + conf + cls.
    """

    def __init__(
        self,
        num_classes: int   = 20,
        width_mult:  float = 0.25,
        depth_mult:  float = 0.33,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        def ch(c: int) -> int:
            return max(round(c * width_mult), 1)

        def dep(n: int) -> int:
            return max(round(n * depth_mult), 1)

        # ── Backbone ──────────────────────────────────────────────────────
        self.stem  = Conv(3,      ch(64),  3, 2)          # /2
        self.b1    = Conv(ch(64), ch(128), 3, 2)          # /4
        self.c1    = C2f(ch(128), ch(128), dep(3))
        self.b2    = Conv(ch(128), ch(256), 3, 2)         # /8  → P3
        self.c2    = C2f(ch(256), ch(256), dep(6))
        self.b3    = Conv(ch(256), ch(512), 3, 2)         # /16 → P4
        self.c3    = C2f(ch(512), ch(512), dep(6))
        self.b4    = Conv(ch(512), ch(512), 3, 2)         # /32 → P5
        self.c4    = C2f(ch(512), ch(512), dep(3))
        self.sppf  = SPPF(ch(512), ch(512))

        # ── Neck: top-down path ───────────────────────────────────────────
        self.up1   = nn.Upsample(scale_factor=2, mode="nearest")
        self.nc1   = C2f(ch(512) + ch(512), ch(512), dep(3), shortcut=False)

        self.up2   = nn.Upsample(scale_factor=2, mode="nearest")
        self.nc2   = C2f(ch(512) + ch(256), ch(256), dep(3), shortcut=False)  # P3 out

        # ── Neck: bottom-up path ──────────────────────────────────────────
        self.dn1   = Conv(ch(256), ch(256), 3, 2)
        self.nc3   = C2f(ch(256) + ch(512), ch(512), dep(3), shortcut=False)  # P4 out

        self.dn2   = Conv(ch(512), ch(512), 3, 2)
        self.nc4   = C2f(ch(512) + ch(512), ch(512), dep(3), shortcut=False)  # P5 out

        # ── Detection heads ───────────────────────────────────────────────
        self.head_p3 = DetHead(ch(256), num_classes)
        self.head_p4 = DetHead(ch(512), num_classes)
        self.head_p5 = DetHead(ch(512), num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        # ── Backbone ──────────────────────────────────────────────────────
        x  = self.stem(x)
        x  = self.b1(x)
        x  = self.c1(x)
        p3 = self.c2(self.b2(x))      # stride /8
        p4 = self.c3(self.b3(p3))     # stride /16
        p5 = self.sppf(self.c4(self.b4(p4)))  # stride /32

        # ── Neck top-down ─────────────────────────────────────────────────
        n1 = self.nc1(torch.cat([self.up1(p5), p4], dim=1))
        n2 = self.nc2(torch.cat([self.up2(n1), p3], dim=1))  # P3 feature

        # ── Neck bottom-up ────────────────────────────────────────────────
        n3 = self.nc3(torch.cat([self.dn1(n2), n1], dim=1))  # P4 feature
        n4 = self.nc4(torch.cat([self.dn2(n3), p5], dim=1))  # P5 feature

        # ── Heads ─────────────────────────────────────────────────────────
        out_p3 = self.head_p3(n2)
        out_p4 = self.head_p4(n3)
        out_p5 = self.head_p5(n4)

        return out_p3, out_p4, out_p5

    @torch.no_grad()
    def decode(
        self,
        outputs,
        img_size:   int   = 640,
        conf_thresh: float = 0.25,
    ) -> Tensor:
        """Decode head outputs to (B, N, 5 + num_classes) for inference.

        Each row: [x1, y1, x2, y2, obj_conf, cls_prob_0, ..., cls_prob_C-1]
        Boxes are in absolute pixel coordinates of the input image.
        """
        all_preds = []
        for cls_logits, reg_preds in outputs:
            B, _, H, W = cls_logits.shape

            # Build grid of cell centres
            gy, gx = torch.meshgrid(
                torch.arange(H, device=cls_logits.device),
                torch.arange(W, device=cls_logits.device),
                indexing="ij",
            )
            stride = img_size // H

            # reg_preds: (B,4,H,W) — cx,cy offset + wh in grid units
            cx = (reg_preds[:, 0] + gx.unsqueeze(0)) * stride
            cy = (reg_preds[:, 1] + gy.unsqueeze(0)) * stride
            bw = reg_preds[:, 2].exp() * stride
            bh = reg_preds[:, 3].exp() * stride

            x1 = (cx - bw / 2).unsqueeze(-1)
            y1 = (cy - bh / 2).unsqueeze(-1)
            x2 = (cx + bw / 2).unsqueeze(-1)
            y2 = (cy + bh / 2).unsqueeze(-1)

            cls_conf = cls_logits.sigmoid()           # (B, C, H, W)
            obj_conf = cls_conf.max(dim=1, keepdim=True).values  # proxy objectness

            # Flatten spatial dims → (B, H*W, ...)
            boxes  = torch.cat([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
            obj    = obj_conf.permute(0, 2, 3, 1).reshape(B, -1, 1)
            clspr  = cls_conf.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

            all_preds.append(torch.cat([boxes, obj, clspr], dim=-1))

        return torch.cat(all_preds, dim=1)  # (B, total_anchors, 5+C)