import torch.nn as nn


class InvertedResidual(nn.Module):
    """Inverted residual block (bottleneck) for MobileNetV2.

    Expands channels with a pointwise conv (if expand_ratio > 1), applies a
    depthwise 3×3 conv, then projects back with a linear pointwise conv (no
    activation on the projection). A residual connection is added only when
    the input and output shapes match (stride == 1 and in_channels == out_channels).

    Args:
        in_channels  (int):              Number of input channels.
        out_channels (int):              Number of output channels.
        stride       (int):              Stride for the depthwise conv (1 or 2).
        expand_ratio (int):              Channel expansion factor before depthwise conv.
        norm         (nn.Module):        Normalisation layer class. Default: nn.BatchNorm2d.

    Shape:
        Input:  (N, in_channels, H, W)
        Output: (N, out_channels, H/stride, W/stride)

    References:
        Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        CVPR 2018. https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int,
        expand_ratio: int,
        norm = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                norm(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            # Depthwise conv
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            norm(hidden),
            nn.ReLU6(inplace=True),
            # Pointwise projection — no activation (linear bottleneck)
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            norm(out_channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 adapted for 32×32 CIFAR-10 input.

    Follows the inverted residual architecture of Sandler et al. (2018) with
    one modification: the initial convolution uses stride=1 instead of stride=2
    so that 32×32 images are not downsampled too aggressively at the stem.
    AdaptiveAvgPool2d at the end handles any resulting spatial size.

    Args:
        norm        (nn.Module): Normalisation layer class. Default: nn.BatchNorm2d.
        num_classes (int):       Number of output classes. Default: 10.

    Attributes:
        first_conv  (nn.Sequential): Stem: Conv 3→32, BN, ReLU6 (stride=1).
        features    (nn.Sequential): Stack of InvertedResidual blocks.
        last_conv   (nn.Sequential): Pointwise conv 320→1280, BN, ReLU6.
        avgpool     (nn.AdaptiveAvgPool2d): Global average pooling to 1×1.
        classifier  (nn.Linear): Final linear head.

    Shape:
        Input:  (N, 3, 32, 32)
        Output: (N, num_classes)

    Example:
        >>> model = MobileNetV2(num_classes=10)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> logits = model(x)   # shape: (4, 10)

    References:
        [1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
            MobileNetV2: Inverted residuals and linear bottlenecks. CVPR 2018.
            https://arxiv.org/abs/1801.04381
    """

    # (expand_ratio, out_channels, num_blocks, stride)
    _CFG = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, norm=nn.BatchNorm2d, num_classes: int = 10) -> None:
        super().__init__()

        # Stem — stride 1 for 32×32 input (original MobileNetV2 uses stride 2)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm(32),
            nn.ReLU6(inplace=True),
        )

        layers = []
        in_ch = 32
        for t, c, n, s in self._CFG:
            for i in range(n):
                layers.append(InvertedResidual(in_ch, c, stride=s if i == 0 else 1,
                                               expand_ratio=t, norm=norm))
                in_ch = c
        self.features = nn.Sequential(*layers)

        # Head
        self.last_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            norm(1280),
            nn.ReLU6(inplace=True),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
