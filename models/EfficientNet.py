import torch.nn as nn
from torchvision import models as tv_models


class EfficientNetB3(nn.Module):
    """EfficientNet-B3 for image classification.

    Wraps torchvision's EfficientNet-B3 with optional ImageNet pre-training and
    stem adaptation for small inputs (e.g. 32×32 CIFAR-10).

    Args:
        num_classes (int): Number of output classes. Default: 10.
        pretrained (bool): Load ImageNet weights. Default: False.
        small_input (bool): Replace the stride-2 stem conv with stride-1 so that
            32×32 inputs are not over-downsampled. Set to True for CIFAR-10,
            False for VOC/ImageNet (224×224). Default: False.

    Attributes:
        features (nn.Sequential): EfficientNet feature extractor (blocks 0-8).
        avgpool (nn.AdaptiveAvgPool2d): Global average pooling.
        classifier (nn.Sequential): Dropout + fully-connected head.

    Shape:
        Input:  (N, 3, H, W)
        Output: (N, num_classes)

    References:
        Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for
        Convolutional Neural Networks. ICML 2019. https://arxiv.org/abs/1905.11946
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained:  bool = False,
        small_input: bool = False,
    ) -> None:
        super().__init__()
        weights = tv_models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base = tv_models.efficientnet_b3(weights=weights)

        if small_input:
            # Stem is features[0]: Conv2dNormActivation → first element is Conv2d
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                old.in_channels, old.out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            )

        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, num_classes)

        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = base.classifier

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.flatten(1)
        return self.classifier(out)
