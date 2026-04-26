import torch.nn as nn
from torchvision import models as tv_models


class DenseNet169(nn.Module):
    """DenseNet-169 for image classification.

    Wraps torchvision's DenseNet-169 with optional ImageNet pre-training and
    stem adaptation for small inputs (e.g. 32×32 CIFAR-10).

    Args:
        num_classes (int): Number of output classes. Default: 10.
        pretrained (bool): Load ImageNet weights. Default: False.
        small_input (bool): Replace the 7×7 stem conv + max-pool with a
            3×3 conv and identity so that 32×32 inputs are not over-downsampled.
            Set to True for CIFAR-10, False for VOC/ImageNet (224×224). Default: False.

    Attributes:
        features (nn.Sequential): DenseNet feature extractor (all dense blocks + transitions).
        classifier (nn.Linear): Final fully-connected classification head.

    Shape:
        Input:  (N, 3, H, W)
        Output: (N, num_classes)

    Example:
        >>> model = DenseNet169(num_classes=10, pretrained=False, small_input=True)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> logits = model(x)

    References:
        Huang, G. et al. (2017). Densely Connected Convolutional Networks.
        CVPR 2017. https://arxiv.org/abs/1608.06993
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained:  bool = False,
        small_input: bool = False,
    ) -> None:
        super().__init__()
        weights = tv_models.DenseNet169_Weights.DEFAULT if pretrained else None
        base = tv_models.densenet169(weights=weights)

        if small_input:
            base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.features.pool0 = nn.Identity()  # type: ignore[assignment]

        base.classifier = nn.Linear(base.classifier.in_features, num_classes)

        self.features   = base.features
        self.classifier = base.classifier

    def forward(self, x):
        import torch.nn.functional as F
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.classifier(out)
