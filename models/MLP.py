from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for image classification.

    Builds a configurable stack of fully-connected blocks using
    ``nn.ModuleList`` and ``nn.Sequential``. Each hidden block contains:
    ``Linear → BatchNorm1d → Activation → Dropout``.

    Args:
        input_size:   Number of input features (e.g. 784 for MNIST).
        hidden_sizes: List of hidden layer widths.
        num_classes:  Number of output classes.
        dropout:      Dropout probability applied after each activation.
        activation:   Activation function to use (``'relu'`` or ``'gelu'``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act_cls = nn.ReLU if activation == "relu" else nn.GELU

        self.flatten = nn.Flatten()

        blocks: List[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            blocks.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                act_cls(),
                nn.Dropout(dropout),
            ))
            in_dim = h

        self.hidden_layers = nn.ModuleList(blocks)
        self.output_layer  = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: flatten → hidden blocks → output layer.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, input_size)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
