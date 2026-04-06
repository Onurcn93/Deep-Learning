"""Grad-CAM visualisation (Selvaraju et al., 2017)."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM using forward/backward hooks on a target layer.

    Args:
        model:        Trained model.
        target_layer: The ``nn.Module`` layer to hook — typically the last
                      convolutional layer before global average pooling.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd = target_layer.register_forward_hook(self._save_activations)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx:  Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Compute a Grad-CAM heatmap for a single image.

        Args:
            img_tensor: Normalised image of shape ``(1, C, H, W)``.
            class_idx:  Class to visualise. Defaults to the argmax prediction.

        Returns:
            Tuple of ``(heatmap, predicted_class)`` where ``heatmap`` is a
            float32 numpy array normalised to ``[0, 1]``.
        """
        self.model.eval()
        output = self.model(img_tensor)
        pred   = int(output.argmax(dim=1).item())
        target = class_idx if class_idx is not None else pred

        self.model.zero_grad()
        output[0, target].backward()

        # Global-average-pool gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1).squeeze()    # (H', W')
        cam     = torch.relu(cam).cpu().numpy().astype(np.float32)

        vmin, vmax = cam.min(), cam.max()
        cam = (cam - vmin) / (vmax - vmin + 1e-8)

        return cam, pred

    def remove_hooks(self) -> None:
        """Remove registered hooks to avoid memory leaks."""
        self._fwd.remove()
        self._bwd.remove()


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the recommended Grad-CAM target layer for common architectures.

    Args:
        model: An instantiated ``nn.Module``.

    Returns:
        The last feature-extraction layer suitable for Grad-CAM.

    Raises:
        ValueError: If no known target layer can be identified.
    """
    if hasattr(model, "layer4"):        # ResNet
        return model.layer4[-1]
    if hasattr(model, "features"):      # VGG
        return model.features[-1]
    if hasattr(model, "conv_layers"):   # SimpleCNN
        return model.conv_layers[-1]
    raise ValueError(
        f"Cannot automatically determine Grad-CAM target layer for "
        f"{type(model).__name__}. Pass target_layer explicitly."
    )