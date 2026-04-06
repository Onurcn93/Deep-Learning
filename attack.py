"""PGD adversarial attack (Madry et al., 2018)."""

import torch
import torch.nn as nn


def pgd_attack(
    model:  nn.Module,
    imgs:   torch.Tensor,
    labels: torch.Tensor,
    eps:    float,
    alpha:  float,
    steps:  int,
    norm:   str   = "linf",
    mean:   tuple = (0.4914, 0.4822, 0.4465),
    std:    tuple = (0.2023, 0.1994, 0.2010),
) -> torch.Tensor:
    """Generate adversarial examples using Projected Gradient Descent.

    Operates in pixel [0, 1] space and returns re-normalised adversarial images
    matching the normalisation expected by the model.

    Args:
        model:  Trained model (set to eval mode internally).
        imgs:   Normalised clean images (B, C, H, W) on device.
        labels: Ground-truth class indices (B,) on device.
        eps:    Perturbation budget in pixel [0, 1] space.
        alpha:  Per-step size in pixel space.
        steps:  Number of PGD iterations.
        norm:   Threat model — ``'linf'`` or ``'l2'``.
        mean:   Per-channel normalisation mean matching model preprocessing.
        std:    Per-channel normalisation std  matching model preprocessing.

    Returns:
        Adversarial images in the same normalised space as ``imgs``.
    """
    model.eval()
    device = imgs.device
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1).to(device)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1).to(device)

    # Denormalise to [0, 1] pixel space
    x = (imgs.detach() * std_t + mean_t)

    # Random initialisation within the epsilon ball
    if norm == "linf":
        delta = torch.empty_like(x).uniform_(-eps, eps)
    else:  # l2 — small random perturbation in a sphere direction
        delta  = torch.randn_like(x)
        d_norm = delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
        delta  = delta / (d_norm + 1e-8) * (eps * 0.01)

    x_adv     = (x + delta).clamp(0.0, 1.0)
    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        x_norm = (x_adv - mean_t) / std_t
        x_norm = x_norm.requires_grad_(True)

        loss = criterion(model(x_norm), labels)
        loss.backward()

        grad = x_norm.grad.detach()

        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + alpha * grad.sign()
                delta  = (x_adv - x).clamp(-eps, eps)
            else:  # l2
                g_norm = grad.view(grad.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                x_adv  = x_adv + alpha * grad / (g_norm + 1e-8)
                delta  = x_adv - x
                d_norm = delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
                delta  = delta * torch.clamp(eps / (d_norm + 1e-8), max=1.0)

            x_adv = (x + delta).clamp(0.0, 1.0).detach()

    return (x_adv - mean_t) / std_t