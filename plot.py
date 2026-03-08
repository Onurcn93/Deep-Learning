"""Plotting utilities for training curves and confusion matrix.

All figures are saved to the ``plots/`` directory (git-ignored).
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
MNIST_CLASSES = [str(i) for i in range(10)]

PLOTS_DIR = "plots"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    train_accs:   List[float],
    val_accs:     List[float],
    out_dir:      str = PLOTS_DIR,
) -> None:
    """Save loss and accuracy curves over epochs.

    Args:
        train_losses: Training loss per epoch.
        val_losses:   Validation loss per epoch.
        train_accs:   Training accuracy per epoch.
        val_accs:     Validation accuracy per epoch.
        out_dir:      Directory to save the figure.
    """
    _ensure_dir(out_dir)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train", marker="o", markersize=3)
    ax1.plot(epochs, val_losses,   label="Val",   marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label="Train", marker="o", markersize=3)
    ax2.plot(epochs, val_accs,   label="Val",   marker="o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved training curves → {path}")


def plot_confusion_matrix(
    all_preds:  List[int],
    all_labels: List[int],
    dataset:    str,
    out_dir:    str = PLOTS_DIR,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        all_preds:  Flat list of predicted class indices.
        all_labels: Flat list of ground-truth class indices.
        dataset:    Dataset name (``'mnist'`` or ``'cifar10'``).
        out_dir:    Directory to save the figure.
    """
    _ensure_dir(out_dir)

    class_names = CIFAR10_CLASSES if dataset == "cifar10" else MNIST_CLASSES
    num_classes = len(class_names)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(all_preds, all_labels):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(10, 8))

    try:
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
    except ImportError:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved confusion matrix → {path}")
