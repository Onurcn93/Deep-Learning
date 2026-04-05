"""Plotting utilities for training curves, confusion matrix, and CIFAR-10-C robustness.

All figures are saved to the ``plots/`` directory (git-ignored).
"""

import os
from typing import Dict, List

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


def _title_to_filename(title: str) -> str:
    """Convert a config title string to a safe filename suffix."""
    import re
    s = title.replace("×", "x").replace(" | ", "_").replace("=", "").replace(" ", "_")
    s = re.sub(r"[^\w\-]", "", s)
    return s


def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    train_accs:   List[float],
    val_accs:     List[float],
    out_dir:      str = PLOTS_DIR,
    title:        str = "",
) -> None:
    """Save loss and accuracy curves over epochs.

    Args:
        train_losses: Training loss per epoch.
        val_losses:   Validation loss per epoch.
        train_accs:   Training accuracy per epoch.
        val_accs:     Validation accuracy per epoch.
        out_dir:      Directory to save the figure.
        title:        Overall figure title describing the training setup.
    """
    _ensure_dir(out_dir)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold")

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

    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])
    suffix = f"_{_title_to_filename(title)}" if title else ""
    path = os.path.join(out_dir, f"training_curves{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved training curves → {path}")


def plot_confusion_matrix(
    all_preds:  List[int],
    all_labels: List[int],
    dataset:    str,
    out_dir:    str = PLOTS_DIR,
    title:      str = "",
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        all_preds:  Flat list of predicted class indices.
        all_labels: Flat list of ground-truth class indices.
        dataset:    Dataset name (``'mnist'`` or ``'cifar10'``).
        out_dir:    Directory to save the figure.
        title:      Overall figure title describing the training setup.
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
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])

    suffix = f"_{_title_to_filename(title)}" if title else ""
    path = os.path.join(out_dir, f"confusion_matrix{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved confusion matrix → {path}")


def plot_cifar10c_results(
    corruption_accs: Dict[str, List[float]],
    clean_acc:       float,
    out_dir:         str = PLOTS_DIR,
    title:           str = "",
) -> None:
    """Save two CIFAR-10-C robustness plots: a bar chart and a severity heatmap.

    Args:
        corruption_accs: Mapping from corruption name to a list of 5 accuracy
                         values (one per severity level, low→high).
        clean_acc:       Clean test accuracy used as the reference baseline.
        out_dir:         Directory to save the figures.
        title:           Config string used in figure titles and filenames.
    """
    _ensure_dir(out_dir)
    suffix = f"_{_title_to_filename(title)}" if title else ""

    corruptions = list(corruption_accs.keys())
    mean_accs   = [sum(v) / len(v) for v in corruption_accs.values()]
    n           = len(corruptions)

    # ── Plot 1: horizontal bar chart (mean acc per corruption) ──────────────
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45)))

    colors = ["#d9534f" if a < clean_acc else "#5cb85c" for a in mean_accs]
    bars   = ax.barh(range(n), mean_accs, color=colors, edgecolor="white", height=0.7)
    ax.axvline(clean_acc, color="steelblue", linewidth=1.8, linestyle="--", label=f"Clean ({clean_acc:.3f})")

    ax.set_yticks(range(n))
    ax.set_yticklabels(corruptions, fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.0)
    ax.set_title("Mean Accuracy per Corruption Type (avg over severities 1–5)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Annotate bar values
    for i, (bar, acc) in enumerate(zip(bars, mean_accs)):
        ax.text(max(acc - 0.02, 0.01), i, f"{acc:.3f}",
                va="center", ha="right", fontsize=8, color="white", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    path = os.path.join(out_dir, f"cifar10c_bar{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved CIFAR-10-C bar chart → {path}")

    # ── Plot 2: heatmap (corruptions × severity) ────────────────────────────
    matrix = np.array([corruption_accs[c] for c in corruptions])  # (n, 5)

    fig, ax = plt.subplots(figsize=(7, max(5, n * 0.45)))

    try:
        import seaborn as sns
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.0, vmax=1.0,
            xticklabels=[f"Sev {s}" for s in range(1, 6)],
            yticklabels=corruptions,
            ax=ax, linewidths=0.4,
        )
    except ImportError:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"Sev {s}" for s in range(1, 6)])
        ax.set_yticks(range(n))
        ax.set_yticklabels(corruptions, fontsize=9)
        for i in range(n):
            for j in range(5):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)

    ax.set_xlabel("Severity")
    ax.set_title("Accuracy per Corruption × Severity")
    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    path = os.path.join(out_dir, f"cifar10c_heatmap{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved CIFAR-10-C heatmap → {path}")
