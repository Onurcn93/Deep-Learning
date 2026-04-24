"""Plotting utilities for training curves, confusion matrix, CIFAR-10-C, and adversarial.

All figures are saved under ``results/<model>/`` (git-ignored, dirs tracked via .gitkeep).
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
MNIST_CLASSES = [str(i) for i in range(10)]

PLOTS_DIR = "results"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _title_to_filename(title: str) -> str:
    """Convert a config title string to a safe filename suffix."""
    import re
    s = title.replace("Ã—", "x").replace(" | ", "_").replace("=", "").replace(" ", "_")
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
    print(f"[plot] Saved training curves â†’ {path}")


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
    print(f"[plot] Saved confusion matrix â†’ {path}")


def plot_cifar10c_results(
    corruption_accs: Dict[str, List[float]],
    clean_acc:       float,
    out_dir:         str = PLOTS_DIR,
    title:           str = "",
) -> None:
    """Save two CIFAR-10-C robustness plots: a bar chart and a severity heatmap.

    Args:
        corruption_accs: Mapping from corruption name to a list of 5 accuracy
                         values (one per severity level, lowâ†’high).
        clean_acc:       Clean test accuracy used as the reference baseline.
        out_dir:         Directory to save the figures.
        title:           Config string used in figure titles and filenames.
    """
    _ensure_dir(out_dir)
    suffix = f"_{_title_to_filename(title)}" if title else ""

    corruptions = list(corruption_accs.keys())
    mean_accs   = [sum(v) / len(v) for v in corruption_accs.values()]
    n           = len(corruptions)

    # â”€â”€ Plot 1: horizontal bar chart (mean acc per corruption) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45)))

    colors = ["#d9534f" if a < clean_acc else "#5cb85c" for a in mean_accs]
    bars   = ax.barh(range(n), mean_accs, color=colors, edgecolor="white", height=0.7)
    ax.axvline(clean_acc, color="steelblue", linewidth=1.8, linestyle="--", label=f"Clean ({clean_acc:.3f})")

    ax.set_yticks(range(n))
    ax.set_yticklabels(corruptions, fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.0)
    ax.set_title("Mean Accuracy per Corruption Type (avg over severities 1â€“5)")
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
    print(f"[plot] Saved CIFAR-10-C bar chart â†’ {path}")

    # â”€â”€ Plot 2: heatmap (corruptions Ã— severity) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    ax.set_title("Accuracy per Corruption Ã— Severity")
    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    path = os.path.join(out_dir, f"cifar10c_heatmap{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved CIFAR-10-C heatmap â†’ {path}")


def plot_gradcam(
    clean_imgs:  List[np.ndarray],
    adv_imgs:    List[np.ndarray],
    clean_cams:  List[np.ndarray],
    adv_cams:    List[np.ndarray],
    labels:      List[int],
    clean_preds: List[int],
    adv_preds:   List[int],
    class_names: List[str],
    out_dir:     str = PLOTS_DIR,
    title:       str = "",
) -> None:
    """Save a Grad-CAM comparison figure: clean vs adversarial for N samples.

    Each row shows one sample: clean image | clean CAM overlay |
    adversarial image | adversarial CAM overlay.

    Args:
        clean_imgs:  List of (H, W, 3) float32 arrays in [0, 1] pixel space.
        adv_imgs:    List of (H, W, 3) float32 arrays in [0, 1] pixel space.
        clean_cams:  List of (H', W') Grad-CAM heatmaps in [0, 1].
        adv_cams:    List of (H', W') Grad-CAM heatmaps in [0, 1].
        labels:      Ground-truth class indices.
        clean_preds: Predicted class indices on clean images.
        adv_preds:   Predicted class indices on adversarial images.
        class_names: Human-readable class name list indexed by class idx.
        out_dir:     Directory to save the figure.
        title:       Config string for figure title and filename.
    """
    from PIL import Image as PILImage

    _ensure_dir(out_dir)
    n = len(clean_imgs)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean", "Clean CAM", "Adversarial", "Adv CAM"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=10, fontweight="bold")

    cmap = plt.get_cmap("jet")

    for i in range(n):
        true_name  = class_names[labels[i]]
        clean_name = class_names[clean_preds[i]]
        adv_name   = class_names[adv_preds[i]]
        H, W       = clean_imgs[i].shape[:2]

        def _overlay(img_hw3, cam_hw):
            cam_pil     = PILImage.fromarray((cam_hw * 255).astype(np.uint8), mode="L")
            cam_resized = np.array(cam_pil.resize((W, H), PILImage.BILINEAR)) / 255.0
            heat        = cmap(cam_resized)[..., :3]             # (H, W, 3)
            return np.clip(0.5 * img_hw3 + 0.5 * heat, 0, 1)

        axes[i, 0].imshow(clean_imgs[i])
        axes[i, 0].set_ylabel(f"True: {true_name}", fontsize=8)
        axes[i, 0].set_xlabel(f"Pred: {clean_name}", fontsize=8, color="green")

        axes[i, 1].imshow(_overlay(clean_imgs[i], clean_cams[i]))
        axes[i, 1].set_xlabel(f"Pred: {clean_name}", fontsize=8, color="green")

        axes[i, 2].imshow(adv_imgs[i])
        axes[i, 2].set_xlabel(f"Pred: {adv_name}", fontsize=8, color="red")

        axes[i, 3].imshow(_overlay(adv_imgs[i], adv_cams[i]))
        axes[i, 3].set_xlabel(f"Pred: {adv_name}", fontsize=8, color="red")

        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])

    suffix = f"_{_title_to_filename(title)}" if title else ""
    path   = os.path.join(out_dir, f"gradcam{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved Grad-CAM figure â†’ {path}")


def plot_tsne(
    feats_clean: np.ndarray,
    feats_adv:   np.ndarray,
    labels:      np.ndarray,
    out_dir:     str = PLOTS_DIR,
    title:       str = "",
) -> None:
    """Save a t-SNE scatter of clean vs adversarial feature embeddings.

    Clean samples are shown as circles and adversarial as crosses, both
    coloured by true class.

    Args:
        feats_clean: (N, D) feature matrix for clean images.
        feats_adv:   (N, D) feature matrix for adversarial images.
        labels:      (N,) ground-truth class indices.
        out_dir:     Directory to save the figure.
        title:       Config string for figure title and filename.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[plot] sklearn not installed â€” skipping t-SNE. Run: pip install scikit-learn")
        return

    _ensure_dir(out_dir)

    all_feats = np.concatenate([feats_clean, feats_adv], axis=0)
    embedded  = TSNE(n_components=2, perplexity=30, random_state=42,
                     max_iter=1000).fit_transform(all_feats)

    n         = len(feats_clean)
    emb_clean = embedded[:n]
    emb_adv   = embedded[n:]

    num_classes = int(labels.max()) + 1
    colors      = plt.get_cmap("tab10")(np.linspace(0, 1, num_classes))
    label_names = CIFAR10_CLASSES if num_classes == 10 else [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(9, 7))

    for c in range(num_classes):
        mask = labels == c
        ax.scatter(emb_clean[mask, 0], emb_clean[mask, 1],
                   color=colors[c], marker="o", s=12, alpha=0.6,
                   label=f"{label_names[c]} clean")
        ax.scatter(emb_adv[mask, 0], emb_adv[mask, 1],
                   color=colors[c], marker="x", s=12, alpha=0.6,
                   label=f"{label_names[c]} adv")

    # Compact legend: one entry per class (ignore clean/adv split in legend)
    handles, leg_labels = ax.get_legend_handles_labels()
    # Take every other entry (clean entries) for the legend
    ax.legend(handles[::2], label_names, fontsize=7, ncol=2,
              loc="upper right", markerscale=1.5)

    ax.set_title("t-SNE: clean (â—‹) vs adversarial (Ã—) features")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])

    suffix = f"_{_title_to_filename(title)}" if title else ""
    path   = os.path.join(out_dir, f"tsne{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved t-SNE figure â†’ {path}")

