from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from train import get_transforms, get_cifar10c_loader
from parameters import DataParams, ModelParams, TrainingParams
from plot import plot_confusion_matrix, plot_cifar10c_results


@torch.no_grad()
def run_test(
    model:           nn.Module,
    data_params:     DataParams,
    model_params:    ModelParams,
    training_params: TrainingParams,
    device:          torch.device,
    config_title:    str = "",
) -> Dict[str, float]:
    """Evaluate a trained model on the test split and print per-class accuracy.

    Loads the best saved weights from ``training_params.save_path`` before
    running evaluation.

    Args:
        model:           The neural network to evaluate.
        data_params:     Dataset parameters used to load test data.
        training_params: Training parameters (save path, batch size).
        device:          Computation device.

    Returns:
        Dictionary with key ``'overall'`` and per-class string keys mapped to
        accuracy values.
    """
    tf = get_transforms(data_params, train=False, transfer_mode=model_params.transfer_mode)

    if data_params.dataset == "mnist":
        test_ds = datasets.MNIST(data_params.data_dir, train=False, download=True, transform=tf)
    else:  # cifar10
        test_ds = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)

    loader = DataLoader(test_ds, batch_size=training_params.batch_size,
                        shuffle=False, num_workers=data_params.num_workers)

    model.load_state_dict(torch.load(training_params.save_path, map_location=device))
    model.eval()

    correct, n  = 0, 0
    class_correct = [0] * data_params.num_classes
    class_total   = [0] * data_params.num_classes
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    results: Dict[str, float] = {"overall": correct / n}
    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(data_params.num_classes):
        acc = class_correct[i] / class_total[i]
        results[str(i)] = acc
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    if training_params.plot:
        plot_confusion_matrix(all_preds, all_labels, data_params.dataset, title=config_title)

    return results


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "gaussian_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


@torch.no_grad()
def run_cifar10c_test(
    model:           nn.Module,
    data_params:     DataParams,
    training_params: TrainingParams,
    device:          torch.device,
    clean_acc:       float = 0.0,
    config_title:    str   = "",
) -> Dict[str, float]:
    """Evaluate a trained model on all CIFAR-10-C corruptions across all severities.

    Prints a table of per-corruption accuracy averaged over severities 1–5, plus
    the overall mean corruption accuracy (mCA). Optionally saves a bar chart and
    heatmap to ``plots/`` when ``training_params.plot`` is enabled.

    Args:
        model:           The neural network to evaluate (weights already loaded).
        data_params:     Dataset parameters (cifar10c_dir, mean, std, num_classes).
        training_params: Training parameters (batch_size, plot flag).
        device:          Computation device.
        clean_acc:       Clean test accuracy for reference line in bar chart.
        config_title:    Config string used in plot titles and filenames.

    Returns:
        Dictionary mapping corruption name to mean accuracy across severities,
        plus ``'mCA'`` for the overall mean corruption accuracy.
    """
    model.eval()

    print("\n=== CIFAR-10-C Robustness Evaluation ===")
    print(f"{'Corruption':<22} {'Sev1':>6} {'Sev2':>6} {'Sev3':>6} {'Sev4':>6} {'Sev5':>6} {'Mean':>6}")
    print("─" * 65)

    results:         Dict[str, float]       = {}
    corruption_accs: Dict[str, List[float]] = {}

    for corruption in CORRUPTIONS:
        sev_accs = []
        for severity in range(1, 6):
            loader = get_cifar10c_loader(corruption, severity, data_params, training_params.batch_size)
            correct, n = 0, 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += preds.eq(labels).sum().item()
                n += imgs.size(0)
            sev_accs.append(correct / n)

        mean_acc = sum(sev_accs) / len(sev_accs)
        results[corruption]         = mean_acc
        corruption_accs[corruption] = sev_accs
        sev_str = "  ".join(f"{a:.3f}" for a in sev_accs)
        print(f"{corruption:<22} {sev_str}  {mean_acc:.3f}")

    mca = sum(results.values()) / len(results)
    print("─" * 65)
    print(f"{'Mean Corruption Acc (mCA)':<22} {'':>42} {mca:.3f}")
    if clean_acc:
        print(f"{'Clean Acc':<22} {'':>42} {clean_acc:.3f}")
    results["mCA"] = mca

    if training_params.plot:
        plot_cifar10c_results(corruption_accs, clean_acc, title=config_title)

    return results
