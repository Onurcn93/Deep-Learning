from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from train import get_transforms
from parameters import DataParams, TrainingParams
from plot import plot_confusion_matrix


@torch.no_grad()
def run_test(
    model:           nn.Module,
    data_params:     DataParams,
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
    tf = get_transforms(data_params, train=False)

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
