from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from train import get_transforms, get_cifar10c_loader
from parameters import DataParams, ModelParams, TrainingParams
from utils.plot import (plot_confusion_matrix, plot_cifar10c_results,
                        plot_gradcam, plot_tsne, CIFAR10_CLASSES)
from attack import pgd_attack
from utils.gradcam import GradCAM, get_target_layer


@torch.no_grad()
def run_test(
    model:            nn.Module,
    data_params:      DataParams,
    model_params:     ModelParams,
    training_params:  TrainingParams,
    device:           torch.device,
    config_title:     str = "",
    checkpoint_path:  str = "",
) -> Dict[str, float]:
    """Evaluate a trained model on the test split and print per-class accuracy.

    Loads the best saved weights from ``checkpoint_path`` if provided, otherwise
    falls back to ``training_params.save_path``.

    Args:
        model:            The neural network to evaluate.
        data_params:      Dataset parameters used to load test data.
        training_params:  Training parameters (save path, batch size).
        device:           Computation device.
        config_title:     Experiment title for confusion matrix filename.
        checkpoint_path:  Override path for model weights (e.g. augmix_save_path).

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

    path = checkpoint_path if checkpoint_path else training_params.save_path
    model.load_state_dict(torch.load(path, map_location=device))
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
    "defocus_blur", "gaussian_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "saturate", "spatter",
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


# ─────────────────────────────────────────────────────────────────────────────
# PGD adversarial evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(
    model:      nn.Module,
    imgs_cpu:   torch.Tensor,
    batch_size: int,
    device:     torch.device,
) -> np.ndarray:
    """Extract penultimate-layer (avgpool) features via a forward hook.

    Args:
        model:      Model with an ``avgpool`` attribute (e.g. ResNet-18).
        imgs_cpu:   Normalised images on CPU, shape (N, C, H, W).
        batch_size: Mini-batch size for the forward passes.
        device:     Computation device.

    Returns:
        (N, D) float32 numpy array of feature vectors.
    """
    feats: List[torch.Tensor] = []

    def _hook(module, inp, out):
        feats.append(out.detach().squeeze(-1).squeeze(-1).cpu())

    hook = model.avgpool.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs_cpu), batch_size):
            model(imgs_cpu[i : i + batch_size].to(device))
    hook.remove()
    return torch.cat(feats).numpy()


def run_pgd_test(
    model:           nn.Module,
    data_params:     DataParams,
    model_params:    ModelParams,
    training_params: TrainingParams,
    device:          torch.device,
    config_title:    str = "",
) -> Dict[str, float]:
    """Evaluate a model under PGD-20 adversarial attack (L∞ and L2).

    Loads weights from ``training_params.model_path``, evaluates clean accuracy
    and adversarial accuracy on a subset of the test set, and optionally
    generates Grad-CAM and t-SNE visualisations.

    Args:
        model:           Instantiated model matching the saved checkpoint.
        data_params:     Dataset parameters (mean, std, data_dir, etc.).
        model_params:    Architecture parameters (used for transforms + GradCAM).
        training_params: Training/evaluation parameters (pgd_* flags).
        device:          Computation device.
        config_title:    Experiment title for plot filenames.

    Returns:
        Dict with keys ``'clean'``, ``'pgd_linf'``, ``'pgd_l2'``.
    """
    # ── Load weights ──────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(training_params.model_path, map_location=device))
    model.eval()

    # ── Build test subset loader ───────────────────────────────────────────────
    tf      = get_transforms(data_params, train=False,
                             transfer_mode=model_params.transfer_mode)
    test_ds = datasets.CIFAR10(data_params.data_dir, train=False,
                               download=True, transform=tf)
    n_use   = min(training_params.pgd_n_samples, len(test_ds))
    loader  = DataLoader(Subset(test_ds, range(n_use)),
                         batch_size=training_params.batch_size,
                         shuffle=False, num_workers=0)

    eps_linf  = training_params.pgd_eps_linf
    eps_l2    = training_params.pgd_eps_l2
    steps     = training_params.pgd_steps
    alpha_linf = 2.5 * eps_linf / steps
    alpha_l2   = 2.5 * eps_l2   / steps

    # ── Accumulators ──────────────────────────────────────────────────────────
    n = clean_correct = linf_correct = l2_correct = 0
    gradcam_samples: List[dict] = []

    clean_batches: List[torch.Tensor] = []
    adv_linf_batches: List[torch.Tensor] = []
    labels_batches: List[torch.Tensor] = []

    print(f"\n=== PGD Adversarial Evaluation  ({n_use} samples) ===")
    print(f"  L∞  eps={eps_linf:.4f}  alpha={alpha_linf:.5f}  steps={steps}")
    print(f"  L2  eps={eps_l2}        alpha={alpha_l2:.5f}  steps={steps}")
    print()

    # ── Main eval loop ─────────────────────────────────────────────────────────
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = model(imgs).argmax(1)

        adv_linf = pgd_attack(model, imgs, labels,
                               eps=eps_linf, alpha=alpha_linf, steps=steps,
                               norm="linf",
                               mean=data_params.mean, std=data_params.std)
        with torch.no_grad():
            linf_preds = model(adv_linf).argmax(1)

        adv_l2 = pgd_attack(model, imgs, labels,
                             eps=eps_l2, alpha=alpha_l2, steps=steps,
                             norm="l2",
                             mean=data_params.mean, std=data_params.std)
        with torch.no_grad():
            l2_preds = model(adv_l2).argmax(1)

        clean_correct += clean_preds.eq(labels).sum().item()
        linf_correct  += linf_preds.eq(labels).sum().item()
        l2_correct    += l2_preds.eq(labels).sum().item()
        n             += labels.size(0)

        clean_batches.append(imgs.cpu())
        adv_linf_batches.append(adv_linf.cpu())
        labels_batches.append(labels.cpu())

        # Collect GradCAM candidates: clean correct, L∞ fooled
        if training_params.gradcam and len(gradcam_samples) < 4:
            for i in range(labels.size(0)):
                if len(gradcam_samples) >= 4:
                    break
                if clean_preds[i] == labels[i] and linf_preds[i] != labels[i]:
                    gradcam_samples.append({
                        "clean_img": imgs[i].cpu(),
                        "adv_img":   adv_linf[i].cpu(),
                        "label":     labels[i].item(),
                        "clean_pred": clean_preds[i].item(),
                        "adv_pred":   linf_preds[i].item(),
                    })

    # ── Results table ─────────────────────────────────────────────────────────
    acc_clean = clean_correct / n
    acc_linf  = linf_correct  / n
    acc_l2    = l2_correct    / n

    print(f"{'Eval':<22} {'Acc':>7}  {'Drop':>7}")
    print("─" * 40)
    print(f"{'Clean':<22} {acc_clean:>7.4f}")
    print(f"{'PGD-L∞'::<22} {acc_linf:>7.4f}  {acc_clean-acc_linf:>+7.4f}")
    print(f"{'PGD-L2':<22} {acc_l2:>7.4f}  {acc_clean-acc_l2:>+7.4f}")
    print("─" * 40)

    results = {"clean": acc_clean, "pgd_linf": acc_linf, "pgd_l2": acc_l2}

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if training_params.gradcam and gradcam_samples:
        mean_t = torch.tensor(data_params.mean).view(3, 1, 1)
        std_t  = torch.tensor(data_params.std).view(3, 1, 1)

        try:
            target_layer = get_target_layer(model)
        except ValueError as e:
            print(f"[gradcam] Skipped — {e}")
        else:
            gradcam = GradCAM(model, target_layer)
            clean_imgs_px, adv_imgs_px = [], []
            clean_cams,    adv_cams    = [], []
            labels_gc, clean_preds_gc, adv_preds_gc = [], [], []

            for s in gradcam_samples:
                img_t = s["clean_img"].unsqueeze(0).to(device)
                cam_c, pred_c = gradcam.generate(img_t)
                adv_t = s["adv_img"].unsqueeze(0).to(device)
                cam_a, pred_a = gradcam.generate(adv_t)

                def _to_pixel(t):
                    return (t * std_t + mean_t).permute(1, 2, 0).clamp(0, 1).numpy()

                clean_imgs_px.append(_to_pixel(s["clean_img"]))
                adv_imgs_px.append(_to_pixel(s["adv_img"]))
                clean_cams.append(cam_c)
                adv_cams.append(cam_a)
                labels_gc.append(s["label"])
                clean_preds_gc.append(pred_c)
                adv_preds_gc.append(pred_a)

            gradcam.remove_hooks()

            if training_params.plot:
                plot_gradcam(
                    clean_imgs_px, adv_imgs_px,
                    clean_cams,    adv_cams,
                    labels_gc, clean_preds_gc, adv_preds_gc,
                    class_names=CIFAR10_CLASSES,
                    title=config_title,
                )

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    if training_params.tsne:
        if not hasattr(model, "avgpool"):
            print("[tsne] Skipped — model has no avgpool layer for feature extraction.")
        else:
            all_clean    = torch.cat(clean_batches)
            all_adv_linf = torch.cat(adv_linf_batches)
            all_labels   = torch.cat(labels_batches).numpy()

            print("[tsne] Extracting features …")
            feats_clean = _extract_features(model, all_clean,    training_params.batch_size, device)
            feats_adv   = _extract_features(model, all_adv_linf, training_params.batch_size, device)

            if training_params.plot:
                plot_tsne(feats_clean, feats_adv, all_labels, title=config_title)

    return results
