import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import DataParams, TrainingParams
from plot import plot_training_curves
from logger import TrainLogger


def get_transforms(
    data_params:   DataParams,
    train:         bool = True,
    transfer_mode: str  = "none",
) -> transforms.Compose:
    """Build a torchvision transform pipeline for a given dataset split.

    Args:
        data_params:   Dataset-related parameters (dataset name, mean, std).
        train:         If ``True``, applies training augmentations (CIFAR-10 only).
        transfer_mode: If ``'resizeFreeze'``, upscales CIFAR-10 images to 224×224
                       to match ImageNet input size expected by pretrained backbones.

    Returns:
        A composed transform pipeline.
    """
    mean, std = data_params.mean, data_params.std

    if data_params.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        resize = [transforms.Resize(224)] if transfer_mode == "resizeFreeze" else []
        if train:
            return transforms.Compose([
                *resize,
                transforms.RandomCrop(224 if transfer_mode == "resizeFreeze" else 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                *resize,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(
    data_params:   DataParams,
    training_params: TrainingParams,
    transfer_mode: str = "none",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        data_params:     Dataset parameters.
        training_params: Training parameters (batch size).
        transfer_mode:   Passed to get_transforms for resize logic.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    train_tf = get_transforms(data_params, train=True,  transfer_mode=transfer_mode)
    val_tf   = get_transforms(data_params, train=False, transfer_mode=transfer_mode)

    if data_params.dataset == "mnist":
        train_ds = datasets.MNIST(data_params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(data_params.data_dir, train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(data_params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=training_params.batch_size,
                              shuffle=True,  num_workers=data_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=training_params.batch_size,
                              shuffle=False, num_workers=data_params.num_workers)
    return train_loader, val_loader


def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    l1_lambda:    float,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one training epoch with optional L1 regularisation.

    Args:
        model:        The neural network to train.
        loader:       Training DataLoader.
        optimizer:    Optimiser instance.
        criterion:    Loss function.
        device:       Computation device.
        l1_lambda:    L1 regularisation coefficient (0 = disabled).
        log_interval: Batches between progress prints.

    Returns:
        Tuple of ``(mean_loss, accuracy)`` over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)

        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f" {batch_idx+1:>4}/{len(loader)} | {total_loss/n:>7.4f} | {correct/n:>6.3f} |       - |      -")

    return total_loss / n, correct / n


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a validation split.

    Args:
        model:     The neural network to evaluate.
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    Computation device.

    Returns:
        Tuple of ``(mean_loss, accuracy)``.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def build_scheduler(
    optimizer:       torch.optim.Optimizer,
    training_params: TrainingParams,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Instantiate an LR scheduler based on training parameters.

    Args:
        optimizer:       The optimiser whose LR will be scheduled.
        training_params: Training parameters (scheduler type, epochs).

    Returns:
        An ``LRScheduler`` instance, or ``None`` if scheduler is ``'none'``.
    """
    if training_params.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if training_params.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params.epochs)
    return None


def run_training(
    model:           nn.Module,
    data_params:     DataParams,
    model_params,
    training_params: TrainingParams,
    device:          torch.device,
    config_title:    str = "",
    logger:          TrainLogger = None,
) -> None:
    """Full training loop with early stopping, LR scheduling, and best-model saving.

    Args:
        model:           The neural network to train.
        data_params:     Dataset parameters.
        training_params: Training hyperparameters.
        device:          Computation device.
    """
    train_loader, val_loader = get_loaders(data_params, training_params, model_params.transfer_mode)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_params.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = training_params.learning_rate,
        weight_decay = training_params.weight_decay,
    )
    scheduler = build_scheduler(optimizer, training_params)

    best_acc      = 0.0
    best_weights  = None
    patience_ctr  = 0

    train_losses: List[float] = []
    val_losses:   List[float] = []
    train_accs:   List[float] = []
    val_accs:     List[float] = []

    logger.log_start(model, data_params, model_params, training_params, device)

    for epoch in range(1, training_params.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            training_params.l1_lambda, training_params.log_interval,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

        logger.log_epoch(epoch, tr_loss, tr_acc, val_loss, val_acc)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, training_params.save_path)
            logger.log_best(best_acc, training_params.save_path)
            patience_ctr = 0
        else:
            patience_ctr += 1

        if training_params.patience > 0 and patience_ctr >= training_params.patience:
            logger._w(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(patience={training_params.patience})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    logger.log_complete(best_acc, training_params.save_path)
    logger.close()

    if training_params.plot:
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, title=config_title)
