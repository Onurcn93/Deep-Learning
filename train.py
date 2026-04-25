import copy
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from parameters import DataParams, TrainingParams
from utils.plot import plot_training_curves
from utils.logger import TrainLogger


def get_transforms(
    data_params:   DataParams,
    train:         bool = True,
    transfer_mode: str  = "none",
) -> transforms.Compose:
    """Build a torchvision transform pipeline for a given dataset split.

    Args:
        data_params:   Dataset-related parameters (dataset name, mean, std).
        train:         If ``True``, applies training augmentations (CIFAR-10 only).
        transfer_mode: If ``'resizeFreeze'``, upscales CIFAR-10 images to 224Ã—224
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
    elif data_params.dataset == "cifar10":
        train_ds = datasets.CIFAR10(data_params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=val_tf)
    elif data_params.dataset == "voc":
        from datasets.voc import VOCClassification
        train_ds = VOCClassification(data_params.voc_dir, split="train",
                                     image_size=data_params.voc_image_size, download=True)
        val_ds   = VOCClassification(data_params.voc_dir, split="val",
                                     image_size=data_params.voc_image_size, download=True)
    else:  # voc_person
        from datasets.voc import VOCBinaryClassification
        train_ds = VOCBinaryClassification(data_params.voc_dir, split="train",
                                           image_size=data_params.voc_image_size, download=True)
        val_ds   = VOCBinaryClassification(data_params.voc_dir, split="val",
                                           image_size=data_params.voc_image_size, download=True)

    train_loader = DataLoader(train_ds, batch_size=training_params.batch_size,
                              shuffle=True,  num_workers=data_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=training_params.batch_size,
                              shuffle=False, num_workers=data_params.num_workers)
    return train_loader, val_loader


class AugMixDataset(Dataset):
    """Dataset wrapper that returns three views of each image for AugMix training.

    For each sample returns ``(img_clean, img_aug1, img_aug2, label)`` where
    ``img_clean`` is the standard normalised image and ``img_aug1`` / ``img_aug2``
    are two independently AugMix-augmented versions.

    Args:
        base_dataset: Underlying dataset returning ``(PIL image, label)`` pairs.
        clean_tf:     Transform applied to produce the clean view (normalise only).
        augmix_tf:    Transform applied twice to produce the two augmented views.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        clean_tf:     transforms.Compose,
        augmix_tf:    transforms.Compose,
    ) -> None:
        self.dataset  = base_dataset
        self.clean_tf = clean_tf
        self.augmix_tf = augmix_tf

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return self.clean_tf(img), self.augmix_tf(img), self.augmix_tf(img), label


def get_augmix_loaders(
    data_params:     DataParams,
    training_params: TrainingParams,
    transfer_mode:   str = "none",
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for AugMix training.

    The train loader returns ``(clean, aug1, aug2, label)`` tuples via
    ``AugMixDataset``.  The val loader is unchanged (standard normalisation).

    Args:
        data_params:     Dataset parameters (mean, std, data_dir).
        training_params: Training parameters (batch size).
        transfer_mode:   Passed through for resize logic.

    Returns:
        Tuple of ``(augmix_train_loader, val_loader)``.
    """
    mean, std = data_params.mean, data_params.std
    resize = [transforms.Resize(224)] if transfer_mode == "resizeFreeze" else []

    clean_tf = transforms.Compose([
        *resize,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    augmix_tf = transforms.Compose([
        *resize,
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = get_transforms(data_params, train=False, transfer_mode=transfer_mode)

    train_base = datasets.CIFAR10(data_params.data_dir, train=True,  download=True, transform=None)
    val_ds     = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=val_tf)

    train_ds = AugMixDataset(train_base, clean_tf, augmix_tf)

    train_loader = DataLoader(train_ds, batch_size=training_params.batch_size,
                              shuffle=True,  num_workers=data_params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=training_params.batch_size,
                              shuffle=False, num_workers=data_params.num_workers)
    return train_loader, val_loader


def train_one_epoch_augmix(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    jsd_lambda: float = 12.0,
    log_interval: int = 100,
) -> Tuple[float, float]:
    """Train one epoch with AugMix: cross-entropy on clean view + JSD consistency loss.

    Each batch contains three views of every image (clean, aug1, aug2).  The
    cross-entropy loss is computed on the clean logits only.  The JSD term
    encourages consistent predictions across all three views.

    Loss = CE(clean) + Î» * JSD(p_clean, p_aug1, p_aug2)

    Args:
        model:        The neural network to train.
        loader:       AugMix DataLoader returning ``(clean, aug1, aug2, labels)``.
        optimizer:    Optimiser instance.
        criterion:    Hard-label loss (CrossEntropyLoss).
        device:       Computation device.
        jsd_lambda:   Weight on the JSD consistency term (paper default: 12.0).
        log_interval: Batches between progress prints.

    Returns:
        Tuple of ``(mean_loss, mean_accuracy)`` over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (img_clean, img_aug1, img_aug2, labels) in enumerate(loader, 1):
        img_clean = img_clean.to(device)
        img_aug1  = img_aug1.to(device)
        img_aug2  = img_aug2.to(device)
        labels    = labels.to(device)

        # Three forward passes
        logits_clean = model(img_clean)
        logits_aug1  = model(img_aug1)
        logits_aug2  = model(img_aug2)

        # Cross-entropy on clean view
        ce_loss = criterion(logits_clean, labels)

        # Jensen-Shannon consistency across three views
        p_clean = F.softmax(logits_clean, dim=1)
        p_aug1  = F.softmax(logits_aug1,  dim=1)
        p_aug2  = F.softmax(logits_aug2,  dim=1)
        p_mix   = (p_clean + p_aug1 + p_aug2) / 3.0
        jsd = (
            F.kl_div(p_mix.log(), p_clean, reduction="batchmean") +
            F.kl_div(p_mix.log(), p_aug1,  reduction="batchmean") +
            F.kl_div(p_mix.log(), p_aug2,  reduction="batchmean")
        ) / 3.0

        loss = ce_loss + jsd_lambda * jsd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size  = labels.size(0)
        total_loss += loss.item() * batch_size
        correct    += logits_clean.argmax(1).eq(labels).sum().item()
        n          += batch_size

        if batch_idx % log_interval == 0:
            print(f"  {batch_idx:>5}/{len(loader)} | {total_loss/n:>8.4f} | {correct/n:>6.3f} |       - |      -")

    return total_loss / n, correct / n


def get_cifar10c_loader(
    corruption:  str,
    severity:    int,
    data_params: DataParams,
    batch_size:  int,
) -> DataLoader:
    """Create a DataLoader for a single CIFAR-10-C corruption at a given severity.

    CIFAR-10-C stores each corruption as an (50000, 32, 32, 3) uint8 numpy array
    where indices [0:10000] correspond to severity 1, [10000:20000] to severity 2,
    etc.  Labels are shared across all corruptions and severities.

    Args:
        corruption:  Corruption name (e.g. ``'gaussian_noise'``).
        severity:    Severity level 1â€“5.
        data_params: Dataset parameters (cifar10c_dir, mean, std).
        batch_size:  Mini-batch size for the returned DataLoader.

    Returns:
        DataLoader yielding normalised tensors and integer labels.
    """
    assert 1 <= severity <= 5, "severity must be between 1 and 5"

    data_path   = os.path.join(data_params.cifar10c_dir, f"{corruption}.npy")
    labels_path = os.path.join(data_params.cifar10c_dir, "labels.npy")

    images = np.load(data_path)   # (50000, 32, 32, 3) uint8
    labels = np.load(labels_path) # (50000,)

    start = (severity - 1) * 10000
    images = images[start : start + 10000]  # (10000, 32, 32, 3)
    labels = labels[start : start + 10000]  # (10000,)

    # HWC uint8 â†’ CHW float32 in [0,1], then normalise
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_params.mean, data_params.std),
    ])

    tensors = torch.stack([tf(img) for img in images])
    label_t = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(tensors, label_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


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


def train_one_epoch_kd(
    student:      nn.Module,
    teacher:      nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    temperature:  float,
    alpha:        float,
    l1_lambda:    float,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one KD training epoch (Hinton et al., 2015).

    Loss = alpha * TÂ² * KL(softmax(s/T) || softmax(t/T))
         + (1 - alpha) * CE(s, y_hard)

    The TÂ² scaling ensures gradient magnitudes stay balanced when T changes.

    Args:
        student:      Student network being trained.
        teacher:      Frozen teacher network providing soft targets.
        loader:       Training DataLoader.
        optimizer:    Optimiser instance.
        criterion:    Hard-label loss (CrossEntropyLoss).
        device:       Computation device.
        temperature:  Distillation temperature T (> 1 softens distributions).
        alpha:        Weight on soft KD loss; ``(1-alpha)`` weights hard CE loss.
        l1_lambda:    L1 regularisation coefficient (0 = disabled).
        log_interval: Batches between progress prints.

    Returns:
        Tuple of ``(mean_loss, accuracy)`` over the epoch.
    """
    student.train()
    teacher.eval()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            t_logits = teacher(imgs)

        s_logits = student(imgs)

        # Soft loss: KL(student_soft || teacher_soft), scaled by TÂ²
        soft_loss = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

        # Hard loss: standard cross-entropy at T=1
        hard_loss = criterion(s_logits, labels)

        loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in student.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += s_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f" {batch_idx+1:>4}/{len(loader)} | {total_loss/n:>7.4f} | {correct/n:>6.3f} |       - |      -")

    return total_loss / n, correct / n


def train_one_epoch_teacher_prob(
    student:      nn.Module,
    teacher:      nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
    l1_lambda:    float,
    log_interval: int,
) -> Tuple[float, float]:
    """Run one training epoch using teacher-probability dynamic label smoothing.

    Combines label smoothing with knowledge distillation: the teacher's softmax
    confidence on the true class becomes a per-sample smoothing weight.

    For each sample the soft target distribution is:
        - true class  â†’ p_teacher(y_true)
        - other classes â†’ (1 - p_teacher(y_true)) / (C - 1)  equally

    Loss = CE(student_logits, soft_target)

    High teacher confidence â†’ near one-hot target (easy sample).
    Low teacher confidence  â†’ spread distribution (hard sample).
    No temperature or alpha â€” the teacher's raw softmax IS the signal.

    Args:
        student:      Student network being trained.
        teacher:      Frozen teacher network providing per-sample confidence.
        loader:       Training DataLoader.
        optimizer:    Optimiser instance.
        device:       Computation device.
        l1_lambda:    L1 regularisation coefficient (0 = disabled).
        log_interval: Batches between progress prints.

    Returns:
        Tuple of ``(mean_loss, accuracy)`` over the epoch.
    """
    student.train()
    teacher.eval()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            t_probs = F.softmax(teacher(imgs), dim=1)           # [B, C] at T=1
            p_true  = t_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]

        # Build soft target: (1-p_true)/(C-1) everywhere, then set true class
        C            = t_probs.size(1)
        soft_targets = ((1.0 - p_true) / (C - 1)).unsqueeze(1).expand(-1, C).clone()
        soft_targets.scatter_(1, labels.unsqueeze(1), p_true.unsqueeze(1))

        s_logits = student(imgs)
        loss     = -(soft_targets * F.log_softmax(s_logits, dim=1)).sum(dim=1).mean()

        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in student.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += s_logits.argmax(1).eq(labels).sum().item()
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

    When ``warmup_epochs > 0`` and scheduler is ``'cosine'``, returns a
    ``SequentialLR`` that linearly ramps LR from 10% to 100% over the warmup
    period, then applies cosine annealing for the remaining epochs.

    Args:
        optimizer:       The optimiser whose LR will be scheduled.
        training_params: Training parameters (scheduler type, epochs, warmup).

    Returns:
        An ``LRScheduler`` instance, or ``None`` if scheduler is ``'none'``.
    """
    if training_params.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if training_params.scheduler == "cosine":
        warmup = training_params.warmup_epochs
        if warmup > 0:
            warmup_sched  = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup,
            )
            cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=training_params.epochs - warmup,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup],
            )
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
    teacher:         Optional[nn.Module] = None,
) -> None:
    """Full training loop with early stopping, LR scheduling, and best-model saving.

    When ``teacher`` is provided and ``training_params.distill`` is ``True``,
    each epoch uses Hinton KD loss instead of plain cross-entropy.

    Args:
        model:           The neural network to train.
        data_params:     Dataset parameters.
        model_params:    Model architecture parameters.
        training_params: Training hyperparameters.
        device:          Computation device.
        config_title:    Human-readable experiment title for logging/plotting.
        logger:          TrainLogger instance.
        teacher:         Optional frozen teacher model for knowledge distillation.
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
        if training_params.distill and teacher is not None:
            if training_params.distill_mode == "teacher_prob":
                tr_loss, tr_acc = train_one_epoch_teacher_prob(
                    student      = model,
                    teacher      = teacher,
                    loader       = train_loader,
                    optimizer    = optimizer,
                    device       = device,
                    l1_lambda    = training_params.l1_lambda,
                    log_interval = training_params.log_interval,
                )
            else:  # hinton
                tr_loss, tr_acc = train_one_epoch_kd(
                    student      = model,
                    teacher      = teacher,
                    loader       = train_loader,
                    optimizer    = optimizer,
                    criterion    = criterion,
                    device       = device,
                    temperature  = training_params.temperature,
                    alpha        = training_params.alpha,
                    l1_lambda    = training_params.l1_lambda,
                    log_interval = training_params.log_interval,
                )
        else:
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
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir="results/resnet", title=config_title)


def run_augmix_training(
    model:           nn.Module,
    data_params:     DataParams,
    model_params,
    training_params: TrainingParams,
    device:          torch.device,
    config_title:    str        = "",
    logger:          TrainLogger = None,
) -> None:
    """Full training loop using AugMix augmentation and JSD consistency loss.

    Mirrors ``run_training`` but uses ``get_augmix_loaders`` to produce three
    views per sample and ``train_one_epoch_augmix`` for the per-epoch update.
    The best checkpoint is saved to ``training_params.augmix_save_path`` to
    keep it separate from the vanilla fine-tuned model.

    Args:
        model:           The neural network to train.
        data_params:     Dataset parameters.
        model_params:    Model architecture parameters.
        training_params: Training hyperparameters (includes jsd_lambda, augmix_save_path).
        device:          Computation device.
        config_title:    Human-readable experiment title for logging/plotting.
        logger:          TrainLogger instance.
    """
    train_loader, val_loader = get_augmix_loaders(data_params, training_params, model_params.transfer_mode)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = training_params.learning_rate,
        weight_decay = training_params.weight_decay,
    )
    scheduler = build_scheduler(optimizer, training_params)

    best_acc     = 0.0
    best_weights = None
    patience_ctr = 0

    train_losses: List[float] = []
    val_losses:   List[float] = []
    train_accs:   List[float] = []
    val_accs:     List[float] = []

    logger.log_start(model, data_params, model_params, training_params, device)

    for epoch in range(1, training_params.epochs + 1):
        tr_loss, tr_acc = train_one_epoch_augmix(
            model        = model,
            loader       = train_loader,
            optimizer    = optimizer,
            criterion    = criterion,
            device       = device,
            jsd_lambda   = training_params.jsd_lambda,
            log_interval = training_params.log_interval,
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
            torch.save(best_weights, training_params.augmix_save_path)
            logger.log_best(best_acc, training_params.augmix_save_path)
            patience_ctr = 0
        else:
            patience_ctr += 1

        if training_params.patience > 0 and patience_ctr >= training_params.patience:
            logger._w(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(patience={training_params.patience})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    logger.log_complete(best_acc, training_params.augmix_save_path)
    logger.close()

    if training_params.plot:
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir="results/resnet", title=config_title)

