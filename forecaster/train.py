"""
forecaster/train.py — Training loop for HW4 financial forecasting.

Supports:
  sub-task b/c : MSE regression  (target_mode='return' | 'rolling')
  sub-task d   : BCE classification (target_mode='turning')
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Single epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
) -> float:
    """Run one training epoch. Returns mean loss over all batches."""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)

        # BCE expects (batch,) or (batch,1); squeeze for scalar targets
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)

        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> float:
    """Evaluate on val/test loader. Returns mean loss."""
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ── Full training run ─────────────────────────────────────────────────────────

def run_training(
    model       : nn.Module,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    args,
    logger      = None,
) -> tuple:
    """
    Train with early stopping; save best checkpoint by val loss.

    Returns
    -------
    train_losses : list[float]  — per-epoch train loss
    val_losses   : list[float]  — per-epoch val loss
    """
    device = next(model.parameters()).device

    # ── Loss ──────────────────────────────────────────────────────────────────
    task_mode = getattr(args, 'task', 'b')
    if task_mode == 'd':
        # Binary classification — use pos_weight to handle class imbalance
        # (buy signals are rare at ~3% of windows)
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    best_val_loss  = float('inf')
    patience_count = 0
    train_losses   = []
    val_losses     = []

    header = (
        f'\n{"Epoch":>6}  {"Train Loss":>12}  {"Val Loss":>12}  '
        f'{"Best":>12}  {"Time":>7}'
    )
    sep = '-' * len(header)

    def _log(msg):
        print(msg)
        if logger:
            logger.log(msg)

    _log(header)
    _log(sep)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(
                {
                    'epoch'           : epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss'        : best_val_loss,
                    'args'            : vars(args),
                },
                args.save_path,
            )
        else:
            patience_count += 1

        best_marker = f'{best_val_loss:.6f}{"*" if is_best else " "}'
        row = (
            f'{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  '
            f'{best_marker:>12}  {elapsed:>6.1f}s'
        )
        _log(row)

        if patience_count >= args.patience:
            _log(f'\nEarly stopping at epoch {epoch} (patience={args.patience})')
            break

    _log(sep)
    _log(f'Best val loss: {best_val_loss:.6f}  saved to {args.save_path}')

    return train_losses, val_losses
