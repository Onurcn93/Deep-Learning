"""
comm/train.py — Training and validation loops for the CommSystem.

There is no fixed dataset: messages are sampled uniformly at random each batch.
An 'epoch' = args.train_batches random batches of size args.batch_size.
Validation = args.val_batches random batches (no overlap guarantee needed —
  all batches are i.i.d. from the same uniform distribution).

Best model is saved by validation Symbol Error Rate (lower is better).
"""

import sys
import time
import torch
import torch.nn.functional as F

# Force UTF-8 stdout (Windows default CP1254 rejects some Unicode chars)
sys.stdout.reconfigure(encoding='utf-8')


# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str, logger=None) -> None:
    """Print to terminal and optionally write to log file."""
    print(msg)
    if logger is not None:
        logger.log(msg)


def _compute_loss(logits: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Sum of cross-entropy over the 4 symbol positions."""
    return sum(F.cross_entropy(logits[:, i, :], m[:, i]) for i in range(4))


# ── per-epoch functions ───────────────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    train_batches: int,
    device: torch.device,
) -> tuple:
    """
    Returns (avg_loss, ser) over train_batches random batches.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_symbols = 0

    for _ in range(train_batches):
        m = torch.randint(0, 8, (batch_size, 4), device=device)

        optimizer.zero_grad()
        logits = model(m)               # (B, 4, 8)
        loss = _compute_loss(logits, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)   # (B, 4)
        total_correct += (preds == m).sum().item()
        total_symbols += batch_size * 4

    avg_loss = total_loss / train_batches
    ser = 1.0 - total_correct / total_symbols
    return avg_loss, ser


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    batch_size: int,
    val_batches: int,
    device: torch.device,
) -> tuple:
    """
    Returns (avg_loss, ser, mer) over val_batches random batches.
      SER: symbol error rate  (fraction of individual symbols wrong)
      MER: message error rate (fraction of complete 4-symbol messages wrong)
    """
    model.eval()
    total_loss = 0.0
    total_correct_sym = 0
    total_correct_msg = 0
    total_symbols = 0
    total_messages = 0

    for _ in range(val_batches):
        m = torch.randint(0, 8, (batch_size, 4), device=device)
        logits = model(m)

        loss = _compute_loss(logits, m)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)           # (B, 4)
        correct_sym = preds == m                # (B, 4) bool
        total_correct_sym += correct_sym.sum().item()
        total_correct_msg += correct_sym.all(dim=-1).sum().item()
        total_symbols  += batch_size * 4
        total_messages += batch_size

    avg_loss = total_loss / val_batches
    ser = 1.0 - total_correct_sym / total_symbols
    mer = 1.0 - total_correct_msg / total_messages
    return avg_loss, ser, mer


# ── full training loop ────────────────────────────────────────────────────────

def run_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    device: torch.device,
    logger=None,
) -> dict:
    """
    Train for up to args.epochs epochs with early stopping on val SER.
    Saves best checkpoint to args.save_path.
    Returns history dict with loss/SER/MER lists.
    """
    best_val_ser = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_sers, val_sers, val_mers = [], [], []

    # Header
    SEP    = '-' * 72
    HEADER = (
        f"{'Epoch':>6} | {'TrLoss':>8} | {'TrSER':>7} |"
        f" {'VaLoss':>8} | {'VaSER':>7} | {'VaMER':>7} | {'Time':>7}"
    )
    _log('', logger)
    _log(SEP, logger)
    _log(HEADER, logger)
    _log(SEP, logger)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_ser = train_one_epoch(
            model, optimizer,
            args.batch_size, args.train_batches, device,
        )
        va_loss, va_ser, va_mer = validate(
            model,
            args.batch_size, args.val_batches, device,
        )

        elapsed = time.time() - t0

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_sers.append(tr_ser)
        val_sers.append(va_ser)
        val_mers.append(va_mer)

        row = (
            f"{epoch:>6} | {tr_loss:>8.4f} | {tr_ser:>7.4f} |"
            f" {va_loss:>8.4f} | {va_ser:>7.4f} | {va_mer:>7.4f} | {elapsed:>6.1f}s"
        )
        _log(row, logger)

        # Save best model by val SER
        if va_ser < best_val_ser:
            best_val_ser = va_ser
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            _log(f"       ^ New best val SER: {va_ser:.4f} -- saved -> {args.save_path}", logger)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                _log(SEP, logger)
                _log(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs).",
                    logger,
                )
                break

    _log(SEP, logger)
    _log(f"Training complete.  Best val SER: {best_val_ser:.4f}", logger)
    _log('', logger)

    return {
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'train_sers':   train_sers,
        'val_sers':     val_sers,
        'val_mers':     val_mers,
        'best_val_ser': best_val_ser,
    }
