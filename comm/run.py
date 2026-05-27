"""
comm/run.py — Entry point for HW4 Part 2: Transformer Communication Protocol.

Usage
-----
  # Default config (T=4, σ²=0.25, d_model=64, 2 layers, 100 epochs)
  python comm/run.py

  # Save plots
  python comm/run.py --plot

  # Custom model size / training length
  python comm/run.py --d_model 128 --num_layers 3 --epochs 150 --batch_size 512

  # Disable logging
  python comm/run.py --no-log
"""

import math
import os
import sys

# Force UTF-8 stdout so Unicode chars render on Windows terminals (CP1254 default)
sys.stdout.reconfigure(encoding='utf-8')

# Allow running from repo root: python comm/run.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from comm.model      import CommSystem
from comm.train      import run_training
from comm.test       import evaluate
from comm.parameters import get_args


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


def _log(msg: str, logger=None) -> None:
    print(msg)
    if logger is not None:
        logger.log(msg)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = get_args()
    device = _resolve_device(args.device)

    # Reproducibility
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Directories
    os.makedirs('weights',      exist_ok=True)
    os.makedirs('results/comm', exist_ok=True)

    # Logger
    logger = None
    if args.log:
        try:
            from utils.logger import TrainLogger
            logger = TrainLogger('comm', enabled=True)
        except ImportError:
            print('[warn] utils.logger not found — terminal output only')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CommSystem(
        d_model    = args.d_model,
        num_heads  = args.num_heads,
        num_layers = args.num_layers,
        ffn_dim    = args.ffn_dim,
        d_embed    = args.d_embed,
        T          = args.T,
        sigma2     = args.sigma2,
        dropout    = args.dropout,
    ).to(device)

    tx_params    = sum(p.numel() for p in model.tx.parameters())
    rx_params    = sum(p.numel() for p in model.rx.parameters())
    total_params = tx_params + rx_params

    # ── Config summary ────────────────────────────────────────────────────────
    SEP = '-' * 72
    header_lines = [
        '',
        SEP,
        '  HW4 Part 2 — Interactive AWGN Communication (Transformer)',
        SEP,
        f'  T={args.T}  σ²={args.sigma2}  d_model={args.d_model}  '
        f'layers={args.num_layers}  heads={args.num_heads}  '
        f'ffn={args.ffn_dim}  d_embed={args.d_embed}',
        f'  batch={args.batch_size}  lr={args.lr}  '
        f'epochs={args.epochs}  patience={args.patience}',
        f'  TX params: {tx_params:,}  |  RX params: {rx_params:,}  '
        f'|  Total: {total_params:,}',
        f'  Device: {device}  |  Seed: {args.seed}',
        f'  Note: Loss in table = sum-of-4 CE (random-chance start ~{4 * math.log(8):.2f})',
        SEP,
        '',
    ]
    for line in header_lines:
        _log(line, logger)

    # ── Train ─────────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = run_training(model, optimizer, args, device, logger)

    # ── Load best checkpoint ──────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(args.save_path, map_location=device, weights_only=True)
    )
    _log(
        f'Loaded best checkpoint from {args.save_path} '
        f'(best val SER: {history["best_val_ser"]:.4f})',
        logger,
    )

    # ── Test ──────────────────────────────────────────────────────────────────
    ser, mer = evaluate(
        model, args.batch_size, args.test_batches, device,
        label='Test', logger=logger,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        from utils.plot_comm import plot_loss_curves, plot_ser_mer_curves
        plot_loss_curves(
            history['train_losses'], history['val_losses'],
            save_dir='results/comm',
        )
        plot_ser_mer_curves(
            history['val_sers'], history['val_mers'],
            save_dir='results/comm',
        )
        _log('Plots saved to results/comm/', logger)

    if logger is not None:
        logger.close()


if __name__ == '__main__':
    main()
