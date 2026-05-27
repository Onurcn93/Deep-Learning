"""comm/parameters.py — Argparse configuration for the CommSystem pipeline."""

import argparse


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Interactive AWGN Communication Protocol — Transformer (HW4 Part 2)'
    )

    # ── Model architecture ────────────────────────────────────────────────────
    p.add_argument('--d_model',    type=int,   default=64,
                   help='Transformer model dimension (default: 64)')
    p.add_argument('--num_layers', type=int,   default=2,
                   help='Number of Transformer layers in TX and RX (default: 2)')
    p.add_argument('--num_heads',  type=int,   default=4,
                   help='Number of attention heads (default: 4; head_dim=d_model/num_heads)')
    p.add_argument('--ffn_dim',    type=int,   default=128,
                   help='FFN hidden dimension inside Transformer (default: 128 = 2×d_model)')
    p.add_argument('--d_embed',    type=int,   default=16,
                   help='Learnable symbol embedding dimension (default: 16)')
    p.add_argument('--dropout',    type=float, default=0.1,
                   help='Dropout in Transformer layers (default: 0.1)')

    # ── Communication protocol ────────────────────────────────────────────────
    p.add_argument('--T',      type=int,   default=4,
                   help='Number of interactive communication rounds (default: 4)')
    p.add_argument('--sigma2', type=float, default=0.25,
                   help='AWGN noise variance σ² (default: 0.25)')

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument('--batch_size',    type=int,   default=256,
                   help='Messages per batch (default: 256)')
    p.add_argument('--epochs',        type=int,   default=100,
                   help='Maximum training epochs (default: 100)')
    p.add_argument('--lr',            type=float, default=1e-3,
                   help='Adam learning rate (default: 1e-3)')
    p.add_argument('--patience',      type=int,   default=20,
                   help='Early-stopping patience in epochs (default: 20)')
    p.add_argument('--train_batches', type=int,   default=200,
                   help='Random batches per training epoch (default: 200)')
    p.add_argument('--val_batches',   type=int,   default=50,
                   help='Random batches per validation epoch (default: 50)')
    p.add_argument('--test_batches',  type=int,   default=200,
                   help='Random batches for final test evaluation (default: 200)')

    # ── System ────────────────────────────────────────────────────────────────
    p.add_argument('--device',    type=str,  default='auto',
                   help='Device: auto | cpu | cuda | mps (default: auto)')
    p.add_argument('--seed',      type=int,  default=42,
                   help='Random seed (default: 42)')
    p.add_argument('--save_path', type=str,  default='weights/comm_best.pth',
                   help='Path to save best model checkpoint')
    p.add_argument('--log',       action='store_true',  default=True,
                   help='Save training log to logs/comm.log (default: on)')
    p.add_argument('--no-log',    dest='log', action='store_false')
    p.add_argument('--plot',      action='store_true',  default=False,
                   help='Save loss and SER/MER curves to results/comm/')

    return p.parse_args()
