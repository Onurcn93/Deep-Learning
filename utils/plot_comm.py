"""
utils/plot_comm.py — Plotting utilities for the Transformer communication system.

Functions
---------
plot_loss_curves    — training and validation loss over epochs
plot_ser_mer_curves — Symbol Error Rate + Message Error Rate over epochs
"""

import os
import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: list,
    val_losses:   list,
    save_dir:     str,
    title:        str = 'Comm System — Training Loss',
) -> None:
    """Plot train and val cross-entropy loss curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label='Train Loss', color='steelblue',  linewidth=1.8)
    ax.plot(epochs, val_losses,   label='Val Loss',   color='tomato',     linewidth=1.8)

    ax.set_xlabel('Epoch',             fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title(title,                fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'loss_curves.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_ser_mer_curves(
    val_sers:  list,
    val_mers:  list,
    save_dir:  str,
    title:     str = 'Comm System — Error Rates',
) -> None:
    """
    Plot Symbol Error Rate and Message Error Rate over validation epochs.
    Includes a dashed baseline at random-chance SER = 7/8 = 0.875.
    """
    epochs = range(1, len(val_sers) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_sers, label='Symbol Error Rate (SER)',  color='steelblue', linewidth=1.8)
    ax.plot(epochs, val_mers, label='Message Error Rate (MER)', color='tomato',    linewidth=1.8)
    ax.axhline(
        y=7 / 8, color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
        label='Random-chance SER = 0.875',
    )

    ax.set_xlabel('Epoch',       fontsize=12)
    ax.set_ylabel('Error Rate',  fontsize=12)
    ax.set_title(title,          fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'ser_mer_curves.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')
