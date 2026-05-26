"""
forecaster/run.py — Entry point for HW4 financial forecasting.

Usage examples
--------------
  # Sub-task b — exact returns, LSTM
  python forecaster/run.py --task b --model lstm

  # Sub-task b — exact returns, GRU
  python forecaster/run.py --task b --model gru

  # Sub-task c — rolling-average returns
  python forecaster/run.py --task c --model lstm --rolling_l 3

  # Sub-task d — turning-point detection (bidirectional)
  python forecaster/run.py --task d --model lstm

  # With plots + extended training
  python forecaster/run.py --task b --model lstm --epochs 150 --plot
"""

import os
import sys
import torch

# Allow running from repo root: python forecaster/run.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecaster.parameters import get_args
from forecaster.data       import get_loaders
from forecaster.train      import run_training
from forecaster.test       import run_test


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(args) -> torch.nn.Module:
    """
    Instantiate StockLSTM or StockGRU.

    Sub-task d uses a bidirectional wrapper; output_size=1 for BCE.
    """
    output_size = 1 if args.task == 'd' else args.D

    if args.model == 'lstm':
        from models.LSTM import StockLSTM
        model = StockLSTM(
            input_size  = 4,
            hidden_size = args.hidden_size,
            num_layers  = args.num_layers,
            dropout     = args.dropout,
            output_size = output_size,
        )
    else:
        from models.GRU import StockGRU
        model = StockGRU(
            input_size  = 4,
            hidden_size = args.hidden_size,
            num_layers  = args.num_layers,
            dropout     = args.dropout,
            output_size = output_size,
        )

    # Sub-task d: wrap as bidirectional by rebuilding with bidirectional=True
    if args.task == 'd':
        model = _make_bidirectional(args, output_size)

    return model


def _make_bidirectional(args, output_size: int) -> torch.nn.Module:
    """
    PDF sub-task (d) requires a bidirectional LSTM or GRU.
    We rebuild using the same hyperparams but with bidirectional=True.
    The hidden_size stays the same; the FC input doubles (forward+backward).
    """
    import torch.nn as nn

    class BidiWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_cls = nn.LSTM if args.model == 'lstm' else nn.GRU
            rnn_dropout = args.dropout if args.num_layers > 1 else 0.0
            self.rnn = rnn_cls(
                input_size    = 4,
                hidden_size   = args.hidden_size,
                num_layers    = args.num_layers,
                dropout       = rnn_dropout,
                batch_first   = True,
                bidirectional = True,
            )
            self.dropout = nn.Dropout(args.dropout)
            # bidirectional doubles the output dimension
            self.fc = nn.Linear(args.hidden_size * 2, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)          # (batch, T, hidden*2)
            out = out[:, -1, :]           # last time step
            out = self.dropout(out)
            return self.fc(out)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def __repr__(self):
            direction = 'Bi' + args.model.upper()
            return (
                f'Stock{direction}(hidden={args.hidden_size}, '
                f'layers={args.num_layers}, '
                f'params={self.count_parameters():,})'
            )

    return BidiWrapper()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = _resolve_device(args.device)

    # ── Auto save_path: derive from model+task if user left the default ───────
    #    Prevents task b and task c checkpoints overwriting each other.
    if args.save_path == 'weights/stock_best.pth':
        args.save_path = f'weights/stock_{args.model}_task{args.task}_best.pth'

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = None
    if args.log:
        try:
            from utils.logger import TrainLogger
            tag    = f'stock_{args.model}_task{args.task}'
            logger = TrainLogger(tag)
        except ImportError:
            print('[warn] utils.logger not found — logging to terminal only')

    # ── Data ──────────────────────────────────────────────────────────────────
    task_to_mode = {'b': 'return', 'c': 'rolling', 'd': 'turning'}
    target_mode  = task_to_mode[args.task]

    train_loader, val_loader, test_loader = get_loaders(
        tickers     = args.tickers,
        T           = args.T,
        D           = args.D,
        target_mode = target_mode,
        rolling_l   = args.rolling_l,
        gamma       = args.gamma,
        batch_size  = args.batch_size,
        cache_dir   = args.cache_dir,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    print(f'\nModel  : {model}')
    print(f'Device : {device}')
    print(f'Task   : {args.task} ({target_mode})')

    # ── Train ─────────────────────────────────────────────────────────────────
    train_losses, val_losses = run_training(
        model, train_loader, val_loader, args, logger,
    )

    # ── Load best checkpoint for test ─────────────────────────────────────────
    ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'\nLoaded best checkpoint (epoch {ckpt["epoch"]}, '
          f'val_loss={ckpt["val_loss"]:.6f})')

    # ── Test ──────────────────────────────────────────────────────────────────
    results = run_test(model, test_loader, device, task=args.task, D=args.D, logger=logger)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        from utils.plot_stock import (
            plot_loss_curves, plot_horizon_mse, plot_predictions,
            plot_all_horizons, plot_pnl,
            plot_confusion_matrix_d, plot_roc_pr_d,
        )
        save_dir = os.path.join('results', args.model, f'task{args.task}')
        label    = f'Stock{args.model.upper()} | task={args.task}'

        plot_loss_curves(train_losses, val_losses, save_dir, title=f'{label} Loss')

        if args.task in ('b', 'c'):
            plot_horizon_mse(results['per_d_mse'], save_dir,
                             title=f'{label} Per-Horizon MSE')
            plot_predictions(results['labels'], results['preds'],
                             horizon=1, save_dir=save_dir,
                             title=f'{label} Predictions')
            plot_all_horizons(results['preds'], results['labels'],
                              tickers=args.tickers, save_dir=save_dir,
                              title=f'{label} — All Stocks & Horizons')
            plot_pnl(results['preds'], results['labels'],
                     tickers=args.tickers, save_dir=save_dir, horizon=1,
                     title=f'{label} Cumulative P&L')

        if args.task == 'd':
            plot_confusion_matrix_d(
                results['tp'], results['fp'], results['tn'], results['fn'],
                save_dir, title=f'{label} Confusion Matrix',
            )
            plot_roc_pr_d(
                results['probs'], results['labels_np'],
                save_dir, title=f'{label} ROC & Precision-Recall',
            )


if __name__ == '__main__':
    main()
