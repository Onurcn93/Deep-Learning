"""
forecaster/parameters.py — argparse for HW4 financial forecasting.
"""

import argparse
from forecaster.data import DEFAULT_TICKERS


def get_args():
    p = argparse.ArgumentParser(
        description='HW4: Financial Forecasting with LSTM / GRU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Sub-task ──────────────────────────────────────────────────────────────
    p.add_argument(
        '--task', choices=['b', 'c', 'd'], default='b',
        help='Sub-task: b=exact returns, c=rolling-avg returns, d=turning-point',
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                   help='S&P 500 ticker symbols to use')
    p.add_argument('--T', type=int, default=20,
                   help='Lookback window length (trading days)')
    p.add_argument('--D', type=int, default=5,
                   help='Number of forecast horizons (d=1..D)')
    p.add_argument('--rolling_l', type=int, default=3,
                   help='Rolling-average window l (sub-task c only)')
    p.add_argument('--gamma', type=float, default=0.10,
                   help='Buy-signal threshold as return ratio, e.g. 0.10 = 10%% gain '
                        '(sub-task d only)')
    p.add_argument('--cache_dir', default='forecaster/cache',
                   help='Directory for cached yfinance downloads')

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument('--model', choices=['lstm', 'gru'], default='lstm',
                   help='Recurrent cell type')
    p.add_argument('--hidden_size', type=int, default=128,
                   help='Hidden dimension of each recurrent layer')
    p.add_argument('--num_layers', type=int, default=2,
                   help='Number of stacked recurrent layers')
    p.add_argument('--dropout', type=float, default=0.2,
                   help='Dropout probability (inter-layer + pre-FC)')

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='AdamW learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='AdamW weight decay')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patience', type=int, default=15,
                   help='Early-stopping patience (epochs without val improvement)')
    p.add_argument('--device', default='auto',
                   help='Device: auto | cpu | cuda | mps')

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument('--save_path', default='weights/stock_best.pth',
                   help='Path to save the best model checkpoint')
    p.add_argument('--log', action=argparse.BooleanOptionalAction, default=True,
                   help='Write training log to logs/')
    p.add_argument('--plot', action='store_true',
                   help='Save loss curve and metric plots to results/<model>/')

    return p.parse_args()
