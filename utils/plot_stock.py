"""
utils/plot_stock.py — Plotting utilities for HW4 financial forecasting.

Saves figures to results/<model_name>/ to stay consistent with the rest of the repo.
"""

import os
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe on Windows / headless)
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] saved to {path}')


def plot_loss_curves(
    train_losses : list,
    val_losses   : list,
    save_dir     : str,
    title        : str = 'Training Loss',
) -> None:
    """Line plot of train vs val loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train', linewidth=1.5)
    ax.plot(epochs, val_losses,   label='Val',   linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'loss_curves.png'))


def plot_horizon_mse(
    per_d_mse : list,     # length D, MSE for d=1..D
    save_dir  : str,
    title     : str = 'Per-Horizon MSE (Test)',
) -> None:
    """Bar chart of MSE for each forecast horizon d=1..D."""
    D = len(per_d_mse)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f'd={d}' for d in range(1, D + 1)],
        per_d_mse,
        color='steelblue',
        edgecolor='white',
    )
    # Annotate bars
    for bar, v in zip(bars, per_d_mse):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(per_d_mse) * 0.01,
            f'{v:.5f}', ha='center', va='bottom', fontsize=8,
        )
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, os.path.join(save_dir, 'horizon_mse.png'))


def plot_predictions(
    actuals    : np.ndarray,   # (N, D)
    preds      : np.ndarray,   # (N, D)
    horizon    : int = 1,      # which d to visualise (1-indexed)
    save_dir   : str = '.',
    title      : str = 'Predicted vs Actual Returns',
    n_samples  : int = 300,    # show first N samples to keep chart readable
) -> None:
    """
    Overlay predicted and actual return ratios for one horizon over time.
    Useful for qualitative inspection of forecast quality.
    """
    d_idx = horizon - 1
    act = actuals[:n_samples, d_idx]
    pre = preds  [:n_samples, d_idx]
    xs  = range(len(act))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, act, label='Actual',    linewidth=1.0, alpha=0.8)
    ax.plot(xs, pre, label='Predicted', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Return ratio')
    ax.set_title(f'{title}  (d={horizon})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, f'predictions_d{horizon}.png'))


def plot_all_horizons(
    preds    : np.ndarray,   # (N, D)  — test windows for all stocks concatenated
    labels   : np.ndarray,   # (N, D)
    tickers  : list,         # e.g. ['AAPL', 'MSFT', 'GOOGL']
    save_dir : str,
    title    : str = 'Return Forecasts — All Stocks & Horizons',
) -> None:
    """
    One row per stock, all D horizons overlaid in each row.

    Solid lines  = actual return ratios.
    Dashed lines = model predictions.
    Each horizon gets its own colour; a shared legend sits at the bottom.

    The test windows are stored stock-by-stock (AAPL first, then MSFT, …)
    because the DataLoader uses shuffle=False and _process_ticker runs in
    ticker order — so we can slice evenly.
    """
    n_stocks    = len(tickers)
    n_per_stock = len(preds) // n_stocks
    D           = preds.shape[1]

    # One distinct colour per horizon
    palette = plt.cm.tab10(np.linspace(0, 0.9, D))

    fig, axes = plt.subplots(
        n_stocks, 1,
        figsize = (13, 4 * n_stocks),
        sharex  = False,
    )
    if n_stocks == 1:
        axes = [axes]

    xs = np.arange(n_per_stock)

    from matplotlib.lines import Line2D

    for row, (ticker, ax) in enumerate(zip(tickers, axes)):
        start = row * n_per_stock
        end   = start + n_per_stock
        act   = labels[start:end]   # (n_per_stock, D)
        pre   = preds [start:end]   # (n_per_stock, D)

        # Twin axis: actual (left, full scale) vs predicted (right, its own scale)
        # This makes the predicted curve visible even when its range is much smaller.
        ax_pred = ax.twinx()

        for d in range(D):
            c = palette[d]
            # Actual on primary axis
            ax.plot(xs, act[:, d], color=c, linewidth=0.9, alpha=0.75)
            # Predicted on secondary axis — now fills its own vertical range
            ax_pred.plot(xs, pre[:, d], color=c, linewidth=1.1, alpha=0.9,
                         linestyle='--')

        ax.axhline(0, color='#888888', linewidth=0.5, linestyle=':')
        ax.set_title(ticker, fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual return ratio', color='#333333')
        ax_pred.set_ylabel('Predicted return ratio', color='#555599', rotation=270,
                           labelpad=14)
        ax_pred.tick_params(axis='y', labelcolor='#555599')
        ax.set_xlabel('Test window index')
        ax.grid(True, alpha=0.18)

    # ── Shared legend ──────────────────────────────────────────────────────────
    # Horizon colours
    horizon_handles = [
        Line2D([0], [0], color=palette[d], linewidth=1.2, label=f'd={d+1}')
        for d in range(D)
    ]
    # Style proxies
    style_handles = [
        Line2D([0], [0], color='gray', linewidth=1.2,              label='actual (left axis)'),
        Line2D([0], [0], color='#555599', linewidth=1.2, linestyle='--',
               label='predicted (right axis)'),
    ]
    fig.legend(
        handles = horizon_handles + style_handles,
        labels  = [h.get_label() for h in horizon_handles + style_handles],
        loc     = 'upper center',
        ncol    = D + 2,
        fontsize= 8,
        framealpha = 0.9,
        bbox_to_anchor = (0.5, 1.01),
    )
    fig.suptitle(title, y=1.05, fontsize=12)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'all_horizons.png'))


def plot_confusion_matrix_d(
    tp       : int,
    fp       : int,
    tn       : int,
    fn       : int,
    save_dir : str,
    title    : str = 'Confusion Matrix — Turning-Point Detection',
) -> None:
    """
    2×2 confusion matrix heatmap for sub-task d (binary buy/pass).

    Rows = actual class, Columns = predicted class.
    Each cell shows count and row-normalised percentage.
    """
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    # Row-normalise for colour intensity (so rare positive class is still visible)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row proportion')

    classes = ['Pass (0)', 'Buy (1)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title(title)

    # Annotate each cell with count + percentage
    for r in range(2):
        for c in range(2):
            count = int(cm[r, c])
            pct   = cm_norm[r, c] * 100
            color = 'white' if cm_norm[r, c] > 0.55 else 'black'
            ax.text(c, r, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center', color=color, fontsize=11)

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'confusion_matrix.png'))


def _roc_points(probs: np.ndarray, labels: np.ndarray):
    """Compute ROC curve (FPR, TPR) and AUC via numpy — no sklearn needed."""
    thresholds = np.linspace(1.0, 0.0, 300)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    fprs, tprs = [], []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp_  = ((pred == 1) & (labels == 1)).sum()
        fp_  = ((pred == 1) & (labels == 0)).sum()
        tprs.append(tp_ / (n_pos + 1e-9))
        fprs.append(fp_ / (n_neg + 1e-9))
    fprs = np.array([0.0] + fprs + [1.0])
    tprs = np.array([0.0] + tprs + [1.0])
    auc  = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc


def _pr_points(probs: np.ndarray, labels: np.ndarray):
    """Compute PR curve (Recall, Precision) and Average Precision."""
    thresholds = np.linspace(1.0, 0.0, 300)
    n_pos = labels.sum()
    recalls, precisions = [], []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp_  = ((pred == 1) & (labels == 1)).sum()
        fp_  = ((pred == 1) & (labels == 0)).sum()
        fn_  = ((pred == 0) & (labels == 1)).sum()
        precisions.append(tp_ / (tp_ + fp_ + 1e-9))
        recalls.append(   tp_ / (n_pos     + 1e-9))
    recalls    = np.array([0.0] + recalls    + [1.0])
    precisions = np.array([1.0] + precisions + [float(n_pos / len(labels))])
    # sort by recall for proper integration
    idx        = np.argsort(recalls)
    recalls, precisions = recalls[idx], precisions[idx]
    ap = float(np.trapezoid(precisions, recalls))
    return recalls, precisions, ap


def plot_roc_pr_d(
    probs    : np.ndarray,   # (N,) sigmoid probabilities
    labels   : np.ndarray,   # (N,) int {0,1}
    save_dir : str,
    title    : str = 'ROC & Precision-Recall — Turning-Point Detection',
) -> None:
    """
    Side-by-side ROC curve (left) and Precision-Recall curve (right).

    Both computed from scratch with numpy — no sklearn dependency.
    ROC AUC and Average Precision are annotated on each subplot.
    The random-classifier baseline is shown as a dashed reference line.
    """
    fprs, tprs, auc = _roc_points(probs, labels)
    recs, precs, ap = _pr_points(probs, labels)
    base_rate        = float(labels.mean())   # fraction of positives

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── ROC ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(fprs, tprs, linewidth=2, label=f'LSTM/GRU  (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── PR ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(recs, precs, linewidth=2, color='darkorange',
            label=f'LSTM/GRU  (AP={ap:.3f})')
    ax.axhline(base_rate, color='k', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Random  ({base_rate:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'roc_pr.png'))


def plot_pnl(
    preds    : np.ndarray,   # (N, D)
    labels   : np.ndarray,   # (N, D)
    tickers  : list,         # e.g. ['AAPL', 'MSFT', 'GOOGL']
    save_dir : str,
    horizon  : int  = 1,     # which d to use for trade signals (1-indexed)
    title    : str  = 'Cumulative P&L — Model vs Benchmarks',
) -> None:
    """
    Simulated long-only trading strategy using the model's d=horizon predictions.

    At each test window:
      - Model strategy : if pred > 0, "buy" and earn the actual return; else sit out.
      - Buy & hold     : always invested, earn actual return every step.
      - Oracle         : perfect foresight — only enters when actual > 0.

    Returns are compounded: wealth[t] = wealth[t-1] * (1 + r[t]).
    Starting wealth = 1.0 for all strategies.

    One subplot per stock; all three strategy curves shown together.
    """
    n_stocks    = len(tickers)
    n_per_stock = len(preds) // n_stocks
    d_idx       = horizon - 1   # 0-based column index

    fig, axes = plt.subplots(
        1, n_stocks,
        figsize = (5 * n_stocks, 4),
        sharey  = False,
    )
    if n_stocks == 1:
        axes = [axes]

    for col, (ticker, ax) in enumerate(zip(tickers, axes)):
        start = col * n_per_stock
        end   = start + n_per_stock

        act  = labels[start:end, d_idx]   # (n_per_stock,) actual returns
        pred = preds [start:end, d_idx]   # (n_per_stock,) predicted returns

        # ── Strategy returns at each step ─────────────────────────────────────
        signal        = (pred > 0).astype(np.float32)   # 1=buy, 0=sit-out
        model_daily   = signal * act                     # earn actual if pred>0
        bnh_daily     = act                              # always invested
        oracle_daily  = np.maximum(act, 0.0)            # only positive days

        # ── Compound wealth curves (start at 1.0) ─────────────────────────────
        def _cum(r):
            return np.cumprod(1.0 + r)

        xs = np.arange(1, n_per_stock + 1)
        ax.plot(xs, _cum(model_daily),  label='Model',       linewidth=1.4)
        ax.plot(xs, _cum(bnh_daily),    label='Buy & Hold',  linewidth=1.2,
                linestyle='--', alpha=0.8)
        ax.plot(xs, _cum(oracle_daily), label='Oracle',      linewidth=1.0,
                linestyle=':', alpha=0.7, color='green')
        ax.axhline(1.0, color='gray', linewidth=0.6, linestyle=':')

        # Shade profit / loss region under model curve
        cum_model = _cum(model_daily)
        ax.fill_between(xs, 1.0, cum_model,
                        where=(cum_model >= 1.0), alpha=0.12, color='steelblue',
                        label='_nolegend_')
        ax.fill_between(xs, 1.0, cum_model,
                        where=(cum_model < 1.0),  alpha=0.12, color='tomato',
                        label='_nolegend_')

        final_model  = float(_cum(model_daily)[-1])
        final_bnh    = float(_cum(bnh_daily)[-1])
        ax.set_title(f'{ticker}\nModel {final_model:.3f}x  |  B&H {final_bnh:.3f}x',
                     fontsize=10)
        ax.set_xlabel('Test window index')
        ax.set_ylabel('Wealth (start=1.0)')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f'{title}  (d={horizon})', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, f'pnl_d{horizon}.png'))
