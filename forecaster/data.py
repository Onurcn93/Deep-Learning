"""
forecaster/data.py — Stock market dataset for HW4 financial forecasting.

Downloads OHLC data from yfinance, applies per-stock z-score normalisation
(fit on train split only), and builds sliding-window input-output pairs.

Tensor shapes
─────────────
  x : (T, F)   — normalised OHLC features, T=20, F=4
  y : (D,)     — d-day return ratios for d=1..D=5  [target_mode='return'|'rolling']
    OR float   — binary buy(1)/pass(0) label        [target_mode='turning']

Splits (chronological, split boundary never shuffled)
──────────────────────────────────────────────────────
  train : 2020-01-01 – 2024-07-31
  val   : 2024-08-01 – 2024-12-31
  test  : 2025-01-01 – 2025-12-31

Shuffling note
──────────────
We use stateless LSTM/GRU (hidden state reset to zero for every window).
Each (x, y) window is therefore an independent sample, so shuffle=True
in the train DataLoader is correct and beneficial — it breaks the heavy
overlap between consecutive windows (19/20 days shared) that would
otherwise produce correlated gradients within each batch.
Temporal ordering WITHIN each window (the T=20 days) is always preserved.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL']

# PDF-specified features: Open, High, Low, Close  (F = 4)
FEATURES  = ['Open', 'High', 'Low', 'Close']
CLOSE_IDX = 3   # column position of Close in OHLC array
HIGH_IDX  = 1   # column position of High  in OHLC array

# Chronological split boundaries (inclusive on both ends)
SPLIT_DATES = {
    'train': ('2020-01-01', '2024-07-31'),
    'val'  : ('2024-08-01', '2024-12-31'),
    'test' : ('2025-01-01', '2025-12-31'),
}

DOWNLOAD_START = '2020-01-01'
# A few weeks past Dec 2025 so D=5 targets are available for the last test windows
DOWNLOAD_END   = '2026-01-20'


# ── Download + cache ─────────────────────────────────────────────────────────

def _download(ticker: str, cache_dir: str):
    """
    Download OHLC for one ticker; cache raw arrays to avoid repeated API calls.

    Returns
    -------
    ohlc  : np.ndarray  (N, 4)  float32  columns = [Open, High, Low, Close]
    dates : np.ndarray  (N,)    str      'YYYY-MM-DD'
    """
    os.makedirs(cache_dir, exist_ok=True)
    ohlc_path  = os.path.join(cache_dir, f'{ticker}_ohlc.npy')
    dates_path = os.path.join(cache_dir, f'{ticker}_dates.npy')

    if os.path.exists(ohlc_path) and os.path.exists(dates_path):
        ohlc  = np.load(ohlc_path)
        dates = np.load(dates_path, allow_pickle=True)
        return ohlc, dates

    import yfinance as yf
    df = yf.download(
        ticker,
        start       = DOWNLOAD_START,
        end         = DOWNLOAD_END,
        auto_adjust = True,    # adjusts prices for splits and dividends
        progress    = False,
    )

    # yfinance ≥ 0.2 can return a MultiIndex; flatten to single level
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    df    = df[FEATURES].dropna()
    ohlc  = df.values.astype(np.float32)                    # (N, 4)
    dates = np.array([str(d.date()) for d in df.index])     # (N,) 'YYYY-MM-DD'

    np.save(ohlc_path,  ohlc)
    np.save(dates_path, dates)

    print(f'  [{ticker}] downloaded {len(ohlc)} trading days '
          f'({dates[0]} to {dates[-1]}), cached to {cache_dir}/')
    return ohlc, dates


# ── Dataset ──────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    """
    Sliding-window dataset over multi-stock daily OHLC data.

    Parameters
    ----------
    tickers     : ticker symbols; each stock contributes its own windows
    T           : lookback window length in trading days   (default 20)
    D           : number of return horizons  d = 1 .. D   (default 5)
    split       : 'train' | 'val' | 'test'
    target_mode : 'return'  — exact d-day return ratios            (sub-task b)
                  'rolling' — weighted rolling-average return       (sub-task c)
                  'turning' — binary buy/pass label via max price   (sub-task d)
    rolling_l   : rolling window l, used when target_mode='rolling' (default 3)
    gamma       : buy threshold as a return ratio; 0.10 = 10% gain
                  used when target_mode='turning'                   (default 0.10)
    cache_dir   : directory for raw OHLC cache files
    """

    def __init__(
        self,
        tickers     = DEFAULT_TICKERS,
        T           = 20,
        D           = 5,
        split       = 'train',
        target_mode = 'return',
        rolling_l   = 3,
        gamma       = 0.10,
        cache_dir   = 'forecaster/cache',
    ):
        assert split in ('train', 'val', 'test'), \
            f"split must be 'train'|'val'|'test', got '{split}'"
        assert target_mode in ('return', 'rolling', 'turning'), \
            f"target_mode must be 'return'|'rolling'|'turning', got '{target_mode}'"

        self.T           = T
        self.D           = D
        self.split       = split
        self.target_mode = target_mode
        self.rolling_l   = rolling_l
        self.gamma       = gamma

        self.xs = []    # list of FloatTensor (T, F=4)
        self.ys = []    # list of FloatTensor (D,) or scalar

        for ticker in tickers:
            self._process_ticker(ticker, cache_dir)

    # ── Per-ticker processing ─────────────────────────────────────────────────

    def _process_ticker(self, ticker: str, cache_dir: str) -> None:
        ohlc, dates = _download(ticker, cache_dir)
        N = len(ohlc)

        # ── 1. Fit z-score normaliser on TRAIN data only ──────────────────────
        #    This prevents any information from val/test leaking into normalisation.
        train_end  = SPLIT_DATES['train'][1]          # '2024-07-31'
        train_mask = dates <= train_end
        train_ohlc = ohlc[train_mask]

        mean = train_ohlc.mean(axis=0)                # (4,) per-feature mean
        std  = train_ohlc.std(axis=0)                 # (4,) per-feature std
        std[std == 0] = 1.0                           # guard: avoid division by zero

        # ── 2. Normalise ALL data with the train statistics ───────────────────
        #    Targets are always computed from RAW prices (below), never normalised.
        norm_ohlc = (ohlc - mean) / std               # (N, 4)

        # ── 3. Raw prices needed for target computation ───────────────────────
        raw_close = ohlc[:, CLOSE_IDX]               # (N,)  for return ratios
        raw_high  = ohlc[:, HIGH_IDX]                # (N,)  for turning-point max

        # ── 4. Identify which days fall inside this split ─────────────────────
        #    A window "belongs" to a split when its LAST INPUT DAY is in that range.
        s_start, s_end = SPLIT_DATES[self.split]
        split_mask = (dates >= s_start) & (dates <= s_end)

        # ── 5. Build sliding windows ──────────────────────────────────────────
        #    Window i: input  = norm_ohlc[i : i+T]     last input day = i+T-1
        #              target = f(raw prices at days  i+T .. i+T+D)
        #
        #    Valid range: last valid i satisfies  i+T-1+D < N
        #                 i.e.  i <= N - T - D
        for i in range(N - self.T - self.D + 1):
            last_in = i + self.T - 1              # index of last input day

            if not split_mask[last_in]:
                continue

            x   = norm_ohlc[i : i + self.T]       # (T, 4)  normalised OHLC
            p_t = float(raw_close[last_in])        # raw close at last input day

            y = self._make_target(raw_close, raw_high, last_in, p_t)

            self.xs.append(torch.tensor(x, dtype=torch.float32))
            self.ys.append(torch.tensor(y, dtype=torch.float32))

    def _make_target(
        self,
        raw_close : np.ndarray,
        raw_high  : np.ndarray,
        t         : int,      # index of last input day in the full series
        p_t       : float,    # raw close price at day t
    ) -> np.ndarray:
        """Return the target array or scalar for one window."""

        if self.target_mode == 'return':
            # y[d-1] = (close[t+d] - close[t]) / close[t]   for d = 1..D
            return np.array(
                [(raw_close[t + d] - p_t) / p_t for d in range(1, self.D + 1)],
                dtype=np.float32,
            )

        elif self.target_mode == 'rolling':
            # PDF formula: ŷ[d-1] = ( Σ_{j=0}^{l} w_j · close[t+d-j] − close[t] ) / close[t]
            #
            # j=0  → future price  close[t+d]           (needs data ahead)
            # j>0  → prices at     close[t+d-j]         (j > d → past; j ≤ d → future)
            # All indices stay within [0, N) given our window validity bounds.
            # Weights: uniform  w_j = 1/(l+1)
            l       = self.rolling_l
            weights = np.ones(l + 1, dtype=np.float32) / (l + 1)
            targets = []
            for d in range(1, self.D + 1):
                avg = sum(
                    float(weights[j]) * float(raw_close[t + d - j])
                    for j in range(l + 1)
                )
                targets.append((avg - p_t) / p_t)
            return np.array(targets, dtype=np.float32)

        else:  # 'turning'
            # Buy (1) if the HIGH price on any day t+d gives a return > gamma.
            #
            # PDF: γ=1.1 applied to r=(pmax−p_t)/p_t.  A 110% gain in 5 trading
            # days is essentially never observed; the intended interpretation is a
            # 10% gain (price ratio pmax/p_t > 1.1), so we use gamma=0.10 as
            # the return-ratio threshold. Exposed as --gamma in parameters.py.
            for d in range(1, self.D + 1):
                r = (float(raw_high[t + d]) - p_t) / p_t
                if r > self.gamma:
                    return np.float32(1.0)
            return np.float32(0.0)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]

    def __repr__(self) -> str:
        n = len(self)
        extra = ''
        if self.target_mode == 'turning' and n > 0:
            labels  = torch.stack(self.ys)
            buy_pct = 100.0 * labels.sum().item() / n
            extra   = f', buy={buy_pct:.1f}%'
        return (
            f'StockDataset(split={self.split}, n={n}, '
            f'T={self.T}, D={self.D}, mode={self.target_mode}{extra})'
        )


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_loaders(
    tickers     = DEFAULT_TICKERS,
    T           = 20,
    D           = 5,
    target_mode = 'return',
    rolling_l   = 3,
    gamma       = 0.10,
    batch_size  = 64,
    num_workers = 0,           # 0 is safest on Windows (avoids multiprocessing issues)
    cache_dir   = 'forecaster/cache',
):
    """
    Build and return (train_loader, val_loader, test_loader).

    train_loader : shuffle=True   — windows are i.i.d. given stateless LSTM
    val_loader   : shuffle=False  — deterministic evaluation order
    test_loader  : shuffle=False  — deterministic evaluation order
    """
    common = dict(
        tickers     = tickers,
        T           = T,
        D           = D,
        target_mode = target_mode,
        rolling_l   = rolling_l,
        gamma       = gamma,
        cache_dir   = cache_dir,
    )

    print(f'\nBuilding datasets  [mode={target_mode}, T={T}, D={D}]')
    train_ds = StockDataset(**common, split='train')
    val_ds   = StockDataset(**common, split='val')
    test_ds  = StockDataset(**common, split='test')

    print(f'  {train_ds}')
    print(f'  {val_ds}')
    print(f'  {test_ds}')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
