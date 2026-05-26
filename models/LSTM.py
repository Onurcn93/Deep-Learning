"""
models/LSTM.py — StockLSTM for financial return forecasting (HW4 sub-task b/c/d).

Architecture (PDF spec):
  Stacked LSTM layers  →  Dropout  →  FC output

Input  : (batch, T, F)   T=20 time steps, F=4 OHLC features
Output : (batch, D)      D=5 return-ratio predictions  [regression]
         (batch, 1)      binary buy/pass logit          [turning-point]
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Stacked LSTM followed by dropout and a fully-connected head.

    Parameters
    ----------
    input_size  : number of input features per time step  (default 4 — OHLC)
    hidden_size : LSTM hidden dimension                    (default 128)
    num_layers  : number of stacked LSTM layers            (default 2)
    dropout     : dropout probability after last LSTM layer and between layers
                  (inter-layer dropout is disabled when num_layers=1)
    output_size : number of output values                  (default 5 — D horizons)
    """

    def __init__(
        self,
        input_size  : int   = 4,
        hidden_size : int   = 128,
        num_layers  : int   = 2,
        dropout     : float = 0.2,
        output_size : int   = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # inter-layer dropout only makes sense when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = lstm_dropout,
            batch_first = True,      # expects (batch, T, F)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T, F)
        returns : (batch, output_size)
        """
        # out: (batch, T, hidden_size)  —  full sequence of hidden states
        # _  : (h_n, c_n)              —  final hidden and cell states
        out, _ = self.lstm(x)

        # Take only the last time step's output — summary of the T-day window
        out = out[:, -1, :]           # (batch, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)            # (batch, output_size)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f'StockLSTM(hidden={self.hidden_size}, layers={self.num_layers}, '
            f'params={self.count_parameters():,})'
        )
