"""
models/GRU.py — StockGRU for financial return forecasting (HW4 sub-task b/c/d).

Architecture (PDF spec):
  Stacked GRU layers  →  Dropout  →  FC output

GRU update equations (from PDF):
  z_t = σ(W_z [h_{t-1}, x_t] + b_z)          update gate
  r_t = σ(W_r [h_{t-1}, x_t] + b_r)          reset gate
  h̃_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t] + b_h) candidate hidden state
  h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t    new hidden state

Input  : (batch, T, F)   T=20 time steps, F=4 OHLC features
Output : (batch, D)      D=5 return-ratio predictions  [regression]
         (batch, 1)      binary buy/pass logit          [turning-point]
"""

import torch
import torch.nn as nn


class StockGRU(nn.Module):
    """
    Stacked GRU followed by dropout and a fully-connected head.

    Mirrors StockLSTM exactly in interface — swap via --model gru at runtime.

    Parameters
    ----------
    input_size  : number of input features per time step  (default 4 — OHLC)
    hidden_size : GRU hidden dimension                    (default 128)
    num_layers  : number of stacked GRU layers            (default 2)
    dropout     : dropout probability after last GRU layer and between layers
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

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = gru_dropout,
            batch_first = True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T, F)
        returns : (batch, output_size)
        """
        # out: (batch, T, hidden_size)
        # h_n: (num_layers, batch, hidden_size)  — GRU has no cell state
        out, _ = self.gru(x)

        out = out[:, -1, :]           # last time step: (batch, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)            # (batch, output_size)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f'StockGRU(hidden={self.hidden_size}, layers={self.num_layers}, '
            f'params={self.count_parameters():,})'
        )
