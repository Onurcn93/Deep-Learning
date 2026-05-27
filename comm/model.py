"""
comm/model.py — TX Encoder, RX Decoder, CommSystem.

Architecture (all validated against design checklist):
  - TX: learnable symbol embedding → MLP_pre → Transformer → MLP_post
        → batch-level power normalization (preserves amplitude modulation)
  - Channel: AWGN y = x + ε, noise detached from graph, gradient flows through x
  - Feedback: simple relay f^(t) = y^(t) (no extra network)
  - RX: MLP_pre (per-symbol T observations) → Transformer → MLP_post → logits
  - TX and RX share zero parameters (fully isolated)
  - History updated AFTER transmission to enforce strict causality
"""

import math
import torch
import torch.nn as nn


# ── TX Encoder ─────────────────────────────────────────────────────────────────

class TXEncoder(nn.Module):
    """
    Transmitter encoder. Called once per round t; weights shared across rounds.

    Input:
        m       (B, 4)        message symbol indices in {0..7}
        x_hist  (B, 4, T-1)  previously transmitted coded symbols (zero-padded)
        f_hist  (B, 4, T-1)  previously received feedback = y^(t') (zero-padded)

    Output:
        x  (B, 4)  coded symbols, batch-power-normalised so E[||x||²] ≈ 1
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        d_embed: int = 16,
        T: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T

        # Learnable symbol embedding: {0..7} → R^d_embed
        # Better geometric expressiveness than a hardcoded one-hot vector
        self.embedding = nn.Embedding(8, d_embed)

        # MLP pre-processing:  [embed(mᵢ) | x_hist_i | f_hist_i]  →  d_model
        # Fixed input dim regardless of round: d_embed + 2*(T-1)
        mlp_in = d_embed + 2 * (T - 1)          # = 16 + 6 = 22 for T=4
        self.mlp_pre = nn.Sequential(
            nn.Linear(mlp_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable positional encoding for the 4 symbol positions
        self.pos_enc = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        # Standard post-norm Transformer encoder (matches PDF Eq 1-3)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # MLP post-processing: d_model → 1 coded scalar per symbol position
        self.mlp_post = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        m: torch.Tensor,
        x_hist: torch.Tensor,
        f_hist: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Embed symbol indices: (B, 4, d_embed)
        emb = self.embedding(m)

        # 2. Concatenate embedding + history: (B, 4, d_embed + 2*(T-1))
        raw = torch.cat([emb, x_hist, f_hist], dim=-1)

        # 3. MLP pre: (B, 4, d_model)
        z = self.mlp_pre(raw)

        # 4. Add learnable positional encoding
        z = z + self.pos_enc                     # (B, 4, d_model)

        # 5. Transformer (post-norm, Eq 1-3 from PDF)
        h = self.transformer(z)                  # (B, 4, d_model)

        # 6. MLP post: (B, 4, 1) → (B, 4)
        x = self.mlp_post(h).squeeze(-1)

        # 7. Batch-level power normalisation  E_batch[||x||²] → 1
        #    Per-sample norms may differ → amplitude modulation is preserved.
        #    (Per-sample unit-norm would destroy amplitude info.)
        power = x.pow(2).sum(dim=-1).mean().clamp(min=1e-8)   # scalar
        x = x / power.sqrt()

        return x    # (B, 4)


# ── RX Decoder ─────────────────────────────────────────────────────────────────

class RXDecoder(nn.Module):
    """
    Receiver decoder. Called ONCE at the end when all T received signals are available.

    Input:
        Y  (B, T, 4)  all received signals stacked across rounds

    Output:
        logits  (B, 4, 8)  per-symbol log-probabilities over 8 classes
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        T: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T

        # MLP pre-processing: T received scalars per symbol position → d_model
        # Organise as 4-token sequence (one token per symbol); each token = T obs
        self.mlp_pre = nn.Sequential(
            nn.Linear(T, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Separate learnable PE — zero shared parameters with TX
        self.pos_enc = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        # Separate Transformer — no weights shared with TXEncoder
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=num_layers)

        # MLP post-processing: d_model → 8 logits per symbol
        self.mlp_post = nn.Linear(d_model, 8)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        # Y: (B, T, 4) → transpose → (B, 4, T)
        # Each of the 4 positions now has T observations across rounds
        Y_sym = Y.permute(0, 2, 1)              # (B, 4, T)

        # MLP pre: (B, 4, d_model)
        z = self.mlp_pre(Y_sym)

        # Add PE
        z = z + self.pos_enc                    # (B, 4, d_model)

        # Transformer
        h = self.transformer(z)                 # (B, 4, d_model)

        # MLP post: (B, 4, 8)
        logits = self.mlp_post(h)

        return logits   # (B, 4, 8)


# ── Communication System ───────────────────────────────────────────────────────

class CommSystem(nn.Module):
    """
    End-to-end interactive AWGN communication system.

    Runs T rounds of:
        TX encode  →  power normalise  →  AWGN channel  →  noiseless feedback

    Then RX decodes once from all collected received signals.

    TX and RX are fully isolated nn.Module instances with no shared parameters.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        d_embed: int = 16,
        T: int = 4,
        sigma2: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.sigma = math.sqrt(sigma2)

        # Fully separate modules — zero shared parameters
        self.tx = TXEncoder(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            ffn_dim=ffn_dim, d_embed=d_embed, T=T, dropout=dropout,
        )
        self.rx = RXDecoder(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            ffn_dim=ffn_dim, T=T, dropout=dropout,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pad_history(
        self,
        hist_list: list,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert a list of (B, 4) tensors into (B, 4, T-1), zero-padding future slots.

        Uses torch.stack + torch.cat (no in-place ops) so autograd is unobstructed.
        Gradients flow back through filled slots to earlier encoder calls.
        """
        max_len = self.T - 1
        n = len(hist_list)
        if n == 0:
            return torch.zeros(B, 4, max_len, device=device)

        filled = torch.stack(hist_list, dim=2)          # (B, 4, n)
        if n < max_len:
            pad = torch.zeros(B, 4, max_len - n, device=device)
            filled = torch.cat([filled, pad], dim=2)    # (B, 4, T-1)
        return filled

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        m: (B, 4) message symbol indices in {0..7}
        Returns logits: (B, 4, 8)

        Round ordering enforces strict causality:
            1. Build input from PAST history only
            2. TX encode → x^(t)
            3. Add AWGN (noise detached, gradient flows through x)
            4. Feedback = received signal (noiseless relay)
            5. Append x^(t) and f^(t) to history   ← AFTER use, never before
        """
        B = m.shape[0]
        device = m.device

        hist_x: list = []   # (B, 4) tensors; NOT detached → multi-round backprop
        hist_f: list = []   # feedback tensors

        received: list = []

        for t in range(self.T):
            # --- CAUSAL: read history that contains rounds 0..t-1 only ---
            x_hist = self._pad_history(hist_x, B, device)   # (B, 4, T-1)
            f_hist = self._pad_history(hist_f, B, device)   # (B, 4, T-1)

            # TX encode; weights shared across all rounds
            x = self.tx(m, x_hist, f_hist)                  # (B, 4), power-normalised

            # AWGN channel: noise has requires_grad=False by default (fresh tensor).
            # Explicit .detach() makes the intent unambiguous.
            # Gradient flows: ∂y/∂x = 1 — unobstructed.
            noise = torch.randn_like(x).detach() * self.sigma
            y = x + noise                                    # (B, 4)

            received.append(y)

            # Feedback = noiseless relay of what RX received (Hint 1)
            # Update history AFTER transmission — strict causal constraint
            if t < self.T - 1:
                hist_x.append(x)    # keep in graph for multi-round gradient
                hist_f.append(y)    # feedback = noisy received signal

        # Stack all received signals: (B, T, 4)
        Y = torch.stack(received, dim=1)

        # RX decodes once from all T received signals
        logits = self.rx(Y)          # (B, 4, 8)

        return logits
