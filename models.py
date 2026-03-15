# models.py
# ─────────────────────────────────────────────────────────────────
# Neural network architectures.
#   PrimeFamilyNet  — deep residual MLP with per-family sigmoid heads
#   ShallowBaseline — 2-layer MLP (depth ablation)
# ─────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from config import FEAT_DIM_CAUSAL, N_FAM


class ResidualBlock(nn.Module):
    """Single residual block: Linear→LayerNorm→GELU→Dropout→Linear→LayerNorm + skip."""
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)


class PrimeFamilyNet(nn.Module):
    """Multi-head residual MLP.

    Shared trunk (two residual blocks + two narrowing projections) learns
    representations common to all prime families. Seven independent sigmoid
    heads produce one membership probability per family.

    Parameters
    ----------
    input_dim : int   Feature dimensionality (25 causal, 29 non-causal).
    hidden    : int   Width of the two residual blocks (default 512).
    n_heads   : int   Number of output heads (default N_FAM = 7).
    dropout   : float Dropout rate inside each residual block.
    """
    def __init__(self, input_dim=FEAT_DIM_CAUSAL, hidden=512,
                 n_heads=None, dropout=0.15):
        super().__init__()
        if n_heads is None:
            n_heads = N_FAM

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4), nn.LayerNorm(hidden // 4), nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden // 4, 32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            ) for _ in range(n_heads)
        ])

    def forward(self, x):
        z = self.trunk(self.input_proj(x))
        return torch.cat([h(z) for h in self.heads], dim=1)

    def trunk_features(self, x):
        """Return the trunk embedding (useful for analysis)."""
        return self.trunk(self.input_proj(x))


class ShallowBaseline(nn.Module):
    """Two-layer MLP baseline — tests whether depth matters.

    Parameters
    ----------
    input_dim : int  Feature dimensionality.
    n_heads   : int  Number of output heads (default N_FAM = 7).
    """
    def __init__(self, input_dim=FEAT_DIM_CAUSAL, n_heads=None):
        super().__init__()
        if n_heads is None:
            n_heads = N_FAM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, n_heads), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
