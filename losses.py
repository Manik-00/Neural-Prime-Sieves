# losses.py
# ─────────────────────────────────────────────────────────────────
# Loss functions for multi-label prime family prediction.
#   weighted_bce    — inverse-frequency class weights
#   FocalLoss       — Lin et al. 2017 (known collapse on sparse families)
#   AsymmetricLoss  — Ridnik et al. 2021 (decoupled pos/neg focusing)
# ─────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from config import device


def compute_pos_weights(Y):
    """Per-class positive weight = neg_count / pos_count, on device."""
    pos = Y.sum(axis=0).clip(min=1)
    neg = len(Y) - pos
    return torch.tensor(neg / pos, dtype=torch.float32).to(device)


def weighted_bce(preds, targets, pw):
    """Frequency-weighted BCE: up-weights rare positive classes.

    Each positive example for class i is weighted by pw[i];
    negative examples receive weight 1.
    """
    loss = nn.functional.binary_cross_entropy(preds, targets, reduction="none")
    w    = targets * pw + (1 - targets)
    return (loss * w).mean()


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017, ICCV).

    Applies a shared (1-p_t)^gamma modulation to both positives and
    negatives.  Known failure mode for rare algebraic prime families:
    the hard-positive gradient is suppressed as aggressively as the
    easy-negative gradient, causing minority-class recall collapse.

    Parameters
    ----------
    alpha : float  Weighting factor (default 0.25).
    gamma : float  Focusing exponent (default 2.0).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ridnik et al. 2021, ICCV).

    Decouples the focusing exponent for positives (gamma_pos) and negatives
    (gamma_neg), plus a probability margin shift that zeros gradient
    contributions from very easy negatives.

    gamma_pos = 0  → no suppression of hard positives
    gamma_neg = 4  → strong down-weighting of easy negatives
    clip      = 0.05 → predicted negatives below 0.05 contribute no loss

    Parameters
    ----------
    gamma_neg : float  Focusing exponent for negatives (default 4).
    gamma_pos : float  Focusing exponent for positives (default 0).
    clip      : float  Probability margin shift (default 0.05).
    eps       : float  Numerical stability floor (default 1e-8).
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, preds, targets):
        xs_pos = preds.clamp(min=self.eps)
        lo_pos = targets * (1.0 - xs_pos) ** self.gamma_pos * torch.log(xs_pos)

        xs_neg = (preds - self.clip).clamp(min=0.0)
        lo_neg = (1.0 - targets) * xs_neg ** self.gamma_neg * \
                 torch.log((1.0 - xs_neg).clamp(min=self.eps))

        return -(lo_pos + lo_neg).mean()
