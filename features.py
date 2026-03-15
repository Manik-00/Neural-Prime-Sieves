# features.py
# ─────────────────────────────────────────────────────────────────
# Causal and non-causal feature vectors, the PrimeDataset wrapper,
# and dataset generation from exact prime sequences.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import torch
from torch.utils.data import Dataset
from sympy import nextprime

from config import (PRIMORIALS, SMALL_PRIMES, FEAT_DIM_CAUSAL,
                    FEAT_DIM_NONCAUSAL)
from families import label_prime


# ── Feature Computation ───────────────────────────────────────────

def compute_features_causal(p, prev_p):
    """25-dim causal feature vector — no access to next prime."""
    f = []
    for m in PRIMORIALS:
        f.append((p % m) / m)
    for sp in SMALL_PRIMES:
        f.append((p % sp) / sp)
    f.append((p - prev_p) / 100.0)
    f += [np.log(p) / 50.0, p.bit_length() / 64.0,
          np.log(np.log(p + 1) + 1) / 5.0]
    f += [(p % 10) / 10.0,
          (sum(int(d) for d in str(p)) % 9) / 9.0,
          len(str(p)) / 20.0]
    f += [(p % 12) / 12.0, (p % 60) / 60.0]
    return np.array(f, dtype=np.float32)


def compute_features_noncausal(p, prev_p, next_p):
    """29-dim non-causal feature vector — includes forward gap g^+."""
    f = []
    for m in PRIMORIALS:
        f.append((p % m) / m)
    for sp in SMALL_PRIMES:
        f.append((p % sp) / sp)
    gm = p - prev_p
    gp = next_p - p
    gt = gm + gp
    f += [gm / 100.0, gp / 100.0, gm / max(gt, 1),
          gt / 100.0, abs(gm - gp) / 100.0]
    f += [np.log(p) / 50.0, p.bit_length() / 64.0,
          np.log(np.log(p + 1) + 1) / 5.0]
    f += [(p % 10) / 10.0,
          (sum(int(d) for d in str(p)) % 9) / 9.0,
          len(str(p)) / 20.0]
    f += [(p % 12) / 12.0, (p % 60) / 60.0]
    return np.array(f, dtype=np.float32)


# Dimension assertions — catch mismatches immediately on import.
assert FEAT_DIM_CAUSAL    == len(compute_features_causal(13, 11)),      "Causal dim mismatch"
assert FEAT_DIM_NONCAUSAL == len(compute_features_noncausal(13,11,17)), "Non-causal dim mismatch"


# ── Dataset Wrapper ───────────────────────────────────────────────

class PrimeDataset(Dataset):
    """Thin PyTorch Dataset wrapping (X, Y) numpy arrays."""
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ── Dataset Generation ────────────────────────────────────────────

def generate_dataset(start, count, tag="", causal=True):
    """Generate `count` labeled primes near `start`.

    Parameters
    ----------
    start  : int   Starting point — primes are drawn from nextprime(start-1).
    count  : int   Number of labeled primes to return.
    tag    : str   Label printed to stdout for progress tracking.
    causal : bool  If True, 25-dim causal features; else 29-dim non-causal.

    Returns
    -------
    X : np.ndarray  shape (count, feat_dim)
    Y : np.ndarray  shape (count, 7)  — multi-label binary matrix
    """
    print(f"  {count:>7,} primes near {start:.2e}  [{tag}]", end=" ", flush=True)

    primes = []
    p      = nextprime(start - 1)
    extra  = count + (1 if causal else 2)
    for _ in range(extra):
        primes.append(p)
        p = nextprime(p)

    X, Y = [], []
    n_labeled = len(primes) - 1 if causal else len(primes) - 2
    for i in range(1, n_labeled + 1):
        if causal:
            X.append(compute_features_causal(primes[i], primes[i-1]))
        else:
            X.append(compute_features_noncausal(primes[i], primes[i-1], primes[i+1]))
        Y.append(label_prime(primes[i]))

    X = np.array(X[:count], dtype=np.float32)
    Y = np.array(Y[:count], dtype=np.float32)
    print()
    return X, Y
