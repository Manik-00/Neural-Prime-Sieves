# train.py
# ─────────────────────────────────────────────────────────────────
# Training loop with cosine annealing, gradient clipping, and
# best-checkpoint restoration.  Prints every 5 epochs.
# ─────────────────────────────────────────────────────────────────

import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score

from config import (device, set_seed, MAIN_SEED, N_FAM,
                    g_torch, _worker_init_fn)
from features import PrimeDataset
from losses import weighted_bce, FocalLoss, AsymmetricLoss, compute_pos_weights

_focal_fn = FocalLoss()
_asl_fn   = AsymmetricLoss()


def train_model(model, X_tr, Y_tr,
                X_val, Y_val,
                loss_name="wbce",
                epochs=60, lr=1e-3, seed=MAIN_SEED,
                verbose=True, tag=""):
    """Train one model to convergence.

    Parameters
    ----------
    model     : nn.Module   Model to train (modified in-place).
    X_tr, Y_tr: np.ndarray  Training features and labels.
    X_val, Y_val: np.ndarray Validation features and labels.
    loss_name : str         One of 'wbce', 'focal', 'asl'.
    epochs    : int         Training epochs.
    lr        : float       Initial learning rate for AdamW.
    seed      : int         Random seed (set before training starts).
    verbose   : bool        Print progress every 5 epochs.
    tag       : str         Label printed in the header.

    Returns
    -------
    train_hist : list[float]  Per-epoch training loss.
    val_hist   : list[float]  Per-epoch validation loss.
    best_val   : float        Best validation loss achieved.
    """
    set_seed(seed)
    pw  = compute_pos_weights(Y_tr)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_ds = PrimeDataset(X_tr, Y_tr)
    vl_ds = PrimeDataset(X_val, Y_val)
    tl = DataLoader(tr_ds, batch_size=1024, shuffle=True,
                    pin_memory=(device.type == "cuda"), num_workers=0,
                    worker_init_fn=_worker_init_fn, generator=g_torch)
    vl = DataLoader(vl_ds, batch_size=1024,
                    pin_memory=(device.type == "cuda"))

    best_val, best_state = float("inf"), None
    train_hist, val_hist = [], []
    t0 = time.time()

    if verbose:
        print(f"\n{'='*80}")
        print(f"  TRAINING: {tag}")
        print(f"  Loss: {loss_name} | Epochs: {epochs} | LR: {lr} | Seed: {seed}")
        print(f"{'='*80}")

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            p = model(xb)
            if loss_name == "wbce":
                loss = weighted_bce(p, yb, pw)
            elif loss_name == "focal":
                loss = _focal_fn(p, yb)
            else:
                loss = _asl_fn(p, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sch.step()

        # ── Validation pass ──────────────────────────────────────
        model.eval()
        vl_loss = 0.0
        vp_all, vl_all = [], []
        with torch.no_grad():
            for xb, yb in vl:
                p_  = model(xb.to(device))
                ybd = yb.to(device)
                if loss_name == "wbce":
                    lv = weighted_bce(p_, ybd, pw)
                elif loss_name == "focal":
                    lv = _focal_fn(p_, ybd)
                else:
                    lv = _asl_fn(p_, ybd)
                vl_loss += lv.item()
                vp_all.append((p_ > 0.5).float().cpu().numpy())
                vl_all.append(yb.numpy())

        vp = np.concatenate(vp_all)
        vy = np.concatenate(vl_all)
        tl_avg = ep_loss / len(tl)
        vl_avg = vl_loss / len(vl)
        train_hist.append(tl_avg)
        val_hist.append(vl_avg)

        if vl_avg < best_val:
            best_val   = vl_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % 5 == 0:
            tw_f1    = f1_score(vy[:, 0], vp[:, 0], zero_division=0)
            co_f1    = f1_score(vy[:, 3], vp[:, 3], zero_division=0)
            sg_rec   = recall_score(vy[:, 1], vp[:, 1], zero_division=0)
            safe_rec = recall_score(vy[:, 2], vp[:, 2], zero_division=0)
            chen_rec = recall_score(vy[:, 5], vp[:, 5], zero_division=0)
            print(f"  Ep {ep+1:3d}/{epochs} | Train {tl_avg:.5f} | Val {vl_avg:.5f} | "
                  f"TwnF1 {tw_f1:.3f} | CoF1 {co_f1:.3f} | "
                  f"SG_R {sg_rec:.3f} | Safe_R {safe_rec:.3f} | "
                  f"Chen_R {chen_rec:.3f} | {int(time.time()-t0)}s")

    model.load_state_dict(best_state)
    if verbose:
        print(f"\n  Best val loss: {best_val:.6f} — weights restored")
    return train_hist, val_hist, best_val
