# evaluate.py
# ─────────────────────────────────────────────────────────────────
# Per-family evaluation at a fixed threshold.
# Returns precision, recall, F1, AUC-PR, Brier score, search-space
# reduction, and raw probabilities/labels for downstream plotting.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, brier_score_loss

from config import device, LABEL_NAMES
from features import PrimeDataset


def evaluate(X, Y, model, threshold=0.5, tag="", verbose=True):
    """Comprehensive per-family evaluation at a fixed decision threshold.

    Parameters
    ----------
    X, Y      : np.ndarray  Feature matrix and label matrix.
    model     : nn.Module   Trained model (set to eval mode internally).
    threshold : float       Decision threshold (default 0.50).
    tag       : str         Label printed in the results header.
    verbose   : bool        If True, print the formatted metrics table.

    Returns
    -------
    dict  Keyed by family name; each value is a dict of metrics including
          raw 'probs' and 'labels' arrays for downstream plotting.
    """
    model.eval()
    ds  = PrimeDataset(X, Y)
    ldr = DataLoader(ds, batch_size=2048,
                     pin_memory=(device.type == "cuda"))
    probs_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in ldr:
            probs_list.append(model(xb.to(device)).cpu().numpy())
            labels_list.append(yb.numpy())

    probs  = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    preds  = (probs >= threshold).astype(float)
    n      = len(labels)

    metrics = {}
    for i, name in enumerate(LABEL_NAMES):
        yt = labels[:, i]
        yp = probs[:, i]
        yh = preds[:, i]
        tp = int(((yh == 1) & (yt == 1)).sum())
        fp = int(((yh == 1) & (yt == 0)).sum())
        fn = int(((yh == 0) & (yt == 1)).sum())
        tn = int(((yh == 0) & (yt == 0)).sum())
        prec   = tp / max(tp + fp, 1)
        rec    = tp / max(tp + fn, 1)
        f1     = 2 * prec * rec / max(prec + rec, 1e-9)
        auc_pr = average_precision_score(yt, yp) if yt.sum() > 0 else 0.0
        brier  = brier_score_loss(yt, yp)
        red    = 100 * (tn + fn) / n
        miss   = 100 * fn / max(tp + fn, 1)
        # 95% Wilson-ish CI for recall (normal approximation on binomial)
        n_pos  = max(tp + fn, 1)
        se     = np.sqrt(rec * (1 - rec) / n_pos)
        ci_lo  = max(0.0, rec - 1.96 * se)
        ci_hi  = min(1.0, rec + 1.96 * se)
        metrics[name] = dict(
            tp=tp, fp=fp, fn=fn, tn=tn, n=n,
            n_pos=n_pos,
            precision=prec, recall=rec, f1=f1,
            recall_se=se, recall_ci_lo=ci_lo, recall_ci_hi=ci_hi,
            auc_pr=auc_pr, brier=brier,
            reduction=red, missed_pct=miss,
            mean_prob_pos=float(yp[yt == 1].mean()) if yt.sum() > 0 else 0.0,
            mean_prob_neg=float(yp[yt == 0].mean()),
            probs=yp, labels=yt,
        )

    if verbose:
        print(f"\n{'='*92}")
        print(f"  {tag}  (n={n:,})  threshold={threshold:.2f}")
        print(f"{'='*92}")
        print(f"  {'Family':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} "
              f"{'AP':>7} {'Brier':>7} {'Reduc%':>7} {'Miss%':>6} "
              f"{'n_pos':>6} {'95% CI':>15}")
        print("  " + "─" * 84)
        for nm, m in metrics.items():
            ci_str = f"[{m['recall_ci_lo']:.3f},{m['recall_ci_hi']:.3f}]"
            print(f"  {nm:<18} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['auc_pr']:>7.3f} {m['brier']:>7.4f} "
                  f"{m['reduction']:>6.1f}% {m['missed_pct']:>5.2f}% "
                  f"{m['n_pos']:>6} {ci_str:>15}")
        print(f"{'='*92}")
    return metrics
