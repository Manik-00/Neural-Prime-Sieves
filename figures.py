# figures.py
# ─────────────────────────────────────────────────────────────────
# One function per publication figure (fig01–fig15).
# Each function takes only the data it needs and calls save_fig().
# ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from config import (CB, FAM_COLORS, FAM_MARKERS, LABEL_NAMES, LABEL_SHORT,
                    N_FAM, GEN_TAG, GEN_SPECS, X_TICKS, X_LABELS,
                    FEATURE_GROUPS_CAUSAL, save_fig)


def fig01_training_curves(hist_tc, hist_vc, hist_ta, hist_va,
                           hist_tf, hist_vf, hist_ts, hist_vs,
                           hist_tn, hist_vn):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))

    ax = axes[0]
    for hist, ls, color, label in [
        (hist_tc, "-",  CB["blue"],   "wBCE train"),
        (hist_vc, "--", CB["blue"],   "wBCE val"),
        (hist_ta, "-",  CB["sky"],    "ASL train"),
        (hist_va, "--", CB["sky"],    "ASL val"),
        (hist_tf, "-",  CB["orange"], "Focal train"),
        (hist_vf, "--", CB["orange"], "Focal val"),
    ]:
        ax.plot(hist, ls=ls, color=color, lw=1.5, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss-function comparison (causal model)"); ax.legend(fontsize=6)

    ax = axes[1]
    for hist, ls, color, label in [
        (hist_tc, "-",  CB["blue"],   "Causal wBCE train"),
        (hist_vc, "--", CB["blue"],   "Causal wBCE val"),
        (hist_ts, "-",  CB["green"],  "Shallow train"),
        (hist_vs, "--", CB["green"],  "Shallow val"),
        (hist_tn, "-",  CB["purple"], "Non-causal train"),
        (hist_vn, "--", CB["purple"], "Non-causal val"),
    ]:
        ax.plot(hist, ls=ls, color=color, lw=1.5, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted BCE loss")
    ax.set_title("Architecture / causality comparison"); ax.legend(fontsize=6)

    save_fig("fig01_training_curves", fig)


def fig02_generalization_causal(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))

    ax = axes[0]
    for i, name in enumerate(LABEL_NAMES):
        vals = [all_results["causal"][t][name]["recall"] for t in GEN_TAG]
        ax.plot(X_TICKS, vals, marker=FAM_MARKERS[i],
                color=FAM_COLORS[i], label=LABEL_SHORT[i])
    ax.axhline(0.9, color="grey", ls=":", lw=1.0, label="90% line")
    ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=8)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
    ax.set_title("Recall vs. scale — causal model (7 families)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    for i, name in enumerate(LABEL_NAMES):
        vals = [all_results["causal"][t][name]["reduction"] for t in GEN_TAG]
        ax.plot(X_TICKS, vals, marker=FAM_MARKERS[i],
                color=FAM_COLORS[i], label=LABEL_SHORT[i])
    ax.axhline(77, color="grey", ls=":", lw=1.0, label="Sieve baseline 77%")
    ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=8)
    ax.set_ylabel("Candidates eliminated (%)"); ax.set_ylim(50, 100)
    ax.set_title("Search space reduction vs. scale"); ax.legend(fontsize=6, ncol=2)

    save_fig("fig02_generalization_causal", fig)


def fig03_causal_vs_noncausal(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for col, (title, metric) in enumerate([("Recall", "recall"),
                                            ("Search Reduction (%)", "reduction")]):
        ax = axes[col]
        for i, name in enumerate(LABEL_NAMES):
            c_vals = [all_results["causal"][t][name][metric]    for t in GEN_TAG]
            n_vals = [all_results["noncausal"][t][name][metric] for t in GEN_TAG]
            ax.plot(X_TICKS, c_vals, marker=FAM_MARKERS[i], color=FAM_COLORS[i],
                    lw=1.6, label=LABEL_SHORT[i])
            ax.plot(X_TICKS, n_vals, marker=FAM_MARKERS[i], color=FAM_COLORS[i],
                    lw=1.0, ls="--", alpha=0.6)
        ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"{title}: causal (solid) vs. non-causal (dashed)")
        if col == 0:
            ax.set_ylim(0, 1.05); ax.legend(fontsize=6, ncol=2)
        else:
            ax.set_ylim(50, 100)
    fig.suptitle("Effect of causality constraint on predictive performance", fontsize=10)
    save_fig("fig03_causal_vs_noncausal", fig)


def fig04_model_comparison(all_results):
    t12     = "1e12"
    models_ = ["causal", "asl", "focal", "shallow", "noncausal", "xgboost"]
    mlabels = ["wBCE", "ASL", "Focal", "Shallow", "Non-causal", "XGBoost"]
    mcolors = [CB["blue"], CB["sky"], CB["orange"], CB["green"], CB["purple"], CB["red"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    for col, (metric, ylabel) in enumerate([
        ("recall",    "Recall"),
        ("precision", "Precision"),
        ("reduction", "Reduction (%)"),
    ]):
        ax = axes[col]
        xi = np.arange(N_FAM)
        bw = 0.12
        for mi, (mn, mc) in enumerate(zip(models_, mcolors)):
            vals = [all_results[mn][t12][n][metric] for n in LABEL_NAMES]
            ax.bar(xi + (mi - 2.5) * bw, vals, bw, color=mc,
                   label=mlabels[mi], alpha=0.9)
        ax.set_xticks(xi)
        ax.set_xticklabels([s[:5] for s in LABEL_SHORT], fontsize=8)
        ax.set_title(f"{ylabel} at " + r"$10^{12}$"); ax.set_ylabel(ylabel)
        if metric in ("recall", "precision"):
            ax.set_ylim(0, 1.15); ax.axhline(0.9, color="grey", ls=":", lw=0.8)
        else:
            ax.set_ylim(50, 100); ax.axhline(77, color="grey", ls=":", lw=0.8)
        if col == 0:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Model comparison at " + r"$10^{12}$" +
                 " — recall, precision, search reduction", fontsize=10)
    save_fig("fig04_model_comparison_1e12", fig)


def fig05_depth_necessity(all_results):
    n_cols = 4
    n_rows = (N_FAM + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13.0, 3.0 * n_rows), sharey=True)
    axes = axes.flatten()
    for col_idx, name in enumerate(LABEL_NAMES):
        ax = axes[col_idx]
        for mn, mc, ls in [("causal",    CB["blue"],   "-"),
                            ("asl",       CB["sky"],    "-."),
                            ("shallow",   CB["green"],  "--"),
                            ("xgboost",   CB["red"],    "-."),
                            ("noncausal", CB["purple"], ":")]:
            vals = [all_results[mn][t][name]["recall"] for t in GEN_TAG]
            ax.plot(X_TICKS, vals, color=mc, ls=ls, lw=1.4,
                    label={"causal": "Causal wBCE", "asl": "Causal ASL",
                           "shallow": "Shallow", "xgboost": "XGBoost",
                           "noncausal": "Non-causal"}[mn])
        ax.set_title(LABEL_SHORT[col_idx], fontsize=9)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([r"$10^{8}$", r"$10^{10}$", r"$10^{12}$",
                            r"$10^{14}$", r"$10^{16}$"], fontsize=7)
        ax.set_ylim(0, 1.05); ax.axhline(0.9, color="grey", ls=":", lw=0.7)
        if col_idx % n_cols == 0:
            ax.set_ylabel("Recall"); ax.legend(fontsize=6)
    for unused in range(N_FAM, len(axes)):
        axes[unused].axis("off")
    fig.suptitle("Recall across scales: depth and loss-function necessity", fontsize=10)
    save_fig("fig05_depth_necessity", fig)


def fig06_ablation_heatmap(ablation_drop):
    groups  = list(FEATURE_GROUPS_CAUSAL.keys())
    abl_mat = np.array([[ablation_drop[g][n] for n in LABEL_NAMES] for g in groups])

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    im = ax.imshow(abl_mat, cmap="RdYlGn_r", vmin=-0.15, vmax=0.65, aspect="auto")
    ax.set_xticks(range(N_FAM)); ax.set_xticklabels(LABEL_SHORT, fontsize=9)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=8)
    for gi in range(len(groups)):
        for ni in range(N_FAM):
            v = abl_mat[gi, ni]
            ax.text(ni, gi, f"{v:+.3f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v) > 0.3 else "black")
    plt.colorbar(im, ax=ax, label="Recall drop when group is zeroed", fraction=0.03)
    ax.set_title("Feature ablation: recall drop per group\n"
                 "(positive = group contributes positively to recall)")
    save_fig("fig06_ablation_heatmap", fig)


def fig07_threshold_sweep(sweep_results, opt_thresholds):
    n_cols = 4
    n_rows = (N_FAM + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 3.5 * n_rows))
    axes = axes.flatten()
    for col_idx, name in enumerate(LABEL_NAMES):
        ax  = axes[col_idx]
        sr  = sweep_results[name]
        ax2 = ax.twinx()
        ax.plot(sr["thresh"], sr["rec"],  color=CB["blue"],   lw=1.5, label="Recall")
        ax.plot(sr["thresh"], sr["prec"], color=CB["orange"], lw=1.5, label="Precision")
        ax2.plot(sr["thresh"], sr["red"], color=CB["green"],  lw=1.5, ls="--",
                 label="Reduction %")
        ax.axvline(opt_thresholds[name]["tau_f1"], color="grey", ls=":", lw=1.0,
                   label="τ(F1)")
        ax.set_xlabel("Threshold τ"); ax.set_ylim(0, 1.05)
        ax2.set_ylim(50, 100); ax2.set_ylabel("Red%", fontsize=7)
        ax.set_title(LABEL_SHORT[col_idx], fontsize=9)
        if col_idx % n_cols == 0:
            ax.set_ylabel("Precision / Recall")
            ax.legend(fontsize=6, loc="lower left")
            ax2.legend(fontsize=6, loc="upper right")
        else:
            ax.tick_params(labelleft=False)
    for unused in range(N_FAM, len(axes)):
        axes[unused].axis("off")
    fig.suptitle("Threshold sweep: precision, recall, and search reduction on val set",
                 fontsize=10)
    save_fig("fig07_threshold_sweep", fig)


def fig08_pr_curves(all_results):
    pr_tags   = ["5e8", "1e12", "1e16"]
    pr_labels = [r"$5{\times}10^8$ (val)", r"$10^{12}$", r"$10^{16}$"]
    pr_ls     = ["-", "--", ":"]
    n_cols = 4
    n_rows = (N_FAM + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 6.5))
    axes = axes.flatten()
    for idx, name in enumerate(LABEL_NAMES):
        ax = axes[idx]
        for tag, lab, ls, color in zip(pr_tags, pr_labels, pr_ls,
                                       [CB["blue"], CB["orange"], CB["green"]]):
            m = all_results["causal"][tag][name]
            pr, rc, _ = precision_recall_curve(m["labels"], m["probs"])
            ax.plot(rc, pr, color=color, ls=ls, lw=1.5,
                    label=f"{lab}  AP={m['auc_pr']:.3f}")
        ax.set_title(LABEL_SHORT[idx])
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(fontsize=6)
    for unused in range(N_FAM, len(axes)):
        axes[unused].axis("off")
    fig.suptitle("Precision-recall curves at three scales — causal model", fontsize=10)
    save_fig("fig08_pr_curves", fig)


def fig09_calibration(all_results):
    family_groups = {
        "Gap-defined families":        ["twin", "cousin", "sexy", "isolated"],
        "Indirect-inference families":  ["sophie_germain", "safe", "chen"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for col, (title, names_) in enumerate(family_groups.items()):
        ax = axes[col]
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1.0, label="Perfect")
        for name in names_:
            m = all_results["causal"]["1e12"][name]
            if m["labels"].sum() < 5:
                continue
            frac, mean_pred = calibration_curve(
                m["labels"], m["probs"], n_bins=10, strategy="uniform")
            i = LABEL_NAMES.index(name)
            ax.plot(mean_pred, frac, marker=FAM_MARKERS[i],
                    color=FAM_COLORS[i], lw=1.5, label=LABEL_SHORT[i])
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle(r"Calibration curves at $10^{12}$", fontsize=10)
    save_fig("fig09_calibration", fig)


def fig10_score_separation(all_results):
    n_cols = 4
    n_rows = (N_FAM + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 6.0))
    axes = axes.flatten()
    for idx, name in enumerate(LABEL_NAMES):
        ax  = axes[idx]
        m   = all_results["causal"]["1e12"][name]
        pos = m["probs"][m["labels"] == 1]
        neg = m["probs"][m["labels"] == 0]
        ax.hist(neg, bins=50, alpha=0.5, color=CB["orange"], density=True, label="Non-member")
        ax.hist(pos, bins=50, alpha=0.8, color=CB["blue"],   density=True,
                histtype="step", lw=2, label="Member")
        ax.axvline(0.5, color="black", ls="--", lw=1.0)
        ax.set_title(LABEL_SHORT[idx]); ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density"); ax.legend(fontsize=7)
    for unused in range(N_FAM, len(axes)):
        axes[unused].axis("off")
    fig.suptitle(r"Score separation histograms at $10^{12}$", fontsize=10)
    save_fig("fig10_score_separation", fig)


def fig11_robustness(seed_store):
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    xi = np.arange(N_FAM); bw = 0.35
    r_means = [np.mean(seed_store[n]["recall"]) for n in LABEL_NAMES]
    r_stds  = [np.std(seed_store[n]["recall"])  for n in LABEL_NAMES]
    f_means = [np.mean(seed_store[n]["f1"])     for n in LABEL_NAMES]
    f_stds  = [np.std(seed_store[n]["f1"])      for n in LABEL_NAMES]
    ax.bar(xi - bw/2, r_means, bw, yerr=r_stds, color=CB["blue"],
           capsize=4, error_kw={"elinewidth": 1.5}, label="Recall", alpha=0.9)
    ax.bar(xi + bw/2, f_means, bw, yerr=f_stds, color=CB["orange"],
           capsize=4, error_kw={"elinewidth": 1.5}, label="F1", alpha=0.9)
    ax.set_xticks(xi); ax.set_xticklabels(LABEL_SHORT, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title(r"Reproducibility across 3 seeds (val set, mean $\pm$ std)")
    ax.legend(fontsize=8)
    save_fig("fig11_robustness", fig)


def fig12_asl_vs_wbce(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))

    ax = axes[0]
    for i, name in enumerate(LABEL_NAMES):
        c_vals = [all_results["causal"][t][name]["recall"] for t in GEN_TAG]
        a_vals = [all_results["asl"][t][name]["recall"]    for t in GEN_TAG]
        ax.plot(X_TICKS, c_vals, marker=FAM_MARKERS[i], color=FAM_COLORS[i],
                lw=1.6, label=LABEL_SHORT[i])
        ax.plot(X_TICKS, a_vals, marker=FAM_MARKERS[i], color=FAM_COLORS[i],
                lw=1.0, ls="--", alpha=0.65)
    ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=8)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
    ax.set_title("wBCE (solid) vs ASL (dashed)"); ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    xi = np.arange(N_FAM); bw = 0.35
    c_brier = [all_results["causal"]["1e12"][n]["brier"] for n in LABEL_NAMES]
    a_brier = [all_results["asl"]["1e12"][n]["brier"]    for n in LABEL_NAMES]
    ax.bar(xi - bw/2, c_brier, bw, color=CB["blue"], label="wBCE", alpha=0.9)
    ax.bar(xi + bw/2, a_brier, bw, color=CB["sky"],  label="ASL",  alpha=0.9)
    ax.set_xticks(xi); ax.set_xticklabels([s[:5] for s in LABEL_SHORT], fontsize=8)
    ax.set_ylabel("Brier score (lower = better)")
    ax.set_title(r"Calibration at $10^{12}$"); ax.legend(fontsize=8)

    fig.suptitle("Asymmetric Loss vs. weighted BCE: recall and calibration", fontsize=10)
    save_fig("fig12_asl_vs_wbce", fig)


def fig13_new_families(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for col, metric in enumerate(["recall", "reduction"]):
        ax = axes[col]
        for name in ["chen", "isolated"]:
            i = LABEL_NAMES.index(name)
            c_vals  = [all_results["causal"][t][name][metric]    for t in GEN_TAG]
            nc_vals = [all_results["noncausal"][t][name][metric] for t in GEN_TAG]
            ax.plot(X_TICKS, c_vals,  marker=FAM_MARKERS[i], color=FAM_COLORS[i],
                    lw=1.6, label=f"{LABEL_SHORT[i]} (causal)")
            ax.plot(X_TICKS, nc_vals, marker=FAM_MARKERS[i], color=FAM_COLORS[i],
                    lw=1.0, ls="--", alpha=0.65, label=f"{LABEL_SHORT[i]} (NC)")
        ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=8)
        ax.set_ylabel("Recall" if metric == "recall" else "Search reduction (%)")
        if metric == "recall":
            ax.set_ylim(0, 1.05); ax.axhline(0.9, color="grey", ls=":", lw=0.8)
        else:
            ax.set_ylim(50, 100)
        ax.set_title(("Recall" if metric == "recall" else "Reduction") +
                     ": new families vs. scale")
        ax.legend(fontsize=7)
    fig.suptitle("Chen and Isolated primes: causal (solid) vs. non-causal (dashed)",
                 fontsize=10)
    save_fig("fig13_new_families", fig)


def fig14_summary_composite(all_results, seed_store, ablation_drop):
    fig = plt.figure(figsize=(13.0, 9.5))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)
    xi  = np.arange(N_FAM)
    t12 = "1e12"
    abl_short = ["A: Primorial", "B: Sm.prime", "C: Bk.gap",
                 "D: Scale", "E: Digit", "F: Ext.mod"]
    groups  = list(FEATURE_GROUPS_CAUSAL.keys())
    abl_mat = np.array([[ablation_drop[g][n] for n in LABEL_NAMES] for g in groups])

    # (0,0) Causal recall vs scale
    ax = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(LABEL_NAMES):
        ax.plot(X_TICKS, [all_results["causal"][t][name]["recall"] for t in GEN_TAG],
                marker=FAM_MARKERS[i], color=FAM_COLORS[i], lw=1.4, label=LABEL_SHORT[i])
    ax.axhline(0.9, color="grey", ls=":", lw=0.8)
    ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=7)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
    ax.set_title("Recall vs. scale"); ax.legend(fontsize=5, ncol=2)

    # (0,1) Search reduction
    ax = fig.add_subplot(gs[0, 1])
    for i, name in enumerate(LABEL_NAMES):
        ax.plot(X_TICKS, [all_results["causal"][t][name]["reduction"] for t in GEN_TAG],
                marker=FAM_MARKERS[i], color=FAM_COLORS[i], lw=1.4, label=LABEL_SHORT[i])
    ax.axhline(77, color="grey", ls=":", lw=0.8, label="Sieve 77%")
    ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=7)
    ax.set_ylabel("Reduction (%)"); ax.set_ylim(50, 100)
    ax.set_title("Search reduction vs. scale"); ax.legend(fontsize=5, ncol=2)

    # (0,2) Causal vs NC at val
    ax = fig.add_subplot(gs[0, 2])
    bw = 0.4
    c_vals  = [all_results["causal"]["5e8"][n]["recall"]    for n in LABEL_NAMES]
    nc_vals = [all_results["noncausal"]["5e8"][n]["recall"] for n in LABEL_NAMES]
    ax.bar(xi - bw/2, c_vals,  bw, label="Causal",     color=CB["blue"],   alpha=0.85)
    ax.bar(xi + bw/2, nc_vals, bw, label="Non-causal", color=CB["purple"], alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels([s[:4] for s in LABEL_SHORT], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Recall (val)")
    ax.set_title("Causality cost at val"); ax.legend(fontsize=7)

    # (1,0:2) Ablation heatmap
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(abl_mat, cmap="RdYlGn_r", vmin=-0.15, vmax=0.65, aspect="auto")
    ax.set_xticks(range(N_FAM)); ax.set_xticklabels(LABEL_SHORT, fontsize=8)
    ax.set_yticks(range(len(abl_short))); ax.set_yticklabels(abl_short, fontsize=7)
    for gi in range(len(abl_short)):
        for ni in range(N_FAM):
            v = abl_mat[gi, ni]
            ax.text(ni, gi, f"{v:+.3f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(v) > 0.3 else "black")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Recall drop")
    ax.set_title("Feature ablation (all 7 families)")

    # (1,2) Loss comparison bar at 1e12
    ax = fig.add_subplot(gs[1, 2])
    bw = 0.25
    c_rec = [all_results["causal"][t12][n]["recall"] for n in LABEL_NAMES]
    a_rec = [all_results["asl"][t12][n]["recall"]    for n in LABEL_NAMES]
    f_rec = [all_results["focal"][t12][n]["recall"]  for n in LABEL_NAMES]
    ax.bar(xi - bw, c_rec, bw, label="wBCE",  color=CB["blue"],   alpha=0.85)
    ax.bar(xi,      a_rec, bw, label="ASL",   color=CB["sky"],    alpha=0.85)
    ax.bar(xi + bw, f_rec, bw, label="Focal", color=CB["orange"], alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels([s[:4] for s in LABEL_SHORT], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Recall")
    ax.set_title(r"Loss comparison at $10^{12}$"); ax.legend(fontsize=6)

    # (2,0) Multi-seed robustness
    ax = fig.add_subplot(gs[2, 0])
    r_means = [np.mean(seed_store[n]["recall"]) for n in LABEL_NAMES]
    r_stds  = [np.std(seed_store[n]["recall"])  for n in LABEL_NAMES]
    ax.bar(xi, r_means, 0.6, yerr=r_stds, color=CB["blue"],
           capsize=3, error_kw={"elinewidth": 1.2}, alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels([s[:4] for s in LABEL_SHORT], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Recall")
    ax.set_title(r"Recall: $\mu\pm\sigma$, 3 seeds")

    # (2,1) Chen and Isolated detail
    ax = fig.add_subplot(gs[2, 1])
    for name in ["chen", "isolated"]:
        i = LABEL_NAMES.index(name)
        ax.plot(X_TICKS, [all_results["causal"][t][name]["recall"] for t in GEN_TAG],
                marker=FAM_MARKERS[i], color=FAM_COLORS[i], lw=1.4,
                label=f"{LABEL_SHORT[i]} (C)")
        ax.plot(X_TICKS, [all_results["noncausal"][t][name]["recall"] for t in GEN_TAG],
                marker=FAM_MARKERS[i], color=FAM_COLORS[i], lw=1.0, ls="--", alpha=0.6,
                label=f"{LABEL_SHORT[i]} (NC)")
    ax.set_xticks(X_TICKS); ax.set_xticklabels(X_LABELS, fontsize=7)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
    ax.set_title("New families: Chen & Isolated")
    ax.axhline(0.9, color="grey", ls=":", lw=0.8); ax.legend(fontsize=7)

    # (2,2) wBCE vs ASL at val
    ax = fig.add_subplot(gs[2, 2])
    bw = 0.35
    c_val = [all_results["causal"]["5e8"][n]["recall"] for n in LABEL_NAMES]
    a_val = [all_results["asl"]["5e8"][n]["recall"]    for n in LABEL_NAMES]
    ax.bar(xi - bw/2, c_val, bw, label="wBCE", color=CB["blue"], alpha=0.85)
    ax.bar(xi + bw/2, a_val, bw, label="ASL",  color=CB["sky"],  alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels([s[:4] for s in LABEL_SHORT], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Recall (val)")
    ax.set_title("wBCE vs ASL at val"); ax.legend(fontsize=7)

    fig.suptitle("PrimeFamilyNet — Main results summary\n"
                 r"Strictly causal features, trained on $[10^7, 10^9]$, 7 prime families",
                 fontsize=11, fontweight="bold", y=1.01)
    save_fig("fig14_summary_composite", fig, tight=False)


def fig15_density_recall(GEN_Y, all_results):
    """Headline figure: density fractions vs recall, with HL prediction."""
    twin_idx     = LABEL_NAMES.index("twin")
    isolated_idx = LABEL_NAMES.index("isolated")

    twin_frac_pct     = [float(100 * GEN_Y[k][:, twin_idx].mean())
                         for k in range(len(GEN_SPECS))]
    isolated_frac_pct = [float(100 * GEN_Y[k][:, isolated_idx].mean())
                         for k in range(len(GEN_SPECS))]
    twin_recall     = [all_results["causal"][t]["twin"]["recall"]     for t in GEN_TAG]
    isolated_recall = [all_results["causal"][t]["isolated"]["recall"] for t in GEN_TAG]

    # ── R² statistics ──────────────────────────────────────────────
    r_iso  = np.corrcoef(isolated_frac_pct, isolated_recall)[0, 1]
    r_twin = np.corrcoef(twin_frac_pct,     twin_recall)[0, 1]
    print(f"\n  Density-recall R²: isolated={r_iso**2:.4f}, twin={r_twin**2:.4f}")

    HL_LOG_N    = [np.log(s[0]) for s in GEN_SPECS]
    hl_raw      = [1.0 / ln for ln in HL_LOG_N]
    hl_scale    = twin_frac_pct[0] / hl_raw[0]
    twin_hl_pct = [v * hl_scale for v in hl_raw]

    C_TWIN     = CB["blue"]
    C_ISOLATED = CB["red"]
    C_HL       = CB["green"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    ax1, ax2, ax3 = axes

    # Left: density fraction
    ax1.plot(X_TICKS, twin_frac_pct,     color=C_TWIN,     lw=2.0, marker="o", ms=6,
             label="Twin fraction (%)")
    ax1.plot(X_TICKS, isolated_frac_pct, color=C_ISOLATED, lw=2.0, marker="s", ms=6,
             label="Isolated fraction (%)")
    ax1.plot(X_TICKS, twin_hl_pct,       color=C_HL,       lw=1.4, ls="--",
             label=r"HL prediction: $C_2\,/\!\log N$")
    ax1.set_xticks(X_TICKS); ax1.set_xticklabels(X_LABELS, fontsize=8)
    ax1.set_xlabel("Evaluation scale"); ax1.set_ylabel("Density fraction (%)", fontsize=9)
    ax1.set_ylim(0, 105); ax1.legend(fontsize=8, loc="center left")
    ax1.set_title("Density fraction vs. scale", fontsize=9)
    ax1.grid(True, alpha=0.25, linewidth=0.5)

    # Middle: recall
    ax2.plot(X_TICKS, twin_recall,     color=C_TWIN,     lw=2.0, marker="o", ms=6,
             label="Twin recall")
    ax2.plot(X_TICKS, isolated_recall, color=C_ISOLATED, lw=2.0, marker="s", ms=6,
             label="Isolated recall")
    ax2.set_xticks(X_TICKS); ax2.set_xticklabels(X_LABELS, fontsize=8)
    ax2.set_xlabel("Evaluation scale"); ax2.set_ylabel("Model recall", fontsize=9)
    ax2.set_ylim(0, 1.10); ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2.legend(fontsize=8, loc="center left")
    ax2.set_title("Model recall vs. scale", fontsize=9)
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    # Right: scatter density vs recall
    for frac, rec in zip(twin_frac_pct, twin_recall):
        ax3.scatter(frac, rec, color=C_TWIN, s=45, zorder=3)
    for frac, rec in zip(isolated_frac_pct, isolated_recall):
        ax3.scatter(frac, rec, color=C_ISOLATED, s=45, zorder=3)

    def _add_trend(ax, x, y, color):
        c  = np.polyfit(x, y, 1)
        xr = np.linspace(min(x) - 0.5, max(x) + 0.5, 50)
        ax.plot(xr, np.polyval(c, xr), color=color, lw=1.2, ls="--", alpha=0.6)

    _add_trend(ax3, twin_frac_pct,     twin_recall,     C_TWIN)
    _add_trend(ax3, isolated_frac_pct, isolated_recall, C_ISOLATED)

    for frac, rec, lbl in zip(twin_frac_pct, twin_recall, X_LABELS):
        ax3.annotate(lbl, (frac, rec), textcoords="offset points",
                     xytext=(-4, 5), fontsize=5.5, color=C_TWIN)
    for frac, rec, lbl in zip(isolated_frac_pct, isolated_recall, X_LABELS):
        ax3.annotate(lbl, (frac, rec), textcoords="offset points",
                     xytext=(3, -9), fontsize=5.5, color=C_ISOLATED)

    twin_proxy = Line2D([0], [0], color=C_TWIN,     marker="o", ms=5, lw=0, label="Twin")
    iso_proxy  = Line2D([0], [0], color=C_ISOLATED, marker="s", ms=5, lw=0, label="Isolated")
    ax3.legend(handles=[twin_proxy, iso_proxy], fontsize=8)
    ax3.set_xlabel("Density fraction (%)", fontsize=9); ax3.set_ylabel("Model recall", fontsize=9)
    ax3.set_title(
        f"Recall vs. density fraction\n"
        f"(Isolated $R^2$={r_iso**2:.3f}, Twin $R^2$={r_twin**2:.3f})",
        fontsize=9)
    ax3.grid(True, alpha=0.25, linewidth=0.5)

    fig.suptitle(
        "Density-driven generalisation: twin prime fraction declines\n"
        r"as $1/\log N$ (Hardy--Littlewood); isolated prime recall follows the complement",
        fontsize=9.5)
    save_fig("fig15_density_recall", fig)
