# main.py
# ─────────────────────────────────────────────────────────────────
# PrimeFamilyNet — orchestration script.
# Run this file to reproduce all results and figures.
#
# Usage (from the project folder):
#   python main.py
#
# Output:
#   paper/figures/   — fig01–fig15 as PDF + PNG
#   paper/results/   — JSON results + terminal_output.txt
#   paper/weights/   — model checkpoints
#
# Full run time: ~25 min on RTX 4060 (local), ~45 min on T4 (Colab)
# ─────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              average_precision_score, brier_score_loss)

# ── Project modules ───────────────────────────────────────────────
from config import (
    device, MAIN_SEED, SEEDS, set_seed,
    LABEL_NAMES, LABEL_SHORT, N_FAM,
    FEAT_DIM_CAUSAL, FEAT_DIM_NONCAUSAL,
    FEATURE_GROUPS_CAUSAL,
    GEN_SPECS, GEN_TAG, GEN_OOD, GEN_LABEL,
    X_TICKS, X_LABELS,
    save_json,
)
from features import generate_dataset, PrimeDataset
from models import PrimeFamilyNet, ShallowBaseline
from losses import compute_pos_weights
from train import train_model
from evaluate import evaluate
from figures import (
    fig01_training_curves, fig02_generalization_causal,
    fig03_causal_vs_noncausal, fig04_model_comparison,
    fig05_depth_necessity, fig06_ablation_heatmap,
    fig07_threshold_sweep, fig08_pr_curves,
    fig09_calibration, fig10_score_separation,
    fig11_robustness, fig12_asl_vs_wbce,
    fig13_new_families, fig14_summary_composite,
    fig15_density_recall,
)

# ── Print model sizes ─────────────────────────────────────────────
n_main    = sum(p.numel() for p in PrimeFamilyNet().parameters())
n_shallow = sum(p.numel() for p in ShallowBaseline().parameters())
print(f"\nFeature dims  : causal={FEAT_DIM_CAUSAL}, non-causal={FEAT_DIM_NONCAUSAL}")
print(f"\n── Model sizes ──────────────────────────────────────────")
print(f"  PrimeFamilyNet (causal, {N_FAM} heads) : {n_main:>10,} parameters")
print(f"  ShallowBaseline ({N_FAM} heads)        : {n_shallow:>10,} parameters")

# ═════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═════════════════════════════════════════════════════════════════
print("\n── Generating datasets ──────────────────────────────────")

# Training sets (three scales, concatenated)
X_t1, Y_t1 = generate_dataset(10_000_000,     100_000, "10^7  train", causal=True)
X_t2, Y_t2 = generate_dataset(100_000_000,     50_000, "10^8  train", causal=True)
X_t3, Y_t3 = generate_dataset(1_000_000_000,   50_000, "10^9  train", causal=True)
X_train = np.concatenate([X_t1, X_t2, X_t3])
Y_train = np.concatenate([Y_t1, Y_t2, Y_t3])

X_t1n, _ = generate_dataset(10_000_000,     100_000, "10^7  train (NC)", causal=False)
X_t2n, _ = generate_dataset(100_000_000,     50_000, "10^8  train (NC)", causal=False)
X_t3n, _ = generate_dataset(1_000_000_000,   50_000, "10^9  train (NC)", causal=False)
X_train_nc = np.concatenate([X_t1n, X_t2n, X_t3n])

# Validation set (causal + NC)
X_val,  Y_val  = generate_dataset(500_000_000, 20_000, "5×10^8 val",       causal=True)
X_valn, Y_valn = generate_dataset(500_000_000, 20_000, "5×10^8 val  (NC)", causal=False)

# OOD evaluation sets at all five scales
GEN_X, GEN_Y, GEN_XN, GEN_YN = [], [], [], []
for start, count, label, tag, ood in GEN_SPECS:
    if ood:
        xc, yc = generate_dataset(start, count, tag,           causal=True)
        xn, yn = generate_dataset(start, count, f"{tag} (NC)", causal=False)
    else:
        xc, yc = X_val,  Y_val
        xn, yn = X_valn, Y_valn
    GEN_X.append(xc);  GEN_Y.append(yc)
    GEN_XN.append(xn); GEN_YN.append(yn)

print(f"\n  Train (causal)    : {len(X_train):,}  |  Val: {len(X_val):,}")
print(f"  Train (non-causal): {len(X_train_nc):,}")

# Label distribution table
print("\n── Label distribution ───────────────────────────────────")
print(f"{'Scale':<18}" + "".join(f"{n:>14}" for n in LABEL_SHORT))
for (_, _, label, tag, _), Y in zip(GEN_SPECS, GEN_Y):
    row = f"{label:<18}"
    for i in range(N_FAM):
        row += f"  {100*Y[:,i].mean():>10.1f}%"
    print(row)

# Positive weights
pw_causal = compute_pos_weights(Y_train)
print(f"\nPositive weights (causal): " +
      "  ".join(f"{n}={pw_causal[i]:.2f}" for i, n in enumerate(LABEL_SHORT)))

print("\n── Loss functions ───────────────────────────────────────")
print("  Primary   : Frequency-weighted BCE (per-class inverse-frequency weights)")
print("  Comparison: Focal Loss (Lin et al. 2017, α=0.25, γ=2.0)")
print("  Comparison: Asymmetric Loss (Ridnik et al. 2021, γ_neg=4, γ_pos=0, clip=0.05)")

# ═════════════════════════════════════════════════════════════════
# PHASE 1: TRAIN ALL PRIMARY MODELS
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 1: TRAINING PRIMARY MODELS")
print("="*80)

model_causal = PrimeFamilyNet(input_dim=FEAT_DIM_CAUSAL).to(device)
hist_tc, hist_vc, _ = train_model(
    model_causal, X_train, Y_train, X_val, Y_val,
    loss_name="wbce", epochs=60,
    tag="PrimeFamilyNet — Causal (wBCE)")

model_focal = PrimeFamilyNet(input_dim=FEAT_DIM_CAUSAL).to(device)
hist_tf, hist_vf, _ = train_model(
    model_focal, X_train, Y_train, X_val, Y_val,
    loss_name="focal", epochs=60,
    tag="PrimeFamilyNet — Focal Loss (comparison)")

model_asl = PrimeFamilyNet(input_dim=FEAT_DIM_CAUSAL).to(device)
hist_ta, hist_va, _ = train_model(
    model_asl, X_train, Y_train, X_val, Y_val,
    loss_name="asl", epochs=60,
    tag="PrimeFamilyNet — Asymmetric Loss (Ridnik et al. 2021)")

model_shallow = ShallowBaseline(input_dim=FEAT_DIM_CAUSAL).to(device)
hist_ts, hist_vs, _ = train_model(
    model_shallow, X_train, Y_train, X_val, Y_val,
    loss_name="wbce", epochs=60, lr=5e-4,
    tag="ShallowBaseline — Causal (wBCE)")

model_noncausal = PrimeFamilyNet(input_dim=FEAT_DIM_NONCAUSAL).to(device)
hist_tn, hist_vn, _ = train_model(
    model_noncausal, X_train_nc, Y_train, X_valn, Y_valn,
    loss_name="wbce", epochs=60,
    tag="PrimeFamilyNet — Non-Causal (wBCE, upper bound)")

for name, m in [("causal",    model_causal),
                ("focal",     model_focal),
                ("asl",       model_asl),
                ("shallow",   model_shallow),
                ("noncausal", model_noncausal)]:
    torch.save({"state": m.state_dict(), "label_names": LABEL_NAMES},
               f"paper/weights/{name}.pt")
print("\n  Weights saved → paper/weights/")

# ═════════════════════════════════════════════════════════════════
# PHASE 2: MULTI-SEED ROBUSTNESS
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 2: MULTI-SEED ROBUSTNESS  (3 seeds × causal wBCE)")
print("="*80)

seed_store = {n: {"recall": [], "f1": [], "auc_pr": []} for n in LABEL_NAMES}
for s in SEEDS:
    print(f"  ── Seed {s} ──")
    m = PrimeFamilyNet(input_dim=FEAT_DIM_CAUSAL).to(device)
    train_model(m, X_train, Y_train, X_val, Y_val,
                epochs=60, seed=s, verbose=False)
    mets = evaluate(X_val, Y_val, m, tag=f"Seed {s}", verbose=False)
    for n in LABEL_NAMES:
        seed_store[n]["recall"].append(mets[n]["recall"])
        seed_store[n]["f1"].append(mets[n]["f1"])
        seed_store[n]["auc_pr"].append(mets[n]["auc_pr"])
    print("    " + "  ".join(f"{n}={mets[n]['recall']:.3f}" for n in LABEL_NAMES))

print(f"\n  {'Family':<18} {'Recall μ±σ':>16} {'F1 μ±σ':>14} {'AUC-PR μ±σ':>16}")
print("  " + "─" * 66)
for n in LABEL_NAMES:
    r  = seed_store[n]["recall"]
    f  = seed_store[n]["f1"]
    ap = seed_store[n]["auc_pr"]
    print(f"  {n:<18} {np.mean(r):.3f}±{np.std(r):.4f}      "
          f"{np.mean(f):.3f}±{np.std(f):.4f}    "
          f"{np.mean(ap):.3f}±{np.std(ap):.4f}")

save_json("robustness", {n: {k: [float(x) for x in v]
          for k, v in seed_store[n].items()} for n in LABEL_NAMES})

# ═════════════════════════════════════════════════════════════════
# PHASE 2b: OOD MULTI-SEED ROBUSTNESS  (3 seeds × 1e12 + 1e16)
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 2b: OOD MULTI-SEED ROBUSTNESS  (3 seeds × 1e12 & 1e16)")
print("="*80)

OOD_ROBUST_TAGS = ["1e12", "1e16"]
ood_robust_idx  = {t: GEN_TAG.index(t) for t in OOD_ROBUST_TAGS}

# Structure: ood_seed_store[tag][family] = list of recall values across seeds
ood_seed_store = {
    t: {n: [] for n in LABEL_NAMES} for t in OOD_ROBUST_TAGS
}

for s in SEEDS:
    print(f"  ── Seed {s} ──")
    m = PrimeFamilyNet(input_dim=FEAT_DIM_CAUSAL).to(device)
    train_model(m, X_train, Y_train, X_val, Y_val,
                epochs=60, seed=s, verbose=False)
    for t in OOD_ROBUST_TAGS:
        idx = ood_robust_idx[t]
        mets = evaluate(GEN_X[idx], GEN_Y[idx], m,
                        tag=f"OOD Seed {s} — {t}", verbose=False)
        for n in LABEL_NAMES:
            ood_seed_store[t][n].append(mets[n]["recall"])
    row = "  ".join(
        f"{t}: " + "  ".join(f"{n}={ood_seed_store[t][n][-1]:.3f}"
                              for n in LABEL_NAMES)
        for t in OOD_ROBUST_TAGS
    )
    print(f"    {row}")

for t in OOD_ROBUST_TAGS:
    print(f"\n  OOD Recall  μ±σ  —  {t}")
    print(f"  {'Family':<18} {'Mean':>8} {'Std':>8} {'95% CI':>18}")
    print("  " + "─" * 56)
    for n in LABEL_NAMES:
        vals  = ood_seed_store[t][n]
        mu    = np.mean(vals)
        sigma = np.std(vals)
        # CI using t-distribution with 2 df (3 seeds)
        ci_hw = 4.303 * sigma / np.sqrt(len(vals))   # t_{0.025, 2}
        print(f"  {n:<18} {mu:>8.3f} {sigma:>8.4f}   "
              f"[{max(0.0,mu-ci_hw):.3f}, {min(1.0,mu+ci_hw):.3f}]")

save_json("ood_robustness", {
    t: {n: {"values": [float(v) for v in ood_seed_store[t][n]],
            "mean":   float(np.mean(ood_seed_store[t][n])),
            "std":    float(np.std(ood_seed_store[t][n]))}
        for n in LABEL_NAMES}
    for t in OOD_ROBUST_TAGS
})


print("\n" + "="*80)
print("  PHASE 3: XGBOOST BASELINE  (same causal features)")
print("="*80)

scaler      = StandardScaler()
X_tr_sc     = scaler.fit_transform(X_train)
X_val_sc    = scaler.transform(X_val)
xgb_models  = {}

print(f"\n  {'Family':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} "
      f"{'AUC-PR':>8} {'Brier':>7} {'Reduc%':>8}")
print("  " + "─" * 68)

for i, name in enumerate(LABEL_NAMES):
    ratio = float((Y_train[:, i] == 0).sum() /
                  max((Y_train[:, i] == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=ratio, eval_metric="logloss",
        tree_method="hist", verbosity=0, random_state=42,
    )
    clf.fit(X_tr_sc, Y_train[:, i])
    pr  = clf.predict_proba(X_val_sc)[:, 1]
    pd_ = (pr > 0.5).astype(float)
    p_  = precision_score(Y_val[:, i], pd_, zero_division=0)
    r_  = recall_score(Y_val[:, i], pd_, zero_division=0)
    f_  = f1_score(Y_val[:, i], pd_, zero_division=0)
    ap  = average_precision_score(Y_val[:, i], pr)
    bri = brier_score_loss(Y_val[:, i], pr)
    tp  = int(((pd_ == 1) & (Y_val[:, i] == 1)).sum())
    fn  = int(((pd_ == 0) & (Y_val[:, i] == 1)).sum())
    tn  = int(((pd_ == 0) & (Y_val[:, i] == 0)).sum())
    red = 100 * (tn + fn) / len(Y_val)
    xgb_models[name] = dict(clf=clf, val_prec=p_, val_rec=r_,
                             val_f1=f_, val_ap=ap, val_red=red)
    print(f"  {name:<18} {p_:>6.3f} {r_:>6.3f} {f_:>6.3f} "
          f"{ap:>8.3f} {bri:>7.4f} {red:>7.1f}%")

# ═════════════════════════════════════════════════════════════════
# PHASE 4: FEATURE ABLATION
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 4: FEATURE ABLATION  (leave-one-group-out, causal model)")
print("="*80)

full_mets = evaluate(X_val, Y_val, model_causal, tag="Full model (val)", verbose=True)
full_rec  = {n: full_mets[n]["recall"] for n in LABEL_NAMES}

ablation_drop = {}
for grp, sl in FEATURE_GROUPS_CAUSAL.items():
    X_abl = X_val.copy()
    X_abl[:, sl] = 0.0
    mets = evaluate(X_abl, Y_val, model_causal, tag=f"Ablate {grp}", verbose=False)
    ablation_drop[grp] = {n: full_rec[n] - mets[n]["recall"] for n in LABEL_NAMES}

print(f"\n  {'Group':<28}" + "".join(f"{s:>13}" for s in LABEL_SHORT))
print("  " + "─" * (28 + 13 * N_FAM))
print(f"  {'Full model recall':<28}" +
      "".join(f"{full_rec[n]:>13.3f}" for n in LABEL_NAMES))
print("  " + "─" * (28 + 13 * N_FAM))
for grp, drops in ablation_drop.items():
    row = f"  {grp:<28}"
    for n in LABEL_NAMES:
        row += f"{drops[n]:>+13.3f}"
    print(row)

save_json("ablation",
          {"full_recall": full_rec,
           "drop": {g: {n: float(v) for n, v in d.items()}
                    for g, d in ablation_drop.items()}})

# ═════════════════════════════════════════════════════════════════
# PHASE 5: THRESHOLD SWEEP
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 5: THRESHOLD SWEEP  (Pareto-optimal thresholds)")
print("="*80)

model_causal.eval()
with torch.no_grad():
    ds_v  = PrimeDataset(X_val, Y_val)
    ldr_v = DataLoader(ds_v, batch_size=2048)
    val_probs_list = []
    for xb, _ in ldr_v:
        val_probs_list.append(model_causal(xb.to(device)).cpu().numpy())
val_probs = np.concatenate(val_probs_list)

thresholds    = np.linspace(0.05, 0.95, 91)
sweep_results = {n: {"thresh": [], "prec": [], "rec": [], "f1": [], "red": []}
                 for n in LABEL_NAMES}

for tau in thresholds:
    for i, name in enumerate(LABEL_NAMES):
        yh = (val_probs[:, i] >= tau).astype(float)
        yt = Y_val[:, i]
        tp = int(((yh == 1) & (yt == 1)).sum())
        fp = int(((yh == 1) & (yt == 0)).sum())
        fn = int(((yh == 0) & (yt == 1)).sum())
        tn = int(((yh == 0) & (yt == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        red  = 100 * (tn + fn) / len(yt)
        sweep_results[name]["thresh"].append(float(tau))
        sweep_results[name]["prec"].append(prec)
        sweep_results[name]["rec"].append(rec)
        sweep_results[name]["f1"].append(f1)
        sweep_results[name]["red"].append(red)

opt_thresholds = {}
print(f"  {'Family':<18} {'τ(F1-max)':>12} {'F1':>8} {'Rec':>8} {'Red%':>8} │ "
      f"{'τ(Rec≥.95)':>12} {'Rec':>8} {'Red%':>8}")
print("  " + "─" * 92)

for name in LABEL_NAMES:
    sr          = sweep_results[name]
    best_f1_idx = int(np.argmax(sr["f1"]))
    tau_f1      = sr["thresh"][best_f1_idx]
    f1_val      = sr["f1"][best_f1_idx]
    rec_f1      = sr["rec"][best_f1_idx]
    red_f1      = sr["red"][best_f1_idx]
    cands       = [(sr["red"][j], sr["thresh"][j], sr["rec"][j])
                   for j in range(len(thresholds)) if sr["rec"][j] >= 0.95]
    if cands:
        best_cand = max(cands, key=lambda x: x[0])
        tau_95, rec_95, red_95 = best_cand[1], best_cand[2], best_cand[0]
        opt_thresholds[name] = {"tau_f1": tau_f1, "tau_95": tau_95}
        print(f"  {name:<18} {tau_f1:>12.2f} {f1_val:>8.3f} {rec_f1:>8.3f} "
              f"{red_f1:>7.1f}% │ {tau_95:>12.2f} {rec_95:>8.3f} {red_95:>7.1f}%")
    else:
        opt_thresholds[name] = {"tau_f1": tau_f1, "tau_95": None}
        print(f"  {name:<18} {tau_f1:>12.2f} {f1_val:>8.3f} {rec_f1:>8.3f} "
              f"{red_f1:>7.1f}% │ {'N/A':>12}")

save_json("threshold_sweep", {n: {k: [float(x) for x in v] if isinstance(v, list)
          else float(v) for k, v in sr.items()} for n, sr in sweep_results.items()})

# ═════════════════════════════════════════════════════════════════
# PHASE 6: CAUSAL vs NON-CAUSAL COMPARISON
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 6: CAUSAL vs NON-CAUSAL COMPARISON")
print("="*80)

nc_mets_val = evaluate(X_valn, Y_valn, model_noncausal,
                       tag="Non-causal model (val)", verbose=True)

print(f"\n  {'Family':<18} {'Causal Rec':>12} {'NC Rec':>10} {'Cost':>8} "
      f"{'Causal Red%':>14} {'NC Red%':>10}")
print("  " + "─" * 74)
for n in LABEL_NAMES:
    c_rec  = full_rec[n]
    nc_rec = nc_mets_val[n]["recall"]
    cost   = c_rec - nc_rec
    c_red  = full_mets[n]["reduction"]
    nc_red = nc_mets_val[n]["reduction"]
    print(f"  {n:<18} {c_rec:>12.3f} {nc_rec:>10.3f} "
          f"{cost:>+8.3f} {c_red:>13.1f}% {nc_red:>9.1f}%")
print("  (Negative cost = causal unexpectedly outperforms; see paper for discussion)")

# ═════════════════════════════════════════════════════════════════
# PHASE 7: MULTI-SCALE GENERALISATION
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 7: MULTI-SCALE GENERALIZATION")
print("="*80)

all_results = {k: {} for k in ["causal", "focal", "asl", "shallow", "noncausal", "xgboost"]}

for idx, (_, _, label, tag, ood) in enumerate(GEN_SPECS):
    Xc, Yc = GEN_X[idx],  GEN_Y[idx]
    Xn, Yn = GEN_XN[idx], GEN_YN[idx]
    print(f"\n  ── {label}  ({'OOD' if ood else 'in-dist'}) ──")

    all_results["causal"][tag]    = evaluate(Xc, Yc, model_causal,
                                             tag=f"Causal wBCE — {label}")
    all_results["focal"][tag]     = evaluate(Xc, Yc, model_focal,    verbose=False)
    all_results["asl"][tag]       = evaluate(Xc, Yc, model_asl,
                                             tag=f"ASL — {label}")
    all_results["shallow"][tag]   = evaluate(Xc, Yc, model_shallow,  verbose=False)
    all_results["noncausal"][tag] = evaluate(Xn, Yn, model_noncausal,
                                             tag=f"Non-causal — {label}")
    # XGBoost
    Xsc   = scaler.transform(Xc)
    xgb_m = {}
    for i, name in enumerate(LABEL_NAMES):
        pr   = xgb_models[name]["clf"].predict_proba(Xsc)[:, 1]
        pd_  = (pr > 0.5).astype(float)
        r_   = recall_score(Yc[:, i], pd_, zero_division=0)
        p_   = precision_score(Yc[:, i], pd_, zero_division=0)
        f_   = f1_score(Yc[:, i], pd_, zero_division=0)
        ap   = average_precision_score(Yc[:, i], pr) if Yc[:, i].sum() > 0 else 0.0
        tp   = int(((pd_ == 1) & (Yc[:, i] == 1)).sum())
        fn   = int(((pd_ == 0) & (Yc[:, i] == 1)).sum())
        tn   = int(((pd_ == 0) & (Yc[:, i] == 0)).sum())
        red  = 100 * (tn + fn) / len(Yc)
        miss = 100 * fn / max(tp + fn, 1)
        xgb_m[name] = dict(recall=r_, precision=p_, f1=f_, auc_pr=ap,
                            reduction=red, missed_pct=miss,
                            probs=pr, labels=Yc[:, i])
    all_results["xgboost"][tag] = xgb_m

# Summary tables
for metric, label_ in [("recall", "Recall"), ("reduction", "Search Reduction %")]:
    print(f"\n{'='*80}")
    print(f"  GENERALIZATION DECAY — {label_} (causal wBCE)")
    print(f"{'='*80}")
    print(f"\n  {'Family':<18}" + "".join(f"{t:>10}" for t in GEN_TAG))
    print("  " + "─" * (18 + 10 * len(GEN_TAG)))
    for name in LABEL_NAMES:
        row = f"  {name:<18}"
        for tag in GEN_TAG:
            v = all_results["causal"][tag][name][metric]
            row += f"{v:>10.3f}" if metric == "recall" else f"{v:>9.1f}%"
        print(row)

print(f"\n{'='*80}")
print("  MODEL COMPARISON AT 10^12 — Recall")
print(f"{'='*80}")
print(f"\n  {'Family':<18}" +
      "".join(f"{m:>10}" for m in ["Causal", "ASL", "Focal", "Shallow", "NC", "XGBoost"]))
print("  " + "─" * 78)
t12 = "1e12"
for name in LABEL_NAMES:
    rc = all_results["causal"][t12][name]["recall"]
    ra = all_results["asl"][t12][name]["recall"]
    rf = all_results["focal"][t12][name]["recall"]
    rs = all_results["shallow"][t12][name]["recall"]
    rn = all_results["noncausal"][t12][name]["recall"]
    rx = all_results["xgboost"][t12][name]["recall"]
    mark = " ◀" if rc == max(rc, ra, rf, rs, rn, rx) or ra == max(rc, ra, rf, rs, rn, rx) else ""
    print(f"  {name:<18} {rc:>10.3f} {ra:>10.3f} {rf:>10.3f} "
          f"{rs:>10.3f} {rn:>10.3f} {rx:>10.3f}{mark}")

print(f"\n{'='*80}")
print("  CAUSAL vs NON-CAUSAL — Full scale comparison (Recall)")
print(f"{'='*80}")
print(f"\n  {'Family / Scale':<28}" + "".join(f"{t:>8}" for t in GEN_TAG))
print("  " + "─" * (28 + 8 * len(GEN_TAG)))
for name in LABEL_NAMES:
    row_c = f"  {name+' (causal)':<28}"
    row_n = f"  {name+' (NC)':<28}"
    for tag in GEN_TAG:
        row_c += f"{all_results['causal'][tag][name]['recall']:>8.3f}"
        row_n += f"{all_results['noncausal'][tag][name]['recall']:>8.3f}"
    print(row_c); print(row_n); print()

# Export (strip non-serialisable arrays)
export = {}
for model_name, scale_dict in all_results.items():
    export[model_name] = {}
    for tag, fam_dict in scale_dict.items():
        export[model_name][tag] = {}
        for fam, m in fam_dict.items():
            export[model_name][tag][fam] = {
                k: float(v) for k, v in m.items()
                if k not in ("probs", "labels")
            }
save_json("all_results", export)

# ═════════════════════════════════════════════════════════════════
# PHASE 7b: ASL vs wBCE STRUCTURAL SPLIT TABLE
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  PHASE 7b: ASL vs wBCE — STRUCTURAL SPLIT BY FAMILY TYPE")
print("="*80)

# Gap-defined families: twin, sophie_germain, safe, cousin, sexy
# Algebraic families:   chen, isolated
GAP_FAMILIES = ["twin", "sophie_germain", "safe", "cousin", "sexy"]
ALG_FAMILIES = ["chen", "isolated"]
OOD_TAGS     = [t for t, ood in zip(GEN_TAG, GEN_OOD) if ood]

asl_split = {"gap_families": {}, "algebraic_families": {}}

print(f"\n  Gap-defined families — winner per scale (ASL vs wBCE recall)")
print(f"  {'Family':<18}" + "".join(f"{t:>10}" for t in OOD_TAGS))
print("  " + "─" * (18 + 10 * len(OOD_TAGS)))
for name in GAP_FAMILIES:
    row = f"  {name:<18}"
    asl_split["gap_families"][name] = {}
    for t in OOD_TAGS:
        rc = all_results["causal"][t][name]["recall"]
        ra = all_results["asl"][t][name]["recall"]
        winner = "ASL" if ra > rc else "wBCE"
        margin = abs(ra - rc)
        row += f"  {winner}({margin:.3f})"
        asl_split["gap_families"][name][t] = {
            "wbce_recall": float(rc), "asl_recall": float(ra),
            "winner": winner, "margin": float(margin)
        }
    print(row)

print(f"\n  Algebraic families — winner per scale (ASL vs wBCE recall)")
print(f"  {'Family':<18}" + "".join(f"{t:>10}" for t in OOD_TAGS))
print("  " + "─" * (18 + 10 * len(OOD_TAGS)))
for name in ALG_FAMILIES:
    row = f"  {name:<18}"
    asl_split["algebraic_families"][name] = {}
    for t in OOD_TAGS:
        rc = all_results["causal"][t][name]["recall"]
        ra = all_results["asl"][t][name]["recall"]
        winner = "ASL" if ra > rc else "wBCE"
        margin = abs(ra - rc)
        row += f"  {winner}({margin:.3f})"
        asl_split["algebraic_families"][name][t] = {
            "wbce_recall": float(rc), "asl_recall": float(ra),
            "winner": winner, "margin": float(margin)
        }
    print(row)

# Summary: count perfect split
gap_asl_wins = sum(
    1 for name in GAP_FAMILIES for t in OOD_TAGS
    if asl_split["gap_families"][name][t]["winner"] == "ASL"
)
alg_wbce_wins = sum(
    1 for name in ALG_FAMILIES for t in OOD_TAGS
    if asl_split["algebraic_families"][name][t]["winner"] == "wBCE"
)
gap_total  = len(GAP_FAMILIES) * len(OOD_TAGS)
alg_total  = len(ALG_FAMILIES) * len(OOD_TAGS)
print(f"\n  Summary:")
print(f"    Gap-defined  — ASL wins {gap_asl_wins}/{gap_total} cells "
      f"({100*gap_asl_wins/gap_total:.0f}%)")
print(f"    Algebraic    — wBCE wins {alg_wbce_wins}/{alg_total} cells "
      f"({100*alg_wbce_wins/alg_total:.0f}%)")

asl_split["summary"] = {
    "gap_asl_wins":    gap_asl_wins,   "gap_total":   gap_total,
    "alg_wbce_wins":   alg_wbce_wins,  "alg_total":   alg_total,
}
save_json("asl_split", asl_split)


print("\n" + "="*80)
print("  GENERATING PUBLICATION FIGURES")
print("="*80)

fig01_training_curves(hist_tc, hist_vc, hist_ta, hist_va,
                      hist_tf, hist_vf, hist_ts, hist_vs,
                      hist_tn, hist_vn)
fig02_generalization_causal(all_results)
fig03_causal_vs_noncausal(all_results)
fig04_model_comparison(all_results)
fig05_depth_necessity(all_results)
fig06_ablation_heatmap(ablation_drop)
fig07_threshold_sweep(sweep_results, opt_thresholds)
fig08_pr_curves(all_results)
fig09_calibration(all_results)
fig10_score_separation(all_results)
fig11_robustness(seed_store)
fig12_asl_vs_wbce(all_results)
fig13_new_families(all_results)
fig14_summary_composite(all_results, seed_store, ablation_drop)
fig15_density_recall(GEN_Y, all_results)

# ── Final listing ─────────────────────────────────────────────────
print("\n" + "="*80)
print("  COMPLETE — all figures and results saved")
print("="*80)
print("\nFigures (upload to LaTeX/Overleaf):")
for f in sorted(os.listdir("paper/figures")):
    if f.endswith(".pdf"):
        size = os.path.getsize(f"paper/figures/{f}") // 1024
        print(f"  paper/figures/{f:<48} ({size} KB)")
print("\nResults (JSON):")
for f in sorted(os.listdir("paper/results")):
    print(f"  paper/results/{f}")
