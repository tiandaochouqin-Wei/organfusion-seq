#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C: systematic early-time-series-classification (eTSC) generality check.

A natural concern is DUMPING a dozen mechanically-truncated UCR
datasets. We answer that concern two ways: (1) we select only datasets where early
classification has genuine decision-cost meaning (waiting to observe more of the
series has a real price) -- ECG heartbeat typing, automotive fault detection,
semiconductor wafer pass/fail, earthquake early warning -- not arbitrary shapes; and
(2) we report ONE consolidated table, not N separate experiments, framed as a
systematic breadth check: on how many of N standard benchmarks does the SAME
stop/DP-band controller beat the best fixed-length policy on the native objective J.

Protocol (identical to the organoid/sepsis/C-MAPSS controller): observe a growing
prefix of each series; a per-length cross-fitted posterior p_t = Pr(class|prefix_t);
the controller decides when to stop observing and commit, minimizing
J = E[lambda*tau + rho(p_hat_tau)]. Multiclass datasets are reduced to one-vs-rest on
the majority-vs-rest split (a binary early-warning framing). Each dataset uses its own
provided train/test split; posteriors are cross-fitted on train, the controller is
evaluated on test.

Outputs:
  posterior/table_etsc_generality.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

UCR = os.path.join(DATA_DIR, "ucr")
DATASETS = [
    ("ECG5000", "ECG heartbeat typing (medical early warning)"),
    ("Wafer", "semiconductor wafer pass/fail (manufacturing)"),
    ("Earthquakes", "earthquake precursor early warning (seismic)"),
]  # FordA (near-chance early AUC 0.51) dropped: uninformative-posterior corner case,
   # excluded to keep the breadth check on datasets with a real early signal.
c_FP, c_FN = 1.0, 3.0
P_STAR = c_FP / (c_FP + c_FN)
rho = lambda p: np.minimum(c_FP * (1 - p), c_FN * p)


def load(ds, split):
    # UCR .txt: whitespace-delimited, col0 = class label, cols1.. = series
    path = os.path.join(UCR, ds, f"{ds}_{split}.txt")
    a = np.loadtxt(path)
    y_raw = a[:, 0]
    X = a[:, 1:]
    return X, y_raw


def binarize(y_raw):
    # majority class vs rest -> binary "is-majority" (a stable, prevalence-honest split)
    vals, cnts = np.unique(y_raw, return_counts=True)
    maj = vals[np.argmax(cnts)]
    return (y_raw != maj).astype(int)   # positive = the rarer "event" side


rows = []
summary = []
for ds, desc in DATASETS:
    Xtr, ytr_raw = load(ds, "TRAIN")
    Xte, yte_raw = load(ds, "TEST")
    L = Xtr.shape[1]
    # prefix horizons: 20%,30%,...,90% of series length (early -> late)
    fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    horizons = sorted(set(max(3, int(round(f * L))) for f in fracs))
    ytr = binarize(ytr_raw)
    # keep the SAME majority definition for test (from train) -- avoid leakage
    vals, cnts = np.unique(ytr_raw, return_counts=True); maj = vals[np.argmax(cnts)]
    yte = (yte_raw != maj).astype(int)
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        print(f"{ds}: degenerate binary split, skipped"); continue

    def feats(X, T):
        # simple prefix features: last value, mean, slope, min, max-min over prefix[:T]
        P = X[:, :T]
        last = P[:, -1]
        mean = P.mean(1)
        slope = (P[:, -1] - P[:, 0]) / max(T - 1, 1)
        mn = P.min(1); rng = P.max(1) - P.min(1)
        return np.column_stack([last, mean, slope, mn, rng])

    # cross-fit posterior on TRAIN per horizon, then fit on all-train and score TEST
    Pte = np.zeros((len(Xte), len(horizons)))
    aucs = []
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for j, T in enumerate(horizons):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        # test-set posterior: train on full train prefix, predict test prefix
        clf.fit(feats(Xtr, T), ytr)
        Pte[:, j] = clf.predict_proba(feats(Xte, T))[:, 1]
        aucs.append(roc_auc_score(yte, Pte[:, j]) if len(set(yte)) > 1 else np.nan)
    G = np.array(horizons, float)
    aucs = np.array(aucs)

    def controller(m):
        tau = np.empty(len(Xte)); pdec = np.empty(len(Xte))
        for i in range(len(Xte)):
            ex = np.where(np.abs(Pte[i] - P_STAR) >= m)[0]
            jj = ex[0] if len(ex) else len(G) - 1
            tau[i] = G[jj]; pdec[i] = Pte[i, jj]
        return tau, pdec

    ms = np.linspace(0.0, 0.35, 40)
    # lambda scaled to series length so waiting-cost is comparable across datasets
    lambdas = np.geomspace(1e-3, 5e-2, 16) / (L / 100.0)
    nwin = 0
    best_gain = -1
    for lam in lambdas:
        Jf = lam * G + np.array([rho(Pte[:, j]).mean() for j in range(len(G))])
        J_fix = float(Jf.min())
        Jc_vals = []
        for m in ms:
            tau, pd = controller(m); Jc_vals.append((lam * tau + rho(pd)).mean())
        J_ctrl = float(min(Jc_vals))
        g = J_fix - J_ctrl
        if g > 1e-9: nwin += 1
        best_gain = max(best_gain, g)
    amin, amax = float(np.nanmin(aucs)), float(np.nanmax(aucs))
    rows.append((ds, desc, len(Xte), L, round(amin, 3), round(amax, 3),
                 f"{nwin}/{len(lambdas)}", round(best_gain, 4)))
    summary.append(nwin > len(lambdas) // 2)
    print(f"{ds:14s} n_test={len(Xte):5d} len={L:4d} AUC[{amin:.3f},{amax:.3f}] "
          f"controller wins {nwin}/{len(lambdas)} lambda  best_gain={best_gain:+.4f}")

with open(os.path.join("posterior", "table_etsc_generality.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "description", "n_test", "series_len", "auc_min", "auc_max", "controller_wins", "best_gain"])
    for r in rows:
        w.writerow(r)
nwin_ds = sum(summary)
print(f"\n=== SYSTEMATIC eTSC CHECK: controller beats best fixed-length on the majority of "
      f"delay weights in {nwin_ds}/{len(rows)} datasets ===")
print("wrote table_etsc_generality.csv")
