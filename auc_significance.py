#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 (AUC significance): pairwise DeLong test + paired bootstrap for every
model pair in Table S11 (fixed-time baselines) at T=24h and T=48h. Answers the
question directly: is RF (0.875 @24h) significantly different from the
other clustered baselines, and from the trajectory encoder (0.772 @24h)? Reuses
build_table2_baselines.py's exact model set, features, and CV folds so
predictions are properly paired (same organoids, same fold membership).

Outputs:
  posterior/table_auc_pairwise_delong.csv    all C(7,2)=21 pairs x 2 horizons,
                                              DeLong z/p and paired-bootstrap
                                              deltaAUC / 95% CI / p
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, sys, csv, itertools, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import openpyxl
from scipy.stats import norm
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

SEED = 0
XLSX = os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP = {"Staurosporine": "Stauro", "Neratinib": "Ner", "Lapatinib": "Lap"}
DRUGS = ["Stauro", "Ner", "Lap"]
OUT = "posterior"


def parse(ws, cell):
    rows = list(ws.iter_rows(values_only=True))
    ci = [i for i, r in enumerate(rows) if isinstance(r[1], str) and any(d in r[1] for d in DMAP)]
    out = []
    for k, idx in enumerate(ci):
        end = ci[k + 1] if k + 1 < len(ci) else len(rows)
        lab = rows[idx][1].split(); conc = float(lab[0]); drug = DMAP[lab[1]]
        blk = [r for r in (rows[j] for j in range(idx, end)) if isinstance(r[2], (int, float)) and r[2] <= 72.5]
        t = np.array([r[2] for r in blk], float); nc = max(len(r) for r in blk)
        for c in range(3, nc):
            m = np.array([(r[c] if c < len(r) and isinstance(r[c], (int, float)) else np.nan) for r in blk], float)
            if np.isfinite(m).sum() >= 100:
                out.append((cell, drug, conc, t, m))
    return out


wb = openpyxl.load_workbook(XLSX, read_only=True)
recs = parse(wb["Supp Fig 9 MCF"], "MCF-7") + parse(wb["Supp Fig 9 BT"], "BT-474")
# data-integrity filter (2026-07-16): drop organoids whose normalized mass ratio
# exceeds 5x at any horizon -- source-corrupted tracks (Lapatinib 10uM columns that
# wrote timestamps instead of mass). Must match build_posterior.py exactly.
def _rmax(t, m):
    ok = np.isfinite(m)
    if ok.sum() < 2: return np.inf
    base = np.interp(6, t[ok], m[ok])
    return np.inf if base <= 1e-3 else float(np.nanmax(m[ok]) / base)
_n0 = len(recs); recs = [r for r in recs if _rmax(r[3], r[4]) <= 5.0]
print(f"data-integrity filter: kept {len(recs)}/{_n0}")
cell = np.array([r[0] for r in recs]); drug = np.array([r[1] for r in recs])
doh = np.array([np.eye(3)[DRUGS.index(d)] for d in drug])
logc = np.array([[np.log(r[2] + 1)] for r in recs])
cond_key = np.array([f"{c}|{d}|{cc}" for c, d, cc in zip(cell, drug, [r[2] for r in recs])])
cond_ids = {k: i for i, k in enumerate(sorted(set(cond_key)))}
groups = np.array([cond_ids[k] for k in cond_key])


def val(t, m, tt):
    ok = np.isfinite(m)
    return np.interp(tt, t[ok], m[ok])


y = np.array([int(val(t, m, 72) / (val(t, m, 6) + 1e-8) < 0.90) for (_, _, _, t, m) in recs])
N = len(y)
folds = GroupKFold(5)


def feats5(t, m, T, k=12):
    ok = np.isfinite(m)
    g = np.linspace(6, T, k); w = np.interp(g, t[ok], m[ok]); w = w / (w[0] + 1e-8)
    dt = (T - 6) / (k - 1); v = np.gradient(w, dt)
    return [w[-1], v.mean(), v[-1], w.min(), w.max() - w.min()]


def traj(t, m, T, k=12):
    ok = np.isfinite(m)
    g = np.linspace(6, T, k); w = np.interp(g, t[ok], m[ok])
    return list(w / (w[0] + 1e-8))



MODELS = {
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "HGB": lambda: HistGradientBoostingClassifier(random_state=SEED),
    "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
    "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "near-Bayes (LDA)": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
}


# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu 2014 fast implementation via midranks); computes AUC
# variance/covariance for a set of paired (same-sample) classifiers at once.
# ---------------------------------------------------------------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(n, dtype=float)
    T2[J] = T
    return T2


def delong_cov(preds_by_model, labels):
    """preds_by_model: dict name -> array[N] of scores, all on the SAME N
    samples/labels (labels can vary in ordering internally, we sort here)."""
    order = np.argsort(-labels, kind="stable")  # positives (y=1) first
    labels_s = labels[order]
    m = int(labels_s.sum())  # positives
    n = len(labels_s) - m    # negatives
    names = list(preds_by_model.keys())
    X = np.vstack([preds_by_model[nm][order] for nm in names])  # [k, N]
    k = X.shape[0]
    pos = X[:, :m]; neg = X[:, m:]
    tx = np.vstack([compute_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([compute_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([compute_midrank(X[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    cov = sx / m + sy / n
    return names, aucs, cov


def delong_pair_test(auc_i, auc_j, cov_ii, cov_jj, cov_ij):
    var = cov_ii + cov_jj - 2 * cov_ij
    if var <= 0:
        return 0.0, 1.0
    z = (auc_i - auc_j) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def paired_bootstrap(pi, pj, yv, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    Nn = len(yv)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, Nn, Nn)
        deltas[b] = roc_auc_score(yv[idx], pi[idx]) - roc_auc_score(yv[idx], pj[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_boot = 2 * min((deltas > 0).mean(), (deltas < 0).mean())
    p_boot = min(p_boot, 1.0)
    return float(deltas.mean()), float(lo), float(hi), float(p_boot)


rows_out = []
for T in (24, 48):
    print(f"\n===== T={T}h =====")
    F5 = np.column_stack([np.array([feats5(t, m, T) for (_, _, _, t, m) in recs], float), doh, logc])
    FT = np.column_stack([np.array([traj(t, m, T) for (_, _, _, t, m) in recs], float), doh, logc])
    preds = {}
    for name, mk in MODELS.items():
        p = cross_val_predict(mk(), F5, y, cv=folds.split(F5, y, groups), method="predict_proba")[:, 1]
        preds[name] = p
        print(f"  {name:18s} AUC={roc_auc_score(y, p):.4f}")
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                            FT, y, cv=folds.split(FT, y, groups), method="predict_proba")[:, 1]
    preds["trajectory encoder"] = p
    print(f"  {'trajectory encoder':18s} AUC={roc_auc_score(y, p):.4f}")

    names, aucs, cov = delong_cov(preds, y)
    idx_of = {nm: i for i, nm in enumerate(names)}
    for a, b in itertools.combinations(names, 2):
        ia, ib = idx_of[a], idx_of[b]
        z, p_delong = delong_pair_test(aucs[ia], aucs[ib], cov[ia, ia], cov[ib, ib], cov[ia, ib])
        d_mean, lo, hi, p_boot = paired_bootstrap(preds[a], preds[b], y)
        rows_out.append((T, a, b, round(float(aucs[ia]), 4), round(float(aucs[ib]), 4),
                          round(float(aucs[ia] - aucs[ib]), 4), round(z, 3), round(p_delong, 4),
                          round(d_mean, 4), round(lo, 4), round(hi, 4), round(p_boot, 4)))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "table_auc_pairwise_delong.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["horizon_h", "model_a", "model_b", "auc_a", "auc_b", "deltaAUC_point",
                "delong_z", "delong_p", "bootstrap_deltaAUC_mean", "bootstrap_lo95", "bootstrap_hi95", "bootstrap_p"])
    for r in rows_out:
        w.writerow(r)
print("\nwrote table_auc_pairwise_delong.csv")

print("\n" + "=" * 70)
print("KEY COMPARISONS (RF vs everything else):")
for r in rows_out:
    T, a, b = r[0], r[1], r[2]
    if "RF" in (a, b):
        other = b if a == "RF" else a
        sign = "" if a == "RF" else "-"
        d = r[5] if a == "RF" else -r[5]
        sig = "***SIGNIFICANT***" if r[7] < 0.05 else "n.s."
        print(f"  T={T:>2}h  RF vs {other:20s}  deltaAUC={d:+.4f}  DeLong p={r[7]:.4f}  bootstrap p={r[11]:.4f}  {sig}")
