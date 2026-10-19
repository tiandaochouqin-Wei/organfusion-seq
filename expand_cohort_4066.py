#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset-expansion check (Stage 2, 2026-09-30): does the graded (not tiered)
discrimination structure of E1/Table S11 -- fit on the n=1,850 dense subset,
leave-condition-out GroupKFold, the paper's main protocol since Stage 1 -- replicate
on the full n=4,066 development cohort that the paper's own Limitations flags as the
source population ("the dense cohort is a public 'randomly selected' subset,
n=1,850<4,066")? Both protocols are computed and reported; leave-condition-out
(p_group) is primary, matching every other headline number in the paper, and the
organoid-level stratified fit (p_strat) is an explicit upper-bound reference.

Verified before running (not assumed): main_development_cohort_4066.csv's `label`
column matches y=1[end_ratio<0.9] with 0 mismatches across all 4,066 rows (same task
definition as the paper); same 2 cell lines x 3 drugs x 3 concentrations = 18
conditions as the 1,888 subset; t6h is always exactly 1.0 (same anchoring
convention). This is the same underlying Tebon cohort at a coarser temporal grid,
not a different task.

Scope limit (deliberate, not an oversight): this file only has 4 timepoints
(6/24/48/72h). t72h IS the value the label thresholds (y=1[x(72)<0.9]), so it cannot
be used as an INPUT feature at any horizon without leaking the label -- exactly the
same same-modality concern flagged in Sec. "Label semantics" of the paper. We
therefore only test T=24h and T=48h (mirroring Table S11's own two columns), never T=72h.
This also means the dense multi-point sequential/stopping-controller experiments
(E2, E13, etc.) CANNOT be extended to this file -- it lacks the 13-point grid they
require. This script checks discrimination only.

Outputs:
  posterior/table_4066_expansion.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

SEED = 0
OUT = "posterior"
DRUGS = ["Stauro", "Ner", "Lap"]

rows = list(csv.DictReader(open(os.path.join(DATA_DIR, "main_development_cohort_4066.csv"),
                                 encoding="utf-8")))
N = len(rows)
print(f"N = {N}")

y = np.array([int(r["label"]) for r in rows])
cell = np.array([r["cell_line"] for r in rows])
drug_raw = [r["drug"].split() for r in rows]  # e.g. ["0.1","Lap"]
conc = np.array([float(d[0]) for d in drug_raw])
drug = np.array([d[1] for d in drug_raw])
t24 = np.array([float(r["t24h"]) for r in rows])
t48 = np.array([float(r["t48h"]) for r in rows])

doh = np.array([np.eye(3)[DRUGS.index(d)] for d in drug])
logc = np.log1p(conc)[:, None]

cond_key = np.array([f"{c}|{d}|{cc}" for c, d, cc in zip(cell, drug, conc)])
uniq = {k: i for i, k in enumerate(sorted(set(cond_key)))}
groups = np.array([uniq[k] for k in cond_key])
print(f"conditions: {len(uniq)}  (matches 18 in the 1,888 subset: {len(uniq) == 18})")

MODELS = {
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "HGB": lambda: HistGradientBoostingClassifier(random_state=SEED),
}

folds_strat = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds_group = GroupKFold(5)

# reference numbers already established on the n=1,850 dense subset (Table S11,
# leave-condition-out GroupKFold -- the paper's main protocol since Stage 1)
REF_1850_GROUP = {
    24: {"LR": 0.851, "RF": 0.856, "HGB": 0.845},
    48: {"LR": 0.941, "RF": 0.939, "HGB": 0.935},
}

rows_out = []
preds_group_for_delong = {}  # (T, model) -> p_group, for leave-condition-out pairwise significance
preds_strat_for_delong = {}  # (T, model) -> p_strat, for the upper-bound reference check
for T, tcol in [(24, t24), (48, t48)]:
    X = np.column_stack([tcol[:, None], doh, logc])
    print(f"\n=== T={T}h (N={N}) ===")
    for name, mk in MODELS.items():
        p_strat = cross_val_predict(mk(), X, y, cv=folds_strat, method="predict_proba")[:, 1]
        p_group = cross_val_predict(mk(), X, y, cv=folds_group.split(X, y, groups), method="predict_proba")[:, 1]
        a_strat = roc_auc_score(y, p_strat)
        a_group = roc_auc_score(y, p_group)
        ref = REF_1850_GROUP[T][name]
        rows_out.append((T, name, N, round(a_group, 4), round(a_strat, 4), ref, round(a_group - ref, 4)))
        preds_group_for_delong[(T, name)] = p_group
        preds_strat_for_delong[(T, name)] = p_strat
        print(f"  {name:5s}  AUC(leave-cond-out,n={N})={a_group:.4f}  AUC(stratified,upper bound)={a_strat:.4f}  "
              f"| n=1850 leave-cond-out ref={ref:.4f}  delta={a_group-ref:+.4f}")

# DeLong-style pairwise check at the larger N: is RF still significantly above LR at N=4066?
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; n = len(x); T_ = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[J] = T_
    return out

def delong_test(p1, p2, yv):
    order = np.argsort(-yv, kind="stable")
    ys = yv[order]; m = int(ys.sum()); n = len(ys) - m
    X = np.vstack([p1[order], p2[order]])
    pos = X[:, :m]; neg = X[:, m:]
    tx = np.vstack([compute_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([compute_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([compute_midrank(X[r]) for r in range(2)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0] - aucs[1]), float(p)

print("\n=== graded-structure check at N=4066, leave-condition-out (same DeLong test as E1/Table S11) ===")
PAIRS = [("RF", "LR"), ("RF", "HGB"), ("LR", "HGB")]
for T in (24, 48):
    for a, b in PAIRS:
        d_auc, p_val = delong_test(preds_group_for_delong[(T, a)], preds_group_for_delong[(T, b)], y)
        sig = "SIGNIFICANT" if p_val < 0.05 else "n.s."
        print(f"  T={T}h  {a}-{b} deltaAUC={d_auc:+.4f}  DeLong p={p_val:.4g}  {sig}  [leave-cond-out]")
        rows_out.append((T, f"{a}_vs_{b}_DeLong_leaveCondOut", N, round(d_auc, 4), round(p_val, 4), "", ""))
    # also report the upper-bound (stratified) version of the headline RF-HGB gap, for comparison
    d_auc_s, p_val_s = delong_test(preds_strat_for_delong[(T, "RF")], preds_strat_for_delong[(T, "HGB")], y)
    print(f"  T={T}h  RF-HGB deltaAUC={d_auc_s:+.4f}  DeLong p={p_val_s:.4g}  "
          f"[stratified, upper bound reference]")
    rows_out.append((T, "RF_vs_HGB_DeLong_stratified", N, round(d_auc_s, 4), round(p_val_s, 4), "", ""))

with open(os.path.join(OUT, "table_4066_expansion.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["horizon_h", "model_or_test", "N", "value1", "value2", "ref_1850_leaveCondOut", "delta_vs_1850"])
    for r in rows_out:
        w.writerow(r)
print("\nwrote table_4066_expansion.csv")
print("\nNote: value1=AUC(leave-condition-out) for model rows, value1=deltaAUC(DeLong) for pairwise rows;")
print("      value2=AUC(stratified, upper bound) for model rows, value2=DeLong p for pairwise rows.")
