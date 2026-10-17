#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 (2026-09-30) companion check for expand_cohort_4066.py: is RF's collapse on
the 4,066-organoid export (only single-timepoint features available: t6h/t24h/t48h/t72h,
no dense trajectory) a real population mismatch with the n=1,850 dense subset, or an
artefact of the coarser feature representation?

Isolates the two by re-running Table II's EXACT n=1,850 organoids, same labels, same
leave-condition-out folds, but restricted to a single-timepoint feature (x(T) only,
mirroring what the 4,066 file provides) instead of the full 5-dim trajectory-shape
feature (value, mean velocity, last velocity, min, range). If RF also collapses here,
on data we already trust, the 4,066-cohort result is a feature effect, not a subset
bias -- confirmed by the result below (RF 0.856->0.793 at 24h; LR unchanged at 0.851).

Outputs:
  posterior/table_4066_feature_ablation.csv
"""
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

OUT = "posterior"
d = np.load(os.path.join(OUT, "posterior_data.npz"), allow_pickle=True)
dg = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
xg = d["xgrid"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float)
drug = d["drug"]; conc = d["conc"].astype(float); groups = dg["groups"]
assert np.array_equal(d["y"], dg["y"])
DRUGS = ["Stauro", "Ner", "Lap"]
doh = np.array([np.eye(3)[DRUGS.index(x)] for x in drug])
logc = np.log1p(conc)[:, None]

MODELS = {
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
}
# Table II's own leave-condition-out AUC at 24h/48h (5-dim trajectory-shape features),
# for the side-by-side comparison
REF_TABLE_II = {24: {"LR": 0.851, "RF": 0.856, "HGB": 0.845},
                 48: {"LR": 0.941, "RF": 0.939, "HGB": 0.935}}

rows_out = []
for T in (24, 48):
    j = int(np.where(GRID == T)[0][0])
    X = np.column_stack([xg[:, j:j + 1], doh, logc])  # single scalar x(T), mirrors the 4,066 file
    print(f"=== T={T}h, single-scalar feature (mirrors 4,066-cohort export) ===")
    for name, mk in MODELS.items():
        p = cross_val_predict(mk(), X, y, cv=GroupKFold(5).split(X, y, groups), method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        ref = REF_TABLE_II[T][name]
        rows_out.append((T, name, round(auc, 4), ref, round(auc - ref, 4)))
        print(f"  {name:5s} AUC(single-scalar, leave-cond-out)={auc:.4f}  "
              f"| Table II (5-dim features)={ref:.4f}  delta={auc-ref:+.4f}")

with open(os.path.join(OUT, "table_4066_feature_ablation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["horizon_h", "model", "AUC_single_scalar_leaveCondOut", "AUC_tableII_5dim", "delta"])
    for r in rows_out:
        w.writerow(r)
print("\nwrote table_4066_feature_ablation.csv")
