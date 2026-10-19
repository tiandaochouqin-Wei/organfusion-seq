#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 (grouped CV): the public Tebon cohort has NO well/plate/field id below
the (cell_line x drug x concentration) condition level (18 conditions total; each
condition IS one experimental block of ~100 organoids). Plate/well-grouped
CV is the natural check; the honest version achievable on this data is
LEAVE-CONDITION-OUT (GroupKFold with group=condition), which is the strictest
grouping the data supports (organoids within a condition share a plate/run).

Produces, alongside the existing stratified (organoid-level) CV as an upper bound:
  posterior/posterior_data_grouped.npz   P_strat, P_group (both [N,G]), y, drug, conc, cell, GRID
  posterior/table_cv_auc_comparison.csv  per-horizon AUC, stratified vs grouped
  posterior/table_matched_ops_grouped.csv  9 matched OPs (as tab:matched) under GroupKFold posterior,
                                            with condition-level cluster-bootstrap 95% CI
Both P matrices are leakage-free WITHIN their own CV scheme; comparing them bounds
how much within-condition (organoid-level) sharing could inflate the headline numbers.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import openpyxl
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 0
XLSX = os.path.join(DATA_DIR, "Tebon_SupplementaryData_MOESM6.xlsx")
DMAP = {"Staurosporine": "Stauro", "Neratinib": "Ner", "Lapatinib": "Lap"}
DRUGS = ["Stauro", "Ner", "Lap"]
OUT = "posterior"
os.makedirs(OUT, exist_ok=True)


def parse(ws, cell):
    rows = list(ws.iter_rows(values_only=True))
    ci = [i for i, r in enumerate(rows) if isinstance(r[1], str) and any(d in r[1] for d in DMAP)]
    out = []
    for k, idx in enumerate(ci):
        end = ci[k + 1] if k + 1 < len(ci) else len(rows)
        lab = rows[idx][1].split()
        conc = float(lab[0]); drug = DMAP[lab[1]]
        blk = [r for r in (rows[j] for j in range(idx, end)) if isinstance(r[2], (int, float)) and r[2] <= 72.5]
        t = np.array([r[2] for r in blk], float)
        nc = max(len(r) for r in blk)
        for c in range(3, nc):
            m = np.array([(r[c] if c < len(r) and isinstance(r[c], (int, float)) else np.nan) for r in blk], float)
            if np.isfinite(m).sum() >= 100:
                out.append((drug, conc, cell, t, m))
    return out


print("parsing Tebon xlsx ...")
wb = openpyxl.load_workbook(XLSX, read_only=True)
recs = parse(wb["Supp Fig 9 MCF"], "MCF-7") + parse(wb["Supp Fig 9 BT"], "BT-474")
# data-integrity filter (2026-07-16): match build_posterior.py -- drop ratio>5x tracks
# (source-corrupted Lapatinib 10uM columns). recs tuple = (drug,conc,cell,t,m); t,m at [3],[4].
def _rmax(t, m):
    ok = np.isfinite(m)
    if ok.sum() < 2: return np.inf
    b = np.interp(6, t[ok], m[ok]); return np.inf if b <= 1e-3 else float(np.nanmax(m[ok]) / b)
_n0 = len(recs); recs = [r for r in recs if _rmax(r[3], r[4]) <= 5.0]
print(f"data-integrity filter: kept {len(recs)}/{_n0}")
N = len(recs)
print("organoids:", N)
drug = np.array([r[0] for r in recs]); conc = np.array([r[1] for r in recs]); cell = np.array([r[2] for r in recs])
doh = np.array([np.eye(3)[DRUGS.index(d)] for d in drug]); logc = np.array([[np.log(c + 1)] for c in conc])


def val(t, m, tt):
    ok = np.isfinite(m)
    return np.interp(tt, t[ok], m[ok])


y = np.array([int(val(t, m, 72) / (val(t, m, 6) + 1e-8) < 0.90) for (_, _, _, t, m) in recs])
GRID = [12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72]


def feats(t, m, T, k=12):
    ok = np.isfinite(m)
    g = np.linspace(6, T, k)
    w = np.interp(g, t[ok], m[ok]); w = w / (w[0] + 1e-8)
    dt = (T - 6) / (k - 1); v = np.gradient(w, dt)
    return [w[-1], v.mean(), v[-1], w.min(), w.max() - w.min()]


# ---- condition = leave-condition-out grouping key (18 conditions; finest unit the
# public data supports since organoids within a condition share one experimental block) ----
cond_key = np.array([f"{c}|{d}|{cc}" for c, d, cc in zip(cell, drug, conc)])
cond_ids = {k: i for i, k in enumerate(sorted(set(cond_key)))}
groups = np.array([cond_ids[k] for k in cond_key])
n_cond = len(cond_ids)
print(f"n conditions (leave-condition-out group key): {n_cond}")

folds_strat = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds_group = GroupKFold(5)

P_strat = np.zeros((N, len(GRID)))
P_group = np.zeros((N, len(GRID)))
rows_auc = []
for j, T in enumerate(GRID):
    X = np.column_stack([np.array([feats(t, m, T) for (_, _, _, t, m) in recs], float), doh, logc])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    P_strat[:, j] = cross_val_predict(clf, X, y, cv=folds_strat, method="predict_proba")[:, 1]
    P_group[:, j] = cross_val_predict(clf, X, y, cv=folds_group.split(X, y, groups), method="predict_proba")[:, 1]
    a_s = roc_auc_score(y, P_strat[:, j]); a_g = roc_auc_score(y, P_group[:, j])
    rows_auc.append((T, a_s, a_g))
    print(f"  T={T:>2}h  AUC stratified(organoid-CV, upper bound)={a_s:.3f}   "
          f"AUC leave-condition-out(no-leakage lower bound)={a_g:.3f}   gap={a_s - a_g:+.3f}")

np.savez(os.path.join(OUT, "posterior_data_grouped.npz"),
          P_strat=P_strat, P_group=P_group, y=y, drug=drug, conc=conc, cell=cell,
          GRID=np.array(GRID), groups=groups)
with open(os.path.join(OUT, "table_cv_auc_comparison.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["horizon_h", "auc_stratified_organoidCV", "auc_groupKFold_leaveConditionOut", "gap"])
    for T, a_s, a_g in rows_auc:
        w.writerow([T, round(a_s, 5), round(a_g, 5), round(a_s - a_g, 5)])
print("wrote table_cv_auc_comparison.csv")

# ---------------------------------------------------------------------------
# Matched operating points under the leakage-free (grouped) posterior, with
# condition-level cluster bootstrap 95% CI (resample the 18 conditions, not
# the 1,888 organoids independently).
# ---------------------------------------------------------------------------
G = np.array(GRID, float)


def controller(P, m):
    times = np.empty(N); dec = np.empty(N, int); ex = np.zeros(N, bool)
    for i in range(N):
        e = np.where(np.abs(P[i] - 0.5) >= m)[0]
        j = e[0] if len(e) else len(G) - 1
        times[i] = G[j]; dec[i] = int(P[i, j] >= 0.5); ex[i] = len(e) > 0
    return times, dec, ex


ft = G.copy()
fe_group = np.array([(np.array(P_group[:, j] >= 0.5, int) != y).mean() for j in range(len(G))])

mvals9 = np.linspace(0.10, 0.42, 9)
cond_list = sorted(set(groups.tolist()))
cond_to_idx = {c: np.where(groups == c)[0] for c in cond_list}

rows_ops = []
np.random.seed(0)
for k, m in enumerate(mvals9, 1):
    tt, dd, exd = controller(P_group, m)
    err = float((dd != y).mean()); mt = float(tt.mean())
    ftm = float(np.interp(-err, -fe_group, ft))
    saved = ftm - mt
    pos = exd & (dd == 1); neg = exd & (dd == 0)
    ppv = float((y[pos] == 1).mean()) if pos.sum() else float("nan")
    npv = float((y[neg] == 0).mean()) if neg.sum() else float("nan")
    # condition-level cluster bootstrap: resample 18 conditions with replacement,
    # pool their organoids, recompute (mean_time, error)
    B = 500
    bt = np.empty(B); be = np.empty(B)
    for b in range(B):
        bc = np.random.choice(cond_list, size=n_cond, replace=True)
        idx = np.concatenate([cond_to_idx[c] for c in bc])
        bt[b] = tt[idx].mean(); be[b] = (dd[idx] != y[idx]).mean()
    lo_t, hi_t = np.percentile(bt, [2.5, 97.5]); lo_e, hi_e = np.percentile(be, [2.5, 97.5])
    rows_ops.append((f"OP{k}", round(m, 3), round(err, 4), round(lo_e, 4), round(hi_e, 4),
                      round(mt, 2), round(lo_t, 2), round(hi_t, 2), round(ftm, 2), round(saved, 2),
                      round(ppv, 4), round(npv, 4)))
    print(f"  OP{k}: m={m:.3f}  err={err:.3f} [{lo_e:.3f},{hi_e:.3f}]  "
          f"t_ctrl={mt:.1f} [{lo_t:.1f},{hi_t:.1f}]h  t_fixed={ftm:.1f}h  saved={saved:.1f}h  "
          f"PPV={ppv:.3f}  NPV={npv:.3f}")

with open(os.path.join(OUT, "table_matched_ops_grouped.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["OP", "band_margin", "err", "err_lo95_condCB", "err_hi95_condCB",
                "t_ctrl", "t_ctrl_lo95_condCB", "t_ctrl_hi95_condCB", "t_fixed", "hours_saved",
                "PPV", "NPV"])
    for r in rows_ops:
        w.writerow(r)
print("wrote table_matched_ops_grouped.csv")

n_dom = sum(1 for r in rows_ops if r[-1] > 0)
print(f"\nSUMMARY: leave-condition-out controller dominates fixed-time at {n_dom}/9 matched operating points.")
print("Compare against the stratified (organoid-CV) headline 9/9 in the main text to bound leakage sensitivity.")
